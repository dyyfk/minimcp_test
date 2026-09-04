"""Export the frozen P7 scorer as an explicitly shadow-only artifact."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


BLOCK = 4096


def load_p3a():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fusion-result", type=Path, required=True)
    parser.add_argument("--semantic-result", type=Path, required=True)
    parser.add_argument("--rtj-result", type=Path, required=True)
    parser.add_argument("--latency-result", type=Path)
    parser.add_argument(
        "--p3a-artifact", type=Path,
        help="Preserve coefficients from an already-frozen shadow artifact.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p3a = load_p3a()
    fusion = json.loads(args.fusion_result.read_text())
    semantic = json.loads(args.semantic_result.read_text())
    rtj = json.loads(args.rtj_result.read_text())
    latency = (json.loads(args.latency_result.read_text())
               if args.latency_result else None)
    winner = fusion["winner"]
    if winner["kind"] != "fixed":
        raise RuntimeError("expected the fixed three-way P7 winner")

    x, y, _ids, groups, blocks = p3a.collect_training(args.data_dir)
    cols = np.arange(BLOCK, 3 * BLOCK)
    if args.p3a_artifact:
        p3a_coefficients = json.loads(args.p3a_artifact.read_text())["p3a"]
    else:
        model = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
        model.fit(x[:, cols], y)
        p3a_coefficients = {
            "w": model.coef_[0].tolist(), "b": float(model.intercept_[0])}
    pools = fusion["pools"]
    external = list(pools.values())

    artifact = {
        "status": "shadow_only",
        "activation_prohibited": True,
        "reason": "aggregate AUC gate passed; fixed-set latency is high and online routing lift is not validated",
        "feature_recipe": {
            "blocks": ["eot_mean8", "user_mean"], "dimension": 2 * BLOCK,
            "training_rows": len(y), "source_groups": len(set(groups)),
            "training_blocks": blocks, "C": 3e-4,
        },
        "p3a": p3a_coefficients,
        "semantic": {
            "extra_native_answers": 2, "temperature": .7, "top_k": 20,
            "metric": semantic["winner"]["target"],
            "cosine_component_threshold": .70,
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "center": winner["semantic_center"],
            "scale": winner["semantic_scale"],
        },
        "rtj": {
            "pipeline": "audio -> MiniCPM-o verbatim transcript -> text p(True)",
            "asr_prompt": "Transcribe the speech in the audio verbatim. Output ONLY the transcription, with no commentary or answer.",
            "ptrue_prompt": "repository PTRUE_PRE exact first-token Yes/No mass",
            "center": winner["ptrue_center"],
            "scale": winner["ptrue_scale"],
            "external_latency_seconds": {"median": .6193, "mean": .9011,
                                          "p90": 1.5580},
        },
        "fusion": {
            "p3a_weight": 1 - winner["semantic_weight"] - winner["ptrue_weight"],
            "semantic_weight": winner["semantic_weight"],
            "rtj_weight": winner["ptrue_weight"],
            "p3a_center": winner["deploy_base_center"],
            "p3a_scale": winner["deploy_base_scale"],
        },
        "threshold_policy": {
            "kind": "rolling exact-quantile by language bucket",
            "rates": [.15, .30, .50],
            "note": "do not reuse live gate thresholds on the fused score",
        },
        "validation": {
            "mean_external_native_auc_delta": float(np.mean([
                row["native_auc_candidate"] - row["native_auc_aligned"]
                for row in external])),
            "mean_external_benefit_auc_delta": float(np.mean([
                row["benefit_auc_candidate"] - row["benefit_auc_aligned"]
                for row in external])),
            "mean_external_cascade_delta": {
                tier: float(np.mean([
                    row["budgets"][tier]["candidate_accuracy"] -
                    row["budgets"][tier]["aligned_accuracy"]
                    for row in external]))
                for tier in ("conservative", "balanced", "aggressive")},
            "fusion_result_sha256": p3a.sha256(args.fusion_result),
            "semantic_result_sha256": p3a.sha256(args.semantic_result),
            "rtj_result_sha256": p3a.sha256(args.rtj_result),
        },
        "shadow_logging_required": [
            "language", "p3a_score", "semantic_score", "rtj_score",
            "fused_score", "latency_ms", "realized_escalation",
            "local_outcome", "expert_outcome"],
    }
    if latency:
        artifact["semantic"]["fixed_50_serial_latency_seconds"] = latency[
            "latency_seconds"]["semantic_two_sample_serial_measured"]
        artifact["latency_benchmark"] = {
            "fixed_rows": latency["fixed_rows"],
            "rows_per_pool": latency["rows_per_pool"],
            "semantic_plus_rtj_serial_estimate_seconds": latency[
                "latency_seconds"]["semantic_plus_rtj_serial_estimate"],
            "three_replica_parallel_estimate_seconds": latency[
                "latency_seconds"]["three_replica_parallel_estimate"],
            "notes": latency["notes"],
            "receipt_sha256": p3a.sha256(args.latency_result),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print("wrote", args.output, "sha256", p3a.sha256(args.output))


if __name__ == "__main__":
    main()
