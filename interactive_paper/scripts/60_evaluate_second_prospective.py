"""One-shot P16 evaluation of the frozen robust ensemble versus live 8bq."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BLOCK = 4096
EXPECTED_ROWS = 450
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
LIVE_SHA = "0e6494c2eeac9bcd86c10b5def3cbd32e98bb0765fa2fd8afc8c1b47915ea372"
CANDIDATE_SHA = "4d75a506a59e6206e4687a6b3630b1dae54034f902d9136112689c9091eeec15"
RESULT_STATUS = "one_shot_second_prospective_source_disjoint_validation"


def support_module():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_sha(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def top_mask(score, rate):
    count = int(round(len(score) * rate))
    output = np.zeros(len(score), dtype=bool)
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    output[order[:count]] = True
    return output


def metrics(y, candidate, live, rng):
    live_auc = float(roc_auc_score(y, live))
    candidate_auc = float(roc_auc_score(y, candidate))
    boot = []
    indices = np.arange(len(y))
    for _ in range(5000):
        sample = rng.choice(indices, len(indices), replace=True)
        if np.unique(y[sample]).size < 2:
            continue
        boot.append(roc_auc_score(y[sample], candidate[sample]) -
                    roc_auc_score(y[sample], live[sample]))
    result = {
        "rows": len(y), "failure_rate": float(y.mean()),
        "live_auc": live_auc, "candidate_auc": candidate_auc,
        "auc_delta": candidate_auc - live_auc,
        "auc_delta_bootstrap_95ci": [float(np.mean(boot)),
                                     float(np.percentile(boot, 2.5)),
                                     float(np.percentile(boot, 97.5))],
        "budgets": {},
    }
    positives = max(1, int(y.sum()))
    for name, rate in RATES.items():
        lm, cm = top_mask(live, rate), top_mask(candidate, rate)
        result["budgets"][name] = {
            "rate": rate,
            "live_precision": float(y[lm].mean()),
            "candidate_precision": float(y[cm].mean()),
            "precision_delta": float(y[cm].mean() - y[lm].mean()),
            "live_recall": float(y[lm].sum() / positives),
            "candidate_recall": float(y[cm].sum() / positives),
            "selection_agreement": float(np.mean(lm == cm)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    support = support_module()
    if sha256(args.live_artifact) != LIVE_SHA or sha256(
            args.candidate_artifact) != CANDIDATE_SHA:
        raise RuntimeError("frozen artifact SHA mismatch")

    selection = pd.read_parquet(args.selection)[["id", "pool"]]
    judged_full = (pd.read_parquet(args.judged)
                   .drop_duplicates("id", keep="last"))
    if judged_full.adequate.isna().any():
        raise RuntimeError("judge output has unresolved rows")
    ids, arrays = [], []
    feature_paths = sorted(args.native_dir.glob(
        "prospective_native_feats.rank*.npz"))
    for path in feature_paths:
        archive = np.load(path, allow_pickle=True)
        ids += [str(value) for value in archive["ids"]]
        arrays.append(archive["X"].astype(np.float32, copy=False))
    features = np.concatenate(arrays)
    feature_rows = pd.DataFrame({"id": ids, "row": range(len(ids))})
    if feature_rows.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    rows = (selection.merge(judged_full[["id", "adequate"]], on="id",
                            validate="one_to_one")
            .merge(feature_rows, on="id", validate="one_to_one")
            .sort_values("id"))
    if len(rows) != EXPECTED_ROWS or features.shape[1] != 3 * BLOCK:
        raise RuntimeError(f"expected {EXPECTED_ROWS}x{3 * BLOCK}; got "
                           f"{len(rows)} and {features.shape}")
    x = features[rows.row.to_numpy()]
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite features")
    live = support.artifact_score(args.live_artifact, x)
    candidate_artifact = json.loads(args.candidate_artifact.read_text())
    candidate = x @ np.asarray(candidate_artifact["w"]) + float(
        candidate_artifact["b"])
    y = 1 - rows.adequate.astype(int).to_numpy()

    rng = np.random.default_rng(48)
    result = {
        "status": RESULT_STATUS,
        "overall": metrics(y, candidate, live, rng), "by_pool": {},
        "decision_rule": {
            "replacement_auc_gate": "macro mean source AUC delta >= .015",
            "broad_support": "positive AUC delta in all three frozen sources",
            "statistical_support": "source-stratified macro bootstrap 95% CI lower bound > 0",
            "no_evaluation_set_retuning": True,
        },
    }
    for pool in sorted(rows.pool.unique()):
        mask = rows.pool.to_numpy() == pool
        result["by_pool"][pool] = metrics(
            y[mask], candidate[mask], live[mask], rng)
    macro_point = float(np.mean([
        value["auc_delta"] for value in result["by_pool"].values()]))
    macro_boot = []
    pool_array = rows.pool.to_numpy()
    for _ in range(5000):
        deltas = []
        for pool in sorted(rows.pool.unique()):
            indices = np.flatnonzero(pool_array == pool)
            sample = rng.choice(indices, len(indices), replace=True)
            if np.unique(y[sample]).size < 2:
                break
            deltas.append(roc_auc_score(y[sample], candidate[sample]) -
                          roc_auc_score(y[sample], live[sample]))
        if len(deltas) == len(result["by_pool"]):
            macro_boot.append(float(np.mean(deltas)))
    result["macro_source_auc_delta"] = {
        "point": macro_point,
        "source_stratified_bootstrap_95ci": [
            float(np.mean(macro_boot)), float(np.percentile(macro_boot, 2.5)),
            float(np.percentile(macro_boot, 97.5))],
    }
    result["decision"] = {
        "replacement_auc_gate": bool(macro_point >= .015),
        "broad_support": bool(all(value["auc_delta"] > 0
                                  for value in result["by_pool"].values())),
        "statistical_support": bool(result["macro_source_auc_delta"]
                                    ["source_stratified_bootstrap_95ci"][1] > 0),
    }
    prompt = int(judged_full.prompt_tokens.fillna(0).sum())
    cached = int(judged_full.cached_prompt_tokens.fillna(0).sum())
    completion = int(judged_full.completion_tokens.fillna(0).sum())
    result["judge_usage"] = {
        "model": "gpt-5.4-mini", "prompt_tokens": prompt,
        "cached_prompt_tokens": cached, "completion_tokens": completion,
        "cost_usd_at_published_standard_rates": float(
            (prompt - cached) * .75 / 1e6 + cached * .075 / 1e6
            + completion * 4.5 / 1e6),
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
        "pricing_checked_utc": "2026-09-02",
    }
    trace_paths = sorted(args.native_dir.glob(
        "prospective_native_traces.rank*.jsonl"))
    result["provenance"] = {
        "selection_sha256": sha256(args.selection),
        "ordered_ids_sha256": hashlib.sha256(
            ("\n".join(rows.id) + "\n").encode()).hexdigest(),
        "trace_manifest_sha256": manifest_sha(trace_paths),
        "feature_manifest_sha256": manifest_sha(feature_paths),
        "judged_sha256": sha256(args.judged),
        "live_artifact_sha256": LIVE_SHA,
        "candidate_artifact_sha256": CANDIDATE_SHA,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
