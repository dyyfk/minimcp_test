"""One-shot evaluation of frozen live and distilled gates on P15."""
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
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
LIVE_SHA256 = "0e6494c2eeac9bcd86c10b5def3cbd32e98bb0765fa2fd8afc8c1b47915ea372"
CANDIDATE_SHA256 = "c85e0697788b2f8ce819fc963aa68c5a5ef34e0ae59c4f58b7917ccbf848dbb0"


def load_support():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_sha(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha(path)))
    return digest.hexdigest()


def top_mask(score, rate):
    count = int(round(len(score) * rate))
    mask = np.zeros(len(score), dtype=bool)
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    mask[order[:count]] = True
    return mask


def metric_block(target, candidate, live, rng, n_boot=5000):
    output = {
        "rows": len(target), "failure_rate": float(np.mean(target)),
        "live_auc": float(roc_auc_score(target, live)),
        "candidate_auc": float(roc_auc_score(target, candidate)),
    }
    output["auc_delta"] = output["candidate_auc"] - output["live_auc"]
    values = []
    indices = np.arange(len(target))
    for _ in range(n_boot):
        sample = rng.choice(indices, len(indices), replace=True)
        if np.unique(target[sample]).size < 2:
            continue
        values.append(roc_auc_score(target[sample], candidate[sample]) -
                      roc_auc_score(target[sample], live[sample]))
    output["auc_delta_bootstrap_95ci"] = [
        float(np.mean(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]
    output["budgets"] = {}
    positives = max(1, int(target.sum()))
    for name, rate in RATES.items():
        candidate_mask = top_mask(candidate, rate)
        live_mask = top_mask(live, rate)
        output["budgets"][name] = {
            "rate": rate,
            "live_precision": float(target[live_mask].mean()),
            "candidate_precision": float(target[candidate_mask].mean()),
            "precision_delta": float(target[candidate_mask].mean() -
                                     target[live_mask].mean()),
            "live_recall": float(target[live_mask].sum() / positives),
            "candidate_recall": float(
                target[candidate_mask].sum() / positives),
            "selection_agreement": float(np.mean(live_mask == candidate_mask)),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    support = load_support()
    live_sha = file_sha(args.live_artifact)
    candidate_sha = file_sha(args.candidate_artifact)
    if live_sha != LIVE_SHA256 or candidate_sha != CANDIDATE_SHA256:
        raise RuntimeError(f"artifact mismatch: live={live_sha}, candidate={candidate_sha}")

    selection = pd.read_parquet(args.selection)[["id", "pool"]]
    judged = (pd.read_parquet(args.judged)
              .drop_duplicates("id", keep="last")[["id", "adequate"]])
    if judged.adequate.isna().any():
        raise RuntimeError("judge output contains unresolved rows")
    ids, arrays = [], []
    feature_paths = sorted(args.native_dir.glob("prospective_native_feats.rank*.npz"))
    for path in feature_paths:
        archive = np.load(path, allow_pickle=True)
        ids.extend(str(value) for value in archive["ids"])
        arrays.append(archive["X"].astype(np.float32, copy=False))
    if not arrays:
        raise RuntimeError("no feature shards found")
    features = np.concatenate(arrays)
    feature_frame = pd.DataFrame({"id": ids, "feature_row": range(len(ids))})
    if feature_frame.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    rows = (selection.merge(judged, on="id", validate="one_to_one")
            .merge(feature_frame, on="id", validate="one_to_one")
            .sort_values("id"))
    if len(rows) != 400 or features.shape[1] != 3 * BLOCK:
        raise RuntimeError(f"expected 400x{3 * BLOCK}, got {len(rows)} / {features.shape}")
    x = features[rows.feature_row.to_numpy()]
    if not np.isfinite(x).all():
        raise RuntimeError("features contain non-finite values")
    live = support.artifact_score(args.live_artifact, x)
    artifact = json.loads(args.candidate_artifact.read_text())
    candidate = x[:, BLOCK:] @ np.asarray(artifact["w"]) + float(artifact["b"])
    target = 1 - rows.adequate.astype(int).to_numpy()

    rng = np.random.default_rng(46)
    result = {
        "status": "one_shot_frozen_source_disjoint_validation",
        "overall": metric_block(target, candidate, live, rng),
        "by_pool": {},
        "decision_rule": {
            "broad_support": "positive aggregate AUC delta and positive delta in both frozen source pools",
            "statistical_support": "aggregate bootstrap 95% CI lower bound above zero",
            "no_retuning": True,
        },
    }
    for pool in sorted(rows.pool.unique()):
        mask = rows.pool.to_numpy() == pool
        result["by_pool"][pool] = metric_block(
            target[mask], candidate[mask], live[mask], rng)
    result["decision"] = {
        "broad_support": bool(
            result["overall"]["auc_delta"] > 0 and all(
                value["auc_delta"] > 0 for value in result["by_pool"].values())),
        "statistical_support": bool(
            result["overall"]["auc_delta_bootstrap_95ci"][1] > 0),
    }
    trace_paths = sorted(args.native_dir.glob("prospective_native_traces.rank*.jsonl"))
    result["provenance"] = {
        "selection_sha256": file_sha(args.selection),
        "ordered_ids_sha256": hashlib.sha256(
            ("\n".join(rows.id) + "\n").encode()).hexdigest(),
        "trace_manifest_sha256": manifest_sha(trace_paths),
        "feature_manifest_sha256": manifest_sha(feature_paths),
        "judged_sha256": file_sha(args.judged),
        "live_artifact_sha256": live_sha,
        "candidate_artifact_sha256": candidate_sha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("receipt_sha256", file_sha(args.output))


if __name__ == "__main__":
    main()
