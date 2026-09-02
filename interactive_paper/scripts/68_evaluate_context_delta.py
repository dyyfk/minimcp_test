"""Evaluate a prior-turn score subtraction on controlled multi-turn traffic.

The candidate is operationally cheap: the previous turn's already-computed
shadow score is retained and the follow-up score is
``current_shadow - beta * previous_shadow``.  No extra model forward is needed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(pattern):
    ids, values = [], []
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise RuntimeError(f"no feature shards: {pattern}")
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        ids.extend(str(value) for value in archive["ids"])
        values.append(archive["X"].astype(np.float32, copy=False))
    index = pd.DataFrame({"id": ids, "row": range(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    matrix = np.concatenate(values)
    if matrix.shape[1] != 12288 or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid feature matrix {matrix.shape}")
    return index, matrix, paths


def artifact_score(path, values, require_inactive=False):
    artifact = json.loads(Path(path).read_text())
    if require_inactive and artifact.get("activation_prohibited") is not True:
        raise RuntimeError("candidate artifact must remain activation-prohibited")
    weights = np.asarray(artifact["w"], dtype=np.float64)
    if weights.shape != (values.shape[1],):
        raise RuntimeError("artifact/feature dimension mismatch")
    return values @ weights + float(artifact["b"])


def top_mask(values, rate):
    count = int(round(len(values) * rate))
    mask = np.zeros(len(values), dtype=bool)
    order = np.lexsort((np.arange(len(values)), -np.asarray(values)))
    mask[order[:count]] = True
    return mask


def metrics(y, live, candidate):
    output = {
        "rows": len(y),
        "failure_rate": float(np.mean(y)),
        "live_auc": float(roc_auc_score(y, live)),
        "candidate_auc": float(roc_auc_score(y, candidate)),
        "budgets": {},
    }
    output["auc_delta"] = output["candidate_auc"] - output["live_auc"]
    positives = max(1, int(np.sum(y)))
    for name, rate in RATES.items():
        live_mask, candidate_mask = top_mask(live, rate), top_mask(candidate, rate)
        output["budgets"][name] = {
            "live_precision": float(np.mean(y[live_mask])),
            "candidate_precision": float(np.mean(y[candidate_mask])),
            "precision_delta": float(np.mean(y[candidate_mask]) -
                                     np.mean(y[live_mask])),
            "live_recall": float(np.sum(y[live_mask]) / positives),
            "candidate_recall": float(np.sum(y[candidate_mask]) / positives),
            "selection_agreement": float(np.mean(live_mask == candidate_mask)),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--carrier-features", required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--shadow-artifact", type=Path, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs)
    judged = pd.read_parquet(args.judged)
    if pairs.id.duplicated().any() or judged.id.duplicated().any():
        raise RuntimeError("duplicate pair/judge IDs")
    target_index, target_values, target_paths = load_features(
        args.target_features)
    carrier_index, carrier_values, carrier_paths = load_features(
        args.carrier_features)
    rows = (pairs.merge(judged[["id", "adequate"]], on="id",
                        validate="one_to_one")
            .merge(target_index.rename(columns={"row": "target_row"}),
                   on="id", validate="one_to_one")
            .merge(carrier_index.rename(columns={"row": "carrier_row"}),
                   on="id", validate="one_to_one")
            .sort_values("id"))
    if len(rows) != len(pairs) or rows.adequate.isna().any():
        raise RuntimeError("incomplete pair/judge/feature join")
    target = target_values[rows.target_row.to_numpy()]
    carrier = carrier_values[rows.carrier_row.to_numpy()]
    live = artifact_score(args.live_artifact, target)
    shadow_target = artifact_score(args.shadow_artifact, target,
                                   require_inactive=True)
    shadow_carrier = artifact_score(args.shadow_artifact, carrier,
                                    require_inactive=True)
    candidate = shadow_target - args.beta * shadow_carrier
    y = 1 - rows.adequate.astype(int).to_numpy()
    pools = rows.target_pool.to_numpy()
    by_pool = {}
    for pool in sorted(set(pools)):
        mask = pools == pool
        by_pool[pool] = metrics(y[mask], live[mask], candidate[mask])

    rng = np.random.default_rng(68)
    bootstrap = []
    for _ in range(5000):
        deltas = []
        for pool in sorted(set(pools)):
            indices = np.flatnonzero(pools == pool)
            sample = rng.choice(indices, len(indices), replace=True)
            if np.unique(y[sample]).size < 2:
                break
            deltas.append(roc_auc_score(y[sample], candidate[sample]) -
                          roc_auc_score(y[sample], live[sample]))
        if len(deltas) == len(set(pools)):
            bootstrap.append(float(np.mean(deltas)))
    deltas = [value["auc_delta"] for value in by_pool.values()]
    interval = [float(np.mean(bootstrap)),
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5))]
    output = {
        "status": "context_delta_evaluation",
        "rows": len(rows),
        "candidate": {
            "formula": "current_p16_score - beta * prior_p16_score",
            "beta": args.beta,
            "extra_model_forwards": 0,
            "first_turn_behavior": "unchanged",
        },
        "pooled": metrics(y, live, candidate),
        "by_target_pool": by_pool,
        "macro_auc_delta": {
            "point": float(np.mean(deltas)),
            "source_stratified_bootstrap_95ci": interval,
        },
        "decision": {
            "broad_support": bool(all(delta > 0 for delta in deltas)),
            "statistical_support": bool(interval[1] > 0),
            "clears_point_threshold": bool(np.mean(deltas) >= .015),
            "activation_recommended": False,
        },
        "provenance": {
            "pairs_sha256": sha256(args.pairs),
            "judged_sha256": sha256(args.judged),
            "live_artifact_sha256": sha256(args.live_artifact),
            "shadow_artifact_sha256": sha256(args.shadow_artifact),
            "target_feature_shards": {Path(path).name: sha256(path)
                                      for path in target_paths},
            "carrier_feature_shards": {Path(path).name: sha256(path)
                                       for path in carrier_paths},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
