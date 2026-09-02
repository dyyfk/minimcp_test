"""Evaluate live and frozen P16 scores on dependent multi-turn fixtures."""
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


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(pattern: str):
    ids, arrays, paths = [], [], sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(f"no feature shards: {pattern}")
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        ids.extend(str(value) for value in archive["ids"])
        arrays.append(archive["X"].astype(np.float32, copy=False))
    index = pd.DataFrame({"id": ids, "feature_row": range(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    matrix = np.concatenate(arrays)
    if matrix.shape[1] != 12288 or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid feature matrix {matrix.shape}")
    return index, matrix, paths


def score(path: Path, values: np.ndarray, require_inactive=False):
    artifact = json.loads(path.read_text())
    if require_inactive and artifact.get("activation_prohibited") is not True:
        raise RuntimeError("P16 artifact must be activation-prohibited")
    weights = np.asarray(artifact["w"], dtype=np.float64)
    if weights.shape != (values.shape[1],):
        raise RuntimeError("artifact/feature dimension mismatch")
    return values @ weights + float(artifact["b"])


def top_mask(values: np.ndarray, rate: float):
    count = int(round(len(values) * rate))
    mask = np.zeros(len(values), dtype=bool)
    order = np.lexsort((np.arange(len(values)), -values))
    mask[order[:count]] = True
    return mask


def metrics(y: np.ndarray, live: np.ndarray, candidate: np.ndarray):
    output = {"rows": len(y), "failures": int(y.sum()),
              "failure_rate": float(y.mean()), "budgets": {}}
    if np.unique(y).size == 2:
        output["live_auc"] = float(roc_auc_score(y, live))
        output["candidate_auc"] = float(roc_auc_score(y, candidate))
        output["auc_delta"] = output["candidate_auc"] - output["live_auc"]
    else:
        output.update(live_auc=None, candidate_auc=None, auc_delta=None)
    positives = max(1, int(y.sum()))
    for name, rate in RATES.items():
        live_mask, candidate_mask = top_mask(live, rate), top_mask(candidate, rate)
        output["budgets"][name] = {
            "live_precision": float(y[live_mask].mean()),
            "candidate_precision": float(y[candidate_mask].mean()),
            "precision_delta": float(y[candidate_mask].mean() -
                                     y[live_mask].mean()),
            "live_recall": float(y[live_mask].sum() / positives),
            "candidate_recall": float(y[candidate_mask].sum() / positives),
            "selection_agreement": float(np.mean(live_mask == candidate_mask)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs)
    judged = pd.read_parquet(args.judged)
    if pairs.id.duplicated().any() or judged.id.duplicated().any():
        raise RuntimeError("duplicate pair/judge IDs")
    feature_index, feature_values, feature_paths = load_features(args.features)
    rows = (pairs.merge(judged[["id", "adequate"]], on="id",
                        validate="one_to_one")
            .merge(feature_index, on="id", validate="one_to_one")
            .sort_values("id"))
    if rows.adequate.isna().any():
        raise RuntimeError("incomplete judged rows")
    scored_ids = set(rows.id.astype(str))
    unscored = sorted(set(pairs.id.astype(str)) - scored_ids)
    values = feature_values[rows.feature_row.to_numpy()]
    live = score(args.live_artifact, values)
    candidate = score(args.candidate_artifact, values, require_inactive=True)
    y = 1 - rows.adequate.astype(int).to_numpy()

    group_metrics = {}
    for pool in sorted(rows.target_pool.unique()):
        mask = rows.target_pool.to_numpy() == pool
        group_metrics[pool] = metrics(y[mask], live[mask], candidate[mask])
    language_metrics = {}
    for language in sorted(rows.language.unique()):
        mask = rows.language.to_numpy() == language
        language_metrics[language] = metrics(y[mask], live[mask], candidate[mask])

    valid_pools = [pool for pool, value in group_metrics.items()
                   if value["auc_delta"] is not None]
    point = (float(np.mean([group_metrics[pool]["auc_delta"]
                            for pool in valid_pools]))
             if valid_pools else None)
    rng = np.random.default_rng(71)
    bootstrap = []
    pool_values = rows.target_pool.to_numpy()
    for _ in range(5000):
        deltas = []
        for pool in valid_pools:
            indices = np.flatnonzero(pool_values == pool)
            sample = rng.choice(indices, len(indices), replace=True)
            if np.unique(y[sample]).size < 2:
                break
            deltas.append(roc_auc_score(y[sample], candidate[sample]) -
                          roc_auc_score(y[sample], live[sample]))
        if len(deltas) == len(valid_pools) and deltas:
            bootstrap.append(float(np.mean(deltas)))
    interval = ([float(np.mean(bootstrap)),
                 float(np.percentile(bootstrap, 2.5)),
                 float(np.percentile(bootstrap, 97.5))]
                if bootstrap else None)

    trace_paths = sorted(args.trace_dir.glob("controlled_multiturn.rank*.jsonl"))
    # Generation is resumable and appends retries.  Match the judge's contract:
    # the last record for an ID is authoritative.
    trace_by_id = {}
    for path in trace_paths:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                trace_by_id[str(record["id"])] = record
    traces = list(trace_by_id.values())
    usage_columns = ["prompt_tokens", "completion_tokens",
                     "cached_prompt_tokens"]
    usage = judged[usage_columns].fillna(0).sum().astype(int)
    judge_cost = float(
        (usage.prompt_tokens - usage.cached_prompt_tokens) * .75 / 1e6
        + usage.cached_prompt_tokens * .075 / 1e6
        + usage.completion_tokens * 4.5 / 1e6)
    deltas = [group_metrics[pool]["auc_delta"] for pool in valid_pools]
    decision_pass = bool(
        point is not None and point >= .015 and interval is not None and
        interval[1] > 0 and deltas and min(deltas) >= -.01)
    output = {
        "status": "dependent_multiturn_p16_evaluation",
        "selected_rows": len(pairs),
        "scored_rows": len(rows),
        "unscored_ids": unscored,
        "pooled": metrics(y, live, candidate),
        "by_pool": group_metrics,
        "by_language": language_metrics,
        "macro_pool_auc_delta": {
            "point": point,
            "source_stratified_bootstrap_95ci": interval,
            "valid_pools": valid_pools,
        },
        "decision": {
            "clears_preregistered_offline_gate": decision_pass,
            "requirements": ("macro delta >= .015, bootstrap lower bound > 0, "
                             "and no pool delta below -.01"),
            "activation_recommended": False,
            "reason": ("Template-generated dependent conversations remain "
                       "controlled evidence; organic shadow traffic is required."),
        },
        "execution": {
            "trace_rows": len(traces),
            "errors": int(sum(row.get("error") is not None for row in traces)),
            "carrier_eot_rate": float(np.mean([
                row.get("carrier_eot_seen", False) for row in traces])),
            "target_eot_rate": float(np.mean([
                row.get("target_eot_seen", False) for row in traces])),
        },
        "judge_usage": {
            **usage.to_dict(),
            "cost_usd_at_published_standard_rates": judge_cost,
            "pricing_checked_utc": "2026-09-02",
        },
        "provenance": {
            "pairs_sha256": sha256(args.pairs),
            "judged_sha256": sha256(args.judged),
            "feature_shards": {Path(path).name: sha256(path)
                               for path in feature_paths},
            "trace_shards": {path.name: sha256(path) for path in trace_paths},
            "live_artifact_sha256": sha256(args.live_artifact),
            "candidate_artifact_sha256": sha256(args.candidate_artifact),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
