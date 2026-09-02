"""Evaluate frozen live/P16 scores on P19 controlled two-turn traffic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(paths, id_column="ids"):
    ids, arrays = [], []
    for path in sorted(paths):
        archive = np.load(path, allow_pickle=True)
        ids += [str(value) for value in archive[id_column]]
        arrays.append(archive["X"].astype(np.float32, copy=False))
    if not arrays:
        raise RuntimeError("no feature shards")
    values = np.concatenate(arrays)
    frame = pd.DataFrame({"feature_id": ids, "row": range(len(ids))})
    if frame.feature_id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    return frame, values


def score(path, values):
    artifact = json.loads(Path(path).read_text())
    return values @ np.asarray(artifact["w"]) + float(artifact["b"])


def top_mask(values, rate):
    count = int(round(len(values) * rate))
    mask = np.zeros(len(values), dtype=bool)
    order = np.lexsort((np.arange(len(values)), -np.asarray(values)))
    mask[order[:count]] = True
    return mask


def metrics(y, live, candidate):
    result = {
        "rows": len(y), "failure_rate": float(y.mean()),
        "live_auc": float(roc_auc_score(y, live)),
        "candidate_auc": float(roc_auc_score(y, candidate)),
        "budgets": {},
    }
    result["auc_delta"] = result["candidate_auc"] - result["live_auc"]
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


def manifest_sha(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(Path(path).name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--context-feature-dir", type=Path, required=True)
    parser.add_argument("--context-trace-dir", type=Path, required=True)
    parser.add_argument("--context-judged", type=Path, required=True)
    parser.add_argument("--standalone-feature-dir", type=Path, required=True)
    parser.add_argument("--standalone-judged", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs)
    judged_context = pd.read_parquet(args.context_judged)
    judged_standalone = pd.read_parquet(args.standalone_judged)
    context_paths = sorted(args.context_feature_dir.glob(
        "controlled_multiturn_feats.rank*.npz"))
    standalone_paths = sorted(args.standalone_feature_dir.glob(
        "prospective_native_feats.rank*.npz"))
    context_index, context_values = load_features(context_paths)
    standalone_index, standalone_values = load_features(standalone_paths)
    context_index = context_index.rename(columns={"feature_id": "id",
                                                  "row": "context_row"})
    standalone_index = standalone_index.rename(
        columns={"feature_id": "target_id", "row": "standalone_row"})
    rows = (pairs.merge(judged_context[["id", "adequate"]].rename(
                columns={"adequate": "context_adequate"}), on="id",
                validate="one_to_one")
            .merge(judged_standalone[["id", "adequate"]].rename(
                columns={"id": "target_id",
                         "adequate": "standalone_adequate"}),
                on="target_id", validate="one_to_one")
            .merge(context_index, on="id", validate="one_to_one")
            .merge(standalone_index, on="target_id", validate="one_to_one")
            .sort_values("id"))
    if len(rows) != len(pairs) or rows[["context_adequate",
                                        "standalone_adequate"]].isna().any().any():
        raise RuntimeError("incomplete pair join")
    xc = context_values[rows.context_row.to_numpy()]
    xs = standalone_values[rows.standalone_row.to_numpy()]
    if xc.shape[1] != 12288 or not np.isfinite(xc).all():
        raise RuntimeError("invalid context features")
    live_c = score(args.live_artifact, xc)
    candidate_c = score(args.candidate_artifact, xc)
    live_s = score(args.live_artifact, xs)
    candidate_s = score(args.candidate_artifact, xs)
    yc = 1 - rows.context_adequate.astype(int).to_numpy()
    ys = 1 - rows.standalone_adequate.astype(int).to_numpy()
    rng = np.random.default_rng(67)
    outcome_bootstrap = []
    for _ in range(5000):
        sample = rng.choice(len(rows), len(rows), replace=True)
        outcome_bootstrap.append(float(np.mean(ys[sample] - yc[sample])))

    result = {
        "status": "controlled_multiturn_paired_evaluation",
        "rows": len(rows),
        "outcome_shift": {
            "standalone_adequate_rate": float(1 - ys.mean()),
            "context_adequate_rate": float(1 - yc.mean()),
            "adequate_rate_delta": float(ys.mean() - yc.mean()),
            "adequate_rate_delta_bootstrap_95ci": [
                float(np.mean(outcome_bootstrap)),
                float(np.percentile(outcome_bootstrap, 2.5)),
                float(np.percentile(outcome_bootstrap, 97.5)),
            ],
            "correct_to_wrong": int(np.sum((ys == 0) & (yc == 1))),
            "wrong_to_correct": int(np.sum((ys == 1) & (yc == 0))),
            "outcome_agreement": float(np.mean(ys == yc)),
        },
        "matched_standalone": metrics(ys, live_s, candidate_s),
        "contextual_target": metrics(yc, live_c, candidate_c),
        "by_target_pool": {},
        "score_context_shift": {
            "live_mean_delta": float(np.mean(live_c - live_s)),
            "candidate_mean_delta": float(np.mean(candidate_c - candidate_s)),
            "live_pair_correlation": float(np.corrcoef(live_s, live_c)[0, 1]),
            "candidate_pair_correlation": float(np.corrcoef(
                candidate_s, candidate_c)[0, 1]),
        },
    }
    pools = rows.target_pool.to_numpy()
    for pool in sorted(set(pools)):
        mask = pools == pool
        result["by_target_pool"][pool] = metrics(
            yc[mask], live_c[mask], candidate_c[mask])

    bootstrap = []
    for _ in range(5000):
        deltas = []
        for pool in sorted(set(pools)):
            indices = np.flatnonzero(pools == pool)
            sample = rng.choice(indices, len(indices), replace=True)
            if np.unique(yc[sample]).size < 2:
                break
            deltas.append(roc_auc_score(yc[sample], candidate_c[sample]) -
                          roc_auc_score(yc[sample], live_c[sample]))
        if len(deltas) == len(set(pools)):
            bootstrap.append(float(np.mean(deltas)))
    points = [value["auc_delta"]
              for value in result["by_target_pool"].values()]
    result["macro_context_auc_delta"] = {
        "point": float(np.mean(points)),
        "source_stratified_bootstrap_95ci": [
            float(np.mean(bootstrap)), float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5))],
    }
    pool_deltas = [value["auc_delta"]
                   for value in result["by_target_pool"].values()]
    result["decision"] = {
        "broad_support": bool(all(value > 0 for value in pool_deltas)),
        "statistical_support": bool(
            result["macro_context_auc_delta"]
            ["source_stratified_bootstrap_95ci"][1] > 0),
        "activation_recommended": False,
        "reason": "controlled context evidence is diagnostic, not organic; "
                  "candidate must also improve every source without a "
                  "material interval regression",
    }
    trace_paths = sorted(args.context_trace_dir.glob(
        "controlled_multiturn.rank*.jsonl"))
    traces = [json.loads(line) for path in trace_paths
              for line in path.read_text().splitlines() if line.strip()]
    result["execution"] = {
        "errors": int(sum(row.get("error") is not None for row in traces)),
        "carrier_eot_rate": float(np.mean([
            row.get("carrier_eot_seen", False) for row in traces])),
        "target_eot_rate": float(np.mean([
            row.get("target_eot_seen", False) for row in traces])),
        "elapsed_s": {
            "median": float(np.median([row["elapsed_s"] for row in traces])),
            "mean": float(np.mean([row["elapsed_s"] for row in traces])),
            "p90": float(np.quantile([row["elapsed_s"] for row in traces], .9)),
        },
    }
    usage = judged_context[["prompt_tokens", "completion_tokens",
                            "cached_prompt_tokens"]].fillna(0).sum().astype(int)
    result["judge_usage"] = {
        **usage.to_dict(),
        "cost_usd_at_published_standard_rates": float(
            (usage.prompt_tokens - usage.cached_prompt_tokens) * .75 / 1e6
            + usage.cached_prompt_tokens * .075 / 1e6
            + usage.completion_tokens * 4.5 / 1e6),
        "pricing_checked_utc": "2026-09-02",
    }
    result["provenance"] = {
        "pairs_sha256": sha256(args.pairs),
        "context_judged_sha256": sha256(args.context_judged),
        "standalone_judged_sha256": sha256(args.standalone_judged),
        "context_feature_manifest_sha256": manifest_sha(context_paths),
        "context_trace_manifest_sha256": manifest_sha(trace_paths),
        "standalone_feature_manifest_sha256": manifest_sha(standalone_paths),
        "live_artifact_sha256": sha256(args.live_artifact),
        "candidate_artifact_sha256": sha256(args.candidate_artifact),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
