"""Fit a follow-up-only failure probe on native contextual features."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


BLOCK = 4096
BLOCKS = {
    "p3a": np.arange(BLOCK, 3 * BLOCK),
    "all": np.arange(0, 3 * BLOCK),
}
CS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(pattern):
    ids, matrices, paths = [], [], sorted(glob.glob(str(pattern)))
    if not paths:
        raise RuntimeError(f"no feature shards: {pattern}")
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        ids.extend(str(value) for value in archive["ids"])
        matrices.append(archive["X"].astype(np.float32, copy=False))
    index = pd.DataFrame({"id": ids, "row": range(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    values = np.concatenate(matrices)
    if values.shape[1] != 3 * BLOCK or not np.isfinite(values).all():
        raise RuntimeError(f"invalid feature matrix: {values.shape}")
    return index, values, paths


def artifact_score(path, values):
    artifact = json.loads(Path(path).read_text())
    weights = np.asarray(artifact["w"], dtype=np.float64)
    if weights.shape != (values.shape[1],):
        raise RuntimeError("live artifact/feature dimension mismatch")
    return values @ weights + float(artifact["b"])


def top_mask(values, rate):
    count = int(round(len(values) * rate))
    mask = np.zeros(len(values), dtype=bool)
    order = np.lexsort((np.arange(len(values)), -np.asarray(values)))
    mask[order[:count]] = True
    return mask


def pool_metrics(rows, y, baseline, candidate):
    output = {}
    for pool in sorted(rows.target_pool.unique()):
        mask = rows.target_pool.eq(pool).to_numpy()
        if np.unique(y[mask]).size < 2:
            output[pool] = {"rows": int(mask.sum()),
                            "failure_rate": float(np.mean(y[mask])),
                            "auc_available": False}
            continue
        live_auc = float(roc_auc_score(y[mask], baseline[mask]))
        candidate_auc = float(roc_auc_score(y[mask], candidate[mask]))
        output[pool] = {
            "rows": int(mask.sum()),
            "failure_rate": float(np.mean(y[mask])),
            "auc_available": True,
            "live_auc": live_auc,
            "candidate_auc": candidate_auc,
            "auc_delta": candidate_auc - live_auc,
        }
    return output


def fit_fold(train, test, values, y, columns, c):
    model = LogisticRegression(C=c, max_iter=5000, tol=1e-5)
    model.fit(values[train][:, columns], y[train])
    return test, model.decision_function(values[test][:, columns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs)
    judged = pd.read_parquet(args.judged)
    index, values, feature_paths = load_features(args.features)
    rows = (pairs.merge(judged[["id", "adequate"]], on="id",
                        validate="one_to_one")
            .merge(index, on="id", validate="one_to_one")
            .sort_values("id").reset_index(drop=True))
    excluded = sorted(set(pairs.id.astype(str)) - set(rows.id))
    x = values[rows.row.to_numpy()]
    y = 1 - rows.adequate.astype(int).to_numpy()
    groups = rows.target_pool.astype(str).to_numpy()
    if len(set(groups)) < 5:
        raise RuntimeError("need at least five target-pool groups")
    live = artifact_score(args.live_artifact, x)
    folds = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=69).split(x, y, groups))
    tasks = [(name, c, train, test) for name in BLOCKS for c in CS
             for train, test in folds]
    fitted = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_fold)(train, test, x, y, BLOCKS[name], c)
        for name, c, train, test in tasks)
    sweep, offset = [], 0
    for name in BLOCKS:
        for c in CS:
            oof = np.full(len(rows), np.nan)
            for test, score in fitted[offset:offset + len(folds)]:
                oof[test] = score
            offset += len(folds)
            if np.isnan(oof).any():
                raise RuntimeError("incomplete OOF scores")
            pools = pool_metrics(rows, y, live, oof)
            available = [value for value in pools.values()
                         if value["auc_available"]]
            macro_auc = float(np.mean([value["candidate_auc"]
                                       for value in available]))
            macro_live = float(np.mean([value["live_auc"]
                                        for value in available]))
            balanced_mask = top_mask(oof, RATES["balanced"])
            sweep.append({
                "blocks": name,
                "C": c,
                "macro_pool_auc": macro_auc,
                "macro_live_auc": macro_live,
                "macro_auc_delta": macro_auc - macro_live,
                "pooled_auc": float(roc_auc_score(y, oof)),
                "pooled_live_auc": float(roc_auc_score(y, live)),
                "balanced_failure_precision": float(np.mean(
                    y[balanced_mask])),
                "pools": pools,
                "_oof": oof,
            })
    winner = max(sweep, key=lambda row: (
        row["macro_pool_auc"], row["balanced_failure_precision"]))
    winner_oof = winner.pop("_oof")
    for row in sweep:
        row.pop("_oof", None)
    winner_copy = {key: value for key, value in winner.items()
                   if key != "pools"}
    columns = BLOCKS[winner["blocks"]]
    final = LogisticRegression(C=winner["C"], max_iter=5000, tol=1e-5)
    final.fit(x[:, columns], y)

    rng = np.random.default_rng(69)
    pool_names = sorted(rows.target_pool.unique())
    bootstrap = []
    for _ in range(5000):
        deltas = []
        for pool in pool_names:
            indices = np.flatnonzero(rows.target_pool.eq(pool).to_numpy())
            sample = rng.choice(indices, len(indices), replace=True)
            if np.unique(y[sample]).size < 2:
                continue
            deltas.append(roc_auc_score(y[sample], winner_oof[sample]) -
                          roc_auc_score(y[sample], live[sample]))
        if deltas:
            bootstrap.append(float(np.mean(deltas)))
    usage = judged[["prompt_tokens", "completion_tokens",
                    "cached_prompt_tokens"]].fillna(0).sum().astype(int)
    cost = float((usage.prompt_tokens - usage.cached_prompt_tokens) * .75 / 1e6
                 + usage.cached_prompt_tokens * .075 / 1e6
                 + usage.completion_tokens * 4.5 / 1e6)
    output = {
        "status": "p21_context_native_grouped_oof",
        "selected_rows": len(pairs),
        "scored_rows": len(rows),
        "excluded_no_onset_ids": excluded,
        "failure_rate": float(np.mean(y)),
        "selection_rule": "max grouped-OOF macro target-pool AUC; 30% failure precision tie-break",
        "winner": winner_copy,
        "winner_pools": winner["pools"],
        "winner_macro_auc_delta_bootstrap_95ci": [
            float(np.mean(bootstrap)),
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "sweep": sweep,
        "judge_usage": {
            **usage.to_dict(),
            "cost_usd_at_published_standard_rates": cost,
            "pricing_checked_utc": "2026-09-02",
        },
        "provenance": {
            "pairs_sha256": sha256(args.pairs),
            "judged_sha256": sha256(args.judged),
            "live_artifact_sha256": sha256(args.live_artifact),
            "feature_shards": {Path(path).name: sha256(path)
                               for path in feature_paths},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    modes = (["eot_mean8", "user_mean"] if winner["blocks"] == "p3a"
             else ["eot_last", "eot_mean8", "user_mean"])
    artifact = {
        "status": "shadow_only",
        "activation_prohibited": True,
        "live_gate_unchanged": True,
        "deployment_scope": "followup_turns_only",
        "feature_recipe": {"blocks": modes, "dimension": len(columns)},
        "model": {"type": "logistic_regression", "C": winner["C"]},
        "w": final.coef_[0].tolist(),
        "b": float(final.intercept_[0]),
        "validation": {
            "stage": "training_grouped_oof_only",
            "macro_auc_delta": winner["macro_auc_delta"],
            "result_sha256": sha256(args.output),
            "independent_acceptance_required": True,
        },
        "provenance": output["provenance"],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "winner": winner_copy,
        "bootstrap_95ci": output["winner_macro_auc_delta_bootstrap_95ci"],
        "result_sha256": sha256(args.output),
        "artifact_sha256": sha256(args.artifact),
        "scored_rows": len(rows),
        "excluded_no_onset": len(excluded),
        "judge_cost_usd": cost,
    }, indent=2))


if __name__ == "__main__":
    main()
