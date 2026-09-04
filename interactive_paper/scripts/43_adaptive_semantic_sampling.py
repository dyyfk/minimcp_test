"""Simulate requesting a second semantic sample only near the gate boundary.

The first- and second-sample score recipes are already frozen by script 42.
For each escalation budget, rows closest to the first-sample quantile boundary
receive the second sample; all other rows retain their first-sample score.
The gray-band fraction is selected on the paired training pilot only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


BLOCK = 4096
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
FRACTIONS = (0., .1, .2, .3, .5, .75, 1.)


def load_module(filename, name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score(labels, base, winner, deploy=False):
    prefix = "deploy_" if deploy else ""
    bc = winner[f"{prefix}base_center"]
    bs = winner[f"{prefix}base_scale"]
    bz = (base - bc) / max(bs, 1e-8)
    semantic = labels[winner["target"]].to_numpy(dtype=float)
    sz = ((semantic - winner["semantic_center"]) /
          max(winner["semantic_scale"], 1e-8))
    return (1 - winner["blend"]) * bz + winner["blend"] * sz


def adaptive_score(first, second, rate, fraction):
    n = len(first)
    k = int(round(n * rate))
    order = np.lexsort((np.arange(n), -np.asarray(first)))
    if k == 0:
        boundary = first[order[0]]
    elif k == n:
        boundary = first[order[-1]]
    else:
        boundary = (first[order[k - 1]] + first[order[k]]) / 2
    gray_n = int(round(n * fraction))
    gray = np.zeros(n, dtype=bool)
    if gray_n:
        distance_order = np.lexsort((np.arange(n), np.abs(first - boundary)))
        gray[distance_order[:gray_n]] = True
    hybrid = np.where(gray, second, first)
    return hybrid, gray


def fit_base_fold(train, test, x, y, cols):
    model = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
    model.fit(x[train][:, cols], y[train])
    return test, model.decision_function(x[test][:, cols])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--train-labels-k1", type=Path, required=True)
    parser.add_argument("--train-labels-k2", type=Path, required=True)
    parser.add_argument("--result-k1", type=Path, required=True)
    parser.add_argument("--result-k2", type=Path, required=True)
    parser.add_argument("--external-selection", type=Path, required=True)
    parser.add_argument("--external-labels-k1", type=Path, required=True)
    parser.add_argument("--external-labels-k2", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    p3a = load_module("35_feature_conditioning.py", "feature_conditioning")
    w1 = json.loads(args.result_k1.read_text())["winner"]
    w2 = json.loads(args.result_k2.read_text())["winner"]
    if w1["kind"] != "blend" or w2["kind"] != "blend":
        raise RuntimeError("adaptive simulation requires scalar blend winners")

    selection = pd.read_parquet(args.train_selection)
    l1 = pd.read_parquet(args.train_labels_k1)
    l2 = pd.read_parquet(args.train_labels_k2)
    train = (selection.merge(l1, on="id", validate="one_to_one")
             .merge(l2, on="id", suffixes=("_k1", "_k2"),
                    validate="one_to_one"))
    x, y, ids, groups, _blocks = p3a.collect_training(args.data_dir)
    id_to_index = {row_id: i for i, row_id in enumerate(ids)}
    train = train[train["id"].isin(id_to_index)].copy()
    index = np.array([id_to_index[row_id] for row_id in train["id"]])
    cols = np.arange(BLOCK, 3 * BLOCK)
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    fold_outputs = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_base_fold)(
            train_fold, test_fold, x, y, cols)
        for train_fold, test_fold in cv)
    base_oof = np.full(len(y), np.nan)
    for test_fold, values in fold_outputs:
        base_oof[test_fold] = values
    first = score(train, base_oof[index], {
        **w1, "target": w1["target"] + "_k1"})
    second = score(train, base_oof[index], {
        **w2, "target": w2["target"] + "_k2"})
    local_ok = 1 - y[index]
    expert_ok = train["adequate"].astype(int).to_numpy()
    gain = expert_ok - local_ok

    sweep = []
    for fraction in FRACTIONS:
        utilities = {}
        for tier, rate in RATES.items():
            hybrid, _gray = adaptive_score(first, second, rate, fraction)
            utilities[tier] = float(
                gain[p3a.top_mask(hybrid, rate)].sum() / len(gain))
        sweep.append({
            "gray_fraction": fraction,
            "average_extra_samples": 1 + fraction,
            "routing_objective": float(np.mean(list(utilities.values()))),
            "utilities": utilities,
        })
    best = max(row["routing_objective"] for row in sweep)
    winner = min((row for row in sweep
                  if row["routing_objective"] >= best - .001),
                 key=lambda row: row["gray_fraction"])
    print("winner", winner, flush=True)

    full = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
    full.fit(x[:, cols], y)
    external_selection = pd.read_parquet(args.external_selection)
    external = (external_selection
                .merge(pd.read_parquet(args.external_labels_k1), on="id",
                       validate="one_to_one")
                .merge(pd.read_parquet(args.external_labels_k2), on="id",
                       suffixes=("_k1", "_k2"), validate="one_to_one"))
    pools = {}
    for pool, frame in external.groupby("pool", sort=True):
        pool_ids, xp = p3a.load_feats(args.data_dir, f"{pool}off")
        positions = {row_id: i for i, row_id in enumerate(pool_ids)}
        frame = frame[frame["id"].isin(positions)].copy()
        xp = xp[[positions[row_id] for row_id in frame["id"]]]
        base = full.decision_function(xp[:, cols])
        first_ext = score(frame, base, {
            **w1, "target": w1["target"] + "_k1"}, deploy=True)
        second_ext = score(frame, base, {
            **w2, "target": w2["target"] + "_k2"}, deploy=True)
        aligned = p3a.artifact_score(args.aligned_artifact, xp)
        local = frame["native_ok"].astype(int).to_numpy()
        expert = frame["expert_ok"].astype(int).to_numpy()
        result = {"n": len(frame), "budgets": {}}
        for tier, rate in RATES.items():
            hybrid, gray = adaptive_score(
                first_ext, second_ext, rate, winner["gray_fraction"])
            af = p3a.top_mask(aligned, rate)
            hf = p3a.top_mask(hybrid, rate)
            f1 = p3a.top_mask(first_ext, rate)
            f2 = p3a.top_mask(second_ext, rate)
            result["budgets"][tier] = {
                "realized_average_extra_samples": float(1 + gray.mean()),
                "aligned_accuracy": float(np.where(af, expert, local).mean()),
                "one_sample_accuracy": float(np.where(f1, expert, local).mean()),
                "two_sample_accuracy": float(np.where(f2, expert, local).mean()),
                "adaptive_accuracy": float(np.where(hf, expert, local).mean()),
                "adaptive_vs_aligned_delta_ci": p3a.bootstrap_cascade_delta(
                    local, expert, hybrid, aligned, rate),
            }
        pools[pool] = result

    out = {
        "inputs": {
            "selection_sha256": p3a.sha256(args.external_selection),
            "labels_k1_sha256": p3a.sha256(args.external_labels_k1),
            "labels_k2_sha256": p3a.sha256(args.external_labels_k2),
            "result_k1_sha256": p3a.sha256(args.result_k1),
            "result_k2_sha256": p3a.sha256(args.result_k2),
        },
        "selection_rule": "smallest gray fraction within .001 routing objective of the best training-pilot fraction",
        "sweep": sweep, "winner": winner, "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
