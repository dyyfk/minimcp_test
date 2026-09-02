"""Distill the frozen semantic+RTJ teacher into one hidden-state pass.

Candidate selection uses only source-family-grouped OOF scores on the fixed
1,000-row semantic pilot.  The official-native external pools are evaluated
once after the block/regularization/blend winner is frozen.
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


BLOCK = 4096
BLOCKS = {
    "eot_last": np.arange(0, BLOCK),
    "p3a": np.arange(BLOCK, 3 * BLOCK),
    "all": np.arange(0, 3 * BLOCK),
}
ALPHAS = (100., 1000., 10000.)
BLENDS = (0., .1, .25, .5, .75)
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}


def load_module(filename, name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def zapply(values, center, scale):
    return (np.asarray(values) - center) / max(scale, 1e-8)


def zfit(values):
    return float(np.mean(values)), float(np.std(values))


def fit_base_fold(train, test, x, y):
    model = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
    model.fit(x[train][:, BLOCK:], y[train])
    return test, model.decision_function(x[test][:, BLOCK:])


def fit_teacher_fold(train, test, x, pilot_index, teacher, columns, alpha):
    train_rows = np.flatnonzero(np.isin(pilot_index, train))
    test_rows = np.flatnonzero(np.isin(pilot_index, test))
    model = Ridge(alpha=alpha)
    model.fit(x[pilot_index[train_rows]][:, columns], teacher[train_rows])
    return test_rows, model.predict(x[pilot_index[test_rows]][:, columns])


def metrics(p3a, local, expert, score):
    gain = expert - local
    return {
        "native_auc": float(roc_auc_score(1 - local, score)),
        "benefit_auc": float(roc_auc_score(gain > 0, score)),
        "routing_objective": p3a.routing_objective(gain, score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--train-semantic", type=Path, required=True)
    parser.add_argument("--train-rtj", type=Path, required=True)
    parser.add_argument("--semantic-result", type=Path, required=True)
    parser.add_argument("--fusion-result", type=Path, required=True)
    parser.add_argument("--external-selection", type=Path, required=True)
    parser.add_argument("--external-semantic", type=Path, required=True)
    parser.add_argument("--external-rtj", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    p3a = load_module("35_feature_conditioning.py", "feature_conditioning")
    ptrue_mod = load_module("45_text_ptrue_fusion.py", "text_ptrue")
    target = json.loads(args.semantic_result.read_text())["winner"]["target"]
    frozen = json.loads(args.fusion_result.read_text())["winner"]
    train = (pd.read_parquet(args.train_selection)
             .merge(pd.read_parquet(args.train_semantic), on="id",
                    validate="one_to_one")
             .merge(ptrue_mod.read_signal(args.train_rtj, "p_yes_rtj"),
                    on="id", validate="one_to_one"))
    x, y, ids, groups, _ = p3a.collect_training(args.data_dir)
    positions = {row_id: index for index, row_id in enumerate(ids)}
    train = train[train.id.isin(positions)].copy()
    pilot_index = np.array([positions[row_id] for row_id in train.id])
    local = 1 - y[pilot_index]
    expert = train["adequate"].astype(int).to_numpy()
    semantic = zapply(train[target], frozen["semantic_center"],
                      frozen["semantic_scale"])
    rtj = zapply(-logit(train.p_yes), frozen["ptrue_center"],
                 frozen["ptrue_scale"])
    teacher = .5 * semantic + .5 * rtj

    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    base_outputs = joblib.Parallel(n_jobs=min(args.jobs, 5), verbose=10)(
        joblib.delayed(fit_base_fold)(tr, te, x, y) for tr, te in cv)
    base_full_oof = np.full(len(y), np.nan)
    for indices, score in base_outputs:
        base_full_oof[indices] = score
    base = base_full_oof[pilot_index]
    base_center = zfit(base)
    bz = zapply(base, *base_center)

    tasks = [(name, alpha, tr, te) for name in BLOCKS for alpha in ALPHAS
             for tr, te in cv]
    outputs = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_teacher_fold)(
            tr, te, x, pilot_index, teacher, BLOCKS[name], alpha)
        for name, alpha, tr, te in tasks)
    candidates, teacher_oof = [], {}
    offset = 0
    for name in BLOCKS:
        for alpha in ALPHAS:
            prediction = np.full(len(train), np.nan)
            for rows, values in outputs[offset:offset + 5]:
                prediction[rows] = values
            offset += 5
            if np.isnan(prediction).any():
                raise RuntimeError(f"OOF prediction incomplete: {name}/{alpha}")
            teacher_oof[(name, alpha)] = prediction
            center = zfit(prediction)
            pz = zapply(prediction, *center)
            correlation = float(spearmanr(teacher, prediction).statistic)
            for blend in BLENDS:
                score = (1 - blend) * bz + blend * pz
                candidates.append({
                    "name": f"{name}_ridge{alpha:g}_blend{blend:g}",
                    "blocks": name, "alpha": alpha, "blend": blend,
                    "teacher_oof_spearman": correlation,
                    "prediction_center": center[0],
                    "prediction_scale": center[1],
                    **metrics(p3a, local, expert, score),
                })
    best_native = max(row["native_auc"] for row in candidates)
    eligible = [row for row in candidates
                if row["native_auc"] >= best_native - .005]
    winner = max(eligible, key=lambda row: row["routing_objective"])
    print("winner", winner, flush=True)

    base_model = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
    base_model.fit(x[:, BLOCK:], y)
    base_deploy_center = zfit(base_model.decision_function(
        x[pilot_index][:, BLOCK:]))
    teacher_model = Ridge(alpha=winner["alpha"])
    teacher_model.fit(x[pilot_index][:, BLOCKS[winner["blocks"]]], teacher)

    external = (pd.read_parquet(args.external_selection)
                .merge(pd.read_parquet(args.external_semantic), on="id",
                       validate="one_to_one")
                .merge(ptrue_mod.read_signal(args.external_rtj, "p_yes_rtj"),
                       on="id", validate="one_to_one"))
    pools = {}
    for pool, frame in external.groupby("pool", sort=True):
        pool_ids, xp = p3a.load_feats(args.data_dir, f"{pool}off")
        pool_positions = {row_id: i for i, row_id in enumerate(pool_ids)}
        frame = frame[frame.id.isin(pool_positions)].copy()
        xp = xp[[pool_positions[row_id] for row_id in frame.id]]
        aligned = p3a.artifact_score(args.aligned_artifact, xp)
        base_external = zapply(base_model.decision_function(xp[:, BLOCK:]),
                               *base_deploy_center)
        teacher_external = zapply(
            teacher_model.predict(xp[:, BLOCKS[winner["blocks"]]]),
            winner["prediction_center"], winner["prediction_scale"])
        candidate = ((1 - winner["blend"]) * base_external +
                     winner["blend"] * teacher_external)
        lo = frame.native_ok.astype(int).to_numpy()
        eo = frame.expert_ok.astype(int).to_numpy()
        ny, by = 1 - lo, (eo - lo) > 0
        result = {
            "n": len(frame),
            "native_auc_aligned": float(roc_auc_score(ny, aligned)),
            "native_auc_candidate": float(roc_auc_score(ny, candidate)),
            "native_auc_delta_ci": p3a.bootstrap_auc_delta(
                ny, candidate, aligned),
            "benefit_auc_aligned": float(roc_auc_score(by, aligned)),
            "benefit_auc_candidate": float(roc_auc_score(by, candidate)),
            "benefit_auc_delta_ci": p3a.bootstrap_auc_delta(
                by, candidate, aligned), "budgets": {},
        }
        for tier, rate in RATES.items():
            aligned_mask = p3a.top_mask(aligned, rate)
            candidate_mask = p3a.top_mask(candidate, rate)
            result["budgets"][tier] = {
                "aligned_accuracy": float(np.where(aligned_mask, eo, lo).mean()),
                "candidate_accuracy": float(np.where(candidate_mask, eo, lo).mean()),
                "candidate_vs_aligned_delta_ci": p3a.bootstrap_cascade_delta(
                    lo, eo, candidate, aligned, rate),
            }
        pools[pool] = result
    result = {
        "signal": "single-pass ridge distillation of frozen semantic+RTJ teacher",
        "selection_rule": "max routing OOF among candidates within .005 of best native OOF AUC",
        "train_n": len(train), "target": target,
        "teacher_definition": ".5*z(two-sample semantic)+.5*z(RTJ uncertainty)",
        "teacher_direct_metrics": metrics(p3a, local, expert, teacher),
        "sweep": candidates, "winner": winner, "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
