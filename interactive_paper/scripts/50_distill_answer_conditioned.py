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
LIVE_ALIGNED_SHA256 = "0e6494c2eeac9bcd86c10b5def3cbd32e98bb0765fa2fd8afc8c1b47915ea372"


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
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--robustness-output", type=Path)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    p3a = load_module("35_feature_conditioning.py", "feature_conditioning")
    ptrue_mod = load_module("45_text_ptrue_fusion.py", "text_ptrue")
    aligned_sha256 = p3a.sha256(args.aligned_artifact)
    if aligned_sha256 != LIVE_ALIGNED_SHA256:
        raise RuntimeError(
            "aligned artifact is not the frozen live 8bq baseline: "
            f"{aligned_sha256}")
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
    pools, external_cache = {}, {}
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
        external_cache[pool] = {
            "x": xp[:, BLOCKS[winner["blocks"]]],
            "aligned": aligned, "base": base_external,
            "local": lo, "expert": eo,
        }
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
    if args.artifact:
        blend = winner["blend"]
        base_center_value, base_scale_value = base_deploy_center
        teacher_center = winner["prediction_center"]
        teacher_scale = winner["prediction_scale"]
        weight = ((1 - blend) * base_model.coef_[0] / base_scale_value +
                  blend * teacher_model.coef_ / teacher_scale)
        bias = float(
            (1 - blend) * (base_model.intercept_[0] - base_center_value) /
            base_scale_value + blend *
            (teacher_model.intercept_ - teacher_center) / teacher_scale)
        folded = x[pilot_index][:, BLOCK:] @ weight + bias
        unfurled = ((1 - blend) * zapply(
            base_model.decision_function(x[pilot_index][:, BLOCK:]),
            *base_deploy_center) + blend * zapply(
                teacher_model.predict(x[pilot_index][:, BLOCK:]),
                teacher_center, teacher_scale))
        fold_max_abs_diff = float(np.max(np.abs(folded - unfurled)))
        if fold_max_abs_diff > 1e-5:
            raise RuntimeError(
                f"algebraic coefficient fold mismatch: {fold_max_abs_diff}")
        validation_rows = list(pools.values())
        artifact = {
            "status": "shadow_only", "activation_prohibited": True,
            "live_gate_unchanged": True,
            "reason": "external AUC gate passed; online routing lift and score calibration are not yet validated",
            "feature_recipe": {
                "blocks": ["eot_mean8", "user_mean"],
                "dimension": 2 * BLOCK,
                "base_training_rows": len(y),
                "distillation_rows": len(train),
                "base_C": 3e-4,
                "teacher": result["teacher_definition"],
                "teacher_ridge_alpha": winner["alpha"],
                "teacher_blend": blend,
            },
            "w": weight.tolist(), "b": bias,
            "threshold_policy": {
                "kind": "rolling exact-quantile by language bucket",
                "rates": [.15, .30, .50],
                "note": "do not reuse live gate thresholds on the distilled score",
            },
            "validation": {
                "mean_external_native_auc_delta": float(np.mean([
                    row["native_auc_candidate"] - row["native_auc_aligned"]
                    for row in validation_rows])),
                "mean_external_benefit_auc_delta": float(np.mean([
                    row["benefit_auc_candidate"] - row["benefit_auc_aligned"]
                    for row in validation_rows])),
                "mean_external_cascade_delta": {
                    tier: float(np.mean([
                        row["budgets"][tier]["candidate_accuracy"] -
                        row["budgets"][tier]["aligned_accuracy"]
                        for row in validation_rows])) for tier in RATES},
                "result_sha256": p3a.sha256(args.output),
                "coefficient_fold_max_abs_diff": fold_max_abs_diff,
            },
            "provenance": {
                "aligned_artifact_sha256": aligned_sha256,
                "semantic_result_sha256": p3a.sha256(args.semantic_result),
                "fusion_result_sha256": p3a.sha256(args.fusion_result),
            },
            "shadow_logging_required": [
                "language", "live_score", "distilled_score", "latency_ms",
                "realized_escalation", "local_outcome", "expert_outcome"],
        }
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
        print("wrote", args.artifact, "sha256", p3a.sha256(args.artifact),
              flush=True)
    if args.robustness_output:
        pilot_groups = np.asarray(groups)[pilot_index]
        rows = []
        for held_out in sorted(set(pilot_groups)):
            keep = pilot_groups != held_out
            model = Ridge(alpha=winner["alpha"]).fit(
                x[pilot_index[keep]][:, BLOCKS[winner["blocks"]]],
                teacher[keep])
            pool_rows = {}
            for pool, cache in external_cache.items():
                distilled = zapply(
                    model.predict(cache["x"]),
                    winner["prediction_center"], winner["prediction_scale"])
                score = ((1 - winner["blend"]) * cache["base"] +
                         winner["blend"] * distilled)
                native = 1 - cache["local"]
                benefit = (cache["expert"] - cache["local"]) > 0
                row = {
                    "native_auc_delta": float(
                        roc_auc_score(native, score) -
                        roc_auc_score(native, cache["aligned"])),
                    "benefit_auc_delta": float(
                        roc_auc_score(benefit, score) -
                        roc_auc_score(benefit, cache["aligned"])),
                    "cascade_delta": {},
                }
                for tier, rate in RATES.items():
                    aligned_mask = p3a.top_mask(cache["aligned"], rate)
                    candidate_mask = p3a.top_mask(score, rate)
                    aligned_accuracy = np.where(
                        aligned_mask, cache["expert"], cache["local"]).mean()
                    candidate_accuracy = np.where(
                        candidate_mask, cache["expert"], cache["local"]).mean()
                    row["cascade_delta"][tier] = float(
                        candidate_accuracy - aligned_accuracy)
                pool_rows[pool] = row
            values = list(pool_rows.values())
            rows.append({
                "held_out_group": str(held_out),
                "held_out_rows": int(np.sum(~keep)),
                "mean_native_auc_delta": float(np.mean([
                    row["native_auc_delta"] for row in values])),
                "mean_benefit_auc_delta": float(np.mean([
                    row["benefit_auc_delta"] for row in values])),
                "mean_cascade_delta": {
                    tier: float(np.mean([
                        row["cascade_delta"][tier] for row in values]))
                    for tier in RATES},
                "positive_native_pools": int(sum(
                    row["native_auc_delta"] > 0 for row in values)),
                "pools": pool_rows,
            })

        def distribution(key):
            values = np.asarray([row[key] for row in rows])
            return {"min": float(values.min()),
                    "median": float(np.median(values)),
                    "max": float(values.max())}

        robustness = {
            "method": "leave one source family out of the 1,000-row teacher fit",
            "families": len(rows),
            "fixed_candidate": winner["name"],
            "mean_native_auc_delta": distribution("mean_native_auc_delta"),
            "mean_benefit_auc_delta": distribution("mean_benefit_auc_delta"),
            "mean_balanced_cascade_delta": {
                "min": float(min(row["mean_cascade_delta"]["balanced"]
                                 for row in rows)),
                "median": float(np.median([
                    row["mean_cascade_delta"]["balanced"] for row in rows])),
                "max": float(max(row["mean_cascade_delta"]["balanced"]
                                 for row in rows)),
            },
            "auc_gate_passes": int(sum(
                row["mean_native_auc_delta"] >= .015 for row in rows)),
            "all_five_native_positive": int(sum(
                row["positive_native_pools"] == 5 for row in rows)),
            "rows": rows,
        }
        args.robustness_output.parent.mkdir(parents=True, exist_ok=True)
        args.robustness_output.write_text(
            json.dumps(robustness, indent=2) + "\n")
        print("wrote", args.robustness_output,
              "sha256", p3a.sha256(args.robustness_output), flush=True)


if __name__ == "__main__":
    main()
