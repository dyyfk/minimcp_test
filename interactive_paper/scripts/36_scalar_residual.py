"""Test a low-capacity nonlinear residual on hidden-state summary scalars.

The high-dimensional branch is the P3a winner (raw eot_mean8 + user_mean).
The residual sees only norms, moments, and cross-block geometry, so it can
model conditioning/domain effects without fitting a 12k-dimensional MLP.
All candidate scores are outer source-family-grouped OOF predictions.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
BLOCK = 4096


def load_p3a_module():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scalar_features(x):
    parts = [x[:, j * BLOCK:(j + 1) * BLOCK].astype(np.float64, copy=False)
             for j in range(3)]
    out, names = [], []
    for j, part in enumerate(parts):
        for name, value in (
                ("mean", part.mean(axis=1)),
                ("std", part.std(axis=1)),
                ("rms", np.sqrt(np.mean(part * part, axis=1))),
                ("maxabs", np.max(np.abs(part), axis=1))):
            out.append(value)
            names.append(f"b{j}_{name}")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pa, pb = parts[a], parts[b]
        denom = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1)
        out.append(np.sum(pa * pb, axis=1) / np.maximum(denom, 1e-12))
        names.append(f"cos_b{a}_b{b}")
        out.append(np.sqrt(np.mean((pa - pb) ** 2, axis=1)))
        names.append(f"rmsdiff_b{a}_b{b}")
    return np.column_stack(out).astype(np.float32), names


def logit(p):
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    return np.log(p) - np.log1p(-p)


def zfit(values):
    return float(np.mean(values)), float(np.std(values))


def zapply(values, center, scale):
    return (values - center) / max(scale, 1e-8)


def fold_fit(train, test, x, scalar, y, leaves):
    cols = np.arange(BLOCK, 3 * BLOCK)
    base = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
    base.fit(x[train][:, cols], y[train])
    base_score = base.predict_proba(x[test][:, cols])[:, 1]
    residual = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_leaf_nodes=leaves, max_iter=150, learning_rate=.05,
            min_samples_leaf=50, l2_regularization=1., random_state=42))
    residual.fit(scalar[train], y[train])
    residual_score = residual.predict_proba(scalar[test])[:, 1]
    return test, base_score, residual_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=15)
    args = parser.parse_args()

    p3a = load_p3a_module()
    x, y, ids, groups, blocks = p3a.collect_training(args.data_dir)
    scalar, scalar_names = scalar_features(x)
    expert_df = pd.read_parquet(args.expert_labels)
    ecol = "adequate" if "adequate" in expert_df else "expert_ok"
    expert = (expert_df.dropna(subset=[ecol]).drop_duplicates("id", keep="last")
              .set_index("id")[ecol].astype(int).to_dict())
    paired = np.array([i for i, row_id in enumerate(ids) if row_id in expert])
    gain = np.array([expert[ids[i]] - (1 - y[i]) for i in paired])
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))

    leaves_grid = (3, 7, 15)
    tasks = [(leaves, train, test) for leaves in leaves_grid
             for train, test in cv]
    outputs = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fold_fit)(train, test, x, scalar, y, leaves)
        for leaves, train, test in tasks)

    candidates = []
    oof_by_candidate = {}
    base_oof = None
    for j, leaves in enumerate(leaves_grid):
        base = np.full(len(y), np.nan)
        residual = np.full(len(y), np.nan)
        for test, bs, rs in outputs[j * 5:(j + 1) * 5]:
            base[test], residual[test] = bs, rs
        if base_oof is None:
            base_oof = base
        bc, bs = zfit(logit(base))
        rc, rs = zfit(logit(residual))
        bz = zapply(logit(base), bc, bs)
        rz = zapply(logit(residual), rc, rs)
        for alpha in (0., .1, .25, .5, .75, 1.):
            score = (1 - alpha) * bz + alpha * rz
            name = f"hgb{leaves}_alpha{alpha:g}"
            row = {
                "name": name, "leaves": leaves, "alpha": alpha,
                "native_group_oof_auc": float(roc_auc_score(y, score)),
                "benefit_oof_auc": float(roc_auc_score(
                    (gain > 0).astype(int), score[paired])),
                "routing_oof_objective": p3a.routing_objective(
                    gain, score[paired]),
                "base_logit_center": bc, "base_logit_scale": bs,
                "residual_logit_center": rc, "residual_logit_scale": rs,
            }
            candidates.append(row)
            oof_by_candidate[name] = score
            print(name, row["native_group_oof_auc"],
                  row["benefit_oof_auc"], row["routing_oof_objective"],
                  flush=True)

    best_native = max(row["native_group_oof_auc"] for row in candidates)
    eligible = [row for row in candidates
                if row["native_group_oof_auc"] >= best_native - .005]
    winner = max(eligible, key=lambda row: row["routing_oof_objective"])
    print("winner", winner["name"], flush=True)

    cols = np.arange(BLOCK, 3 * BLOCK)
    base_model = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
    base_model.fit(x[:, cols], y)
    residual_model = make_pipeline(
        StandardScaler(), HistGradientBoostingClassifier(
            max_leaf_nodes=winner["leaves"], max_iter=150,
            learning_rate=.05, min_samples_leaf=50,
            l2_regularization=1., random_state=42))
    residual_model.fit(scalar, y)

    def candidate_score(xp):
        sp, _ = scalar_features(xp)
        b = base_model.predict_proba(xp[:, cols])[:, 1]
        r = residual_model.predict_proba(sp)[:, 1]
        bz = zapply(logit(b), winner["base_logit_center"],
                    winner["base_logit_scale"])
        rz = zapply(logit(r), winner["residual_logit_center"],
                    winner["residual_logit_scale"])
        return (1 - winner["alpha"]) * bz + winner["alpha"] * rz

    pools = {}
    for name, tag, col in [("internal_test", "testoff", None)] + [
            (pool, pool + "off", col) for pool, col in p3a.EXTERNAL]:
        pool_ids, xp = p3a.load_feats(args.data_dir, tag)
        failure = p3a.native_failure(args.data_dir, tag)
        expert_out = p3a.expert_outcomes(args.data_dir, name, col)
        keep = [i for i, row_id in enumerate(pool_ids)
                if row_id in failure and row_id in expert_out]
        kept_ids = [pool_ids[i] for i in keep]
        xp = xp[keep]
        local_ok = np.array([1 - failure[i] for i in kept_ids])
        expert_ok = np.array([expert_out[i] for i in kept_ids])
        gain_p = expert_ok - local_ok
        aligned = p3a.artifact_score(args.aligned_artifact, xp)
        candidate = candidate_score(xp)
        native_y = 1 - local_ok
        benefit_y = (gain_p > 0).astype(int)
        result = {
            "n": len(keep),
            "native_auc_aligned": float(roc_auc_score(native_y, aligned)),
            "native_auc_candidate": float(roc_auc_score(native_y, candidate)),
            "native_auc_delta_ci": p3a.bootstrap_auc_delta(
                native_y, candidate, aligned),
            "benefit_auc_aligned": float(roc_auc_score(benefit_y, aligned)),
            "benefit_auc_candidate": float(roc_auc_score(benefit_y, candidate)),
            "benefit_auc_delta_ci": p3a.bootstrap_auc_delta(
                benefit_y, candidate, aligned),
            "budgets": {},
        }
        for tier, rate in RATES.items():
            af = p3a.top_mask(aligned, rate)
            cf = p3a.top_mask(candidate, rate)
            result["budgets"][tier] = {
                "aligned_accuracy": float(
                    np.where(af, expert_ok, local_ok).mean()),
                "candidate_accuracy": float(
                    np.where(cf, expert_ok, local_ok).mean()),
                "candidate_vs_aligned_delta_ci": p3a.bootstrap_cascade_delta(
                    local_ok, expert_ok, candidate, aligned, rate),
            }
        pools[name] = result

    out = {
        "inputs": {
            "aligned_sha256": p3a.sha256(args.aligned_artifact),
            "expert_labels_sha256": p3a.sha256(args.expert_labels),
        },
        "train": {"n": len(y), "groups": len(set(groups)),
                  "paired_expert_n": len(paired), "scalar_features": scalar_names,
                  "blocks": blocks},
        "selection_rule": "max routing OOF among configs within .005 of best native grouped OOF AUC",
        "sweep": candidates,
        "winner": winner,
        "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
