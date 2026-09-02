"""Source-grouped low-rank probe sweep on the P3a feature blocks.

PCA is fit strictly inside each outer fold.  One 512-component decomposition
per fold is reused for all smaller dimensions and logistic regularization
values.  The six official-native evaluation pools are opened only after the
winner is frozen from grouped OOF native/routing metrics.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


BLOCK = 4096
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
DIMS = (128, 256, 512)
CS = (3e-4, 1e-3, 3e-3, 1e-2)


def load_p3a_module():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fit_fold(train, test, x, y):
    cols = np.arange(BLOCK, 3 * BLOCK)
    pca = PCA(n_components=max(DIMS), svd_solver="randomized",
              iterated_power=3, random_state=42)
    z_train = pca.fit_transform(x[train][:, cols])
    z_test = pca.transform(x[test][:, cols])
    scores = {}
    for dim in DIMS:
        for whiten in (False, True):
            scale = (np.sqrt(pca.explained_variance_[:dim])
                     if whiten else np.ones(dim))
            train_d = z_train[:, :dim] / np.maximum(scale, 1e-8)
            test_d = z_test[:, :dim] / np.maximum(scale, 1e-8)
            for c_value in CS:
                model = LogisticRegression(
                    C=c_value, max_iter=3000, tol=1e-5)
                model.fit(train_d, y[train])
                name = f"pca{dim}_{'white' if whiten else 'raw'}_C{c_value:g}"
                scores[name] = model.predict_proba(test_d)[:, 1]
    return test, scores, float(pca.explained_variance_ratio_.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    p3a = load_p3a_module()
    x, y, ids, groups, blocks = p3a.collect_training(args.data_dir)
    expert_df = pd.read_parquet(args.expert_labels)
    ecol = "adequate" if "adequate" in expert_df else "expert_ok"
    expert = (expert_df.dropna(subset=[ecol]).drop_duplicates("id", keep="last")
              .set_index("id")[ecol].astype(int).to_dict())
    paired = np.array([i for i, row_id in enumerate(ids) if row_id in expert])
    gain = np.array([expert[ids[i]] - (1 - y[i]) for i in paired])
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))

    outputs = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_fold)(train, test, x, y)
        for train, test in cv)
    names = sorted(outputs[0][1])
    rows = []
    for name in names:
        oof = np.full(len(y), np.nan)
        for test, scores, _variance in outputs:
            oof[test] = scores[name]
        row = {
            "name": name,
            "native_group_oof_auc": float(roc_auc_score(y, oof)),
            "benefit_oof_auc": float(roc_auc_score(
                (gain > 0).astype(int), oof[paired])),
            "routing_oof_objective": p3a.routing_objective(
                gain, oof[paired]),
        }
        rows.append(row)
        print(name, row["native_group_oof_auc"], row["benefit_oof_auc"],
              row["routing_oof_objective"], flush=True)
    best_native = max(row["native_group_oof_auc"] for row in rows)
    eligible = [row for row in rows
                if row["native_group_oof_auc"] >= best_native - .005]
    winner = max(eligible, key=lambda row: row["routing_oof_objective"])
    print("winner", winner, flush=True)

    fields = winner["name"].split("_")
    dim = int(fields[0].replace("pca", ""))
    whiten = fields[1] == "white"
    c_value = float(fields[2].replace("C", ""))
    cols = np.arange(BLOCK, 3 * BLOCK)
    pca = PCA(n_components=dim, svd_solver="randomized",
              iterated_power=3, random_state=42)
    z = pca.fit_transform(x[:, cols])
    scale = (np.sqrt(pca.explained_variance_)
             if whiten else np.ones(dim))
    model = LogisticRegression(C=c_value, max_iter=5000, tol=1e-5)
    model.fit(z / np.maximum(scale, 1e-8), y)

    def candidate_score(xp):
        zp = pca.transform(xp[:, cols]) / np.maximum(scale, 1e-8)
        return model.predict_proba(zp)[:, 1]

    pools = {}
    eval_specs = [("internal_test", "testoff", None)] + [
        (pool, pool + "off", col) for pool, col in p3a.EXTERNAL]
    for name, tag, col in eval_specs:
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
                  "paired_expert_n": len(paired), "blocks": blocks,
                  "fold_pca512_variance": [row[2] for row in outputs]},
        "selection_rule": "max routing OOF among configs within .005 of best native grouped OOF AUC",
        "sweep": rows,
        "winner": winner,
        "full_fit_explained_variance": float(
            pca.explained_variance_ratio_.sum()),
        "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
