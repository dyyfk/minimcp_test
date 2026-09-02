"""Fuse the two-sample semantic signal with text p(True) and P3a."""
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


BLOCK = 4096
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}


def load_module(filename, name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fit_fold(train, test, x, y, cols):
    model = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
    model.fit(x[train][:, cols], y[train])
    return test, model.decision_function(x[test][:, cols])


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def zfit(values):
    return float(np.mean(values)), float(np.std(values))


def zapply(values, center, scale):
    return (np.asarray(values) - center) / max(scale, 1e-8)


def metrics(p3a, local, expert, values):
    gain = expert - local
    return {"native_auc": float(roc_auc_score(1 - local, values)),
            "benefit_auc": float(roc_auc_score(gain > 0, values)),
            "routing_objective": p3a.routing_objective(gain, values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--train-semantic", type=Path, required=True)
    parser.add_argument("--train-ptrue", type=Path, required=True)
    parser.add_argument("--semantic-result", type=Path, required=True)
    parser.add_argument("--external-selection", type=Path, required=True)
    parser.add_argument("--external-semantic", type=Path, required=True)
    parser.add_argument("--external-ptrue", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    p3a = load_module("35_feature_conditioning.py", "feature_conditioning")
    ptrue_mod = load_module("45_text_ptrue_fusion.py", "text_ptrue")
    semantic_winner = json.loads(args.semantic_result.read_text())["winner"]
    target = semantic_winner["target"]
    train = (pd.read_parquet(args.train_selection)
             .merge(pd.read_parquet(args.train_semantic), on="id",
                    validate="one_to_one")
             .merge(ptrue_mod.read_signal(args.train_ptrue), on="id",
                    validate="one_to_one"))
    x, y, ids, groups, _ = p3a.collect_training(args.data_dir)
    id_to_index = {row_id: i for i, row_id in enumerate(ids)}
    train = train[train["id"].isin(id_to_index)].copy()
    index = np.array([id_to_index[row_id] for row_id in train["id"]])
    local = 1 - y[index]
    expert = train["adequate"].astype(int).to_numpy()
    semantic = train[target].to_numpy(dtype=float)
    ptrue = -logit(train["p_yes_textq"])
    cols = np.arange(BLOCK, 3 * BLOCK)
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    folds = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_fold)(tr, te, x, y, cols) for tr, te in cv)
    base_oof = np.full(len(y), np.nan)
    for te, values in folds:
        base_oof[te] = values
    base = base_oof[index]
    centers = {"base": zfit(base), "semantic": zfit(semantic),
               "ptrue": zfit(ptrue)}
    bz = zapply(base, *centers["base"])
    sz = zapply(semantic, *centers["semantic"])
    pz = zapply(ptrue, *centers["ptrue"])

    candidates = []
    weights = (0., .1, .25, .5)
    for ws in weights:
        for wp in weights:
            if ws + wp > .75:
                continue
            values = (1 - ws - wp) * bz + ws * sz + wp * pz
            row = {"name": f"fixed_sem{ws:g}_ptrue{wp:g}",
                   "kind": "fixed", "semantic_weight": ws,
                   "ptrue_weight": wp, **metrics(p3a, local, expert, values)}
            candidates.append(row)
            print(row, flush=True)
    raw = np.column_stack([base, semantic, ptrue])
    position = {row_index: i for i, row_index in enumerate(index)}
    for c_value in (.01, .1, 1.):
        oof = np.full(len(index), np.nan)
        for tr, te in cv:
            tr_set, te_set = set(tr), set(te)
            tp = np.array([position[i] for i in index if i in tr_set])
            vp = np.array([position[i] for i in index if i in te_set])
            scaler = StandardScaler().fit(raw[tp])
            model = LogisticRegression(C=c_value, max_iter=3000, tol=1e-6)
            model.fit(scaler.transform(raw[tp]), y[index[tp]])
            oof[vp] = model.predict_proba(scaler.transform(raw[vp]))[:, 1]
        row = {"name": f"meta_C{c_value:g}", "kind": "meta",
               "c_value": c_value, **metrics(p3a, local, expert, oof)}
        candidates.append(row)
        print(row, flush=True)
    best_native = max(row["native_auc"] for row in candidates)
    eligible = [row for row in candidates
                if row["native_auc"] >= best_native - .005]
    winner = max(eligible, key=lambda row: row["routing_objective"])

    full = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
    full.fit(x[:, cols], y)
    base_full = full.decision_function(x[index][:, cols])
    deploy_center = zfit(base_full)
    meta_scaler = meta_model = None
    if winner["kind"] == "meta":
        full_raw = np.column_stack([base_full, semantic, ptrue])
        meta_scaler = StandardScaler().fit(full_raw)
        meta_model = LogisticRegression(
            C=winner["c_value"], max_iter=3000, tol=1e-6)
        meta_model.fit(meta_scaler.transform(full_raw), y[index])
    winner["deploy_base_center"], winner["deploy_base_scale"] = deploy_center
    winner["semantic_center"], winner["semantic_scale"] = centers["semantic"]
    winner["ptrue_center"], winner["ptrue_scale"] = centers["ptrue"]
    print("winner", winner, flush=True)

    external = (pd.read_parquet(args.external_selection)
                .merge(pd.read_parquet(args.external_semantic), on="id",
                       validate="one_to_one")
                .merge(ptrue_mod.read_signal(args.external_ptrue), on="id",
                       validate="one_to_one"))
    pools = {}
    for pool, frame in external.groupby("pool", sort=True):
        pool_ids, xp = p3a.load_feats(args.data_dir, f"{pool}off")
        positions = {row_id: i for i, row_id in enumerate(pool_ids)}
        frame = frame[frame["id"].isin(positions)].copy()
        xp = xp[[positions[row_id] for row_id in frame["id"]]]
        aligned = p3a.artifact_score(args.aligned_artifact, xp)
        base_e = full.decision_function(xp[:, cols])
        sem_e = frame[target].to_numpy(dtype=float)
        pt_e = -logit(frame["p_yes_textq"])
        if winner["kind"] == "fixed":
            ws, wp = winner["semantic_weight"], winner["ptrue_weight"]
            candidate = ((1 - ws - wp) * zapply(base_e, *deploy_center) +
                         ws * zapply(sem_e, *centers["semantic"]) +
                         wp * zapply(pt_e, *centers["ptrue"]))
        else:
            candidate = meta_model.predict_proba(meta_scaler.transform(
                np.column_stack([base_e, sem_e, pt_e])))[:, 1]
        lo = frame["native_ok"].astype(int).to_numpy()
        eo = frame["expert_ok"].astype(int).to_numpy()
        ny, by = 1 - lo, (eo - lo) > 0
        result = {"n": len(frame),
                  "native_auc_aligned": float(roc_auc_score(ny, aligned)),
                  "native_auc_candidate": float(roc_auc_score(ny, candidate)),
                  "native_auc_delta_ci": p3a.bootstrap_auc_delta(
                      ny, candidate, aligned),
                  "benefit_auc_aligned": float(roc_auc_score(by, aligned)),
                  "benefit_auc_candidate": float(roc_auc_score(by, candidate)),
                  "benefit_auc_delta_ci": p3a.bootstrap_auc_delta(
                      by, candidate, aligned), "budgets": {}}
        for tier, rate in RATES.items():
            af, cf = p3a.top_mask(aligned, rate), p3a.top_mask(candidate, rate)
            result["budgets"][tier] = {
                "aligned_accuracy": float(np.where(af, eo, lo).mean()),
                "candidate_accuracy": float(np.where(cf, eo, lo).mean()),
                "candidate_vs_aligned_delta_ci": p3a.bootstrap_cascade_delta(
                    lo, eo, candidate, aligned, rate)}
        pools[pool] = result
    out = {"signal": "two-sample semantic + original-query text p(True)",
           "selection_rule": "max routing OOF among configs within .005 of best native OOF AUC",
           "train_n": len(train), "target": target, "sweep": candidates,
           "winner": winner, "pools": pools}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
