"""Freeze a direct semantic-uncertainty router and test external transfer.

Unlike the learned P4 surrogate, this experiment pays the inference cost of
three additional local native-duplex samples.  Threshold/metric/blend choice
uses only the fixed 1,000-row paired training pilot; the five official-native
external pools remain untouched until the winner is frozen.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


BLOCK = 4096
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
TARGETS = ("entropy_70", "entropy_78", "entropy_85",
           "mean_pairwise_dissimilarity",
           "official_sample_dissimilarity",
           "sampled_pairwise_dissimilarity")


def load_module(filename, name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def zfit(values):
    return float(np.mean(values)), float(np.std(values))


def zapply(values, center, scale):
    return (np.asarray(values) - center) / max(scale, 1e-8)


def metrics(p3a, local_ok, expert_ok, score):
    native_y = 1 - local_ok
    gain = expert_ok - local_ok
    return {
        "native_auc": float(roc_auc_score(native_y, score)),
        "benefit_auc": float(roc_auc_score((gain > 0).astype(int), score)),
        "routing_objective": p3a.routing_objective(gain, score),
    }


def fit_base_fold(train, test, x, y, cols):
    model = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
    model.fit(x[train][:, cols], y[train])
    return test, model.decision_function(x[test][:, cols])


def fit_meta_oof(base_score, semantic, y, sem_index, cv, c_value):
    features = np.column_stack([base_score[sem_index], semantic])
    oof = np.full(len(sem_index), np.nan)
    position = {row_index: i for i, row_index in enumerate(sem_index)}
    for train_fold, test_fold in cv:
        train_set, test_set = set(train_fold), set(test_fold)
        train_pos = np.array([position[i] for i in sem_index
                              if i in train_set])
        test_pos = np.array([position[i] for i in sem_index
                             if i in test_set])
        scaler = StandardScaler().fit(features[train_pos])
        model = LogisticRegression(
            C=c_value, max_iter=3000, tol=1e-6).fit(
                scaler.transform(features[train_pos]), y[sem_index[train_pos]])
        oof[test_pos] = model.predict_proba(
            scaler.transform(features[test_pos]))[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("incomplete meta OOF predictions")
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--train-samples", type=Path)
    parser.add_argument("--external-selection", type=Path, required=True)
    parser.add_argument("--external-samples", type=Path, required=True)
    parser.add_argument("--external-labels", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-count", type=int, choices=(1, 2, 3),
                        default=3)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--reuse-train-labels", action="store_true")
    parser.add_argument("--reuse-external-labels", action="store_true")
    args = parser.parse_args()

    p3a = load_module("35_feature_conditioning.py", "feature_conditioning")
    sem = load_module("40_semantic_entropy_refit.py", "semantic_refit")
    train_sel = pd.read_parquet(args.train_selection)
    if args.reuse_train_labels:
        train_lab = pd.read_parquet(args.train_labels)
    else:
        if args.train_samples is None:
            parser.error("--train-samples is required unless --reuse-train-labels")
        train_lab = sem.build_labels(
            args.train_selection, args.train_samples, args.embedding_model,
            args.train_labels, args.batch_size, args.sample_count)
    train = train_sel.merge(train_lab, on="id", validate="one_to_one")
    x, y, ids, groups, _blocks = p3a.collect_training(args.data_dir)
    id_to_index = {row_id: i for i, row_id in enumerate(ids)}
    train = train[train["id"].isin(id_to_index)].copy()
    index = np.array([id_to_index[row_id] for row_id in train["id"]])
    local_ok = 1 - y[index]
    expert_ok = train["adequate"].astype(int).to_numpy()

    base_columns = {
        "aligned": np.arange(0, 3 * BLOCK),
        "p3a": np.arange(BLOCK, 3 * BLOCK),
    }
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    tasks = [(name, cols, train_fold, test_fold)
             for name, cols in base_columns.items()
             for train_fold, test_fold in cv]
    outputs = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_base_fold)(
            train_fold, test_fold, x, y, cols)
        for _name, cols, train_fold, test_fold in tasks)
    base_oof = {name: np.full(len(y), np.nan) for name in base_columns}
    offset = 0
    for name in base_columns:
        for test_fold, score in outputs[offset:offset + 5]:
            base_oof[name][test_fold] = score
        offset += 5

    cols = base_columns["p3a"]
    p3a_model = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
    p3a_model.fit(x[:, cols], y)
    bases = {name: score[index] for name, score in base_oof.items()}

    targets = TARGETS if args.sample_count >= 2 else TARGETS[:-1]
    candidates = []
    for base_name, base_score in bases.items():
        bc, bs = zfit(base_score)
        bz = zapply(base_score, bc, bs)
        for target in targets:
            semantic_score = train[target].to_numpy(dtype=float)
            sc, ss = zfit(semantic_score)
            sz = zapply(semantic_score, sc, ss)
            for blend in (0., .1, .25, .5, .75, 1.):
                score = (1 - blend) * bz + blend * sz
                row = {
                    "name": f"{base_name}+{target}@{blend:g}",
                    "kind": "blend",
                    "base": base_name, "target": target, "blend": blend,
                    "base_center": bc, "base_scale": bs,
                    "semantic_center": sc, "semantic_scale": ss,
                    **metrics(p3a, local_ok, expert_ok, score),
                }
                candidates.append(row)
                print(row, flush=True)
    semantic_train = train[list(targets)].to_numpy(dtype=float)
    for base_name in bases:
        for c_value in (.01, .1, 1.):
            score = fit_meta_oof(
                base_oof[base_name], semantic_train, y, index, cv, c_value)
            row = {
                "name": f"{base_name}+semantic_meta_C{c_value:g}",
                "kind": "meta", "base": base_name, "c_value": c_value,
                **metrics(p3a, local_ok, expert_ok, score),
            }
            candidates.append(row)
            print(row, flush=True)
    best_native = max(row["native_auc"] for row in candidates)
    eligible = [row for row in candidates
                if row["native_auc"] >= best_native - .005]
    winner = max(eligible, key=lambda row: row["routing_objective"])
    full_base_train = (
        p3a.artifact_score(args.aligned_artifact, x[index])
        if winner["base"] == "aligned" else
        p3a_model.decision_function(x[index][:, cols]))
    meta_scaler = meta_model = None
    if winner["kind"] == "blend":
        deploy_center, deploy_scale = zfit(full_base_train)
        winner["deploy_base_center"] = deploy_center
        winner["deploy_base_scale"] = deploy_scale
    else:
        meta_features = np.column_stack([full_base_train, semantic_train])
        meta_scaler = StandardScaler().fit(meta_features)
        meta_model = LogisticRegression(
            C=winner["c_value"], max_iter=3000, tol=1e-6).fit(
                meta_scaler.transform(meta_features), y[index])
        winner["deploy_scaler_mean"] = meta_scaler.mean_.tolist()
        winner["deploy_scaler_scale"] = meta_scaler.scale_.tolist()
        winner["deploy_coef"] = meta_model.coef_[0].tolist()
        winner["deploy_intercept"] = float(meta_model.intercept_[0])
    print("winner", winner, flush=True)

    external_sel = pd.read_parquet(args.external_selection)
    external_lab = (pd.read_parquet(args.external_labels)
                    if args.reuse_external_labels else
                    sem.build_labels(
                        args.external_selection, args.external_samples,
                        args.embedding_model, args.external_labels,
                        args.batch_size, args.sample_count))
    external = external_sel.merge(
        external_lab, on="id", validate="one_to_one")
    pools = {}
    for pool, frame in external.groupby("pool", sort=True):
        pool_ids, xp = p3a.load_feats(args.data_dir, f"{pool}off")
        positions = {row_id: i for i, row_id in enumerate(pool_ids)}
        frame = frame[frame["id"].isin(positions)].copy()
        xp = xp[[positions[row_id] for row_id in frame["id"]]]
        aligned = p3a.artifact_score(args.aligned_artifact, xp)
        base = (aligned if winner["base"] == "aligned" else
                p3a_model.decision_function(xp[:, cols]))
        if winner["kind"] == "blend":
            bz = zapply(base, winner["deploy_base_center"],
                        winner["deploy_base_scale"])
            sz = zapply(frame[winner["target"]].to_numpy(dtype=float),
                        winner["semantic_center"], winner["semantic_scale"])
            candidate = ((1 - winner["blend"]) * bz +
                         winner["blend"] * sz)
        else:
            meta_features = np.column_stack(
                [base, frame[list(targets)].to_numpy(dtype=float)])
            candidate = meta_model.predict_proba(
                meta_scaler.transform(meta_features))[:, 1]
        local = frame["native_ok"].astype(int).to_numpy()
        expert = frame["expert_ok"].astype(int).to_numpy()
        native_y = 1 - local
        benefit_y = ((expert - local) > 0).astype(int)
        result = {
            "n": len(frame),
            "native_auc_aligned": float(roc_auc_score(native_y, aligned)),
            "native_auc_candidate": float(roc_auc_score(native_y, candidate)),
            "native_auc_delta_ci": p3a.bootstrap_auc_delta(
                native_y, candidate, aligned),
            "benefit_auc_aligned": float(roc_auc_score(benefit_y, aligned)),
            "benefit_auc_candidate": float(
                roc_auc_score(benefit_y, candidate)),
            "benefit_auc_delta_ci": p3a.bootstrap_auc_delta(
                benefit_y, candidate, aligned),
            "budgets": {},
        }
        for tier, rate in RATES.items():
            af = p3a.top_mask(aligned, rate)
            cf = p3a.top_mask(candidate, rate)
            result["budgets"][tier] = {
                "aligned_accuracy": float(np.where(af, expert, local).mean()),
                "candidate_accuracy": float(np.where(cf, expert, local).mean()),
                "candidate_vs_aligned_delta_ci": p3a.bootstrap_cascade_delta(
                    local, expert, candidate, aligned, rate),
            }
        pools[pool] = result

    output = {
        "inputs": {
            "train_selection_sha256": p3a.sha256(args.train_selection),
            "train_labels_sha256": p3a.sha256(args.train_labels),
            "external_selection_sha256": p3a.sha256(args.external_selection),
            "external_labels_sha256": p3a.sha256(args.external_labels),
            "aligned_sha256": p3a.sha256(args.aligned_artifact),
        },
        "selection_rule": "max routing objective among configs within .005 of best native AUC on fixed training pilot",
        "sample_count": args.sample_count, "targets": list(targets),
        "train_n": len(train), "sweep": candidates, "winner": winner,
        "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
