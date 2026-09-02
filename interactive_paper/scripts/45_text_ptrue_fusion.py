"""Fuse original-query text p(True) with grouped-OOF native gate scores.

This is a fast ceiling diagnostic for repeat-then-judge: the p(True) prompt
sees the ground-truth query text rather than an ASR transcript.  Candidate
selection uses only the fixed 1,000-row paired training pilot.
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


def load_p3a():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_signal(directory, column="p_yes_textq"):
    rows = []
    for path in sorted(directory.glob("rtj.rank*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    frame = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    if frame.empty or frame["error"].notna().any():
        raise RuntimeError("missing or errored p(True) rows")
    mass_column = "mass_rtj" if column == "p_yes_rtj" else "mass_textq"
    return frame[["id", column, mass_column, "elapsed_s"]].rename(
        columns={column: "p_yes", mass_column: "yes_no_mass"})


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(values) - np.log1p(-values)


def zfit(values):
    return float(np.mean(values)), float(np.std(values))


def zapply(values, center, scale):
    return (np.asarray(values) - center) / max(scale, 1e-8)


def fit_base_fold(train, test, x, y, cols):
    model = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
    model.fit(x[train][:, cols], y[train])
    return test, model.decision_function(x[test][:, cols])


def candidate_metrics(p3a, local_ok, expert_ok, score):
    native = 1 - local_ok
    gain = expert_ok - local_ok
    return {
        "native_auc": float(roc_auc_score(native, score)),
        "benefit_auc": float(roc_auc_score((gain > 0).astype(int), score)),
        "routing_objective": p3a.routing_objective(gain, score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--train-signal", type=Path, required=True)
    parser.add_argument("--external-selection", type=Path, required=True)
    parser.add_argument("--external-signal", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--signal-column", choices=("p_yes_textq", "p_yes_rtj"),
                        default="p_yes_textq")
    args = parser.parse_args()

    p3a = load_p3a()
    train = pd.read_parquet(args.train_selection).merge(
        read_signal(args.train_signal, args.signal_column), on="id",
        validate="one_to_one")
    x, y, ids, groups, _blocks = p3a.collect_training(args.data_dir)
    id_to_index = {row_id: i for i, row_id in enumerate(ids)}
    train = train[train["id"].isin(id_to_index)].copy()
    index = np.array([id_to_index[row_id] for row_id in train["id"]])
    local_ok = 1 - y[index]
    expert_ok = train["adequate"].astype(int).to_numpy()
    uncertainty = -logit(train["p_yes"])

    columns = {"aligned": np.arange(3 * BLOCK),
               "p3a": np.arange(BLOCK, 3 * BLOCK)}
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    tasks = [(name, cols, tr, te) for name, cols in columns.items()
             for tr, te in cv]
    outputs = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_base_fold)(tr, te, x, y, cols)
        for _name, cols, tr, te in tasks)
    bases = {name: np.full(len(y), np.nan) for name in columns}
    offset = 0
    for name in columns:
        for test, values in outputs[offset:offset + 5]:
            bases[name][test] = values
        offset += 5

    candidates = []
    for name, base_all in bases.items():
        base = base_all[index]
        bc, bs = zfit(base)
        pc, ps = zfit(uncertainty)
        bz, pz = zapply(base, bc, bs), zapply(uncertainty, pc, ps)
        for blend in (0., .1, .25, .5, .75, 1.):
            values = (1 - blend) * bz + blend * pz
            row = {"name": f"{name}+text_ptrue@{blend:g}",
                   "kind": "blend", "base": name, "blend": blend,
                   "base_center": bc, "base_scale": bs,
                   "ptrue_center": pc, "ptrue_scale": ps,
                   **candidate_metrics(p3a, local_ok, expert_ok, values)}
            candidates.append(row)
            print(row, flush=True)

        features = np.column_stack([base, uncertainty])
        position = {row_index: i for i, row_index in enumerate(index)}
        for c_value in (.01, .1, 1.):
            oof = np.full(len(index), np.nan)
            for tr, te in cv:
                tr_set, te_set = set(tr), set(te)
                tp = np.array([position[i] for i in index if i in tr_set])
                vp = np.array([position[i] for i in index if i in te_set])
                scaler = StandardScaler().fit(features[tp])
                model = LogisticRegression(
                    C=c_value, max_iter=3000, tol=1e-6).fit(
                        scaler.transform(features[tp]), y[index[tp]])
                oof[vp] = model.predict_proba(
                    scaler.transform(features[vp]))[:, 1]
            row = {"name": f"{name}+text_ptrue_meta_C{c_value:g}",
                   "kind": "meta", "base": name, "c_value": c_value,
                   **candidate_metrics(p3a, local_ok, expert_ok, oof)}
            candidates.append(row)
            print(row, flush=True)

    best_native = max(row["native_auc"] for row in candidates)
    eligible = [row for row in candidates
                if row["native_auc"] >= best_native - .005]
    winner = max(eligible, key=lambda row: row["routing_objective"])
    print("winner", winner, flush=True)

    p3a_cols = columns["p3a"]
    p3a_model = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
    p3a_model.fit(x[:, p3a_cols], y)
    base_train = (p3a.artifact_score(args.aligned_artifact, x[index])
                  if winner["base"] == "aligned" else
                  p3a_model.decision_function(x[index][:, p3a_cols]))
    meta_scaler = meta_model = None
    if winner["kind"] == "blend":
        winner["deploy_base_center"], winner["deploy_base_scale"] = zfit(
            base_train)
    else:
        features = np.column_stack([base_train, uncertainty])
        meta_scaler = StandardScaler().fit(features)
        meta_model = LogisticRegression(
            C=winner["c_value"], max_iter=3000, tol=1e-6).fit(
                meta_scaler.transform(features), y[index])

    external = pd.read_parquet(args.external_selection).merge(
        read_signal(args.external_signal, args.signal_column), on="id",
        validate="one_to_one")
    pools = {}
    for pool, frame in external.groupby("pool", sort=True):
        pool_ids, xp = p3a.load_feats(args.data_dir, f"{pool}off")
        positions = {row_id: i for i, row_id in enumerate(pool_ids)}
        frame = frame[frame["id"].isin(positions)].copy()
        xp = xp[[positions[row_id] for row_id in frame["id"]]]
        aligned = p3a.artifact_score(args.aligned_artifact, xp)
        base = (aligned if winner["base"] == "aligned" else
                p3a_model.decision_function(xp[:, p3a_cols]))
        ptrue = -logit(frame["p_yes"])
        if winner["kind"] == "blend":
            candidate = ((1 - winner["blend"]) * zapply(
                base, winner["deploy_base_center"],
                winner["deploy_base_scale"]) + winner["blend"] * zapply(
                    ptrue, winner["ptrue_center"], winner["ptrue_scale"]))
        else:
            candidate = meta_model.predict_proba(meta_scaler.transform(
                np.column_stack([base, ptrue])))[:, 1]
        local = frame["native_ok"].astype(int).to_numpy()
        expert = frame["expert_ok"].astype(int).to_numpy()
        native = 1 - local
        benefit = ((expert - local) > 0).astype(int)
        result = {
            "n": len(frame),
            "native_auc_aligned": float(roc_auc_score(native, aligned)),
            "native_auc_ptrue": float(roc_auc_score(native, ptrue)),
            "native_auc_candidate": float(roc_auc_score(native, candidate)),
            "native_auc_delta_ci": p3a.bootstrap_auc_delta(
                native, candidate, aligned),
            "benefit_auc_aligned": float(roc_auc_score(benefit, aligned)),
            "benefit_auc_candidate": float(roc_auc_score(benefit, candidate)),
            "benefit_auc_delta_ci": p3a.bootstrap_auc_delta(
                benefit, candidate, aligned), "budgets": {},
        }
        for tier, rate in RATES.items():
            af, cf = p3a.top_mask(aligned, rate), p3a.top_mask(candidate, rate)
            result["budgets"][tier] = {
                "aligned_accuracy": float(np.where(af, expert, local).mean()),
                "candidate_accuracy": float(np.where(cf, expert, local).mean()),
                "candidate_vs_aligned_delta_ci": p3a.bootstrap_cascade_delta(
                    local, expert, candidate, aligned, rate),
            }
        pools[pool] = result

    out = {"inputs": {
        "train_signal_sha256": [p3a.sha256(p) for p in sorted(
            args.train_signal.glob("rtj.rank*.jsonl"))],
        "external_signal_sha256": [p3a.sha256(p) for p in sorted(
            args.external_signal.glob("rtj.rank*.jsonl"))],
        "aligned_sha256": p3a.sha256(args.aligned_artifact)},
        "selection_rule": "max routing OOF among configs within .005 of best native OOF AUC",
        "signal": args.signal_column,
        "train_n": len(train), "sweep": candidates, "winner": winner,
        "pools": pools}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
