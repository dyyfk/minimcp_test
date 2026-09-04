"""Cheap feature-conditioning sweep for the official native gate.

This is the first post-P2 experiment: keep the 5,228 official-native labels
and the untouched evaluation pools fixed, but test whether the deployed
12,288-dimensional linear head is losing transfer through conditioning or
feature-block choice.  Candidate selection uses source-family-grouped OOF
scores only.  The staged expert outcomes are used as an internal routing
validation signal, never as fit labels.

Nothing overwrites ``gate_native.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


CORE = [
    ("caliboff", "calib_features.parquet"),
    ("expoff", "expansion_labels.parquet"),
    ("exp2off", "expansion2_labels.parquet"),
    ("exp3off", "expansion3_labels.parquet"),
    ("exp3zhoff", "expansion3zh_labels.parquet"),
]
EXTERNAL = [
    ("striviaqa", "oab_ok"),
    ("swebq", "oab_ok"),
    ("sllama", "oab_ok"),
    ("sdqa", "heard_ok"),
    ("sreason", "heard_ok"),
]
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
BLOCK = 4096
RNG = np.random.default_rng(42)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feats(data_dir: Path, tag: str):
    ids, arrays = [], []
    for path in sorted(data_dir.glob(
            f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(path, allow_pickle=True)
        ids += [str(row_id) for row_id in z["ids"]]
        arrays.append(z["X"].astype(np.float32, copy=False))
    if not arrays:
        raise FileNotFoundError(f"no official-native features for {tag}")
    x = np.concatenate(arrays)
    rows = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    rows = rows.drop_duplicates("id", keep="last")
    return list(rows["id"]), x[rows["row"].to_numpy()]


def native_failure(data_dir: Path, tag: str):
    df = pd.read_parquet(
        data_dir / f"frozen_native_{tag}_judged.parquet")
    df = df.dropna(subset=["adequate"]).drop_duplicates("id", keep="last")
    return (1 - df.set_index("id")["adequate"].astype(int)).to_dict()


def group_for(meta: pd.DataFrame, tag: str, row_id: str) -> str:
    if row_id not in meta.index:
        return f"{tag}:unknown"
    row = meta.loc[row_id]
    for col in ("source", "pool"):
        if col in meta.columns and pd.notna(row.get(col)):
            # A family recurring in expansion rounds must remain in one fold.
            return str(row[col])
    return f"{tag}:unknown"


def collect_training(data_dir: Path):
    xs, ys, ids_out, groups = [], [], [], []
    blocks = []
    for tag, metadata_file in CORE:
        ids, x = load_feats(data_dir, tag)
        ymap = native_failure(data_dir, tag)
        meta = (pd.read_parquet(data_dir / metadata_file)
                .drop_duplicates("id", keep="last").set_index("id"))
        rows = [j for j, row_id in enumerate(ids) if row_id in ymap]
        xs.append(x[rows])
        ys.append(np.array([ymap[ids[j]] for j in rows], dtype=np.int8))
        ids_out += [ids[j] for j in rows]
        groups += [group_for(meta, tag, ids[j]) for j in rows]
        blocks.append({"tag": tag, "n": len(rows)})

    tag = "freshoff"
    ids, x = load_feats(data_dir, tag)
    ymap = native_failure(data_dir, tag)
    meta = (pd.read_parquet(data_dir / "fresh_labels.parquet")
            .drop_duplicates("id", keep="last").set_index("id"))
    rows, y = [], []
    for j, row_id in enumerate(ids):
        if row_id not in meta.index or meta.loc[row_id].get("split") != "train":
            continue
        pool = meta.loc[row_id].get("pool")
        if pool == "fresh_fast":
            value = 1  # retained policy-positive contract of live 8bq
        elif row_id in ymap:
            value = ymap[row_id]
        else:
            continue
        rows.append(j)
        y.append(value)
    xs.append(x[rows])
    ys.append(np.asarray(y, dtype=np.int8))
    ids_out += [ids[j] for j in rows]
    groups += [group_for(meta, tag, ids[j]) for j in rows]
    blocks.append({"tag": tag, "n": len(rows)})

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if len(y) != 5228 or x.shape[1] != 3 * BLOCK:
        raise RuntimeError(f"unexpected training shape {x.shape}")
    return x, y, np.asarray(ids_out), np.asarray(groups), blocks


def top_mask(score, rate):
    n = len(score)
    k = int(round(n * rate))
    out = np.zeros(n, dtype=bool)
    order = np.lexsort((np.arange(n), -np.asarray(score)))
    out[order[:k]] = True
    return out


def routing_objective(gain, score):
    return float(np.mean([
        gain[top_mask(score, rate)].sum() / len(gain)
        for rate in RATES.values()
    ]))


def bootstrap_auc_delta(y, candidate, baseline, n_boot=3000):
    values = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        sample = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[sample])) < 2:
            continue
        values.append(roc_auc_score(y[sample], candidate[sample]) -
                      roc_auc_score(y[sample], baseline[sample]))
    return [float(np.mean(values)),
            *[float(v) for v in np.percentile(values, [2.5, 97.5])]]


def bootstrap_cascade_delta(local_ok, expert_ok, candidate, baseline,
                            rate, n_boot=3000):
    values = []
    idx = np.arange(len(local_ok))
    for _ in range(n_boot):
        sample = RNG.choice(idx, len(idx), replace=True)
        lo, eo = local_ok[sample], expert_ok[sample]
        cf = top_mask(candidate[sample], rate)
        bf = top_mask(baseline[sample], rate)
        values.append(np.where(cf, eo, lo).mean() -
                      np.where(bf, eo, lo).mean())
    return [float(np.mean(values)),
            *[float(v) for v in np.percentile(values, [2.5, 97.5])]]


def block_l2(x):
    parts = []
    for j in range(3):
        part = x[:, j * BLOCK:(j + 1) * BLOCK]
        norm = np.linalg.norm(part, axis=1, keepdims=True)
        parts.append(part / np.maximum(norm, 1e-8) * np.sqrt(BLOCK))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


@dataclass(frozen=True)
class Spec:
    name: str
    blocks: tuple[int, ...]
    transform: str
    c_value: float


def columns_for(blocks):
    return np.concatenate([
        np.arange(j * BLOCK, (j + 1) * BLOCK) for j in blocks])


def prepare_train_test(x_train, x_test, spec):
    cols = columns_for(spec.blocks)
    train = x_train[:, cols]
    test = x_test[:, cols]
    scaler = None
    if spec.transform == "standard":
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train).astype(np.float32, copy=False)
        test = scaler.transform(test).astype(np.float32, copy=False)
    elif spec.transform != "raw":
        raise ValueError(spec.transform)
    return train, test, scaler


def fit_fold(spec, train, test, x, y):
    xt, xv, _ = prepare_train_test(x[train], x[test], spec)
    model = LogisticRegression(
        C=spec.c_value, max_iter=3000, tol=1e-5).fit(xt, y[train])
    return test, model.predict_proba(xv)[:, 1], int(model.n_iter_[0])


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"]) + float(artifact["b"])


def fit_full(spec, x, y):
    cols = columns_for(spec.blocks)
    xx = x[:, cols]
    scaler = None
    if spec.transform == "standard":
        scaler = StandardScaler().fit(xx)
        xx = scaler.transform(xx).astype(np.float32, copy=False)
    model = LogisticRegression(
        C=spec.c_value, max_iter=5000, tol=1e-5).fit(xx, y)
    return model, scaler, cols


def score_full(model, scaler, cols, x):
    xx = x[:, cols]
    if scaler is not None:
        xx = scaler.transform(xx).astype(np.float32, copy=False)
    return model.predict_proba(xx)[:, 1]


def expert_outcomes(data_dir: Path, pool: str, col=None):
    if pool == "internal_test":
        df = pd.read_parquet(data_dir / "frozen_v3_traces.parquet")
        out = (df[df["mode"] == "escalated"].groupby("id")["heard_ok"]
               .max().dropna().astype(int))
    else:
        df = pd.read_parquet(data_dir / f"{pool}_conclive_traces.parquet")
        out = df[df["tier"] == "always"].dropna(subset=[col])
        out = out.drop_duplicates("id", keep="last").set_index("id")[col]
        out = out.astype(int)
    return out.to_dict()


def eval_pool(data_dir, name, tag, expert_col, fitted, aligned_artifact):
    ids, x = load_feats(data_dir, tag)
    failure = native_failure(data_dir, tag)
    expert = expert_outcomes(data_dir, name, expert_col)
    rows = [j for j, row_id in enumerate(ids)
            if row_id in failure and row_id in expert]
    row_ids = [ids[j] for j in rows]
    x = x[rows]
    local_ok = np.array([1 - failure[row_id] for row_id in row_ids])
    expert_ok = np.array([expert[row_id] for row_id in row_ids])
    gain = expert_ok - local_ok
    model, scaler, cols = fitted
    candidate = score_full(model, scaler, cols, x)
    aligned = artifact_score(aligned_artifact, x)
    failure_y = 1 - local_ok
    positive = (gain > 0).astype(int)
    result = {
        "n": len(rows),
        "native_auc_aligned": float(roc_auc_score(failure_y, aligned)),
        "native_auc_candidate": float(roc_auc_score(failure_y, candidate)),
        "native_auc_delta_ci": bootstrap_auc_delta(
            failure_y, candidate, aligned),
        "benefit_auc_aligned": float(roc_auc_score(positive, aligned)),
        "benefit_auc_candidate": float(roc_auc_score(positive, candidate)),
        "benefit_auc_delta_ci": bootstrap_auc_delta(
            positive, candidate, aligned),
        "budgets": {},
    }
    for tier, rate in RATES.items():
        result["budgets"][tier] = {}
        for label, score in (("aligned", aligned), ("candidate", candidate)):
            fire = top_mask(score, rate)
            result["budgets"][tier][label] = {
                "accuracy": float(np.where(fire, expert_ok, local_ok).mean()),
                "harmful_selected": float((gain[fire] < 0).mean()),
            }
        result["budgets"][tier]["candidate_vs_aligned_delta_ci"] = (
            bootstrap_cascade_delta(
                local_ok, expert_ok, candidate, aligned, rate))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()

    x, y, ids, groups, blocks = collect_training(args.data_dir)
    expert_df = pd.read_parquet(args.expert_labels)
    expert_col = "adequate" if "adequate" in expert_df else "expert_ok"
    expert = (expert_df.dropna(subset=[expert_col])
              .drop_duplicates("id", keep="last").set_index("id")[expert_col]
              .astype(int).to_dict())
    paired = np.array([i for i, row_id in enumerate(ids) if row_id in expert])
    gain = np.array([expert[ids[i]] - (1 - y[i]) for i in paired])

    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    block_sets = {
        "all": (0, 1, 2), "last": (0,), "tail8": (1,), "user": (2,),
        "last_tail8": (0, 1), "last_user": (0, 2),
        "tail8_user": (1, 2),
    }
    specs = []
    for name, selected in block_sets.items():
        for c_value in (1e-4, 3e-4, 1e-3):
            specs.append(Spec(f"raw_{name}_C{c_value:g}", selected,
                              "raw", c_value))
    for c_value in (1e-4, 3e-4, 1e-3):
        specs.append(Spec(f"standard_all_C{c_value:g}", (0, 1, 2),
                          "standard", c_value))

    tasks = [(spec, fold, train, test) for spec in specs
             for fold, (train, test) in enumerate(cv)]
    print(f"train={x.shape}, groups={len(set(groups))}, paired={len(paired)}, "
          f"candidates={len(specs)}, fold fits={len(tasks)}", flush=True)
    fitted_folds = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_fold)(spec, train, test, x, y)
        for spec, _fold, train, test in tasks)

    rows = []
    for j, spec in enumerate(specs):
        oof = np.full(len(y), np.nan)
        iterations = []
        for test, score, n_iter in fitted_folds[j * 5:(j + 1) * 5]:
            oof[test] = score
            iterations.append(n_iter)
        native_auc = float(roc_auc_score(y, oof))
        benefit_auc = float(roc_auc_score((gain > 0).astype(int), oof[paired]))
        route = routing_objective(gain, oof[paired])
        row = {"name": spec.name, "blocks": spec.blocks,
               "transform": spec.transform, "C": spec.c_value,
               "native_group_oof_auc": native_auc,
               "benefit_oof_auc": benefit_auc,
               "routing_oof_objective": route,
               "max_iterations": max(iterations)}
        rows.append(row)
        print(f"{spec.name:<28} native {native_auc:.4f} benefit "
              f"{benefit_auc:.4f} route {route:+.4f}", flush=True)

    # Routing utility is the primary purpose; native AUC is a guard.  Only
    # candidates within .005 native AUC of the best are eligible.
    best_native = max(row["native_group_oof_auc"] for row in rows)
    eligible = [row for row in rows
                if row["native_group_oof_auc"] >= best_native - .005]
    winner_row = max(eligible, key=lambda row: row["routing_oof_objective"])
    winner = next(spec for spec in specs if spec.name == winner_row["name"])
    print(f"winner: {winner.name}", flush=True)
    fitted = fit_full(winner, x, y)

    pools = {"internal_test": eval_pool(
        args.data_dir, "internal_test", "testoff", None, fitted,
        args.aligned_artifact)}
    for pool, col in EXTERNAL:
        pools[pool] = eval_pool(
            args.data_dir, pool, pool + "off", col, fitted,
            args.aligned_artifact)

    result = {
        "inputs": {
            "data_dir": str(args.data_dir),
            "aligned_artifact": str(args.aligned_artifact),
            "aligned_sha256": sha256(args.aligned_artifact),
            "expert_labels": str(args.expert_labels),
            "expert_labels_sha256": sha256(args.expert_labels),
        },
        "train": {"n": len(y), "dim": x.shape[1],
                  "groups": len(set(groups)), "paired_expert_n": len(paired),
                  "blocks": blocks},
        "selection_rule": "max routing OOF among configs within .005 of best native grouped OOF AUC",
        "sweep": rows,
        "winner": winner_row,
        "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
