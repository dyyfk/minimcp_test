"""Nested source-family evaluation of causal layer/trajectory views.

This development-only diagnostic compares one layer and one causal temporal
view at a time using strongly regularized logistic heads.  Selection uses only
the already-opened P25-B standalone cohort and never changes a gate artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


RATES = (.15, .30, .50)
HIDDEN = 4096
EXPECTED_LAYERS = (18, 22, 26, 30)
EXPECTED_VIEWS = ("last", "prefill_mean8", "last_prev",
                  "running_prefill_mean")
STRONG_L2_CS = (1e-5, 3e-5, 1e-4, 3e-4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(directory: Path, pattern: str):
    ids, arrays, paths = [], [], sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    layers, views = None, None
    for path in paths:
        value = np.load(path, allow_pickle=True)
        ids.extend(str(row_id) for row_id in value["ids"])
        arrays.append(value["X"].astype(np.float32, copy=False))
        current_layers = tuple(int(item) for item in value["layers"])
        current_views = tuple(str(item) for item in value["views"])
        if layers is not None and current_layers != layers:
            raise RuntimeError("inconsistent layer metadata")
        if views is not None and current_views != views:
            raise RuntimeError("inconsistent view metadata")
        layers, views = current_layers, current_views
    matrix = np.concatenate(arrays)
    index = pd.DataFrame({"id": ids, "row": range(len(ids))})
    index = index.drop_duplicates("id", keep="last")
    return (index.id.to_list(), matrix[index.row.to_numpy()], layers, views,
            paths)


def load_original(directory: Path):
    ids, arrays, paths = [], [], sorted(
        directory.glob("prospective_native_feats.rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"no original features under {directory}")
    for path in paths:
        value = np.load(path, allow_pickle=True)
        ids.extend(str(row_id) for row_id in value["ids"])
        arrays.append(value["X"].astype(np.float32, copy=False))
    matrix = np.concatenate(arrays)
    index = pd.DataFrame({"id": ids, "row": range(len(ids))})
    index = index.drop_duplicates("id", keep="last")
    return index.id.to_list(), matrix[index.row.to_numpy()], paths


def top_mask(score, rate):
    count = min(len(score), max(0, int(round(len(score) * rate))))
    mask = np.zeros(len(score), dtype=bool)
    if count:
        order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
        mask[order[:count]] = True
    return mask


def routing_objective(gain, score):
    return float(np.mean([
        gain[top_mask(score, rate)].sum() / len(gain) for rate in RATES
    ]))


def macro_routing(gain, score, groups):
    return float(np.mean([
        routing_objective(gain[groups == group], score[groups == group])
        for group in sorted(set(groups))
    ]))


def macro_auc(y, score, groups):
    values = []
    for group in sorted(set(groups)):
        mask = groups == group
        if len(np.unique(y[mask])) == 2:
            values.append(roc_auc_score(y[mask], score[mask]))
    return (float(np.mean(values)) if values else None), len(values)


def summary(local_ok, expert_ok, score, groups):
    y = 1 - local_ok
    macro, valid = macro_auc(y, score, groups)
    return {
        "routing_objective_pooled": routing_objective(expert_ok - local_ok,
                                                       score),
        "routing_objective_macro_source": macro_routing(
            expert_ok - local_ok, score, groups),
        "native_failure_auc_pooled": float(roc_auc_score(y, score)),
        "native_failure_auc_macro_source": macro,
        "native_failure_valid_sources": valid,
    }


@dataclass(frozen=True)
class Spec:
    layer: int
    view: str
    c: float

    @property
    def name(self):
        return f"trajectory_l{self.layer}_{self.view}_c{self.c:g}"

    @property
    def parameter_count(self):
        return HIDDEN + 1


def columns(spec, layers, views):
    block = (layers.index(spec.layer) * len(views) + views.index(spec.view))
    return slice(block * HIDDEN, (block + 1) * HIDDEN)


def fit_predict(spec, layers, views, x_train, local_train, expert_train,
                x_test):
    cols = columns(spec, layers, views)
    train, test = x_train[:, cols], x_test[:, cols]
    mean = train.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    train = np.clip((train - mean) / scale, -10., 10.)
    test = np.clip((test - mean) / scale, -10., 10.)
    gain = expert_train - local_train
    weight = np.where(gain < 0, 2., 1.)
    model = LogisticRegression(C=spec.c, max_iter=3000, tol=1e-5)
    model.fit(train, 1 - local_train, sample_weight=weight)
    return model.predict_proba(test)[:, 1]


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"], dtype=np.float32) + float(artifact["b"])


def choose(rows):
    primary = max(row["inner_routing_macro_mean"] for row in rows)
    eligible = [row for row in rows
                if row["inner_routing_macro_mean"] >= primary - .001]
    return max(eligible, key=lambda row: (
        row["inner_native_macro_mean"], -row["parameter_count"], row["name"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--c-values", default=",".join(
        f"{value:g}" for value in STRONG_L2_CS))
    args = parser.parse_args()

    c_values = tuple(float(value) for value in args.c_values.split(","))
    if (not c_values or len(set(c_values)) != len(c_values)
            or any(value <= 0 or value > 3e-4 for value in c_values)):
        raise ValueError("c-values must be unique positive strong-L2 values <=3e-4")

    trajectory_ids, trajectory, layers, views, trajectory_paths = load_npz(
        args.trajectory_dir, "causal_trajectory_feats.rank*.npz")
    original_ids, original, original_paths = load_original(args.original_dir)
    if layers != EXPECTED_LAYERS or views != EXPECTED_VIEWS:
        raise RuntimeError(f"unexpected feature layout layers={layers} views={views}")
    expected_width = len(layers) * len(views) * HIDDEN
    if trajectory.shape[1] != expected_width:
        raise RuntimeError(f"unexpected trajectory shape {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise RuntimeError("trajectory features contain non-finite values")
    trajectory_index = {row_id: i for i, row_id in enumerate(trajectory_ids)}
    original_index = {row_id: i for i, row_id in enumerate(original_ids)}

    frame = pd.read_parquet(args.selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    local = pd.read_parquet(args.local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    frame = frame.merge(local[["id", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    frame = frame[frame.id.astype(str).isin(trajectory_index) &
                  frame.id.astype(str).isin(original_index)]
    frame = frame.sort_values("id").reset_index(drop=True)
    x = np.stack([trajectory[trajectory_index[str(row_id)]]
                  for row_id in frame.id])
    xo = np.stack([original[original_index[str(row_id)]] for row_id in frame.id])
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    groups = frame.source_family.astype(str).to_numpy()
    gain = expert_ok - local_ok
    strat = (gain + 1) * 2 + (1 - local_ok)
    live = artifact_score(args.live_artifact, xo)

    specs = tuple(Spec(layer, view, c) for layer in layers for view in views
                  for c in c_values)
    outer = list(StratifiedGroupKFold(
        args.outer_folds, shuffle=True, random_state=42).split(x, strat, groups))
    nested = np.full(len(frame), np.nan)
    outer_rows, history = [], {spec.name: [] for spec in specs}
    for outer_index, (outer_train, outer_test) in enumerate(outer):
        inner = list(StratifiedGroupKFold(
            args.inner_folds, shuffle=True,
            random_state=100 + outer_index).split(
                x[outer_train], strat[outer_train], groups[outer_train]))
        sweep = []
        for spec in specs:
            routing, native = [], []
            for train_rel, valid_rel in inner:
                train, valid = outer_train[train_rel], outer_train[valid_rel]
                score = fit_predict(spec, layers, views, x[train],
                                    local_ok[train], expert_ok[train], x[valid])
                routing.append(macro_routing(gain[valid], score, groups[valid]))
                native.append(macro_auc(1 - local_ok[valid], score,
                                        groups[valid])[0])
            row = {"name": spec.name, "spec": asdict(spec),
                   "parameter_count": spec.parameter_count,
                   "inner_routing_macro_mean": float(np.mean(routing)),
                   "inner_native_macro_mean": float(np.mean(native))}
            sweep.append(row)
            history[spec.name].append(row)
            print(f"outer={outer_index} {spec.name} "
                  f"routing={row['inner_routing_macro_mean']:+.6f} "
                  f"native={row['inner_native_macro_mean']:.6f}", flush=True)
        winner = choose(sweep)
        spec = next(item for item in specs if item.name == winner["name"])
        nested[outer_test] = fit_predict(
            spec, layers, views, x[outer_train], local_ok[outer_train],
            expert_ok[outer_train], x[outer_test])
        outer_rows.append({"outer_fold": outer_index,
                           "test_groups": sorted(set(groups[outer_test])),
                           "winner": winner, "sweep": sweep,
                           "test": summary(local_ok[outer_test],
                                           expert_ok[outer_test],
                                           nested[outer_test],
                                           groups[outer_test])})
        print(f"outer={outer_index} winner={winner['name']}", flush=True)

    global_rows = []
    for spec in specs:
        rows = history[spec.name]
        global_rows.append({"name": spec.name, "spec": asdict(spec),
                            "parameter_count": spec.parameter_count,
                            "inner_routing_macro_mean": float(np.mean([
                                row["inner_routing_macro_mean"] for row in rows])),
                            "inner_native_macro_mean": float(np.mean([
                                row["inner_native_macro_mean"] for row in rows]))})
    global_winner = choose(global_rows)
    summaries = {"live": summary(local_ok, expert_ok, live, groups),
                 "trajectory_nested": summary(local_ok, expert_ok, nested, groups)}
    keys = ("routing_objective_pooled", "routing_objective_macro_source",
            "native_failure_auc_pooled", "native_failure_auc_macro_source")
    deltas = {key: summaries["trajectory_nested"][key] - summaries["live"][key]
              for key in keys}
    strata = {}
    for column in ("language", "pool"):
        values = frame[column].astype(str).to_numpy()
        strata[column] = {}
        for value in sorted(set(values)):
            mask = values == value
            candidate = routing_objective(gain[mask], nested[mask])
            baseline = routing_objective(gain[mask], live[mask])
            strata[column][value] = {"rows": int(mask.sum()),
                                      "candidate_routing": candidate,
                                      "live_routing": baseline,
                                      "delta": candidate - baseline}
    gates = {
        "macro_routing_delta_ge_0.005":
            deltas["routing_objective_macro_source"] >= .005,
        "pooled_routing_nonnegative":
            deltas["routing_objective_pooled"] >= 0,
        "macro_native_auc_delta_ge_0.010":
            deltas["native_failure_auc_macro_source"] >= .01,
        "pooled_native_auc_nonnegative":
            deltas["native_failure_auc_pooled"] >= 0,
        "language_routing_nonnegative": all(
            row["delta"] >= 0 for row in strata["language"].values()),
        "broad_pool_routing_ge_minus_0.010": all(
            row["delta"] >= -.01 for row in strata["pool"].values()),
    }
    result = {
        "status": "causal_within_layer_trajectory_development",
        "inputs": {"rows": len(frame), "source_families": int(
            frame.source_family.nunique()), "layers": layers, "views": views,
            "trajectory_dimension": x.shape[1],
            "selection_sha256": sha256(args.selection),
            "local_sha256": sha256(args.local),
            "expert_sha256": sha256(args.expert),
            "live_artifact_sha256": sha256(args.live_artifact),
            "trajectory_files": [
                {"path": str(path), "sha256": sha256(path)}
                for path in trajectory_paths],
            "original_files": [{"path": str(path), "sha256": sha256(path)}
                               for path in original_paths]},
        "protocol": {"outer_folds": args.outer_folds,
                     "inner_folds": args.inner_folds,
                     "primary": "macro within-source routing objective",
                     "tie_break": "within .001: native macro AUC then name",
                     "heads": "one layer/view, strong-L2 logistic",
                     "development_only": True},
        "specs": [{**asdict(spec), "name": spec.name,
                   "parameter_count": spec.parameter_count} for spec in specs],
        "outer_folds": outer_rows, "global_sweep": global_rows,
        "global_winner": global_winner, "summaries": summaries,
        "deltas_vs_live": deltas, "strata": strata, "gates": gates,
        "advance": bool(all(gates.values())), "live_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"winner": global_winner, "summaries": summaries,
                      "deltas_vs_live": deltas, "gates": gates,
                      "advance": result["advance"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
