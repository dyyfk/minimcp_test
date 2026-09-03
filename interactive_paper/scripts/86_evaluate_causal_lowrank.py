"""Fold-local supervised low-rank readout of causal pre-generation states.

The base model remains frozen.  Each outer source-family fold learns at most
eight covariance directions from the outer training rows only: four outcome
targets under pooled and source-balanced weighting.  A fixed low-capacity
logistic head is fit on those factors.  The primary score conservatively
anchors that readout to the live score with a predeclared 0.5 z-score weight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


RATES = (.15, .30, .50)
EXPECTED_LAYERS = (18, 22, 26, 30)
HIDDEN = 4096
VIEWS_PER_LAYER = 3
LOWRANK_C = .1
ANCHOR_ALPHA = .5


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
    layers = None
    for path in paths:
        value = np.load(path, allow_pickle=True)
        ids.extend(str(row_id) for row_id in value["ids"])
        arrays.append(value["X"].astype(np.float32, copy=False))
        if "layers" in value:
            current = tuple(int(item) for item in value["layers"])
            if layers is not None and current != layers:
                raise RuntimeError("inconsistent layer metadata")
            layers = current
    matrix = np.concatenate(arrays)
    index = pd.DataFrame({"id": ids, "row": range(len(ids))})
    index = index.drop_duplicates("id", keep="last")
    return index.id.to_list(), matrix[index.row.to_numpy()], layers, paths


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
    if not values:
        raise RuntimeError("no source family has both native outcome classes")
    return float(np.mean(values)), len(values)


def summary(local_ok, expert_ok, score, groups):
    failure = 1 - local_ok
    macro, valid = macro_auc(failure, score, groups)
    return {
        "routing_objective_pooled": routing_objective(
            expert_ok - local_ok, score),
        "routing_objective_macro_source": macro_routing(
            expert_ok - local_ok, score, groups),
        "native_failure_auc_pooled": float(roc_auc_score(failure, score)),
        "native_failure_auc_macro_source": macro,
        "native_failure_valid_sources": valid,
    }


def zfit(value):
    mean = float(np.mean(value))
    scale = max(float(np.std(value)), 1e-8)
    return mean, scale


def zapply(value, fitted):
    mean, scale = fitted
    return (np.asarray(value) - mean) / scale


def source_balanced_weights(groups):
    groups = np.asarray(groups)
    counts = {group: int(np.sum(groups == group)) for group in set(groups)}
    weights = np.asarray([1. / counts[group] for group in groups],
                         dtype=np.float64)
    return weights / weights.mean()


def residual_target(live, failure):
    model = LogisticRegression(C=1., max_iter=3000, tol=1e-8)
    model.fit(np.asarray(live).reshape(-1, 1), failure)
    return failure - model.predict_proba(
        np.asarray(live).reshape(-1, 1))[:, 1]


def fit_projection(x, local_ok, expert_ok, live, groups):
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    standardized = np.clip((x - mean) / scale, -10., 10.)
    failure = 1 - local_ok
    gain = expert_ok - local_ok
    targets = np.column_stack([
        failure,
        (gain > 0).astype(float),
        (gain < 0).astype(float),
        residual_target(live, failure),
    ]).astype(np.float64)
    directions = []
    for weights in (np.ones(len(x)), source_balanced_weights(groups)):
        target_mean = np.average(targets, axis=0, weights=weights)
        centered = targets - target_mean
        directions.append(standardized.T @ (centered * weights[:, None]))
    raw = np.concatenate(directions, axis=1).astype(np.float64)
    norms = np.linalg.norm(raw, axis=0)
    if np.any(norms <= 1e-12):
        raise RuntimeError("degenerate supervised contrast direction")
    raw /= norms
    basis, singular, _ = np.linalg.svd(raw, full_matrices=False)
    keep = singular > max(float(singular[0]) * 1e-7, 1e-10)
    basis = basis[:, keep].astype(np.float32)
    train_factor = standardized @ basis
    factor_mean = train_factor.mean(0, dtype=np.float64).astype(np.float32)
    factor_scale = np.maximum(
        train_factor.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    train_factor = np.clip(
        (train_factor - factor_mean) / factor_scale, -10., 10.)
    return (mean, scale, basis, factor_mean, factor_scale,
            train_factor, singular[keep])


def project(x, fitted):
    mean, scale, basis, factor_mean, factor_scale, _, _ = fitted
    standardized = np.clip((x - mean) / scale, -10., 10.)
    factor = standardized @ basis
    return np.clip((factor - factor_mean) / factor_scale, -10., 10.)


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"], dtype=np.float32) + float(
        artifact["b"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    causal_ids, causal, layers, causal_paths = load_npz(
        args.causal_dir, "causal_prefill_feats.rank*.npz")
    original_ids, original, _, original_paths = load_npz(
        args.original_dir, "prospective_native_feats.rank*.npz")
    if layers != EXPECTED_LAYERS:
        raise RuntimeError(f"unexpected causal layers {layers}")
    expected = len(layers) * VIEWS_PER_LAYER * HIDDEN
    if causal.shape[1] != expected or not np.isfinite(causal).all():
        raise RuntimeError(f"invalid causal feature matrix {causal.shape}")
    causal_index = {row_id: i for i, row_id in enumerate(causal_ids)}
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
    frame = frame[frame.id.astype(str).isin(causal_index)
                  & frame.id.astype(str).isin(original_index)]
    frame = frame.sort_values("id").reset_index(drop=True)
    x = np.stack([causal[causal_index[str(row_id)]] for row_id in frame.id])
    xo = np.stack([original[original_index[str(row_id)]] for row_id in frame.id])
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    groups = frame.source_family.astype(str).to_numpy()
    gain = expert_ok - local_ok
    failure = 1 - local_ok
    strat = (gain + 1) * 2 + failure
    live = artifact_score(args.live_artifact, xo)

    scores = {
        "anchored_lowrank_alpha0.5": np.full(len(frame), np.nan),
        "lowrank_only_diagnostic": np.full(len(frame), np.nan),
    }
    outer_rows = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(outer.split(x, strat, groups)):
        fitted = fit_projection(
            x[train], local_ok[train], expert_ok[train], live[train],
            groups[train])
        train_factor = fitted[5]
        test_factor = project(x[test], fitted)
        weights = np.where(gain[train] < 0, 2., 1.)
        head = LogisticRegression(C=LOWRANK_C, max_iter=3000, tol=1e-8)
        head.fit(train_factor, failure[train], sample_weight=weights)
        lowrank_train = head.decision_function(train_factor)
        lowrank_test = head.decision_function(test_factor)
        lowrank_z = zapply(lowrank_test, zfit(lowrank_train))
        live_z = zapply(live[test], zfit(live[train]))
        anchored = live_z + ANCHOR_ALPHA * lowrank_z
        scores["anchored_lowrank_alpha0.5"][test] = anchored
        scores["lowrank_only_diagnostic"][test] = lowrank_z
        outer_rows.append({
            "outer_fold": fold,
            "test_groups": sorted(set(groups[test])),
            "projection_rank": int(fitted[2].shape[1]),
            "projection_singular_values": [float(v) for v in fitted[6]],
            "head_coef": [float(v) for v in head.coef_[0]],
            "head_intercept": float(head.intercept_[0]),
            "primary_test": summary(local_ok[test], expert_ok[test],
                                    anchored, groups[test]),
        })
        print(f"outer={fold} rank={fitted[2].shape[1]} "
              f"train={len(train)} test={len(test)}", flush=True)

    if any(not np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("outer cross-fit did not score every row")
    summaries = {"live": summary(local_ok, expert_ok, live, groups)}
    summaries.update({name: summary(local_ok, expert_ok, value, groups)
                      for name, value in scores.items()})
    primary_name = "anchored_lowrank_alpha0.5"
    keys = ("routing_objective_pooled", "routing_objective_macro_source",
            "native_failure_auc_pooled", "native_failure_auc_macro_source")
    deltas = {key: summaries[primary_name][key] - summaries["live"][key]
              for key in keys}
    strata = {}
    for column in ("language", "pool"):
        strata[column] = {}
        for value in sorted(frame[column].astype(str).unique()):
            mask = frame[column].astype(str).to_numpy() == value
            candidate = routing_objective(gain[mask], scores[primary_name][mask])
            baseline = routing_objective(gain[mask], live[mask])
            strata[column][value] = {
                "rows": int(mask.sum()), "candidate_routing": candidate,
                "live_routing": baseline, "delta": candidate - baseline,
            }
    gates = {
        "macro_routing_delta_ge_0.005":
            deltas["routing_objective_macro_source"] >= .005,
        "pooled_routing_nonnegative":
            deltas["routing_objective_pooled"] >= 0.,
        "macro_native_auc_delta_ge_0.010":
            deltas["native_failure_auc_macro_source"] >= .010,
        "pooled_native_auc_nonnegative":
            deltas["native_failure_auc_pooled"] >= 0.,
        "language_routing_nonnegative": all(
            row["delta"] >= 0. for row in strata["language"].values()),
        "broad_pool_routing_ge_minus_0.010": all(
            row["delta"] >= -.010 for row in strata["pool"].values()),
    }
    payload = {
        "status": "development_only",
        "inputs": {
            "selection": {"path": str(args.selection),
                          "sha256": sha256(args.selection)},
            "causal_npz": [{"path": str(path), "sha256": sha256(path)}
                           for path in causal_paths],
            "original_npz": [{"path": str(path), "sha256": sha256(path)}
                             for path in original_paths],
            "local": {"path": str(args.local), "sha256": sha256(args.local)},
            "expert": {"path": str(args.expert),
                       "sha256": sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact),
                              "sha256": sha256(args.live_artifact)},
        },
        "protocol": {
            "rows": len(frame), "outer_folds": 5,
            "grouping": "source_family", "projection_max_rank": 8,
            "projection_targets": ["native_failure", "positive_benefit",
                                   "harmful_escalation", "live_residual"],
            "projection_weighting": ["pooled", "source_balanced"],
            "head": {"family": "logistic", "C": LOWRANK_C,
                     "harm_weight": 2.},
            "primary": {"formula": "z(live)+0.5*z(lowrank)",
                        "alpha": ANCHOR_ALPHA},
            "diagnostic": "lowrank_only",
        },
        "outer_folds": outer_rows,
        "summaries": summaries,
        "primary_deltas_vs_live": deltas,
        "strata": strata,
        "gates": gates,
        "advance": bool(all(gates.values())),
        "live_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summaries": summaries,
                      "primary_deltas_vs_live": deltas,
                      "gates": gates, "advance": payload["advance"]},
                     indent=2))


if __name__ == "__main__":
    main()
