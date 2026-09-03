"""Fit frozen P28 on standalone development data and test P22/P23 causally.

P22/P23 labels were previously opened, so these are development transfer
checks only.  Passing cannot authorize activation or prospective claims.
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


EXPECTED_LAYERS = (18, 22, 26, 30)
HIDDEN = 4096
VIEWS_PER_LAYER = 3
LOWRANK_C = .1
ANCHOR_ALPHA = .5
RATES = (.15, .30, .50)


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
        ids.extend(str(item) for item in value["ids"])
        arrays.append(value["X"].astype(np.float32, copy=False))
        if "layers" in value:
            current = tuple(int(item) for item in value["layers"])
            if layers is not None and current != layers:
                raise RuntimeError("inconsistent layer metadata")
            layers = current
    matrix = np.concatenate(arrays)
    index = pd.DataFrame({"id": ids, "row": range(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    return index, matrix, layers, paths


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"], dtype=np.float32) + float(
        artifact["b"])


def zfit(value):
    return float(np.mean(value)), max(float(np.std(value)), 1e-8)


def zapply(value, fitted):
    return (np.asarray(value) - fitted[0]) / fitted[1]


def source_balanced_weights(groups):
    counts = {group: int(np.sum(groups == group)) for group in set(groups)}
    weights = np.asarray([1. / counts[group] for group in groups])
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
        failure, (gain > 0).astype(float), (gain < 0).astype(float),
        residual_target(live, failure),
    ]).astype(np.float64)
    directions = []
    for weights in (np.ones(len(x)), source_balanced_weights(groups)):
        centered = targets - np.average(targets, axis=0, weights=weights)
        directions.append(standardized.T @ (centered * weights[:, None]))
    raw = np.concatenate(directions, axis=1).astype(np.float64)
    norms = np.linalg.norm(raw, axis=0)
    if np.any(norms <= 1e-12):
        raise RuntimeError("degenerate supervised contrast direction")
    raw /= norms
    basis, singular, _ = np.linalg.svd(raw, full_matrices=False)
    keep = singular > max(float(singular[0]) * 1e-7, 1e-10)
    basis = basis[:, keep].astype(np.float32)
    factor = standardized @ basis
    factor_mean = factor.mean(0, dtype=np.float64).astype(np.float32)
    factor_scale = np.maximum(
        factor.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    factor = np.clip((factor - factor_mean) / factor_scale, -10., 10.)
    return mean, scale, basis, factor_mean, factor_scale, factor, singular[keep]


def project(x, fitted):
    mean, scale, basis, factor_mean, factor_scale, _, _ = fitted
    standardized = np.clip((x - mean) / scale, -10., 10.)
    return np.clip(((standardized @ basis) - factor_mean) / factor_scale,
                   -10., 10.)


def top_mask(score, rate):
    count = min(len(score), max(0, int(round(len(score) * rate))))
    mask = np.zeros(len(score), dtype=bool)
    if count:
        order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
        mask[order[:count]] = True
    return mask


def metrics(failure, live, candidate):
    result = {"rows": len(failure), "failures": int(failure.sum()),
              "failure_rate": float(failure.mean()), "budgets": {}}
    if len(np.unique(failure)) == 2:
        result["live_auc"] = float(roc_auc_score(failure, live))
        result["candidate_auc"] = float(roc_auc_score(failure, candidate))
        result["auc_delta"] = result["candidate_auc"] - result["live_auc"]
    else:
        result.update(live_auc=None, candidate_auc=None, auc_delta=None)
    positives = max(1, int(failure.sum()))
    for rate in RATES:
        live_mask, candidate_mask = top_mask(live, rate), top_mask(candidate, rate)
        result["budgets"][str(rate)] = {
            "live_precision": float(failure[live_mask].mean()),
            "candidate_precision": float(failure[candidate_mask].mean()),
            "live_recall": float(failure[live_mask].sum() / positives),
            "candidate_recall": float(failure[candidate_mask].sum() / positives),
            "selection_agreement": float(np.mean(live_mask == candidate_mask)),
        }
    return result


def read_latest(directory: Path, pattern: str):
    records, paths = {}, sorted(directory.glob(pattern))
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return records, paths


def replay_qc(causal_dir: Path, original_dir: Path, expected_ids):
    causal, causal_paths = read_latest(causal_dir, "causal_multiturn.rank*.jsonl")
    original, original_paths = read_latest(
        original_dir, "controlled_multiturn.rank*.jsonl")
    fields = ("seed", "carrier_answer", "carrier_eot_seen", "target_answer",
              "target_eot_seen", "target_onset_chunk", "error")
    mismatches = []
    for row_id in expected_ids:
        if row_id not in causal or row_id not in original:
            mismatches.append({"id": row_id, "field": "missing_trace"})
            continue
        for field in fields:
            if causal[row_id].get(field) != original[row_id].get(field):
                mismatches.append({"id": row_id, "field": field,
                                   "causal": causal[row_id].get(field),
                                   "original": original[row_id].get(field)})
    return {"rows": len(expected_ids), "fields": list(fields),
            "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
            "exact": not mismatches}, causal_paths, original_paths


def parse_external(value):
    parts = value.split("::")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "external must be name::pairs::judged::causal_dir::original_dir::seed")
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--train-causal-dir", type=Path, required=True)
    parser.add_argument("--train-original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--external", action="append", type=parse_external,
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_ci, train_x, layers, train_causal_paths = load_npz(
        args.train_causal_dir, "causal_prefill_feats.rank*.npz")
    train_oi, train_xo, _, train_original_paths = load_npz(
        args.train_original_dir, "prospective_native_feats.rank*.npz")
    if layers != EXPECTED_LAYERS or train_x.shape[1] != (
            len(layers) * VIEWS_PER_LAYER * HIDDEN):
        raise RuntimeError(f"invalid training causal matrix {train_x.shape}")
    if not np.isfinite(train_x).all():
        raise RuntimeError("non-finite training causal features")

    frame = pd.read_parquet(args.selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    local = pd.read_parquet(args.local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    frame = frame.merge(local[["id", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    ci = dict(zip(train_ci.id.astype(str), train_ci.row))
    oi = dict(zip(train_oi.id.astype(str), train_oi.row))
    frame = frame[frame.id.astype(str).isin(ci) & frame.id.astype(str).isin(oi)]
    frame = frame.sort_values("id").reset_index(drop=True)
    x = np.stack([train_x[ci[str(row_id)]] for row_id in frame.id])
    xo = np.stack([train_xo[oi[str(row_id)]] for row_id in frame.id])
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    groups = frame.source_family.astype(str).to_numpy()
    gain = expert_ok - local_ok
    failure = 1 - local_ok
    live = artifact_score(args.live_artifact, xo)
    fitted = fit_projection(x, local_ok, expert_ok, live, groups)
    weights = np.where(gain < 0, 2., 1.)
    head = LogisticRegression(C=LOWRANK_C, max_iter=3000, tol=1e-8)
    head.fit(fitted[5], failure, sample_weight=weights)
    lowrank_train = head.decision_function(fitted[5])
    lowrank_zfit, live_zfit = zfit(lowrank_train), zfit(live)

    results, all_transfer_gates = {}, []
    provenance_external = {}
    for name, pairs_value, judged_value, causal_value, original_value, seed in args.external:
        pairs_path, judged_path = Path(pairs_value), Path(judged_value)
        causal_dir, original_dir = Path(causal_value), Path(original_value)
        pairs = pd.read_parquet(pairs_path)
        judged = pd.read_parquet(judged_path).drop_duplicates("id", keep="last")
        causal_i, causal_x, external_layers, causal_paths = load_npz(
            causal_dir, "causal_multiturn_feats.rank*.npz")
        original_i, original_x, _, original_paths = load_npz(
            original_dir, "controlled_multiturn_feats.rank*.npz")
        if external_layers != EXPECTED_LAYERS or causal_x.shape[1] != x.shape[1]:
            raise RuntimeError(f"{name}: invalid causal matrix {causal_x.shape}")
        if not np.isfinite(causal_x).all():
            raise RuntimeError(f"{name}: non-finite causal features")
        cidx = dict(zip(causal_i.id.astype(str), causal_i.row))
        oidx = dict(zip(original_i.id.astype(str), original_i.row))
        rows = pairs.merge(judged[["id", "adequate"]], on="id",
                           validate="one_to_one")
        rows = rows[rows.id.astype(str).isin(cidx) & rows.id.astype(str).isin(oidx)]
        rows = rows.sort_values("id").reset_index(drop=True)
        ids = rows.id.astype(str).tolist()
        cx = np.stack([causal_x[cidx[row_id]] for row_id in ids])
        ox = np.stack([original_x[oidx[row_id]] for row_id in ids])
        external_live = artifact_score(args.live_artifact, ox)
        factor = project(cx, fitted)
        lowrank = head.decision_function(factor)
        candidate = (zapply(external_live, live_zfit)
                     + ANCHOR_ALPHA * zapply(lowrank, lowrank_zfit))
        y = 1 - rows.adequate.astype(int).to_numpy()
        pooled = metrics(y, external_live, candidate)
        by_pool = {}
        for pool in sorted(rows.target_pool.astype(str).unique()):
            mask = rows.target_pool.astype(str).to_numpy() == pool
            by_pool[pool] = metrics(y[mask], external_live[mask], candidate[mask])
        by_language = {}
        for language in sorted(rows.language.astype(str).unique()):
            mask = rows.language.astype(str).to_numpy() == language
            by_language[language] = metrics(
                y[mask], external_live[mask], candidate[mask])
        valid_pool_deltas = [value["auc_delta"] for value in by_pool.values()
                             if value["auc_delta"] is not None]
        valid_language_deltas = [value["auc_delta"] for value in by_language.values()
                                 if value["auc_delta"] is not None]
        macro_delta = float(np.mean(valid_pool_deltas))
        qc, causal_trace_paths, original_trace_paths = replay_qc(
            causal_dir, original_dir, ids)
        expected_seed = [int(hashlib.sha256(
            f"{seed}:{row_id}".encode()).hexdigest()[:8], 16) for row_id in ids]
        causal_records, _ = read_latest(causal_dir, "causal_multiturn.rank*.jsonl")
        seed_exact = all(causal_records[row_id]["seed"] == value
                         for row_id, value in zip(ids, expected_seed))
        gates = {
            "replay_exact": qc["exact"] and seed_exact,
            "pooled_native_auc_nonnegative": pooled["auc_delta"] >= 0.,
            "macro_pool_native_auc_nonnegative": macro_delta >= 0.,
            "language_native_auc_nonnegative": all(
                value >= 0. for value in valid_language_deltas),
            "minimum_pool_native_auc_ge_minus_0.010": min(
                valid_pool_deltas) >= -.010,
        }
        all_transfer_gates.extend(gates.values())
        results[name] = {"rows": len(rows), "pooled": pooled,
                         "macro_pool_auc_delta": macro_delta,
                         "by_pool": by_pool, "by_language": by_language,
                         "replay_qc": {**qc, "seed_namespace": seed,
                                       "seed_exact": seed_exact},
                         "transfer_gates": gates,
                         "all_transfer_gates_pass": all(gates.values())}
        provenance_external[name] = {
            "pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "judged": {"path": str(judged_path), "sha256": sha256(judged_path)},
            "causal_features": [{"path": str(path), "sha256": sha256(path)}
                                for path in causal_paths],
            "original_features": [{"path": str(path), "sha256": sha256(path)}
                                  for path in original_paths],
            "causal_traces": [{"path": str(path), "sha256": sha256(path)}
                              for path in causal_trace_paths],
            "original_traces": [{"path": str(path), "sha256": sha256(path)}
                                for path in original_trace_paths],
        }

    payload = {
        "status": "opened_development_multiturn_transfer_only",
        "protocol": {
            "train_rows": len(frame), "projection_rank": int(fitted[2].shape[1]),
            "projection_targets": ["native_failure", "positive_benefit",
                                   "harmful_escalation", "live_residual"],
            "head": {"family": "logistic", "C": LOWRANK_C,
                     "harm_weight": 2.},
            "score": "z(live)+0.5*z(lowrank)",
            "claim_boundary": ("P22/P23 labels were already opened; passing "
                               "only supports development transfer and cannot "
                               "authorize activation."),
        },
        "results": results,
        "all_transfer_gates_pass": all(all_transfer_gates),
        "activation_recommended": False,
        "live_unchanged": True,
        "provenance": {
            "selection": {"path": str(args.selection),
                          "sha256": sha256(args.selection)},
            "local": {"path": str(args.local), "sha256": sha256(args.local)},
            "expert": {"path": str(args.expert), "sha256": sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact),
                              "sha256": sha256(args.live_artifact)},
            "train_causal_features": [
                {"path": str(path), "sha256": sha256(path)}
                for path in train_causal_paths],
            "train_original_features": [
                {"path": str(path), "sha256": sha256(path)}
                for path in train_original_paths],
            "external": provenance_external,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"results": results,
                      "all_transfer_gates_pass": payload["all_transfer_gates_pass"]},
                     indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
