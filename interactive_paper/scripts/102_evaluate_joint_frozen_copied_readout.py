"""P33: jointly fit live score and the frozen copied-block representation.

P31 fitted a failure head independently and combined it with the live score
afterward.  This fixed control instead gives one strongly regularized logistic
readout the fold-local standardized live score and the 4096-dimensional frozen
L30-to-checkpoint-L31 transform together.  MiniCPM and the copied block remain
fully frozen; there is no candidate or hyperparameter sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoConfig


LOGISTIC_C = 3e-4
TAP_LAYER = 30
COPIED_LAYER = 31


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_external(value):
    parts = value.split("::")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "external must be name::selection::judged::windows::original::seed")
    return parts


def read_latest(directory, pattern):
    rows, paths = {}, sorted(directory.glob(pattern))
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows, paths


def replay_qc(causal_dir, original_dir, ids, namespace):
    causal, causal_paths = read_latest(
        causal_dir, "causal_windows.rank*.jsonl")
    original, original_paths = read_latest(
        original_dir, "prospective_native_traces.rank*.jsonl")
    fields = ("seed", "onset_chunk", "answer_text", "eot_seen", "error")
    mismatches = []
    for row_id in ids:
        if row_id not in causal or row_id not in original:
            mismatches.append({"id": row_id, "field": "missing_trace"})
            continue
        for field in fields:
            if causal[row_id].get(field) != original[row_id].get(field):
                mismatches.append({"id": row_id, "field": field})
    seed_exact = all(
        row_id in causal and causal[row_id].get("seed") == int(
            hashlib.sha256(f"{namespace}:{row_id}".encode()).hexdigest()[:8], 16)
        for row_id in ids)
    return ({"rows": len(ids), "fields": list(fields),
             "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
             "seed_namespace": namespace, "seed_exact": seed_exact,
             "exact": not mismatches and seed_exact},
            causal_paths, original_paths)


def standardizer(values):
    mean = values.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        values.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    return mean, scale


def apply_standardizer(values, fit):
    mean, scale = fit
    return np.clip((values - mean) / scale, -10., 10.)


def design(x, live, x_fit, live_fit):
    standardized = apply_standardizer(x, x_fit)
    live_z = ((live - live_fit[0]) / live_fit[1]).astype(np.float32)
    return np.column_stack([live_z, standardized])


def fit_head(x, live, failure, gain):
    x_fit = standardizer(x)
    live_fit = (float(np.mean(live)), max(float(np.std(live)), 1e-8))
    matrix = design(x, live, x_fit, live_fit)
    head = LogisticRegression(C=LOGISTIC_C, max_iter=3000, tol=1e-5)
    head.fit(matrix, failure, sample_weight=np.where(gain < 0, 2., 1.))
    return head, x_fit, live_fit, head.decision_function(matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--train-windows-dir", type=Path, required=True)
    parser.add_argument("--train-original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--external", action="append", type=parse_external,
                        required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    script_path = Path(__file__)
    p31_path = script_path.with_name("97_evaluate_frozen_copied_block_probe.py")
    common_path = script_path.with_name("95_evaluate_copied_block_multiturn.py")
    p31 = import_file("p31_for_p33", p31_path)
    common = import_file("p29_common_for_p33", common_path)
    probe, probe_path = common.load_probe_module()

    wi, windows, lengths, layers, window_paths = common.load_npz(
        args.train_windows_dir, "causal_windows_feats.rank*.npz", "window_row")
    oi, original, _, _, original_paths = common.load_npz(
        args.train_original_dir, "prospective_native_feats.rank*.npz",
        "original_row")
    frame = pd.read_parquet(args.selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    local = pd.read_parquet(args.local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    frame = frame.merge(local[["id", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(wi, on="id", validate="one_to_one")
    frame = frame.merge(oi, on="id", validate="one_to_one")
    frame = frame.sort_values("id").reset_index(drop=True)
    if windows.shape[2:] != (8, 4096) or np.any(lengths != 8):
        raise RuntimeError(f"invalid training windows {windows.shape}")
    train_windows = windows[
        frame.window_row.to_numpy(), layers.index(TAP_LAYER)].astype(np.float16)
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    failure, gain = 1 - local_ok, expert_ok - local_ok
    groups = frame.source_family.astype(str).to_numpy()
    strat = (gain + 1) * 2 + failure
    live = common.artifact_score(
        args.live_artifact,
        original[frame.original_row.to_numpy()].astype(np.float32))

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    state, index_path, checkpoint_paths = probe.load_block_weights(
        args.model_dir, COPIED_LAYER)
    transform = probe.CopiedBlockProbe(
        config, COPIED_LAYER, state, "attention_only").cuda()
    for parameter in transform.parameters():
        parameter.requires_grad = False
    x = p31.copied_features(transform, train_windows, args.batch_size)
    if x.shape != (len(frame), 4096) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid copied features {x.shape}")

    splits = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, strat, groups))
    oof = np.full(len(frame), np.nan)
    folds = []
    for fold, (train, test) in enumerate(splits):
        head, x_fit, live_fit, _ = fit_head(
            x[train], live[train], failure[train], gain[train])
        oof[test] = head.decision_function(
            design(x[test], live[test], x_fit, live_fit))
        folds.append({"fold": fold, "train_rows": len(train),
                      "test_rows": len(test),
                      "test_sources": sorted(set(groups[test]))})
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete OOF predictions")
    oof_rows = frame[["id", "source_family", "pool", "language",
                      "local_ok", "expert_ok"]].copy()
    oof_rows["live"], oof_rows["candidate"] = live, oof
    pooled_oof = p31.metrics(oof_rows)
    source_oof = {name: p31.metrics(rows) for name, rows in
                  oof_rows.groupby("source_family", sort=True)}
    language_oof = {name: p31.metrics(rows) for name, rows in
                    oof_rows.groupby("language", sort=True)}
    pool_oof = {name: p31.metrics(rows) for name, rows in
                oof_rows.groupby("pool", sort=True)}
    valid_sources = [row for row in source_oof.values()
                     if row["auc_delta"] is not None]
    macro_auc = float(np.mean([row["auc_delta"] for row in valid_sources]))
    macro_routing = float(np.mean(
        [row["routing_delta"] for row in source_oof.values()]))
    oof_gates = {
        "macro_routing_delta_ge_0.005": macro_routing >= .005,
        "pooled_routing_nonnegative": pooled_oof["routing_delta"] >= 0.,
        "macro_native_auc_delta_ge_0.010": macro_auc >= .010,
        "pooled_native_auc_nonnegative": pooled_oof["auc_delta"] >= 0.,
        "language_routing_nonnegative": all(
            row["routing_delta"] >= 0. for row in language_oof.values()),
        "broad_pool_routing_ge_minus_0.010": all(
            row["routing_delta"] >= -.010 for row in pool_oof.values()),
    }

    full_head, x_fit, live_fit, _ = fit_head(x, live, failure, gain)
    external_results, external_provenance, combined = {}, {}, []
    all_pool_deltas, replay_exact = [], []
    for name, selection_value, judged_value, windows_value, native_value, seed in args.external:
        selection_path, judged_path = Path(selection_value), Path(judged_value)
        windows_dir, native_dir = Path(windows_value), Path(native_value)
        ci, cx, clengths, external_layers, causal_paths = common.load_npz(
            windows_dir, "causal_windows_feats.rank*.npz", "window_row")
        ni, nx, _, _, native_paths = common.load_npz(
            native_dir, "prospective_native_feats.rank*.npz", "original_row")
        if external_layers != layers or cx.shape[2:] != (8, 4096):
            raise RuntimeError(f"{name}: invalid external windows {cx.shape}")
        if np.any(clengths != 8) or not np.isfinite(cx).all():
            raise RuntimeError(f"{name}: invalid external window values")
        rows = pd.read_parquet(selection_path).drop_duplicates("id")
        judged = pd.read_parquet(judged_path).drop_duplicates("id", keep="last")
        rows = rows.merge(judged[["id", "adequate"]], on="id",
                          validate="one_to_one")
        rows = rows.merge(ci, on="id", validate="one_to_one")
        rows = rows.merge(ni, on="id", validate="one_to_one")
        rows = rows.sort_values("id").reset_index(drop=True)
        ids = rows.id.astype(str).tolist()
        qc, causal_traces, native_traces = replay_qc(
            windows_dir, native_dir, ids, seed)
        ex = p31.copied_features(
            transform, cx[rows.window_row.to_numpy(),
                          layers.index(TAP_LAYER)].astype(np.float16),
            args.batch_size)
        ex_live = common.artifact_score(
            args.live_artifact, nx[rows.original_row.to_numpy()].astype(np.float32))
        candidate = full_head.decision_function(
            design(ex, ex_live, x_fit, live_fit))
        y = 1 - rows.adequate.astype(int).to_numpy()
        pooled = common.metrics(y, ex_live, candidate)
        by_pool = common.grouped_metrics(rows, y, ex_live, candidate, "pool")
        pool_deltas = [value["auc_delta"] for value in by_pool.values()
                       if value["auc_delta"] is not None]
        precision_delta_30 = (pooled["budgets"]["0.3"]["candidate_precision"]
                              - pooled["budgets"]["0.3"]["live_precision"])
        external_results[name] = {
            "rows": len(rows), "pooled": pooled,
            "macro_pool_auc_delta": float(np.mean(pool_deltas)),
            "budget_0.30_precision_delta": float(precision_delta_30),
            "by_pool": by_pool, "replay_qc": qc,
        }
        all_pool_deltas.extend(pool_deltas)
        replay_exact.append(qc["exact"])
        combined.append(pd.DataFrame({"dataset": name, "failure": y,
                                      "live": ex_live, "candidate": candidate}))
        external_provenance[name] = {
            "selection": {"path": str(selection_path),
                          "sha256": common.sha256(selection_path)},
            "judged": {"path": str(judged_path),
                       "sha256": common.sha256(judged_path)},
            "causal_windows": [{"path": str(path), "sha256": common.sha256(path)}
                               for path in causal_paths],
            "native_features": [{"path": str(path), "sha256": common.sha256(path)}
                                for path in native_paths],
            "causal_traces": [{"path": str(path), "sha256": common.sha256(path)}
                              for path in causal_traces],
            "native_traces": [{"path": str(path), "sha256": common.sha256(path)}
                              for path in native_traces],
        }
    combined = pd.concat(combined, ignore_index=True)
    combined_metrics = common.metrics(
        combined.failure.to_numpy(), combined.live.to_numpy(),
        combined.candidate.to_numpy())
    combined_precision_30 = (
        combined_metrics["budgets"]["0.3"]["candidate_precision"]
        - combined_metrics["budgets"]["0.3"]["live_precision"])
    transfer_gates = {
        "all_replays_exact": all(replay_exact),
        "aggregate_macro_pool_auc_delta_ge_0.010":
            float(np.mean(all_pool_deltas)) >= .010,
        "minimum_pool_auc_delta_ge_minus_0.010": min(all_pool_deltas) >= -.010,
        "each_dataset_pooled_auc_nonnegative": all(
            row["pooled"]["auc_delta"] >= 0.
            for row in external_results.values()),
        "combined_budget_0.30_precision_nonnegative": combined_precision_30 >= 0.,
        "p32_budget_0.30_precision_nonnegative":
            external_results["p32"]["budget_0.30_precision_delta"] >= 0.,
    }
    payload = {
        "status": "opened_development_joint_frozen_copied_readout",
        "protocol": {
            "candidate": "joint logistic over live score plus frozen L30-to-L31 transform",
            "base_model_trainable_parameters": 0,
            "copied_block_trainable_parameters": 0,
            "classifier_parameters": 4098,
            "logistic_C": LOGISTIC_C,
            "selection": "none; P31 representation and strong-L2 C inherited",
            "claim_boundary": ("All external sets are opened development evidence. "
                               "Passing only authorizes one new prospective set."),
        },
        "oof": {"rows": len(frame), "folds": folds, "pooled": pooled_oof,
                "macro_source": {"auc_delta": macro_auc,
                                 "routing_delta": macro_routing},
                "by_source": source_oof, "by_language": language_oof,
                "by_pool": pool_oof, "gates": oof_gates,
                "all_gates_pass": all(oof_gates.values())},
        "external": external_results,
        "aggregate_external_macro_pool_auc_delta": float(np.mean(all_pool_deltas)),
        "combined_external": {
            "metrics": combined_metrics,
            "budget_0.30_precision_delta": float(combined_precision_30)},
        "transfer_gates": transfer_gates,
        "all_gates_pass": all(oof_gates.values()) and all(transfer_gates.values()),
        "activation_recommended": False,
        "live_unchanged": True,
        "provenance": {
            "script": {"path": str(script_path), "sha256": common.sha256(script_path)},
            "p31": {"path": str(p31_path), "sha256": common.sha256(p31_path)},
            "probe": {"path": str(probe_path), "sha256": common.sha256(probe_path)},
            "selection": {"path": str(args.selection),
                          "sha256": common.sha256(args.selection)},
            "windows": [{"path": str(path), "sha256": common.sha256(path)}
                        for path in window_paths],
            "original": [{"path": str(path), "sha256": common.sha256(path)}
                         for path in original_paths],
            "local": {"path": str(args.local), "sha256": common.sha256(args.local)},
            "expert": {"path": str(args.expert), "sha256": common.sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact),
                              "sha256": common.sha256(args.live_artifact)},
            "checkpoint_index": {"path": str(index_path),
                                 "sha256": common.sha256(index_path)},
            "checkpoint_shards": [{"path": str(path), "sha256": common.sha256(path)}
                                  for path in checkpoint_paths],
            "external": external_provenance,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"oof": payload["oof"]["macro_source"],
                      "oof_gates": oof_gates,
                      "external": {name: {"pooled_auc_delta": row["pooled"]["auc_delta"],
                                           "macro_pool_auc_delta": row["macro_pool_auc_delta"],
                                           "budget_0.30_precision_delta": row["budget_0.30_precision_delta"]}
                                   for name, row in external_results.items()},
                      "transfer_gates": transfer_gates,
                      "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    print("receipt_sha256", common.sha256(args.output))


if __name__ == "__main__":
    main()
