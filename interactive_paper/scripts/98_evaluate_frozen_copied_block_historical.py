"""Fit P31 on all development rows and test P15--P17 unchanged."""
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
from transformers import AutoConfig


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


def read_latest(directory: Path, pattern: str):
    records, paths = {}, sorted(directory.glob(pattern))
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return records, paths


def replay_qc(causal_dir, original_dir, ids, namespace):
    causal, causal_paths = read_latest(causal_dir, "causal_windows.rank*.jsonl")
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
                mismatches.append({"id": row_id, "field": field,
                                   "causal": causal[row_id].get(field),
                                   "original": original[row_id].get(field)})
    seed_exact = all(
        row_id in causal and causal[row_id].get("seed") == int(hashlib.sha256(
            f"{namespace}:{row_id}".encode()).hexdigest()[:8], 16)
        for row_id in ids)
    return ({"rows": len(ids), "fields": list(fields),
             "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
             "seed_namespace": namespace, "seed_exact": seed_exact,
             "exact": not mismatches and seed_exact},
            causal_paths, original_paths)


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

    p31_path = Path(__file__).with_name(
        "97_evaluate_frozen_copied_block_probe.py")
    common_path = Path(__file__).with_name(
        "95_evaluate_copied_block_multiturn.py")
    p31 = import_file("p31", p31_path)
    common = import_file("p29_common_for_p31", common_path)
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
        frame.window_row.to_numpy(), layers.index(p31.TAP_LAYER)].astype(np.float16)
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    failure = 1 - local_ok
    gain = expert_ok - local_ok
    train_live = common.artifact_score(
        args.live_artifact,
        original[frame.original_row.to_numpy()].astype(np.float32))

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    state, index_path, checkpoint_paths = probe.load_block_weights(
        args.model_dir, p31.COPIED_LAYER)
    transform = probe.CopiedBlockProbe(
        config, p31.COPIED_LAYER, state, "attention_only").cuda()
    for parameter in transform.parameters():
        parameter.requires_grad = False
    train_x = p31.copied_features(transform, train_windows, args.batch_size)
    mean = train_x.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        train_x.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    standardized = np.clip((train_x - mean) / scale, -10., 10.)
    head = LogisticRegression(C=p31.LOGISTIC_C, max_iter=3000, tol=1e-5)
    head.fit(standardized, failure, sample_weight=np.where(gain < 0, 2., 1.))
    train_head = head.decision_function(standardized)
    head_fit, live_fit = common.zfit(train_head), common.zfit(train_live)

    results, provenance_external, all_pool_deltas, dataset_gates = {}, {}, [], []
    for name, selection_value, judged_value, windows_value, original_value, seed in args.external:
        selection_path, judged_path = Path(selection_value), Path(judged_value)
        windows_dir, native_dir = Path(windows_value), Path(original_value)
        ci, cx, clengths, external_layers, causal_paths = common.load_npz(
            windows_dir, "causal_windows_feats.rank*.npz", "window_row")
        ni, nx, _, _, native_paths = common.load_npz(
            native_dir, "prospective_native_feats.rank*.npz", "original_row")
        if external_layers != layers or cx.shape[2:] != (8, 4096):
            raise RuntimeError(f"{name}: invalid external windows {cx.shape}")
        if np.any(clengths != 8) or not np.isfinite(cx).all():
            raise RuntimeError(f"{name}: invalid window values")
        rows = pd.read_parquet(selection_path).drop_duplicates("id")
        judged = pd.read_parquet(judged_path).drop_duplicates("id", keep="last")
        rows = rows.merge(judged[["id", "adequate"]], on="id",
                          validate="one_to_one")
        rows = rows.merge(ci, on="id", validate="one_to_one")
        rows = rows.merge(ni, on="id", validate="one_to_one")
        rows = rows.sort_values("id").reset_index(drop=True)
        ids = rows.id.astype(str).tolist()
        qc, causal_traces, original_traces = replay_qc(
            windows_dir, native_dir, ids, seed)
        external_windows = cx[
            rows.window_row.to_numpy(), layers.index(p31.TAP_LAYER)].astype(
                np.float16)
        external_x = p31.copied_features(
            transform, external_windows, args.batch_size)
        external_head = head.decision_function(
            np.clip((external_x - mean) / scale, -10., 10.))
        live = common.artifact_score(
            args.live_artifact,
            nx[rows.original_row.to_numpy()].astype(np.float32))
        candidate = common.zapply(live, live_fit) + p31.ANCHOR_ALPHA * common.zapply(
            external_head, head_fit)
        y = 1 - rows.adequate.astype(int).to_numpy()
        pooled = common.metrics(y, live, candidate)
        by_pool = common.grouped_metrics(rows, y, live, candidate, "pool")
        pool_deltas = [value["auc_delta"] for value in by_pool.values()
                       if value["auc_delta"] is not None]
        all_pool_deltas.extend(pool_deltas)
        gates = {"replay_exact": qc["exact"],
                 "pooled_native_auc_nonnegative": pooled["auc_delta"] >= 0.,
                 "minimum_pool_native_auc_ge_minus_0.010": min(pool_deltas) >= -.010}
        dataset_gates.extend(gates.values())
        results[name] = {"rows": len(rows), "pooled": pooled,
                         "macro_pool_auc_delta": float(np.mean(pool_deltas)),
                         "by_pool": by_pool, "replay_qc": qc,
                         "gates": gates, "all_dataset_gates_pass": all(gates.values())}
        provenance_external[name] = {
            "selection": {"path": str(selection_path),
                          "sha256": common.sha256(selection_path)},
            "judged": {"path": str(judged_path),
                       "sha256": common.sha256(judged_path)},
            "causal_windows": [{"path": str(path), "sha256": common.sha256(path)}
                               for path in causal_paths],
            "original_features": [{"path": str(path), "sha256": common.sha256(path)}
                                  for path in native_paths],
            "causal_traces": [{"path": str(path), "sha256": common.sha256(path)}
                              for path in causal_traces],
            "original_traces": [{"path": str(path), "sha256": common.sha256(path)}
                                for path in original_traces],
        }
    aggregate = {
        "all_dataset_gates_pass": all(dataset_gates),
        "macro_pool_native_auc_delta_ge_0.010": float(
            np.mean(all_pool_deltas)) >= .010,
    }
    payload = {
        "status": "opened_historical_frozen_transform_transfer",
        "protocol": {
            "candidate": "frozen checkpoint L30-to-L31 transform plus logistic",
            "train_rows": len(frame), "copied_block_trainable_parameters": 0,
            "base_model_trainable_parameters": 0, "classifier_parameters": 4097,
            "logistic_C": p31.LOGISTIC_C, "anchor_alpha": p31.ANCHOR_ALPHA,
            "claim_boundary": ("Historical opened P15--P17 transfer only. "
                               "Passing authorizes a new prospective set, not activation."),
        },
        "results": results,
        "aggregate_macro_pool_auc_delta": float(np.mean(all_pool_deltas)),
        "aggregate_gates": aggregate, "all_gates_pass": all(aggregate.values()),
        "activation_recommended": False, "live_unchanged": True,
        "provenance": {
            "script": {"path": str(Path(__file__)),
                       "sha256": common.sha256(Path(__file__))},
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
            "external": provenance_external,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"results": results,
                      "aggregate_macro_pool_auc_delta":
                          payload["aggregate_macro_pool_auc_delta"],
                      "aggregate_gates": aggregate,
                      "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    print("receipt_sha256", common.sha256(args.output))


if __name__ == "__main__":
    main()
