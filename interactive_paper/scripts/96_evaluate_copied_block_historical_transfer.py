"""Evaluate frozen primary P29 probe on source-disjoint historical sets.

P15--P17 labels are already known globally but were not inputs to P29 model or
hyperparameter selection.  This is therefore a historical transfer audit, not
new prospective evidence and not an activation authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig


TAP_LAYER = 30
COPIED_LAYER = 31
TRAIN_MODE = "attention_only"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = .01
EPOCHS = 3
ANCHOR_ALPHA = .5


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_latest(directory: Path, pattern: str):
    records, paths = {}, sorted(directory.glob(pattern))
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return records, paths


def replay_qc(causal_dir: Path, original_dir: Path, ids, namespace):
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


def parse_external(value):
    parts = value.split("::")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "external must be name::selection::judged::windows::original::seed")
    return parts


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

    transfer_path = Path(__file__).with_name(
        "95_evaluate_copied_block_multiturn.py")
    common = import_file("p29_transfer", transfer_path)
    probe, probe_path = common.load_probe_module()
    train_i, windows, lengths, tap_layers, window_paths = common.load_npz(
        args.train_windows_dir, "causal_windows_feats.rank*.npz", "window_row")
    original_i, original, _, _, original_paths = common.load_npz(
        args.train_original_dir, "prospective_native_feats.rank*.npz",
        "original_row")
    if windows.shape[2:] != (8, 4096) or np.any(lengths != 8):
        raise RuntimeError(f"invalid training windows {windows.shape}")
    if not np.isfinite(windows).all() or TAP_LAYER not in tap_layers:
        raise RuntimeError("invalid training window values or layers")

    frame = pd.read_parquet(args.selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    local = pd.read_parquet(args.local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    frame = frame.merge(local[["id", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(train_i, on="id", validate="one_to_one")
    frame = frame.merge(original_i, on="id", validate="one_to_one")
    frame = frame.sort_values("id").reset_index(drop=True)
    x = windows[frame.window_row.to_numpy(), tap_layers.index(TAP_LAYER)].astype(
        np.float16)
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    failure = 1 - local_ok
    gain = expert_ok - local_ok
    train_live = common.artifact_score(
        args.live_artifact,
        original[frame.original_row.to_numpy()].astype(np.float32))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    state, index_path, checkpoint_paths = probe.load_block_weights(
        args.model_dir, COPIED_LAYER)
    model = probe.CopiedBlockProbe(
        config, COPIED_LAYER, state, TRAIN_MODE).cuda()
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    dataset = TensorDataset(
        torch.from_numpy(x), torch.from_numpy(failure).float(),
        torch.from_numpy(np.where(gain < 0, 2., 1.)).float())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    history = []
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch_x, batch_y, batch_weight in loader:
            batch_x = batch_x.cuda().float()
            batch_y = batch_y.cuda()
            batch_weight = batch_weight.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch_x)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, batch_y, weight=batch_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1,
                        "train_loss": float(np.mean(losses))})
    train_raw = probe.predict(model, x, args.batch_size)
    head_fit, live_fit = common.zfit(train_raw), common.zfit(train_live)

    results, provenance_external, all_pool_deltas = {}, {}, []
    dataset_gates = []
    for name, selection_value, judged_value, windows_value, original_value, seed in args.external:
        selection_path, judged_path = Path(selection_value), Path(judged_value)
        windows_dir, native_dir = Path(windows_value), Path(original_value)
        ci, cx, clengths, layers, causal_paths = common.load_npz(
            windows_dir, "causal_windows_feats.rank*.npz", "window_row")
        oi, ox, _, _, native_paths = common.load_npz(
            native_dir, "prospective_native_feats.rank*.npz", "original_row")
        if layers != tap_layers or cx.shape[2:] != (8, 4096):
            raise RuntimeError(f"{name}: invalid windows {cx.shape}")
        if np.any(clengths != 8) or not np.isfinite(cx).all():
            raise RuntimeError(f"{name}: invalid window values")
        rows = pd.read_parquet(selection_path).drop_duplicates("id")
        judged = pd.read_parquet(judged_path).drop_duplicates("id", keep="last")
        rows = rows.merge(judged[["id", "adequate"]], on="id",
                          validate="one_to_one")
        rows = rows.merge(ci, on="id", validate="one_to_one")
        rows = rows.merge(oi, on="id", validate="one_to_one")
        rows = rows.sort_values("id").reset_index(drop=True)
        ids = rows.id.astype(str).tolist()
        qc, causal_traces, original_traces = replay_qc(
            windows_dir, native_dir, ids, seed)
        raw = probe.predict(
            model, cx[rows.window_row.to_numpy(),
                      tap_layers.index(TAP_LAYER)].astype(np.float16),
            args.batch_size)
        live = common.artifact_score(
            args.live_artifact,
            ox[rows.original_row.to_numpy()].astype(np.float32))
        candidate = common.zapply(live, live_fit) + ANCHOR_ALPHA * common.zapply(
            raw, head_fit)
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

    aggregate_gates = {
        "all_dataset_gates_pass": all(dataset_gates),
        "macro_pool_native_auc_delta_ge_0.010": float(
            np.mean(all_pool_deltas)) >= .010,
    }
    payload = {
        "status": "opened_historical_source_disjoint_transfer",
        "protocol": {
            "candidate": "P29 primary L30 to copied L31 attention-only",
            "tap_layer": TAP_LAYER, "copied_layer": COPIED_LAYER,
            "train_mode": TRAIN_MODE, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS,
            "train_rows": len(frame), "trainable_parameters": int(sum(
                parameter.numel() for parameter in trainable)),
            "base_model_trainable_parameters": 0,
            "score": "z(live)+0.5*z(copied_block_probe)",
            "claim_boundary": ("P15--P17 labels were globally opened before "
                               "P29, though unused by P29 selection. Passing "
                               "can authorize a new prospective standalone "
                               "set, not activation."),
        },
        "history": history, "results": results,
        "aggregate_macro_pool_auc_delta": float(np.mean(all_pool_deltas)),
        "aggregate_gates": aggregate_gates,
        "all_gates_pass": all(aggregate_gates.values()),
        "activation_recommended": False, "live_unchanged": True,
        "provenance": {
            "script": {"path": str(Path(__file__)),
                       "sha256": common.sha256(Path(__file__))},
            "transfer_implementation": {"path": str(transfer_path),
                                        "sha256": common.sha256(transfer_path)},
            "probe_implementation": {"path": str(probe_path),
                                     "sha256": common.sha256(probe_path)},
            "selection": {"path": str(args.selection),
                          "sha256": common.sha256(args.selection)},
            "local": {"path": str(args.local), "sha256": common.sha256(args.local)},
            "expert": {"path": str(args.expert), "sha256": common.sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact),
                              "sha256": common.sha256(args.live_artifact)},
            "train_windows": [{"path": str(path), "sha256": common.sha256(path)}
                              for path in window_paths],
            "train_original": [{"path": str(path), "sha256": common.sha256(path)}
                               for path in original_paths],
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
                      "aggregate_gates": aggregate_gates,
                      "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    print("receipt_sha256", common.sha256(args.output))


if __name__ == "__main__":
    main()
