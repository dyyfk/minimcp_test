"""Fit frozen P29 copied-block finalists and test P22/P23 transfer.

The MiniCPM base model is not instantiated for training.  Each probe is a
separate checkpoint-initialized next decoder block plus a classifier.  P22 and
P23 labels are already opened, so these results are development transfer
evidence only and cannot authorize activation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig


RATES = (.15, .30, .50)
ANCHOR_ALPHA = .5
FINALISTS = (
    {"name": "l22_full", "tap_layer": 22, "copied_layer": 23,
     "train_mode": "full_block", "learning_rate": 1e-4,
     "weight_decay": 0., "epochs": 2},
    {"name": "l30_attention", "tap_layer": 30, "copied_layer": 31,
     "train_mode": "attention_only", "learning_rate": 3e-5,
     "weight_decay": .01, "epochs": 3},
)


def load_probe_module():
    path = Path(__file__).with_name("90_train_copied_block_probe.py")
    spec = importlib.util.spec_from_file_location("p29_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(directory: Path, pattern: str, row_name: str):
    ids, values, lengths, paths = [], [], [], sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    tap_layers = None
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        ids.extend(str(value) for value in archive["ids"])
        values.append(archive["X"])
        if "lengths" in archive:
            lengths.extend(int(value) for value in archive["lengths"])
        if "tap_layers" in archive:
            current = tuple(int(value) for value in archive["tap_layers"])
            if tap_layers is not None and current != tap_layers:
                raise RuntimeError("inconsistent tap layers")
            tap_layers = current
    index = pd.DataFrame({"id": ids, row_name: range(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError(f"duplicate IDs in {directory}")
    return (index, np.concatenate(values), np.asarray(lengths), tap_layers,
            paths)


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"], dtype=np.float32) + float(
        artifact["b"])


def zfit(value):
    return float(np.mean(value)), max(float(np.std(value)), 1e-8)


def zapply(value, fitted):
    return (np.asarray(value) - fitted[0]) / fitted[1]


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
        live_mask = top_mask(live, rate)
        candidate_mask = top_mask(candidate, rate)
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


def replay_qc(causal_dir: Path, original_dir: Path, expected_ids, seed):
    causal, causal_paths = read_latest(
        causal_dir, "causal_multiturn_windows.rank*.jsonl")
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
    expected_seeds = {row_id: int(hashlib.sha256(
        f"{seed}:{row_id}".encode()).hexdigest()[:8], 16)
                      for row_id in expected_ids}
    seed_exact = all(row_id in causal and causal[row_id].get("seed") == value
                     for row_id, value in expected_seeds.items())
    return ({"rows": len(expected_ids), "fields": list(fields),
             "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
             "seed_namespace": seed, "seed_exact": seed_exact,
             "exact": not mismatches and seed_exact},
            causal_paths, original_paths)


def parse_external(value):
    parts = value.split("::")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "external must be name::pairs::judged::windows_dir::original_dir::seed")
    return parts


def grouped_metrics(rows, failure, live, candidate, column):
    output = {}
    groups = rows[column].astype(str).to_numpy()
    for group in sorted(set(groups)):
        mask = groups == group
        output[group] = metrics(failure[mask], live[mask], candidate[mask])
    return output


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

    probe, probe_path = load_probe_module()
    train_i, train_windows, train_lengths, tap_layers, train_window_paths = load_npz(
        args.train_windows_dir, "causal_windows_feats.rank*.npz", "window_row")
    original_i, original, _, _, train_original_paths = load_npz(
        args.train_original_dir, "prospective_native_feats.rank*.npz", "original_row")
    if train_windows.shape[2:] != (8, 4096):
        raise RuntimeError(f"invalid training windows {train_windows.shape}")
    if np.any(train_lengths != 8) or not np.isfinite(train_windows).all():
        raise RuntimeError("training windows must be finite eight-token reads")

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
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    failure = 1 - local_ok
    gain = expert_ok - local_ok
    original_x = original[frame.original_row.to_numpy()].astype(np.float32)
    train_live = artifact_score(args.live_artifact, original_x)
    live_fit = zfit(train_live)

    external = {}
    external_provenance = {}
    for name, pairs_value, judged_value, windows_value, original_value, seed in args.external:
        pairs_path, judged_path = Path(pairs_value), Path(judged_value)
        windows_dir, original_dir = Path(windows_value), Path(original_value)
        windows_i, windows, lengths, layers, window_paths = load_npz(
            windows_dir, "causal_multiturn_windows_feats.rank*.npz", "window_row")
        native_i, native, _, _, native_paths = load_npz(
            original_dir, "controlled_multiturn_feats.rank*.npz", "original_row")
        if layers != tap_layers or windows.shape[2:] != (8, 4096):
            raise RuntimeError(f"{name}: invalid window data {windows.shape}")
        if np.any(lengths != 8) or not np.isfinite(windows).all():
            raise RuntimeError(f"{name}: invalid causal windows")
        pairs = pd.read_parquet(pairs_path)
        judged = pd.read_parquet(judged_path).drop_duplicates("id", keep="last")
        rows = pairs.merge(judged[["id", "adequate"]], on="id",
                           validate="one_to_one")
        rows = rows.merge(windows_i, on="id", validate="one_to_one")
        rows = rows.merge(native_i, on="id", validate="one_to_one")
        rows = rows.sort_values("id").reset_index(drop=True)
        ids = rows.id.astype(str).tolist()
        qc, causal_trace_paths, original_trace_paths = replay_qc(
            windows_dir, original_dir, ids, seed)
        external[name] = {
            "rows": rows, "windows": windows, "native": native,
            "failure": 1 - rows.adequate.astype(int).to_numpy(), "qc": qc,
        }
        external_provenance[name] = {
            "pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "judged": {"path": str(judged_path), "sha256": sha256(judged_path)},
            "causal_windows": [{"path": str(path), "sha256": sha256(path)}
                               for path in window_paths],
            "original_features": [{"path": str(path), "sha256": sha256(path)}
                                  for path in native_paths],
            "causal_traces": [{"path": str(path), "sha256": sha256(path)}
                              for path in causal_trace_paths],
            "original_traces": [{"path": str(path), "sha256": sha256(path)}
                                for path in original_trace_paths],
        }

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    finalist_results = {}
    checkpoint_inputs = {}
    for finalist in FINALISTS:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        tap_layer = finalist["tap_layer"]
        if tap_layer not in tap_layers:
            raise RuntimeError(f"tap layer {tap_layer} absent from {tap_layers}")
        x = train_windows[frame.window_row.to_numpy(),
                          tap_layers.index(tap_layer)].astype(np.float16)
        state, index_path, checkpoint_paths = probe.load_block_weights(
            args.model_dir, finalist["copied_layer"])
        model = probe.CopiedBlockProbe(
            config, finalist["copied_layer"], state,
            finalist["train_mode"]).cuda()
        trainable = [parameter for parameter in model.parameters()
                     if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=finalist["learning_rate"],
            weight_decay=finalist["weight_decay"])
        dataset = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(failure).float(),
            torch.from_numpy(np.where(gain < 0, 2., 1.)).float())
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        history = []
        for epoch in range(finalist["epochs"]):
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
        head_fit = zfit(train_raw)
        dataset_results = {}
        all_gates = []
        for name, value in external.items():
            rows = value["rows"]
            external_x = value["windows"][
                rows.window_row.to_numpy(), tap_layers.index(tap_layer)
            ].astype(np.float16)
            raw = probe.predict(model, external_x, args.batch_size)
            native_x = value["native"][rows.original_row.to_numpy()].astype(
                np.float32)
            live = artifact_score(args.live_artifact, native_x)
            candidate = zapply(live, live_fit) + ANCHOR_ALPHA * zapply(
                raw, head_fit)
            y = value["failure"]
            pooled = metrics(y, live, candidate)
            by_pool = grouped_metrics(rows, y, live, candidate, "target_pool")
            by_language = grouped_metrics(rows, y, live, candidate, "language")
            pool_deltas = [item["auc_delta"] for item in by_pool.values()
                           if item["auc_delta"] is not None]
            language_deltas = [item["auc_delta"] for item in by_language.values()
                               if item["auc_delta"] is not None]
            macro_delta = float(np.mean(pool_deltas))
            gates = {
                "replay_exact": value["qc"]["exact"],
                "pooled_native_auc_nonnegative": pooled["auc_delta"] >= 0.,
                "macro_pool_native_auc_nonnegative": macro_delta >= 0.,
                "language_native_auc_nonnegative": all(
                    item >= 0. for item in language_deltas),
                "minimum_pool_native_auc_ge_minus_0.010": min(
                    pool_deltas) >= -.010,
            }
            all_gates.extend(gates.values())
            dataset_results[name] = {
                "rows": len(rows), "pooled": pooled,
                "macro_pool_auc_delta": macro_delta,
                "by_pool": by_pool, "by_language": by_language,
                "replay_qc": value["qc"], "transfer_gates": gates,
                "all_transfer_gates_pass": all(gates.values()),
            }
        finalist_results[finalist["name"]] = {
            "configuration": {**finalist, "window": 8,
                              "anchor_alpha": ANCHOR_ALPHA,
                              "train_rows": len(frame),
                              "trainable_parameters": int(sum(
                                  parameter.numel() for parameter in trainable)),
                              "base_model_trainable_parameters": 0},
            "history": history, "results": dataset_results,
            "all_transfer_gates_pass": all(all_gates),
        }
        checkpoint_inputs[finalist["name"]] = {
            "checkpoint_index": {"path": str(index_path),
                                 "sha256": sha256(index_path)},
            "checkpoint_shards": [{"path": str(path), "sha256": sha256(path)}
                                  for path in checkpoint_paths],
        }
        del model, optimizer, dataset, loader, state, trainable
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "status": "opened_development_multiturn_transfer_only",
        "protocol": {
            "finalists": list(FINALISTS), "training": "all 3000 standalone rows",
            "epoch_rule": "fixed median selected epoch from five-fold OOF",
            "loss": "failure BCE with 2x harmful-escalation weight",
            "score": "z(live)+0.5*z(copied_block_probe)",
            "claim_boundary": ("P22/P23 labels were already opened; passing "
                               "supports development transfer only and cannot "
                               "authorize activation."),
        },
        "finalists": finalist_results,
        "any_finalist_all_transfer_gates_pass": any(
            value["all_transfer_gates_pass"]
            for value in finalist_results.values()),
        "activation_recommended": False,
        "live_unchanged": True,
        "provenance": {
            "script": {"path": str(Path(__file__)),
                       "sha256": sha256(Path(__file__))},
            "probe_implementation": {"path": str(probe_path),
                                     "sha256": sha256(probe_path)},
            "selection": {"path": str(args.selection),
                          "sha256": sha256(args.selection)},
            "local": {"path": str(args.local), "sha256": sha256(args.local)},
            "expert": {"path": str(args.expert), "sha256": sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact),
                              "sha256": sha256(args.live_artifact)},
            "train_windows": [{"path": str(path), "sha256": sha256(path)}
                              for path in train_window_paths],
            "train_original": [{"path": str(path), "sha256": sha256(path)}
                               for path in train_original_paths],
            "checkpoint": checkpoint_inputs,
            "external": external_provenance,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"finalists": finalist_results,
                      "any_finalist_all_transfer_gates_pass":
                          payload["any_finalist_all_transfer_gates_pass"]},
                     indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
