"""Evaluate a frozen copied L31 transform with a 4K-parameter linear probe.

This P31 control tests whether P29's 41.9M trainable attention branch
overfit 3,000 rows.  The copied checkpoint block and MiniCPM base are frozen;
only a fold-local logistic classifier on the copied block's mean output is fit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig


TAP_LAYER = 30
COPIED_LAYER = 31
LOGISTIC_C = 3e-4
ANCHOR_ALPHA = .5
RATES = (.15, .30, .50)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def copied_features(model, values, batch_size):
    loader = DataLoader(TensorDataset(torch.from_numpy(values)),
                        batch_size=batch_size, shuffle=False)
    output = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            x = batch.cuda().float()
            length = x.shape[1]
            positions = torch.arange(length, device=x.device)
            position_ids = positions.unsqueeze(0).expand(len(x), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                position_embeddings = model.rotary(x, position_ids)
                mask = torch.full(
                    (length, length), torch.finfo(x.dtype).min,
                    device=x.device, dtype=x.dtype)
                mask = torch.triu(mask, diagonal=1)[None, None]
                transformed = model.block(
                    x, attention_mask=mask, position_ids=position_ids,
                    cache_position=positions,
                    position_embeddings=position_embeddings,
                    use_cache=False)[0].mean(1)
            output.append(transformed.float().cpu().numpy())
    return np.concatenate(output)


def fit_predict(x_train, local_train, expert_train, x_test):
    mean = x_train.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        x_train.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    train = np.clip((x_train - mean) / scale, -10., 10.)
    test = np.clip((x_test - mean) / scale, -10., 10.)
    gain = expert_train - local_train
    weights = np.where(gain < 0, 2., 1.)
    head = LogisticRegression(C=LOGISTIC_C, max_iter=3000, tol=1e-5)
    head.fit(train, 1 - local_train, sample_weight=weights)
    return head.decision_function(train), head.decision_function(test)


def top_mask(score, rate):
    count = int(round(len(score) * rate))
    mask = np.zeros(len(score), dtype=bool)
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    mask[order[:count]] = True
    return mask


def routing(gain, score):
    return float(np.mean([gain[top_mask(score, rate)].sum() / len(gain)
                          for rate in RATES]))


def metrics(rows):
    local = rows.local_ok.to_numpy()
    expert = rows.expert_ok.to_numpy()
    failure, gain = 1 - local, expert - local
    live, candidate = rows.live.to_numpy(), rows.candidate.to_numpy()
    result = {
        "rows": len(rows),
        "routing_delta": routing(gain, candidate) - routing(gain, live),
    }
    if len(np.unique(failure)) == 2:
        result.update({
            "live_auc": float(roc_auc_score(failure, live)),
            "candidate_auc": float(roc_auc_score(failure, candidate)),
            "auc_delta": float(roc_auc_score(failure, candidate)
                               - roc_auc_score(failure, live)),
        })
    else:
        result["auc_delta"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common_path = Path(__file__).with_name(
        "95_evaluate_copied_block_multiturn.py")
    common = import_file("p29_common", common_path)
    probe, probe_path = common.load_probe_module()
    wi, windows, lengths, tap_layers, window_paths = common.load_npz(
        args.windows_dir, "causal_windows_feats.rank*.npz", "window_row")
    oi, original, _, _, original_paths = common.load_npz(
        args.original_dir, "prospective_native_feats.rank*.npz", "original_row")
    if windows.shape[2:] != (8, 4096) or np.any(lengths != 8):
        raise RuntimeError(f"invalid causal windows {windows.shape}")
    if not np.isfinite(windows).all() or TAP_LAYER not in tap_layers:
        raise RuntimeError("invalid causal window values or tap layers")

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
    raw_windows = windows[
        frame.window_row.to_numpy(), tap_layers.index(TAP_LAYER)].astype(
            np.float16)
    original_x = original[frame.original_row.to_numpy()].astype(np.float32)
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    gain = expert_ok - local_ok
    failure = 1 - local_ok
    groups = frame.source_family.astype(str).to_numpy()
    strat = (gain + 1) * 2 + failure
    live = common.artifact_score(args.live_artifact, original_x)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    state, index_path, checkpoint_paths = probe.load_block_weights(
        args.model_dir, COPIED_LAYER)
    transform = probe.CopiedBlockProbe(
        config, COPIED_LAYER, state, "attention_only").cuda()
    for parameter in transform.parameters():
        parameter.requires_grad = False
    x = copied_features(transform, raw_windows, args.batch_size)
    if x.shape != (len(frame), 4096) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid copied-block features {x.shape}")

    splits = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, strat, groups))
    predictions = np.full(len(frame), np.nan)
    fold_rows = []
    for fold, (train, test) in enumerate(splits):
        train_head, test_head = fit_predict(
            x[train], local_ok[train], expert_ok[train], x[test])
        live_fit = common.zfit(live[train])
        head_fit = common.zfit(train_head)
        predictions[test] = (common.zapply(live[test], live_fit)
                             + ANCHOR_ALPHA * common.zapply(test_head, head_fit))
        fold_rows.append({"fold": fold, "train_rows": len(train),
                          "test_rows": len(test),
                          "test_sources": sorted(set(groups[test]))})
    if not np.isfinite(predictions).all():
        raise RuntimeError("incomplete OOF predictions")
    rows = frame[["id", "source_family", "pool", "language",
                  "local_ok", "expert_ok"]].copy()
    rows["live"] = live
    rows["candidate"] = predictions
    pooled = metrics(rows)
    by_source = {name: metrics(value) for name, value in
                 rows.groupby("source_family", sort=True)}
    by_language = {name: metrics(value) for name, value in
                   rows.groupby("language", sort=True)}
    by_pool = {name: metrics(value) for name, value in
               rows.groupby("pool", sort=True)}
    valid_sources = [value for value in by_source.values()
                     if value["auc_delta"] is not None]
    macro_auc = float(np.mean([value["auc_delta"]
                               for value in valid_sources]))
    macro_routing = float(np.mean([value["routing_delta"]
                                   for value in by_source.values()]))
    gates = {
        "macro_routing_delta_ge_0.005": macro_routing >= .005,
        "pooled_routing_nonnegative": pooled["routing_delta"] >= 0.,
        "macro_native_auc_delta_ge_0.010": macro_auc >= .010,
        "pooled_native_auc_nonnegative": pooled["auc_delta"] >= 0.,
        "language_routing_nonnegative": all(
            value["routing_delta"] >= 0. for value in by_language.values()),
        "broad_pool_routing_ge_minus_0.010": all(
            value["routing_delta"] >= -.010 for value in by_pool.values()),
    }
    payload = {
        "status": "opened_development_fixed_frozen_transform_oof",
        "configuration": {
            "tap_layer": TAP_LAYER, "copied_layer": COPIED_LAYER,
            "copied_block_trainable_parameters": 0,
            "base_model_trainable_parameters": 0,
            "classifier_parameters": 4097, "logistic_C": LOGISTIC_C,
            "anchor_alpha": ANCHOR_ALPHA, "window": 8,
            "selection": "none; fixed P29 primary layer and P26 strong-L2 C",
        },
        "rows": len(rows), "feature_shape": list(x.shape), "folds": fold_rows,
        "pooled": pooled,
        "macro_source": {"auc_delta": macro_auc,
                         "routing_delta": macro_routing,
                         "valid_auc_sources": len(valid_sources)},
        "by_source": by_source, "by_language": by_language,
        "by_pool": by_pool, "gates": gates,
        "all_gates_pass": all(gates.values()),
        "activation_recommended": False, "live_unchanged": True,
        "claim_boundary": ("Opened-development OOF control only. Passing "
                           "authorizes unchanged historical transfer, not "
                           "activation."),
        "provenance": {
            "script": {"path": str(Path(__file__)),
                       "sha256": common.sha256(Path(__file__))},
            "common": {"path": str(common_path),
                       "sha256": common.sha256(common_path)},
            "probe": {"path": str(probe_path),
                      "sha256": common.sha256(probe_path)},
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
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled": pooled, "macro_source": payload["macro_source"],
                      "gates": gates}, indent=2))
    print("receipt_sha256", common.sha256(args.output))


if __name__ == "__main__":
    main()
