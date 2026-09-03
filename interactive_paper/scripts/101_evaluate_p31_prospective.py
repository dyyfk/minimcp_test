"""Fit frozen P31 and evaluate it once on prospective P32."""
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
from transformers import AutoConfig


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bootstrap(rows, count=2000):
    rng = np.random.default_rng(53)
    failure = 1 - rows.adequate.astype(int).to_numpy()
    live = rows.live.to_numpy()
    candidate = rows.candidate.to_numpy()
    pools = rows.pool.astype(str).to_numpy()
    unique = sorted(set(pools))
    pooled, macro = [], []
    for _ in range(count):
        indices = np.concatenate([
            rng.choice(np.flatnonzero(pools == pool),
                       size=int(np.sum(pools == pool)), replace=True)
            for pool in unique])
        if len(np.unique(failure[indices])) != 2:
            continue
        pooled.append(roc_auc_score(failure[indices], candidate[indices])
                      - roc_auc_score(failure[indices], live[indices]))
        values = []
        for pool in unique:
            chosen = indices[pools[indices] == pool]
            if len(np.unique(failure[chosen])) == 2:
                values.append(roc_auc_score(failure[chosen], candidate[chosen])
                              - roc_auc_score(failure[chosen], live[chosen]))
        macro.append(float(np.mean(values)))
    return {
        "replicates": len(pooled),
        "pooled_auc_delta_ci95": [float(np.quantile(pooled, .025)),
                                   float(np.quantile(pooled, .975))],
        "macro_pool_auc_delta_ci95": [float(np.quantile(macro, .025)),
                                       float(np.quantile(macro, .975))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--train-windows-dir", type=Path, required=True)
    parser.add_argument("--train-original-dir", type=Path, required=True)
    parser.add_argument("--train-local", type=Path, required=True)
    parser.add_argument("--train-expert", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).parent
    p31_path = here / "97_evaluate_frozen_copied_block_probe.py"
    p98_path = here / "98_evaluate_frozen_copied_block_historical.py"
    common_path = here / "95_evaluate_copied_block_multiturn.py"
    p31 = import_file("p31_for_p32", p31_path)
    p98 = import_file("p98_for_p32", p98_path)
    common = import_file("common_for_p32", common_path)
    probe, probe_path = common.load_probe_module()

    wi, windows, lengths, layers, window_paths = common.load_npz(
        args.train_windows_dir, "causal_windows_feats.rank*.npz", "window_row")
    oi, original, _, _, original_paths = common.load_npz(
        args.train_original_dir, "prospective_native_feats.rank*.npz",
        "original_row")
    frame = pd.read_parquet(args.train_selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    local = pd.read_parquet(args.train_local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.train_expert).drop_duplicates("id", keep="last")
    frame = frame.merge(local[["id", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(wi, on="id", validate="one_to_one")
    frame = frame.merge(oi, on="id", validate="one_to_one")
    frame = frame.sort_values("id").reset_index(drop=True)
    if windows.shape[2:] != (8, 4096) or np.any(lengths != 8):
        raise RuntimeError("invalid training windows")
    train_windows = windows[
        frame.window_row.to_numpy(), layers.index(p31.TAP_LAYER)].astype(np.float16)
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    failure, gain = 1 - local_ok, expert_ok - local_ok
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
    scale = np.maximum(train_x.std(0, dtype=np.float64).astype(np.float32), 1e-4)
    standardized = np.clip((train_x - mean) / scale, -10., 10.)
    head = LogisticRegression(C=p31.LOGISTIC_C, max_iter=3000, tol=1e-5)
    head.fit(standardized, failure, sample_weight=np.where(gain < 0, 2., 1.))
    train_head = head.decision_function(standardized)
    head_fit, live_fit = common.zfit(train_head), common.zfit(train_live)

    ci, cx, clengths, external_layers, causal_paths = common.load_npz(
        args.windows_dir, "causal_windows_feats.rank*.npz", "window_row")
    ni, nx, _, _, native_paths = common.load_npz(
        args.original_dir, "prospective_native_feats.rank*.npz", "original_row")
    if external_layers != layers or cx.shape[2:] != (8, 4096):
        raise RuntimeError("invalid prospective windows")
    if np.any(clengths != 8) or not np.isfinite(cx).all():
        raise RuntimeError("invalid prospective window values")
    rows = pd.read_parquet(args.selection).drop_duplicates("id")
    judged = pd.read_parquet(args.judged).drop_duplicates("id", keep="last")
    if judged.adequate.isna().any():
        raise RuntimeError("prospective judgments contain errors")
    rows = rows.merge(judged[["id", "adequate"]], on="id",
                      validate="one_to_one")
    rows = rows.merge(ci, on="id", validate="one_to_one")
    rows = rows.merge(ni, on="id", validate="one_to_one")
    rows = rows.sort_values("id").reset_index(drop=True)
    ids = rows.id.astype(str).tolist()
    qc, causal_traces, original_traces = p98.replay_qc(
        args.windows_dir, args.original_dir, ids, "p15-native")
    external_windows = cx[
        rows.window_row.to_numpy(), layers.index(p31.TAP_LAYER)].astype(np.float16)
    external_x = p31.copied_features(transform, external_windows, args.batch_size)
    raw = head.decision_function(np.clip((external_x - mean) / scale, -10., 10.))
    live = common.artifact_score(
        args.live_artifact, nx[rows.original_row.to_numpy()].astype(np.float32))
    candidate = common.zapply(live, live_fit) + p31.ANCHOR_ALPHA * common.zapply(
        raw, head_fit)
    y = 1 - rows.adequate.astype(int).to_numpy()
    pooled = common.metrics(y, live, candidate)
    by_pool = common.grouped_metrics(rows, y, live, candidate, "pool")
    pool_deltas = [value["auc_delta"] for value in by_pool.values()
                   if value["auc_delta"] is not None]
    macro = float(np.mean(pool_deltas))
    rows["live"], rows["candidate"] = live, candidate
    uncertainty = bootstrap(rows)
    precision_30 = (pooled["budgets"]["0.3"]["candidate_precision"]
                    - pooled["budgets"]["0.3"]["live_precision"])
    gates = {
        "replay_exact": qc["exact"],
        "pooled_native_auc_delta_ge_0.010": pooled["auc_delta"] >= .010,
        "macro_pool_native_auc_delta_ge_0.015": macro >= .015,
        "minimum_pool_native_auc_delta_ge_minus_0.005": min(pool_deltas) >= -.005,
        "budget_0.30_precision_nonnegative": precision_30 >= 0.,
        "pooled_auc_delta_ci95_low_gt_0":
            uncertainty["pooled_auc_delta_ci95"][0] > 0.,
    }
    payload = {
        "status": "prospective_source_disjoint_p31_evaluation",
        "protocol": {
            "candidate": "frozen checkpoint L30-to-L31 transform plus logistic",
            "train_rows": len(frame), "prospective_rows": len(rows),
            "copied_block_trainable_parameters": 0,
            "base_model_trainable_parameters": 0,
            "classifier_parameters": 4097, "logistic_C": p31.LOGISTIC_C,
            "anchor_alpha": p31.ANCHOR_ALPHA,
            "claim_boundary": ("One prospective standalone evaluation. Passing "
                               "supports shadow packaging and latency measurement, "
                               "not automatic live activation."),
        },
        "pooled": pooled, "macro_pool_auc_delta": macro,
        "by_pool": by_pool, "budget_0.30_precision_delta": precision_30,
        "bootstrap": uncertainty, "replay_qc": qc, "gates": gates,
        "all_gates_pass": all(gates.values()),
        "activation_recommended": False, "live_unchanged": True,
        "provenance": {
            "script": {"path": str(Path(__file__)),
                       "sha256": common.sha256(Path(__file__))},
            "p31": {"path": str(p31_path), "sha256": common.sha256(p31_path)},
            "probe": {"path": str(probe_path), "sha256": common.sha256(probe_path)},
            "train_selection": {"path": str(args.train_selection),
                                "sha256": common.sha256(args.train_selection)},
            "train_windows": [{"path": str(path), "sha256": common.sha256(path)}
                              for path in window_paths],
            "train_original": [{"path": str(path), "sha256": common.sha256(path)}
                               for path in original_paths],
            "selection": {"path": str(args.selection),
                          "sha256": common.sha256(args.selection)},
            "judged": {"path": str(args.judged), "sha256": common.sha256(args.judged)},
            "windows": [{"path": str(path), "sha256": common.sha256(path)}
                        for path in causal_paths],
            "original": [{"path": str(path), "sha256": common.sha256(path)}
                         for path in native_paths],
            "causal_traces": [{"path": str(path), "sha256": common.sha256(path)}
                              for path in causal_traces],
            "original_traces": [{"path": str(path), "sha256": common.sha256(path)}
                                for path in original_traces],
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
    print(json.dumps({"pooled": pooled, "macro_pool_auc_delta": macro,
                      "by_pool": {key: value["auc_delta"]
                                  for key, value in by_pool.items()},
                      "bootstrap": uncertainty, "gates": gates,
                      "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    print("receipt_sha256", common.sha256(args.output))


if __name__ == "__main__":
    main()
