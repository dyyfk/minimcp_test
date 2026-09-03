"""Screen a copied-next-decoder-block probe on causal token windows.

This is an opened-development structural screen.  The MiniCPM base model is
never instantiated for training: one next decoder block is initialized from
the checkpoint into a separate branch, and only that branch plus a classifier
may be optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding,
)


RATES = (.15, .30, .50)
ANCHOR_ALPHA = .5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_windows(directory: Path):
    ids, values, lengths, paths = [], [], [], sorted(
        directory.glob("causal_windows_feats.rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"no causal window shards under {directory}")
    tap_layers = None
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        ids.extend(str(value) for value in archive["ids"])
        values.append(archive["X"].astype(np.float16, copy=False))
        lengths.extend(int(value) for value in archive["lengths"])
        current = tuple(int(value) for value in archive["tap_layers"])
        if tap_layers is not None and current != tap_layers:
            raise RuntimeError("inconsistent tap layers")
        tap_layers = current
    index = pd.DataFrame({"id": ids, "feature_row": range(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    matrix = np.concatenate(values)
    return index, matrix, np.asarray(lengths), tap_layers, paths


def load_original(directory: Path):
    ids, values = [], []
    paths = sorted(directory.glob("prospective_native_feats.rank*.npz"))
    for path in paths:
        archive = np.load(path, allow_pickle=True)
        ids.extend(str(value) for value in archive["ids"])
        values.append(archive["X"].astype(np.float32, copy=False))
    if not paths:
        raise FileNotFoundError(f"no native feature shards under {directory}")
    index = pd.DataFrame({"id": ids, "original_row": range(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError("duplicate original feature IDs")
    return index, np.concatenate(values), paths


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"], dtype=np.float32) + float(
        artifact["b"])


def load_block_weights(model_dir: Path, layer: int):
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = f"llm.model.layers.{layer}."
    selected = {key: shard for key, shard in weight_map.items()
                if key.startswith(prefix)}
    if not selected:
        raise RuntimeError(f"no checkpoint weights for layer {layer}")
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for key, key_shard in selected.items():
                if key_shard == shard:
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
    return state, index_path, sorted({model_dir / value for value in selected.values()})


class CopiedBlockProbe(nn.Module):
    def __init__(self, config, next_layer, state, mode):
        super().__init__()
        self.mode = mode
        self.block = Qwen3DecoderLayer(config, layer_idx=next_layer)
        missing, unexpected = self.block.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
        self.rotary = Qwen3RotaryEmbedding(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        if mode == "attention_only":
            for parameter in self.block.parameters():
                parameter.requires_grad = False
            for parameter in self.block.self_attn.parameters():
                parameter.requires_grad = True
        elif mode != "full_block":
            raise ValueError(f"unknown train mode {mode}")

    def forward(self, x):
        batch, length, _ = x.shape
        positions = torch.arange(length, device=x.device)
        position_ids = positions.unsqueeze(0).expand(batch, -1)
        position_embeddings = self.rotary(x, position_ids)
        mask = torch.full((length, length), torch.finfo(x.dtype).min,
                          device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)[None, None]
        output = self.block(
            x, attention_mask=mask, position_ids=position_ids,
            cache_position=positions, position_embeddings=position_embeddings,
            use_cache=False)[0]
        return self.classifier(output.mean(1)).squeeze(-1).float()


def top_mask(score, rate):
    count = int(round(len(score) * rate))
    mask = np.zeros(len(score), dtype=bool)
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    mask[order[:count]] = True
    return mask


def routing(gain, score):
    return float(np.mean([gain[top_mask(score, rate)].sum() / len(gain)
                          for rate in RATES]))


def summarize(local_ok, expert_ok, live, candidate, groups):
    failure, gain = 1 - local_ok, expert_ok - local_ok
    pool_rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        if len(np.unique(failure[mask])) != 2:
            continue
        pool_rows.append({
            "group": group,
            "live_auc": float(roc_auc_score(failure[mask], live[mask])),
            "candidate_auc": float(roc_auc_score(failure[mask], candidate[mask])),
            "live_routing": routing(gain[mask], live[mask]),
            "candidate_routing": routing(gain[mask], candidate[mask]),
        })
    live_macro_auc = float(np.mean([row["live_auc"] for row in pool_rows]))
    candidate_macro_auc = float(np.mean([
        row["candidate_auc"] for row in pool_rows]))
    live_macro_routing = float(np.mean([
        row["live_routing"] for row in pool_rows]))
    candidate_macro_routing = float(np.mean([
        row["candidate_routing"] for row in pool_rows]))
    return {
        "rows": len(local_ok),
        "live": {
            "native_auc_pooled": float(roc_auc_score(failure, live)),
            "native_auc_macro": live_macro_auc,
            "routing_pooled": routing(gain, live),
            "routing_macro": live_macro_routing,
        },
        "candidate": {
            "native_auc_pooled": float(roc_auc_score(failure, candidate)),
            "native_auc_macro": candidate_macro_auc,
            "routing_pooled": routing(gain, candidate),
            "routing_macro": candidate_macro_routing,
        },
        "delta": {
            "native_auc_pooled": float(roc_auc_score(failure, candidate)
                                       - roc_auc_score(failure, live)),
            "native_auc_macro": candidate_macro_auc - live_macro_auc,
            "routing_pooled": routing(gain, candidate) - routing(gain, live),
            "routing_macro": candidate_macro_routing - live_macro_routing,
        },
        "by_source": pool_rows,
    }


def predict(model, x, batch_size):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)),
                        batch_size=batch_size, shuffle=False)
    output = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch.cuda().float())
            output.append(logits.float().cpu().numpy())
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tap-layer", type=int, required=True)
    parser.add_argument("--train-mode", choices=("attention_only", "full_block"),
                        required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    window_index, windows, lengths, tap_layers, window_paths = load_windows(
        args.windows_dir)
    if args.tap_layer not in tap_layers:
        raise RuntimeError(f"tap layer {args.tap_layer} absent from {tap_layers}")
    if windows.shape[2:] != (8, 4096) or not np.isfinite(windows).all():
        raise RuntimeError(f"invalid causal windows {windows.shape}")
    if np.any(lengths != 8):
        raise RuntimeError("screen protocol requires eight valid tokens per row")
    original_index, original, original_paths = load_original(args.original_dir)
    frame = pd.read_parquet(args.selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    local = pd.read_parquet(args.local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    frame = frame.merge(local[["id", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(window_index, on="id", validate="one_to_one")
    frame = frame.merge(original_index, on="id", validate="one_to_one")
    frame = frame.sort_values("id").reset_index(drop=True)
    layer_column = tap_layers.index(args.tap_layer)
    x = windows[frame.feature_row.to_numpy(), layer_column]
    xo = original[frame.original_row.to_numpy()]
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    failure, gain = 1 - local_ok, expert_ok - local_ok
    groups = frame.source_family.astype(str).to_numpy()
    strat = (gain + 1) * 2 + failure
    live = artifact_score(args.live_artifact, xo)
    splits = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, strat, groups))
    if not 0 <= args.fold_index < len(splits):
        raise ValueError("fold index must be in [0, 4]")
    train, validation = splits[args.fold_index]

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    state, index_path, checkpoint_paths = load_block_weights(
        args.model_dir, args.tap_layer + 1)
    model = CopiedBlockProbe(config, args.tap_layer + 1, state,
                             args.train_mode).cuda()
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    dataset = TensorDataset(
        torch.from_numpy(x[train]), torch.from_numpy(failure[train]).float(),
        torch.from_numpy(np.where(gain[train] < 0, 2., 1.)).float())
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=generator, drop_last=False)
    best = None
    history, stale = [], 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y, batch_weight in loader:
            batch_x = batch_x.cuda().float()
            batch_y, batch_weight = batch_y.cuda(), batch_weight.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch_x)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, batch_y, weight=batch_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_raw = predict(model, x[validation], args.batch_size)
        validation_auc = float(roc_auc_score(failure[validation], validation_raw))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_failure_auc": validation_auc})
        if best is None or validation_auc > best[0] + 1e-5:
            best = (validation_auc, epoch + 1,
                    {key: value.detach().cpu().clone()
                     for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best[2])
    train_raw = predict(model, x[train], args.batch_size)
    validation_raw = predict(model, x[validation], args.batch_size)
    live_mean, live_scale = float(live[train].mean()), float(live[train].std())
    head_mean, head_scale = float(train_raw.mean()), float(train_raw.std())
    candidate = ((live[validation] - live_mean) / max(live_scale, 1e-8)
                 + ANCHOR_ALPHA * (validation_raw - head_mean)
                 / max(head_scale, 1e-8))
    result = summarize(local_ok[validation], expert_ok[validation],
                       live[validation], candidate, groups[validation])
    result["gates"] = {
        "macro_routing_delta_ge_0.005": result["delta"]["routing_macro"] >= .005,
        "pooled_routing_nonnegative": result["delta"]["routing_pooled"] >= 0.,
        "macro_native_auc_delta_ge_0.010": result["delta"]["native_auc_macro"] >= .010,
        "pooled_native_auc_nonnegative": result["delta"]["native_auc_pooled"] >= 0.,
    }
    payload = {
        "status": "opened_development_structural_screen",
        "configuration": {
            "tap_layer": args.tap_layer, "copied_layer": args.tap_layer + 1,
            "train_mode": args.train_mode, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "window": 8,
            "anchor_alpha": ANCHOR_ALPHA, "max_epochs": args.epochs,
            "patience": args.patience, "selected_epoch": best[1],
            "trainable_parameters": int(sum(p.numel() for p in trainable)),
            "base_model_trainable_parameters": 0,
        },
        "split": {"method": "5-fold stratified source-family CV",
                  "fold_index": args.fold_index,
                  "train_rows": len(train), "validation_rows": len(validation),
                  "validation_sources": sorted(set(groups[validation]))},
        "history": history, "result": result,
        "predictions": [
            {"id": str(frame.iloc[index].id),
             "source_family": str(groups[index]),
             "pool": str(frame.iloc[index].pool),
             "language": str(frame.iloc[index].language),
             "local_ok": int(local_ok[index]),
             "expert_ok": int(expert_ok[index]),
             "live": float(live[index]),
             "branch_raw": float(raw), "candidate": float(score)}
            for index, raw, score in zip(validation, validation_raw, candidate)
        ],
        "claim_boundary": ("Screening only on opened development data; winners "
                           "require full grouped OOF and independent evidence."),
        "provenance": {
            "selection": {"path": str(args.selection),
                          "sha256": sha256(args.selection)},
            "windows": [{"path": str(path), "sha256": sha256(path)}
                        for path in window_paths],
            "original": [{"path": str(path), "sha256": sha256(path)}
                         for path in original_paths],
            "local": {"path": str(args.local), "sha256": sha256(args.local)},
            "expert": {"path": str(args.expert), "sha256": sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact),
                              "sha256": sha256(args.live_artifact)},
            "checkpoint_index": {"path": str(index_path),
                                 "sha256": sha256(index_path)},
            "checkpoint_shards": [{"path": str(path), "sha256": sha256(path)}
                                  for path in checkpoint_paths],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"configuration": payload["configuration"],
                      "result": result}, indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
