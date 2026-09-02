"""Assemble and optionally TTS-render the fixed external semantic set.

The set is the exact ID intersection used by the official-native evaluation:
query metadata, frozen native outcome, cached always-expert outcome, and frozen
hidden features must all exist.  No outcome-dependent sampling is performed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


POOLS = ("striviaqa", "swebq", "sllama", "sdqa", "sreason")
EXPERT_COLUMNS = {
    "striviaqa": "oab_ok", "swebq": "oab_ok", "sllama": "oab_ok",
    "sdqa": "heard_ok", "sreason": "heard_ok",
}


def load_prepare_module():
    path = Path(__file__).with_name("38_prepare_semantic_entropy.py")
    spec = importlib.util.spec_from_file_location("prepare_semantic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def feature_ids(data_dir: Path, pool: str):
    ids = set()
    for path in sorted(data_dir.glob(
            f"frozen_native_{pool}off_feats.shard*.npz")):
        with np.load(path, allow_pickle=True) as shard:
            ids.update(str(row_id) for row_id in shard["ids"])
    return ids


def assemble(data_dir: Path):
    frames = []
    for pool in POOLS:
        queries = pd.read_json(data_dir / f"queries_{pool}.jsonl", lines=True)
        native = (pd.read_parquet(
            data_dir / f"frozen_native_{pool}off_judged.parquet")
            .dropna(subset=["adequate"]).drop_duplicates("id", keep="last")
            [["id", "adequate"]].rename(columns={"adequate": "native_ok"}))
        expert_col = EXPERT_COLUMNS[pool]
        expert = pd.read_parquet(data_dir / f"{pool}_conclive_traces.parquet")
        expert = (expert[expert["tier"] == "always"]
                  .dropna(subset=[expert_col])
                  .drop_duplicates("id", keep="last")[["id", expert_col]]
                  .rename(columns={expert_col: "expert_ok"}))
        frame = queries.merge(native, on="id").merge(expert, on="id")
        frame = frame[frame["id"].astype(str).isin(feature_ids(data_dir, pool))]
        frame["native_ok"] = frame["native_ok"].astype(int)
        frame["expert_ok"] = frame["expert_ok"].astype(int)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values("id")
    if result["id"].duplicated().any():
        raise RuntimeError("duplicate external IDs")
    return result


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    frame = assemble(args.data_dir)
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.selection, index=False)
    characters = int(frame["query"].fillna("").str.len().sum())
    receipt = {
        "selection_sha256": sha256(args.selection),
        "n": len(frame),
        "counts_by_pool": frame["pool"].value_counts().sort_index().to_dict(),
        "characters": characters,
        "tts_1_usd_at_15_per_million_chars": characters * 15 / 1_000_000,
    }
    args.selection.with_suffix(".selection.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if args.render:
        if args.audio_dir is None:
            parser.error("--audio-dir is required with --render")
        prepare = load_prepare_module()
        asyncio.run(prepare.render(
            args.selection, args.audio_dir, args.concurrency))


if __name__ == "__main__":
    main()
