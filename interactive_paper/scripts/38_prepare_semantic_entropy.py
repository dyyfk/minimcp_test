"""Prepare and render a fixed semantic-entropy pilot sample.

The input is the already fixed P2 paired set.  Selection is pre-outcome for
the new target: proportional within source x language x native-failure, using
a stable SHA256 ordering.  TTS is resumable and records an exact character
receipt so API cost is auditable ($15 / 1M characters for tts-1 at execution
time; the rate is metadata, not fetched by this script).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI


def stable_key(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def allocate(counts: pd.Series, target: int) -> dict:
    raw = counts / counts.sum() * target
    base = raw.astype(int).clip(lower=1)
    while base.sum() > target:
        candidates = [key for key in base.index if base[key] > 1]
        key = min(candidates, key=lambda item: raw[item] - base[item])
        base[key] -= 1
    while base.sum() < target:
        candidates = [key for key in base.index if base[key] < counts[key]]
        key = max(candidates, key=lambda item: raw[item] - base[item])
        base[key] += 1
    return {str(key): int(value) for key, value in base.items()}


def local_failure_map(data_dir: Path, rows: pd.DataFrame) -> dict:
    result = {}
    for tag in sorted(rows["training_tag"].unique()):
        judged = (pd.read_parquet(
            data_dir / f"frozen_native_{tag}_judged.parquet")
            .dropna(subset=["adequate"]).drop_duplicates("id", keep="last")
            .set_index("id")["adequate"].astype(int))
        for row in rows[rows["training_tag"] == tag].itertuples():
            if str(row.id) in judged:
                result[str(row.id)] = 1 - int(judged[str(row.id)])
    return result


def select(input_path: Path, data_dir: Path, output_path: Path,
           target: int, seed: int):
    df = pd.read_parquet(input_path).copy()
    df["id"] = df["id"].astype(str)
    df["language"] = df["query"].map(
        lambda value: "zh" if re.search(r"[\u3400-\u9fff]", str(value))
        else "en")
    failure = local_failure_map(data_dir, df)
    if len(failure) != len(df):
        raise RuntimeError(
            f"native/policy labels matched {len(failure)}/{len(df)} rows")
    df["native_failure"] = df["id"].map(failure).astype(int)
    family = df["pool"].fillna(df["source"]).fillna(df["training_tag"])
    df["stratum"] = (family.astype(str) + "|" +
                     df["language"] + "|" +
                     df["native_failure"].astype(str))
    counts = df.groupby("stratum").size()
    allocation = allocate(counts, target)
    chosen = []
    for stratum, part in df.groupby("stratum", sort=True):
        n = allocation[str(stratum)]
        part = part.assign(_key=[stable_key(seed, row_id)
                                for row_id in part["id"]])
        chosen.append(part.sort_values("_key").head(n))
    sample = pd.concat(chosen).sort_values("id").drop(columns="_key")
    ids_blob = "\n".join(sample["id"]) + "\n"
    receipt = {
        "source": str(input_path), "source_sha256": file_sha(input_path),
        "target": target, "seed": seed, "strata": len(counts),
        "id_list_sha256": hashlib.sha256(ids_blob.encode()).hexdigest(),
        "characters": int(sample["query"].fillna("").str.len().sum()),
        "tts_1_usd_at_15_per_million_chars": float(
            sample["query"].fillna("").str.len().sum() * 15 / 1_000_000),
        "counts_by_language": sample["language"].value_counts().to_dict(),
        "counts_by_native_failure": sample["native_failure"].value_counts().to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_path)
    output_path.with_suffix(".selection.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    return receipt


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def render(selection: Path, audio_dir: Path, concurrency: int):
    df = pd.read_parquet(selection)
    audio_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    failed = []

    async def one(row):
        nonlocal completed
        path = audio_dir / f"{row.id}.wav"
        if path.exists() and path.stat().st_size > 44:
            completed += 1
            return
        try:
            async with sem:
                response = await client.audio.speech.create(
                    model="tts-1", voice="alloy", input=str(row.query),
                    response_format="wav")
            path.write_bytes(response.content)
            completed += 1
            if completed % 50 == 0:
                print(f"tts {completed}/{len(df)}", flush=True)
        except Exception as exc:
            failed.append({"id": str(row.id), "error": type(exc).__name__})

    try:
        await asyncio.gather(*(one(row) for row in df.itertuples()))
    finally:
        await client.close()
    receipt = {
        "selection_sha256": file_sha(selection),
        "requested": len(df), "completed": completed,
        "failed": failed,
        "characters": int(df["query"].fillna("").str.len().sum()),
    }
    (audio_dir.parent / "tts_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    receipt = select(
        args.input, args.data_dir, args.selection, args.target, args.seed)
    print(json.dumps(receipt, indent=2), flush=True)
    if args.render:
        if args.audio_dir is None:
            parser.error("--audio-dir is required with --render")
        asyncio.run(render(args.selection, args.audio_dir, args.concurrency))


if __name__ == "__main__":
    main()
