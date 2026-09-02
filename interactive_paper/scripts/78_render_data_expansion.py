"""Resumably render the frozen P25-B utterances with OpenAI tts-1."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main_async(args):
    single = pd.read_parquet(args.root / "single.parquet")
    pairs = pd.read_parquet(args.root / "pairs.parquet")
    requests = [
        {"id": str(row.id), "text": str(row.query), "kind": "standalone"}
        for row in single.itertuples()
    ]
    requests += [
        {"id": str(row.target_id), "text": str(row.query), "kind": "target"}
        for row in pairs.itertuples()
    ]
    requests += [
        {"id": str(row.carrier_id), "text": str(row.carrier_query),
         "kind": "carrier"}
        for row in pairs[pairs.carrier_audio_kind == "tts"].itertuples()
    ]
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = 0
    rendered = 0
    failed = []

    async def one(item):
        nonlocal completed, rendered
        path = args.audio_dir / f"{item['id']}.wav"
        if path.exists() and path.stat().st_size > 44:
            completed += 1
            return
        try:
            async with semaphore:
                response = await client.audio.speech.create(
                    model="tts-1", voice=args.voice, input=item["text"],
                    response_format="wav")
            path.write_bytes(response.content)
            rendered += 1
            completed += 1
            if completed % 100 == 0:
                print(f"tts {completed}/{len(requests)}", flush=True)
        except Exception as exc:
            failed.append({"id": item["id"], "kind": item["kind"],
                           "error": f"{type(exc).__name__}: {exc}"})

    try:
        await asyncio.gather(*(one(item) for item in requests))
    finally:
        await client.close()
    receipt = {
        "selection_sha256": sha256(args.root / "selection.parquet"),
        "requested_files": len(requests),
        "completed_files": completed,
        "newly_rendered_files": rendered,
        "failed": failed,
        "characters": sum(len(item["text"]) for item in requests),
        "tts_model": "tts-1", "voice": args.voice,
        "cost_usd_at_15_per_million_characters":
            sum(len(item["text"]) for item in requests) * 15 / 1_000_000,
    }
    (args.root / "tts_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} TTS requests failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--voice", default="alloy")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
