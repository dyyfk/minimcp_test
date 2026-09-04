"""Resumably render a frozen standalone selection with OpenAI tts-1."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI


async def main_async(args):
    selection = pd.read_parquet(args.selection).sort_values("id")
    rows = [{"id": str(row.id), "text": str(row.query)}
            for row in selection.itertuples()]
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    next_start, rendered, failed = 0., 0, []

    async def one(row):
        nonlocal next_start, rendered
        path = args.audio_dir / f"{row['id']}.wav"
        if path.exists() and path.stat().st_size > 44:
            return
        error = None
        for attempt in range(args.retries):
            try:
                async with semaphore:
                    async with lock:
                        loop = asyncio.get_running_loop()
                        now = loop.time()
                        if next_start > now:
                            await asyncio.sleep(next_start - now)
                        next_start = loop.time() + 60. / args.rpm
                    response = await client.audio.speech.create(
                        model="tts-1", voice=args.voice, input=row["text"],
                        response_format="wav")
                path.write_bytes(response.content)
                rendered += 1
                if rendered % 100 == 0:
                    print(f"rendered {rendered}", flush=True)
                return
            except Exception as exc:
                error = exc
                if attempt + 1 < args.retries:
                    await asyncio.sleep(min(8., .5 * 2 ** attempt))
        failed.append({"id": row["id"],
                       "error": f"{type(error).__name__}: {error}"})
    try:
        await asyncio.gather(*(one(row) for row in rows))
    finally:
        await client.close()
    complete = sum((args.audio_dir / f"{row['id']}.wav").stat().st_size > 44
                   for row in rows if (args.audio_dir / f"{row['id']}.wav").exists())
    receipt = {
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "requested_files": len(rows), "completed_files": complete,
        "newly_rendered_files": rendered, "failed": failed,
        "characters": sum(len(row["text"]) for row in rows),
        "tts_model": "tts-1", "voice": args.voice,
        "cost_usd_at_15_per_million_characters":
            sum(len(row["text"]) for row in rows) * 15 / 1_000_000,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if failed or complete != len(rows):
        raise RuntimeError("TTS rendering incomplete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--rpm", type=float, default=450.)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
