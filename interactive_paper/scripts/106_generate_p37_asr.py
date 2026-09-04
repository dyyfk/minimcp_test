"""Generate P37's reference-blind second transcript locally on B300.

The selection parquet is read only for ``id`` and the pre-existing taxonomy
label.  Ground-truth question and reference columns are never passed to the
model.  Rank-local JSONL files are append-only and resumable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import librosa
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


ASR_INSTR = (
    "Transcribe the speech in the audio verbatim. Output ONLY the "
    "transcription, with no commentary or answer."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"p37_asr.rank{rank}.jsonl"
    done = set()
    if output.exists():
        done = {
            str(json.loads(line)["id"])
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    taxonomy = pd.read_parquet(args.taxonomy, columns=["id", "ftype"])
    ids = sorted(taxonomy.loc[taxonomy["ftype"].eq("perception"), "id"].astype(str))
    if args.limit:
        ids = ids[: args.limit]
    ids = [sample_id for i, sample_id in enumerate(ids)
           if i % world == rank and sample_id not in done]
    if not ids:
        print(f"rank {rank}: nothing pending", flush=True)
        return

    cache = (Path.home() / ".cache/huggingface/modules/transformers_modules" /
             args.model_dir.name)
    cache.mkdir(parents=True, exist_ok=True)
    for path in glob.glob(str(args.model_dir / "*.py")):
        (cache / Path(path).name).write_bytes(Path(path).read_bytes())
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
        init_tts=False,
    ).eval().cuda(rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    sys.path.insert(0, str(args.source_dir.resolve()))
    import decode

    chat_kwargs = decode._chat_kwargs(model, tokenizer)
    with output.open("a", encoding="utf-8") as handle:
        for i, sample_id in enumerate(ids):
            started = time.perf_counter()
            record = {"id": sample_id, "rank": rank, "error": None}
            try:
                audio, _ = librosa.load(
                    args.audio_dir / f"{sample_id}.wav", sr=16000, mono=True
                )
                transcript = model.chat(
                    msgs=[{"role": "user", "content": [audio, ASR_INSTR]}],
                    max_new_tokens=512,
                    **chat_kwargs,
                )
                record["transcript"] = (
                    transcript.strip() if isinstance(transcript, str)
                    else str(transcript)
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_s"] = time.perf_counter() - started
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"rank {rank}: {i + 1}/{len(ids)} {sample_id} "
                f"error={record['error'] is not None} "
                f"seconds={record['elapsed_s']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
