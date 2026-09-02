"""Generate stochastic native-duplex answers for semantic-entropy labels.

Launch with ``torchrun --nproc-per-node 8`` on the allocated GPU node.  Each
rank owns a deterministic shard and appends one JSON object per completed ID,
so interrupted runs resume without repeating completed local inference.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def trace_answers(data_dir: Path):
    answers = {}
    for path in sorted(data_dir.glob("frozen_native_*off_traces.jsonl.shard*")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    answers[str(row["id"])] = str(row.get("answer_text") or "")
    return answers


def chunks(audio):
    out = [audio[i:i + 16000] for i in range(0, len(audio), 16000)]
    return [np.pad(chunk, (0, 16000 - len(chunk))).astype(np.float32)
            if len(chunk) < 16000 else chunk.astype(np.float32)
            for chunk in out]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"semantic_samples.rank{rank}.jsonl"
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(str(json.loads(line)["id"]))

    selection = pd.read_parquet(args.selection).sort_values("id")
    if args.limit:
        selection = selection.head(args.limit)
    rows = [row for i, row in enumerate(selection.itertuples())
            if i % world == rank and str(row.id) not in done]
    if not rows:
        print(f"rank {rank}: nothing pending", flush=True)
        return

    # Match the validated native-dump load path.
    cache = Path.home() / ".cache/huggingface/modules/transformers_modules" / args.model_dir.name
    cache.mkdir(parents=True, exist_ok=True)
    for path in glob.glob(str(args.model_dir / "*.py")):
        target = cache / Path(path).name
        target.write_bytes(Path(path).read_bytes())
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
        init_tts=True).eval().cuda(rank)
    _ = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    duplex = model.as_duplex(generate_audio=False)
    duplex.force_listen_count = 3
    ref, _ = librosa.load(
        args.model_dir / "assets/system_ref_audio.wav", sr=16000, mono=True)
    official = trace_answers(args.data_dir)

    rng = np.random.default_rng(43 + rank)

    def silence():
        return rng.normal(0, .003, 16000).astype(np.float32)

    def generate(audio_chunks, seed):
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        duplex.prepare(prefix_system_prompt="You are a friendly assistant.",
                       ref_audio=ref, prompt_wav_path=None)
        texts, onset, n_answer = [], None, 0
        feed = list(audio_chunks) + [None] * 72
        for chunk_index, chunk in enumerate(feed):
            ok = duplex.streaming_prefill(
                audio_waveform=silence() if chunk is None else chunk)
            if not ok.get("success"):
                continue
            result = duplex.streaming_generate(
                temperature=args.temperature, top_k=20)
            if result["is_listen"]:
                if onset is None and chunk_index >= len(audio_chunks) + 11:
                    break
                continue
            onset = chunk_index if onset is None else onset
            if result.get("text"):
                texts.append(result["text"])
            n_answer += 1
            if result.get("end_of_turn") or n_answer >= 60:
                break
        return "".join(texts).strip(), onset, n_answer

    with output.open("a", encoding="utf-8") as fh:
        for index, row in enumerate(rows):
            row_id = str(row.id)
            row_started = time.perf_counter()
            record = {
                "id": row_id, "rank": rank,
                "official_answer": official.get(row_id, ""),
                "samples": [], "temperature": args.temperature,
                "error": None,
            }
            try:
                audio, _ = librosa.load(
                    args.audio_dir / f"{row_id}.wav", sr=16000, mono=True)
                audio_chunks = chunks(audio)
                for sample_index in range(args.samples):
                    seed = 430000 + sample_index * 100000 + int(
                        row_id.encode().hex()[:8], 16)
                    sample_started = time.perf_counter()
                    answer, onset, n_answer = generate(audio_chunks, seed)
                    record["samples"].append({
                        "seed": seed, "answer": answer,
                        "onset_chunk": onset, "n_answer_chunks": n_answer,
                        "elapsed_s": time.perf_counter() - sample_started})
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_s"] = time.perf_counter() - row_started
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if index < 2 or (index + 1) % 10 == 0:
                print(f"rank {rank}: {index + 1}/{len(rows)} {row_id} "
                      f"error={record['error'] is not None}", flush=True)


if __name__ == "__main__":
    main()
