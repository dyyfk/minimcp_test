"""Capture the deployed L22 feature after four answer chunks.

This P34 diagnostic adds no model forward: the ordinary duplex generation is
replayed with the original content-derived seed and the existing L22 hook is
read again after answer chunk four.  A row is available only if chunk four did
not already end the turn, so the resulting score could still trigger a relay.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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


LAYER = 22
K_EOT = 8
LATE_CHUNK = 4
MAX_WAIT = 12
MAX_ANSWER = 60


def audio_chunks(audio):
    values = [audio[i:i + 16000] for i in range(0, len(audio), 16000)]
    return [np.pad(value, (0, 16000 - len(value))).astype(np.float32)
            if len(value) < 16000 else value.astype(np.float32)
            for value in values]


def row_seed(namespace, row_id):
    return int(hashlib.sha256(
        f"{namespace}:{row_id}".encode()).hexdigest()[:8], 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-namespace", default="p15-native")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    trace_path = args.output_dir / f"midanswer_traces.rank{rank}.jsonl"

    frame = pd.read_parquet(args.selection).sort_values("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    if args.limit:
        frame = frame.head(args.limit)
    rows = [row for index, row in enumerate(frame.itertuples())
            if index % world == rank]
    completed = {}
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[str(record["id"])] = record
    pending = [row for row in rows if str(row.id) not in completed]

    cache = (Path.home() / ".cache/huggingface/modules/transformers_modules"
             / args.model_dir.name)
    cache.mkdir(parents=True, exist_ok=True)
    for source in glob.glob(str(args.model_dir / "*.py")):
        (cache / Path(source).name).write_bytes(Path(source).read_bytes())
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
        init_tts=True).eval().cuda(rank)
    _ = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    duplex = model.as_duplex(generate_audio=False)
    duplex.force_listen_count = 3
    ref, _ = librosa.load(args.model_dir / "assets/system_ref_audio.wav",
                          sr=16000, mono=True)

    state = {"accum": False, "tail": None, "sum": None, "count": 0}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden[0].detach().float()
        tail = hidden[-K_EOT:].cpu()
        state["tail"] = (tail if state["tail"] is None else
                         torch.cat([state["tail"], tail])[-K_EOT:])
        if state["accum"]:
            value = hidden.sum(0).cpu()
            state["sum"] = (value if state["sum"] is None else
                            state["sum"] + value)
            state["count"] += hidden.shape[0]

    handle = model.llm.model.layers[LAYER].register_forward_hook(hook)

    def feature_now():
        if state["tail"] is None or state["sum"] is None:
            raise RuntimeError("feature state is incomplete")
        return torch.cat([
            state["tail"][-1], state["tail"].mean(0),
            state["sum"] / max(1, state["count"]),
        ]).numpy().astype(np.float32)

    print(f"rank {rank}: {len(rows)} owned, {len(pending)} pending", flush=True)
    try:
        with trace_path.open("a", encoding="utf-8") as stream:
            for index, row in enumerate(pending):
                started = time.perf_counter()
                row_id = str(row.id)
                seed = row_seed(args.seed_namespace, row_id)
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                noise_rng = np.random.default_rng(seed ^ 0x5A17)
                state.update(accum=False, tail=None, sum=None, count=0)
                duplex.prepare(
                    prefix_system_prompt="You are a friendly assistant.",
                    ref_audio=ref, prompt_wav_path=None)
                audio, _ = librosa.load(args.audio_dir / f"{row_id}.wav",
                                        sr=16000, mono=True)
                chunks = audio_chunks(audio)
                texts, onset, late_feature = [], None, None
                answer_chunks, eot_seen, error = 0, False, None
                try:
                    feed = list(chunks) + [None] * (MAX_WAIT + MAX_ANSWER)
                    with torch.inference_mode():
                        for chunk_index, chunk in enumerate(feed):
                            waveform = (noise_rng.normal(0, .003, 16000)
                                        .astype(np.float32)
                                        if chunk is None else chunk)
                            state["accum"] = True
                            ok = duplex.streaming_prefill(audio_waveform=waveform)
                            state["accum"] = False
                            if not ok.get("success"):
                                continue
                            result = duplex.streaming_generate(top_k=20)
                            if result["is_listen"]:
                                if onset is None and chunk_index >= (
                                        len(chunks) + MAX_WAIT - 1):
                                    break
                                continue
                            if onset is None:
                                onset = chunk_index
                            if result.get("text"):
                                texts.append(result["text"])
                            answer_chunks += 1
                            if (answer_chunks == LATE_CHUNK
                                    and not result.get("end_of_turn")):
                                late_feature = feature_now()
                            if result.get("end_of_turn"):
                                eot_seen = True
                                break
                            if answer_chunks >= MAX_ANSWER:
                                break
                    if onset is None:
                        raise RuntimeError("model never reached speak onset")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    state["accum"] = False

                if late_feature is not None:
                    path = feature_dir / f"{row_id}.npy"
                    temporary = feature_dir / f".{row_id}.rank{rank}.tmp"
                    with temporary.open("wb") as output:
                        np.save(output, late_feature)
                    temporary.replace(path)
                record = {
                    "id": row_id, "rank": rank, "seed": seed,
                    "pool": str(row.pool), "n_q_chunks": len(chunks),
                    "onset_chunk": onset,
                    "late_answer_chunk": LATE_CHUNK,
                    "late_available": late_feature is not None,
                    "answer_text": "".join(texts).strip(),
                    "eot_seen": eot_seen, "n_answer_chunks": answer_chunks,
                    "elapsed_s": time.perf_counter() - started,
                    "error": error,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if index < 2 or (index + 1) % 10 == 0:
                    print(f"rank {rank}: {index + 1}/{len(pending)} {row_id} "
                          f"late={late_feature is not None} "
                          f"error={error is not None}", flush=True)
    finally:
        handle.remove()

    ids = [str(row.id) for row in rows
           if (feature_dir / f"{row.id}.npy").exists()]
    values = [np.load(feature_dir / f"{row_id}.npy") for row_id in ids]
    output = args.output_dir / f"midanswer_feats.rank{rank}.npz"
    if values:
        np.savez_compressed(output, ids=np.asarray(ids), X=np.stack(values),
                            late_answer_chunk=LATE_CHUNK)
    print(f"rank {rank}: consolidated {len(ids)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
