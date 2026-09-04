"""Capture strictly pre-generation MiniCPM hidden features.

The deployed/offline onset recipe snapshots layer state after the first
``streaming_generate`` call that commits to speaking.  This diagnostic instead
keeps the snapshot produced by the immediately preceding
``streaming_prefill`` call.  It adds no model forward and cannot contain an
answer token.  Per-row files and content-derived seeds make torchrun resumable.
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


K_TAIL = 8
MAX_WAIT = 12
MAX_ANSWER = 60


def audio_chunks(audio):
    values = [audio[i:i + 16000] for i in range(0, len(audio), 16000)]
    return [np.pad(value, (0, 16000 - len(value))).astype(np.float32)
            if len(value) < 16000 else value.astype(np.float32)
            for value in values]


def row_seed(namespace: str, row_id: str) -> int:
    return int(hashlib.sha256(f"{namespace}:{row_id}".encode()).hexdigest()[:8],
               16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="18,22,26,30")
    parser.add_argument("--seed-namespace", default="p15-native")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    layers = tuple(int(value) for value in args.layers.split(","))
    if len(set(layers)) != len(layers) or any(value < 0 for value in layers):
        raise ValueError("layers must be unique nonnegative integers")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    trace_path = args.output_dir / f"causal_prefill.rank{rank}.jsonl"

    frame = pd.read_parquet(args.selection)
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    frame = frame.sort_values("id")
    if args.limit:
        frame = frame.head(args.limit)
    rows = [row for index, row in enumerate(frame.itertuples())
            if index % world == rank]
    completed = {}
    if trace_path.exists():
        with trace_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    completed[str(record["id"])] = record
    pending = [row for row in rows
               if str(row.id) not in completed
               or not (feature_dir / f"{row.id}.npy").exists()]

    cache = (Path.home() / ".cache/huggingface/modules/transformers_modules"
             / args.model_dir.name)
    cache.mkdir(parents=True, exist_ok=True)
    for source in glob.glob(str(args.model_dir / "*.py")):
        target = cache / Path(source).name
        target.write_bytes(Path(source).read_bytes())
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
        init_tts=True).eval().cuda(rank)
    _ = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    duplex = model.as_duplex(generate_audio=False)
    duplex.force_listen_count = 3
    ref, _ = librosa.load(args.model_dir / "assets/system_ref_audio.wav",
                          sr=16000, mono=True)

    states = {layer: {"tail": None, "sum": None, "count": 0}
              for layer in layers}
    capture = {"prefill": False}

    def make_hook(layer):
        def hook(_module, _inputs, output):
            if not capture["prefill"]:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden[0].detach().float().cpu()
            state = states[layer]
            tail = hidden[-K_TAIL:]
            state["tail"] = (tail if state["tail"] is None else
                             torch.cat([state["tail"], tail])[-K_TAIL:])
            value = hidden.sum(0)
            state["sum"] = (value if state["sum"] is None else
                            state["sum"] + value)
            state["count"] += hidden.shape[0]
        return hook

    handles = [model.llm.model.layers[layer].register_forward_hook(
        make_hook(layer)) for layer in layers]

    def snapshot():
        values = []
        for layer in layers:
            state = states[layer]
            if state["tail"] is None or state["sum"] is None:
                raise RuntimeError(f"layer {layer} prefill state is incomplete")
            values.extend([state["tail"][-1], state["tail"].mean(0),
                           state["sum"] / max(1, state["count"])])
        return torch.cat(values).numpy().astype(np.float32)

    print(f"rank {rank}: {len(rows)} owned, {len(pending)} pending", flush=True)
    try:
        with trace_path.open("a", encoding="utf-8") as trace_stream:
            for index, row in enumerate(pending):
                started = time.perf_counter()
                row_id = str(row.id)
                seed = row_seed(args.seed_namespace, row_id)
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                noise_rng = np.random.default_rng(seed ^ 0x5A17)
                for state in states.values():
                    state.update(tail=None, sum=None, count=0)
                duplex.prepare(prefix_system_prompt="You are a friendly assistant.",
                               ref_audio=ref, prompt_wav_path=None)
                audio, _ = librosa.load(args.audio_dir / f"{row_id}.wav",
                                        sr=16000, mono=True)
                chunks = audio_chunks(audio)
                texts, onset, onset_feature = [], None, None
                answer_chunks, eot_seen, error = 0, False, None
                try:
                    feed = list(chunks) + [None] * (MAX_WAIT + MAX_ANSWER)
                    with torch.inference_mode():
                        for chunk_index, chunk in enumerate(feed):
                            waveform = (noise_rng.normal(0, .003, 16000)
                                        .astype(np.float32)
                                        if chunk is None else chunk)
                            capture["prefill"] = True
                            ok = duplex.streaming_prefill(audio_waveform=waveform)
                            capture["prefill"] = False
                            if not ok.get("success"):
                                continue
                            before_generate = snapshot()
                            result = duplex.streaming_generate(top_k=20)
                            if result["is_listen"]:
                                if onset is None and chunk_index >= (
                                        len(chunks) + MAX_WAIT - 1):
                                    break
                                continue
                            if onset is None:
                                onset = chunk_index
                                onset_feature = before_generate
                            if result.get("text"):
                                texts.append(result["text"])
                            answer_chunks += 1
                            if result.get("end_of_turn"):
                                eot_seen = True
                                break
                            if answer_chunks >= MAX_ANSWER:
                                break
                    if onset_feature is None:
                        raise RuntimeError("model never reached speak onset")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    capture["prefill"] = False

                if onset_feature is not None:
                    feature_path = feature_dir / f"{row_id}.npy"
                    temporary = feature_dir / f".{row_id}.rank{rank}.tmp"
                    with temporary.open("wb") as stream:
                        np.save(stream, onset_feature)
                    temporary.replace(feature_path)
                record = {
                    "id": row_id, "rank": rank, "seed": seed,
                    "layers": layers, "read": "prefill_before_first_speak_generate",
                    "pool": str(row.pool), "n_q_chunks": len(chunks),
                    "onset_chunk": onset, "no_speak": onset is None,
                    "answer_text": "".join(texts).strip(),
                    "eot_seen": eot_seen, "n_answer_chunks": answer_chunks,
                    "elapsed_s": time.perf_counter() - started, "error": error,
                }
                trace_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace_stream.flush()
                if index < 2 or (index + 1) % 10 == 0:
                    print(f"rank {rank}: {index + 1}/{len(pending)} {row_id} "
                          f"onset={onset} error={error is not None}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    ids = [str(row.id) for row in rows
           if (feature_dir / f"{row.id}.npy").exists()]
    values = [np.load(feature_dir / f"{row_id}.npy") for row_id in ids]
    output = args.output_dir / f"causal_prefill_feats.rank{rank}.npz"
    if values:
        np.savez_compressed(output, ids=np.asarray(ids), X=np.stack(values),
                            layers=np.asarray(layers))
    print(f"rank {rank}: consolidated {len(ids)}/{len(rows)} features", flush=True)


if __name__ == "__main__":
    main()
