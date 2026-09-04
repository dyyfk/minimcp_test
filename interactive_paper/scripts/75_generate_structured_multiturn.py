"""Capture multi-layer and within-turn trajectory features in duplex replay.

The model output path is unchanged. Hooks observe five existing decoder layers
during the same streaming forwards and save target-turn features at speak
onset. This is an offline diagnostic; it cannot affect ``fired``.
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


LAYERS = [14, 18, 22, 26, 30]
K_EOT = 8
MAX_WAIT = 12
MAX_ANSWER = 60


def chunks(audio):
    output = [audio[i:i + 16000] for i in range(0, len(audio), 16000)]
    return [np.pad(value, (0, 16000 - len(value))).astype(np.float32)
            if len(value) < 16000 else value.astype(np.float32)
            for value in output]


def row_seed(namespace, row_id):
    return int(hashlib.sha256(f"{namespace}:{row_id}".encode()).hexdigest()[:8],
               16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed-namespace", default="structured-context")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "structured_features"
    feature_dir.mkdir(exist_ok=True)
    trace_path = args.output_dir / f"structured_multiturn.rank{rank}.jsonl"

    frame = pd.read_parquet(args.pairs).sort_values("id")
    if args.limit:
        frame = frame.head(args.limit)
    owned = [row for index, row in enumerate(frame.itertuples())
             if index % world == rank]
    done = set()
    if trace_path.exists():
        done = {json.loads(line)["id"] for line in trace_path.read_text().splitlines()
                if line.strip()}
    pending = [row for row in owned if row.id not in done or not (
        feature_dir / f"{row.id}.npz").exists()]

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
    state = {"accum": False, "tail": {}, "sum": {}, "count": 0,
             "chunk_last": {}}

    def reset_turn():
        state.update(accum=False, sum={}, count=0, chunk_last={})

    def mk_hook(layer, count_here):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden[0].detach().float()
            tail = hidden[-K_EOT:].cpu()
            previous = state["tail"].get(layer)
            state["tail"][layer] = (tail if previous is None else
                                     torch.cat([previous, tail])[-K_EOT:])
            if state["accum"]:
                value = hidden.sum(0).cpu()
                previous_sum = state["sum"].get(layer)
                state["sum"][layer] = (value if previous_sum is None else
                                        previous_sum + value)
                state["chunk_last"].setdefault(layer, []).append(
                    hidden[-1].cpu())
                if count_here:
                    state["count"] += hidden.shape[0]
        return hook

    handles = [model.llm.model.layers[layer].register_forward_hook(
        mk_hook(layer, layer == LAYERS[0])) for layer in LAYERS]

    def feature_now(onset_wait):
        if state["count"] <= 0:
            return None
        dimension = state["tail"][LAYERS[0]].shape[1]
        eot = np.zeros((len(LAYERS), K_EOT, dimension), dtype=np.float16)
        turn_mean = np.zeros((len(LAYERS), dimension), dtype=np.float16)
        chunk_mean = np.zeros_like(turn_mean)
        chunk_delta = np.zeros_like(turn_mean)
        eot_len = min(K_EOT, state["tail"][LAYERS[0]].shape[0])
        chunk_count = len(state["chunk_last"].get(LAYERS[0], []))
        for index, layer in enumerate(LAYERS):
            tail = state["tail"][layer].numpy()
            eot[index, K_EOT - len(tail):] = tail.astype(np.float16)
            turn_mean[index] = (state["sum"][layer].numpy() /
                                state["count"]).astype(np.float16)
            values = torch.stack(state["chunk_last"][layer]).numpy()
            chunk_mean[index] = values.mean(0).astype(np.float16)
            chunk_delta[index] = (values[-1] - values[0]).astype(np.float16)
        return {
            "H_eot": eot, "H_turn_mean": turn_mean,
            "H_chunk_mean": chunk_mean, "H_chunk_delta": chunk_delta,
            "eot_len": np.asarray(eot_len, dtype=np.int16),
            "chunk_count": np.asarray(chunk_count, dtype=np.int16),
            "onset_wait": np.asarray(onset_wait, dtype=np.int16),
            "layers": np.asarray(LAYERS, dtype=np.int16),
        }

    def run_turn(audio, noise_rng, capture):
        reset_turn()
        texts, onset, feature, answer_chunks, ended = [], None, None, 0, False
        source_chunks = chunks(audio)
        feed = list(source_chunks) + [None] * (MAX_WAIT + MAX_ANSWER)
        for chunk_index, chunk in enumerate(feed):
            waveform = (noise_rng.normal(0, .003, 16000).astype(np.float32)
                        if chunk is None else chunk)
            state["accum"] = True
            ok = duplex.streaming_prefill(audio_waveform=waveform)
            state["accum"] = False
            if not ok.get("success"):
                continue
            result = duplex.streaming_generate(top_k=20)
            if result["is_listen"]:
                if (onset is None and
                        chunk_index >= len(source_chunks) + MAX_WAIT - 1):
                    break
                continue
            if onset is None:
                onset = chunk_index
                if capture:
                    feature = feature_now(onset - len(source_chunks))
            if result.get("text"):
                texts.append(result["text"])
            answer_chunks += 1
            if result.get("end_of_turn"):
                ended = True
                break
            if answer_chunks >= MAX_ANSWER:
                break
        return {"onset": onset, "feature": feature,
                "answer": "".join(texts).strip(),
                "answer_chunks": answer_chunks, "eot_seen": ended,
                "audio_chunks": len(source_chunks)}

    print(f"rank {rank}: {len(owned)} owned, {len(pending)} pending", flush=True)
    try:
        with trace_path.open("a", encoding="utf-8") as stream:
            for index, row in enumerate(pending):
                started = time.perf_counter()
                seed = row_seed(args.seed_namespace, str(row.id))
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                noise_rng = np.random.default_rng(seed ^ 0xA519)
                state.update(accum=False, tail={}, sum={}, count=0,
                             chunk_last={})
                duplex.prepare(prefix_system_prompt="You are a friendly assistant.",
                               ref_audio=ref, prompt_wav_path=None)
                error = None
                carrier = target = None
                try:
                    carrier_audio, _ = librosa.load(
                        args.audio_dir / f"{row.carrier_id}.wav", sr=16000,
                        mono=True)
                    target_audio, _ = librosa.load(
                        args.audio_dir / f"{row.target_id}.wav", sr=16000,
                        mono=True)
                    with torch.inference_mode():
                        carrier = run_turn(carrier_audio, noise_rng,
                                           capture=False)
                        if not carrier["eot_seen"]:
                            raise RuntimeError("carrier did not reach end of turn")
                        target = run_turn(target_audio, noise_rng, capture=True)
                        if target["feature"] is None:
                            raise RuntimeError("target never reached speak onset")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    state["accum"] = False
                if target is not None and target["feature"] is not None:
                    destination = feature_dir / f"{row.id}.npz"
                    temporary = feature_dir / f".{row.id}.rank{rank}.tmp"
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **target["feature"])
                    temporary.replace(destination)
                record = {
                    "id": str(row.id), "rank": rank, "seed": seed,
                    "target_id": str(row.target_id),
                    "target_pool": str(row.target_pool),
                    "carrier_id": str(row.carrier_id),
                    "carrier_pool": str(row.carrier_pool),
                    "carrier_eot_seen": (False if carrier is None else
                                         carrier["eot_seen"]),
                    "target_answer": (None if target is None else
                                      target["answer"]),
                    "target_eot_seen": (False if target is None else
                                        target["eot_seen"]),
                    "target_onset_chunk": (None if target is None else
                                           target["onset"]),
                    "target_audio_chunks": (None if target is None else
                                           target["audio_chunks"]),
                    "target_onset_wait": (None if target is None or
                                          target["onset"] is None else
                                          target["onset"] -
                                          target["audio_chunks"]),
                    "elapsed_s": time.perf_counter() - started,
                    "error": error,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if index < 2 or (index + 1) % 5 == 0:
                    print(f"rank {rank}: {index + 1}/{len(pending)} {row.id} "
                          f"error={error}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    ids = [str(row.id) for row in owned if (
        feature_dir / f"{row.id}.npz").exists()]
    if ids:
        values = [np.load(feature_dir / f"{row_id}.npz") for row_id in ids]
        np.savez_compressed(
            args.output_dir / f"structured_multiturn_feats.rank{rank}.npz",
            ids=np.asarray(ids),
            H_eot=np.stack([value["H_eot"] for value in values]),
            H_turn_mean=np.stack([value["H_turn_mean"] for value in values]),
            H_chunk_mean=np.stack([value["H_chunk_mean"] for value in values]),
            H_chunk_delta=np.stack([value["H_chunk_delta"] for value in values]),
            eot_len=np.asarray([value["eot_len"] for value in values]),
            chunk_count=np.asarray([value["chunk_count"] for value in values]),
            onset_wait=np.asarray([value["onset_wait"] for value in values]),
            layers=np.asarray(LAYERS, dtype=np.int16))
    print(f"rank {rank}: consolidated {len(ids)}/{len(owned)}", flush=True)


if __name__ == "__main__":
    main()
