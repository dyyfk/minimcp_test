"""Capture strictly pre-generation features on the target turn of a dialogue.

The carrier turn is replayed normally to establish model context.  Hook state is
then cleared, and only target-turn ``streaming_prefill`` calls are observed.  A
snapshot is taken immediately before the first ``streaming_generate`` call
that speaks.  The base model and its output path are unchanged.
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
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="18,22,26,30")
    parser.add_argument("--seed-namespace", required=True)
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
    trace_path = args.output_dir / f"causal_multiturn.rank{rank}.jsonl"

    frame = pd.read_parquet(args.pairs).sort_values("id")
    if args.limit:
        frame = frame.head(args.limit)
    owned = [row for index, row in enumerate(frame.itertuples())
             if index % world == rank]
    completed = {}
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[str(record["id"])] = record
    pending = [row for row in owned
               if str(row.id) not in completed
               or not (feature_dir / f"{row.id}.npy").exists()]

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

    states = {layer: {"tail": None, "sum": None, "count": 0}
              for layer in layers}
    capture = {"prefill": False}

    def reset_capture():
        for state in states.values():
            state.update(tail=None, sum=None, count=0)

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

    def run_turn(audio, noise_rng, observe):
        chunks = audio_chunks(audio)
        feed = list(chunks) + [None] * (MAX_WAIT + MAX_ANSWER)
        texts, onset, feature = [], None, None
        answer_chunks, eot_seen = 0, False
        for chunk_index, chunk in enumerate(feed):
            waveform = (noise_rng.normal(0, .003, 16000).astype(np.float32)
                        if chunk is None else chunk)
            capture["prefill"] = observe
            ok = duplex.streaming_prefill(audio_waveform=waveform)
            capture["prefill"] = False
            if not ok.get("success"):
                continue
            before_generate = snapshot() if observe else None
            result = duplex.streaming_generate(top_k=20)
            if result["is_listen"]:
                if onset is None and chunk_index >= len(chunks) + MAX_WAIT - 1:
                    break
                continue
            if onset is None:
                onset = chunk_index
                feature = before_generate
            if result.get("text"):
                texts.append(result["text"])
            answer_chunks += 1
            if result.get("end_of_turn"):
                eot_seen = True
                break
            if answer_chunks >= MAX_ANSWER:
                break
        return {"answer": "".join(texts).strip(), "onset": onset,
                "feature": feature, "eot_seen": eot_seen,
                "answer_chunks": answer_chunks, "audio_chunks": len(chunks)}

    print(f"rank {rank}: {len(owned)} owned, {len(pending)} pending", flush=True)
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
                noise_rng = np.random.default_rng(seed ^ 0xA519)
                reset_capture()
                duplex.prepare(prefix_system_prompt="You are a friendly assistant.",
                               ref_audio=ref, prompt_wav_path=None)
                carrier = target = None
                error = None
                try:
                    carrier_audio, _ = librosa.load(
                        args.audio_dir / f"{row.carrier_id}.wav", sr=16000,
                        mono=True)
                    target_audio, _ = librosa.load(
                        args.audio_dir / f"{row.target_id}.wav", sr=16000,
                        mono=True)
                    with torch.inference_mode():
                        carrier = run_turn(carrier_audio, noise_rng, False)
                        if not carrier["eot_seen"]:
                            raise RuntimeError("carrier did not reach end of turn")
                        reset_capture()
                        target = run_turn(target_audio, noise_rng, True)
                        if target["feature"] is None:
                            raise RuntimeError("target never reached speak onset")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    capture["prefill"] = False

                if target is not None and target["feature"] is not None:
                    destination = feature_dir / f"{row_id}.npy"
                    temporary = feature_dir / f".{row_id}.rank{rank}.tmp"
                    with temporary.open("wb") as handle:
                        np.save(handle, target["feature"])
                    temporary.replace(destination)
                record = {
                    "id": row_id, "rank": rank, "seed": seed,
                    "layers": layers,
                    "read": "target_prefill_before_first_speak_generate",
                    "carrier_answer": None if carrier is None else carrier["answer"],
                    "carrier_eot_seen": False if carrier is None else carrier["eot_seen"],
                    "target_answer": None if target is None else target["answer"],
                    "target_eot_seen": False if target is None else target["eot_seen"],
                    "target_onset_chunk": None if target is None else target["onset"],
                    "target_audio_chunks": None if target is None else target["audio_chunks"],
                    "target_answer_chunks": None if target is None else target["answer_chunks"],
                    "elapsed_s": time.perf_counter() - started, "error": error,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if index < 2 or (index + 1) % 5 == 0:
                    print(f"rank {rank}: {index + 1}/{len(pending)} {row_id} "
                          f"error={error}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    ids = [str(row.id) for row in owned
           if (feature_dir / f"{row.id}.npy").exists()]
    values = [np.load(feature_dir / f"{row_id}.npy") for row_id in ids]
    if values:
        np.savez_compressed(
            args.output_dir / f"causal_multiturn_feats.rank{rank}.npz",
            ids=np.asarray(ids), X=np.stack(values), layers=np.asarray(layers))
    print(f"rank {rank}: consolidated {len(ids)}/{len(owned)}", flush=True)


if __name__ == "__main__":
    main()
