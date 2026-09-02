"""Run frozen two-turn audio sessions through the native duplex model.

Each pair starts a fresh session, completes the carrier turn, then captures
both turns' exact deployed L22 speak-onset feature recipe and the target answer.
Launch with ``torchrun --standalone --nproc-per-node 8``.
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
    parser.add_argument("--seed-namespace", default="p19-context")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    carrier_feature_dir = args.output_dir / "carrier_features"
    carrier_feature_dir.mkdir(exist_ok=True)
    trace_path = args.output_dir / f"controlled_multiturn.rank{rank}.jsonl"

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
        feature_dir / f"{row.id}.npy").exists() or not (
        carrier_feature_dir / f"{row.id}.npy").exists()]

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
            state["sum"] = value if state["sum"] is None else state["sum"] + value
            state["count"] += hidden.shape[0]

    handle = model.llm.model.layers[LAYER].register_forward_hook(hook)

    def feature_now():
        return torch.cat([
            state["tail"][-1], state["tail"].mean(0),
            state["sum"] / max(1, state["count"]),
        ]).numpy().astype(np.float32)

    def run_turn(audio, noise_rng, capture):
        texts, onset, vector, answer_chunks, ended = [], None, None, 0, False
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
                    vector = feature_now()
            if result.get("text"):
                texts.append(result["text"])
            answer_chunks += 1
            if result.get("end_of_turn"):
                ended = True
                break
            if answer_chunks >= MAX_ANSWER:
                break
        return {"onset": onset, "feature": vector, "answer": "".join(texts).strip(),
                "answer_chunks": answer_chunks, "eot_seen": ended}

    print(f"rank {rank}: {len(owned)} owned, {len(pending)} pending", flush=True)
    try:
        with trace_path.open("a", encoding="utf-8") as stream:
            for index, row in enumerate(pending):
                started = time.perf_counter()
                seed = row_seed(args.seed_namespace, str(row.id))
                random.seed(seed); np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                noise_rng = np.random.default_rng(seed ^ 0xA519)
                state.update(accum=False, tail=None, sum=None, count=0)
                duplex.prepare(prefix_system_prompt="You are a friendly assistant.",
                               ref_audio=ref, prompt_wav_path=None)
                error = None; carrier = target = None
                try:
                    carrier_audio, _ = librosa.load(
                        args.audio_dir / f"{row.carrier_id}.wav", sr=16000,
                        mono=True)
                    target_audio, _ = librosa.load(
                        args.audio_dir / f"{row.target_id}.wav", sr=16000,
                        mono=True)
                    with torch.inference_mode():
                        carrier = run_turn(carrier_audio, noise_rng, capture=True)
                        if not carrier["eot_seen"]:
                            raise RuntimeError("carrier did not reach end of turn")
                        state.update(sum=None, count=0)
                        target = run_turn(target_audio, noise_rng, capture=True)
                        if target["feature"] is None:
                            raise RuntimeError("target never reached speak onset")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    state["accum"] = False
                for value, directory in [
                        (None if target is None else target["feature"],
                         feature_dir),
                        (None if carrier is None else carrier["feature"],
                         carrier_feature_dir)]:
                    if value is None:
                        continue
                    destination = directory / f"{row.id}.npy"
                    temporary = directory / f".{row.id}.rank{rank}.tmp"
                    with temporary.open("wb") as fh:
                        np.save(fh, value)
                    temporary.replace(destination)
                record = {
                    "id": str(row.id), "rank": rank, "seed": seed,
                    "target_id": str(row.target_id),
                    "target_pool": str(row.target_pool),
                    "carrier_id": str(row.carrier_id),
                    "carrier_pool": str(row.carrier_pool),
                    "turn_index": 2, "has_context": True,
                    "prior_escalations": 0,
                    "carrier_answer": None if carrier is None else carrier["answer"],
                    "carrier_eot_seen": False if carrier is None else carrier["eot_seen"],
                    "target_answer": None if target is None else target["answer"],
                    "target_eot_seen": False if target is None else target["eot_seen"],
                    "target_onset_chunk": None if target is None else target["onset"],
                    "elapsed_s": time.perf_counter() - started,
                    "error": error,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if index < 2 or (index + 1) % 5 == 0:
                    print(f"rank {rank}: {index + 1}/{len(pending)} {row.id} "
                          f"error={error}", flush=True)
    finally:
        handle.remove()

    ids = [str(row.id) for row in owned if (
        feature_dir / f"{row.id}.npy").exists() and (
        carrier_feature_dir / f"{row.id}.npy").exists()]
    values = [np.load(feature_dir / f"{row_id}.npy") for row_id in ids]
    carrier_values = [np.load(carrier_feature_dir / f"{row_id}.npy")
                      for row_id in ids]
    if values:
        np.savez_compressed(args.output_dir /
                            f"controlled_multiturn_feats.rank{rank}.npz",
                            ids=np.asarray(ids), X=np.stack(values))
        np.savez_compressed(args.output_dir /
                            f"controlled_multiturn_carrier_feats.rank{rank}.npz",
                            ids=np.asarray(ids), X=np.stack(carrier_values))
    print(f"rank {rank}: consolidated {len(ids)}/{len(owned)}", flush=True)


if __name__ == "__main__":
    main()
