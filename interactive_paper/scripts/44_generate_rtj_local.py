"""Generate repeat-then-judge p(True) scalars locally on B300.

Each row is transcribed by MiniCPM-o from the same fixed TTS waveform used by
the semantic experiment.  The model then self-assesses the transcript through
the repository's first-token Yes/No protocol.  Original-query p(True) is also
stored as a ceiling diagnostic.  Launch with torchrun; rank shards resume.
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


ASR_INSTR = ("Transcribe the speech in the audio verbatim. Output ONLY the "
             "transcription, with no commentary or answer.")
PTRUE_PRE = ("I am going to show you a question. Do NOT answer it. "
             "Judge honestly whether you yourself would answer it correctly.\n\n"
             "Question:\n{q}\n\nWould you answer this question correctly? "
             "Reply with exactly one word: Yes or No.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rtj.rank{rank}.jsonl"
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as fh:
            done = {str(json.loads(line)["id"]) for line in fh if line.strip()}
    selection = pd.read_parquet(args.selection).sort_values("id")
    if args.limit:
        selection = selection.head(args.limit)
    rows = [row for i, row in enumerate(selection.itertuples())
            if i % world == rank and str(row.id) not in done]
    if not rows:
        print(f"rank {rank}: nothing pending", flush=True)
        return

    cache = (Path.home() / ".cache/huggingface/modules/transformers_modules" /
             args.model_dir.name)
    cache.mkdir(parents=True, exist_ok=True)
    for path in glob.glob(str(args.model_dir / "*.py")):
        (cache / Path(path).name).write_bytes(Path(path).read_bytes())
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False,
        init_audio=not args.text_only,
        init_tts=False).eval().cuda(rank)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True)
    sys.path.insert(0, str(args.source_dir.resolve()))
    import decode

    chat_kwargs = decode._chat_kwargs(model, tokenizer)

    def token_ids(words):
        ids = set()
        for word in words:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        return sorted(ids)

    yes = token_ids(["Yes", "yes", "YES", " Yes", " yes",
                     "是", "能", "对", "会"])
    no = token_ids(["No", "no", "NO", " No", " no",
                    "否", "不", "错"])

    def p_yes(text):
        logits = decode.first_token_logits(
            model, tokenizer, PTRUE_PRE.format(q=text))
        probability = torch.softmax(logits, dim=-1)
        py, pn = float(probability[yes].sum()), float(probability[no].sum())
        mass = py + pn
        return (py / mass if mass else .5), mass

    with output.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            started = time.perf_counter()
            record = {"id": str(row.id), "rank": rank, "error": None}
            try:
                if args.text_only:
                    pt, mt = p_yes(str(row.query))
                    record.update({"p_yes_textq": pt, "mass_textq": mt})
                else:
                    audio, _ = librosa.load(
                        args.audio_dir / f"{row.id}.wav", sr=16000, mono=True)
                    transcript = model.chat(
                        msgs=[{"role": "user",
                               "content": [audio, ASR_INSTR]}],
                        max_new_tokens=256, **chat_kwargs)
                    transcript = (transcript.strip()
                                  if isinstance(transcript, str)
                                  else str(transcript))
                    pr, mr = p_yes(transcript)
                    pt, mt = p_yes(str(row.query))
                    record.update({
                        "transcript": transcript, "p_yes_rtj": pr,
                        "mass_rtj": mr, "p_yes_textq": pt, "mass_textq": mt,
                    })
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_s"] = time.perf_counter() - started
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if i < 2 or (i + 1) % 10 == 0:
                print(f"rank {rank}: {i + 1}/{len(rows)} {row.id} "
                      f"error={record['error'] is not None} "
                      f"seconds={record['elapsed_s']:.2f}", flush=True)


if __name__ == "__main__":
    main()
