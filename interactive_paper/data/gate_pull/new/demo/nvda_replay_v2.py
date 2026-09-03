"""nvda_replay_v2: nvda_replay.py + two CAUSAL reads (2026-09-01 review round).

Adds, per layer per query, on top of the pass-2 arrays (H_eot, H_mean, H_onset):
  H_pre  (L, 8, d)  the 8 frames immediately BEFORE the commit frame
                    (positions [commit-8, commit), clipped at prompt_len) —
                    a VAD-free, strictly pre-commit replacement for H_eot
  H_run  (L, d)     mean over ALL frames before the commit frame
                    ([prompt_len, commit)) — the user_mean the online
                    state machine (src/nvda_duplex_probe.py) actually computes
Everything else (model, prompt, batching, commit rule, dtype) is byte-identical
to nvda_replay.py so pass-2 labels remain valid up to the known ~15 %% answer
drift. Configure paths with the NVDA_ROOT, NVDA_DATA, and NVDA_OUT
environment variables.

Original header follows.
Standalone port of modal_nvda.py::answer_shard for a GPU worker.

Offline NVDA inference per wav: agent text (-> local-floor label after
judging) + eoth2-format hidden reads in the same pass. Every constant,
the batching policy, the capture slicing, and the output schema are
copied verbatim from interactive_paper/modal_nvda.py (8ac pipeline) so
the expansion shards are byte-compatible with the existing
nvda_h_{frozen,external}.shard*.npz captures.

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=0 python nvda_replay.py \
      --tag expansion --shard-id 0 --num-shards 8
"""
import argparse
import glob
import json
import os
import re
import sys
import time

ROOT = os.environ.get("NVDA_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("NVDA_NEMO_ROOT", f"{ROOT}/nemo-speech"))

MODEL_DIR = os.environ.get("NVDA_MODEL_DIR", f"{ROOT}/weights")
DATA = os.environ.get("NVDA_DATA", f"{ROOT}/data")
OUT = os.environ.get("NVDA_OUT", f"{ROOT}/out3")

# ---- constants copied from modal_nvda.py (do not tune) -------------------
K_EOT = 8
NVDA_LAYERS = list(range(2, 56, 4))          # 14 layers
FRAME_S = 0.08
TAIL_SIL_S = 12.0        # answer room appended to each query wav
SYS_PROMPT = ("You are a helpful voice assistant. Listen to the "
              "user's question and answer it directly and concisely. "
              "Do not greet the user; wait for the question.")
POOLS = {
    "expansion":  (f"{DATA}/audio_expansion",
                   f"{DATA}/queries_expansion.jsonl"),
    "expansion2": (f"{DATA}/audio_expansion2",
                   f"{DATA}/queries_expansion2.jsonl"),
    "frozen":     (f"{DATA}/audio_pool", f"{DATA}/queries.jsonl"),
    "flooract":   (f"{DATA}/flooract_audio",
                   f"{DATA}/queries_flooract.jsonl"),
    "striviaqa":  (f"{DATA}/bench_audio", f"{DATA}/queries_striviaqa.jsonl"),
    "swebq":      (f"{DATA}/bench_audio", f"{DATA}/queries_swebq.jsonl"),
    "sllama":     (f"{DATA}/bench_audio", f"{DATA}/queries_sllama.jsonl"),
    "valpaca":    (f"{DATA}/bench_audio", f"{DATA}/queries_valpaca.jsonl"),
    "sdqa":       (f"{DATA}/sdqa_audio",  f"{DATA}/queries_sdqa.jsonl"),
}


def _read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_model(device="cuda"):
    from nemo.collections.speechlm2.inference.utils.offline_voicechat \
        import build_model
    import torch
    m = build_model(MODEL_DIR, device=device)
    # bf16 on the STT stack only, TTS stays fp32 (same as modal_nvda)
    m.stt_model.to(torch.bfloat16)
    return m


def _infer_batch(model, wav_paths, capture=True):
    import numpy as np
    import librosa
    import torch
    from nemo.collections.speechlm2.inference.utils.offline_voicechat \
        import encode_system_prompt, run_offline_inference

    B = len(wav_paths)
    aus = []
    for w in wav_paths:
        au, _ = librosa.load(w, sr=16000, mono=True)
        aus.append(au)
    qlens = [len(a) for a in aus]
    tail = int(TAIL_SIL_S * 16000)
    full = max(qlens) + tail
    sig = torch.zeros(B, full)
    for b, a in enumerate(aus):
        sig[b, :len(a)] = torch.tensor(a)
    sig = sig.cuda()
    lens = torch.full((B,), full, dtype=torch.long, device="cuda")

    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.no_grad(), amp:
        q_sig = torch.zeros(B, max(qlens), device="cuda")
        for b, a in enumerate(aus):
            q_sig[b, :len(a)] = torch.tensor(a, device="cuda")
        q_len = torch.tensor(qlens, dtype=torch.long, device="cuda")
        out = model.stt_model.perception(
            input_signal=q_sig, input_signal_length=q_len)
        n_frames_q = [int(x) for x in out[1]]

    prompt_tokens, prompt_token_lens = encode_system_prompt(
        model, SYS_PROMPT, device="cuda")
    if prompt_tokens.shape[0] == 1 and B > 1:
        prompt_tokens = prompt_tokens.expand(B, -1).contiguous()
        prompt_token_lens = prompt_token_lens.expand(B).contiguous()
    prompt_len = int(prompt_token_lens[0].item())

    store = {}
    handles = []
    if capture:
        def mk(L):
            def hook(_m, _i, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                store[L] = hs.detach()            # (B, T_cur, d) GPU
            return hook
        handles = [model.stt_model.llm.layers[L].register_forward_hook(
            mk(L)) for L in NVDA_LAYERS]

    t0 = time.time()
    try:
        with amp:
            result = run_offline_inference(
                model, input_signal=sig, input_signal_lens=lens,
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_token_lens)
    finally:
        for h in handles:
            h.remove()
    secs = time.time() - t0

    # commit-to-speak frame (8be-analog read point): tokens_text is one
    # agent-channel token per 80 ms frame, pad id fills listen frames.
    # Onset = start of the first sustained (>=3-frame) non-pad run —
    # isolated marker tokens (e.g. the <$t$> transcription mark) are
    # single frames and must not count as the commit.
    TEXT_PAD_ID = 12
    tok = result.get("tokens_text")
    onsets = [None] * B
    if tok is not None:
        for b in range(B):
            row = tok[b].tolist()
            for i in range(len(row) - 2):
                if all(t != TEXT_PAD_ID for t in row[i:i + 3]):
                    onsets[b] = i
                    break

    texts = result.get("text", [""] * B)
    outs = []
    for b in range(B):
        t_end = prompt_len + n_frames_q[b]
        eot = mean = onset_h = pre_h = run_h = None
        onset = onsets[b]
        if capture and store:
            d = store[NVDA_LAYERS[0]].shape[-1]
            eot = np.zeros((len(NVDA_LAYERS), K_EOT, d),
                           dtype=np.float16)
            mean = np.zeros((len(NVDA_LAYERS), d), dtype=np.float16)
            onset_h = np.zeros((len(NVDA_LAYERS), K_EOT, d),
                               dtype=np.float16)
            pre_h = np.zeros((len(NVDA_LAYERS), K_EOT, d), dtype=np.float16)
            run_h = np.zeros((len(NVDA_LAYERS), d), dtype=np.float16)
            for j, L in enumerate(NVDA_LAYERS):
                h = store[L][b].float()           # (T, d)
                hi = min(t_end, h.shape[0])
                lo = max(prompt_len, hi - K_EOT)
                w = h[lo:hi].cpu().numpy().astype(np.float16)
                eot[j, K_EOT - w.shape[0]:] = w
                mean[j] = (h[prompt_len:hi].mean(0).cpu().numpy()
                           .astype(np.float16))
                if onset is not None:
                    # first K_EOT frames from the commit-to-speak
                    # frame; positions = prompt_len + frame index
                    olo = prompt_len + onset
                    ohi = min(olo + K_EOT, h.shape[0])
                    ow = (h[olo:ohi].cpu().numpy()
                          .astype(np.float16))
                    onset_h[j, K_EOT - ow.shape[0]:] = ow
                    # v2: strictly pre-commit windows
                    plo = max(prompt_len, olo - K_EOT)
                    if olo > plo:
                        pw = h[plo:olo].cpu().numpy().astype(np.float16)
                        pre_h[j, K_EOT - pw.shape[0]:] = pw
                        run_h[j] = (h[prompt_len:olo].mean(0).cpu().numpy()
                                    .astype(np.float16))
        raw = texts[b] if b < len(texts) else ""
        clean = re.sub(r"<[^>]{0,24}>", " ", raw)
        clean = re.sub(r"  +", " ", clean).strip()
        # user-transcription channel (additive; not written to npz/jsonl)
        src = ""
        for k in ("src_text", "src_text_asr_head", "src_text_rnnt"):
            v = result.get(k)
            if v is not None and b < len(v) and str(v[b]).strip():
                src = str(v[b])
                break
        outs.append(dict(text=clean, text_raw=raw, src_text=src,
                         secs=secs / B, batch_secs=secs,
                         n_frames_query=n_frames_q[b],
                         prompt_len=prompt_len, t_end=t_end,
                         onset_frame=-1 if onset is None else onset,
                         eot=eot, mean=mean, onset_h=onset_h,
                         pre_h=pre_h, run_h=run_h))
    store.clear()
    return outs


def main():
    import numpy as np
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="comma-separated pool tags; the model is loaded once")
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tags = [t.strip() for t in args.tag.split(",") if t.strip()]
    for t in tags:
        if t not in POOLS:
            raise SystemExit(f"unknown tag {t}; choices {sorted(POOLS)}")
    model = None
    for tag in tags:
        args.tag = tag
        model = _run_tag(model, args)


def _run_tag(model, args):
    import numpy as np
    import torch

    adir, qfile = POOLS[args.tag]
    qs = _read_jsonl(qfile)
    if args.limit:
        qs = qs[:args.limit]
    shard = qs[args.shard_id::args.num_shards]

    # resume: skip ids already captured in ANY existing shard of this tag
    done = set()
    for sh in glob.glob(f"{OUT}/nvda_h_{args.tag}.shard*.npz"):
        try:
            done |= {str(x) for x in np.load(sh, allow_pickle=True)["ids"]}
        except Exception as e:      # another shard may still be writing it
            print(f"  (resume: skipping unreadable {sh}: {type(e).__name__})", flush=True)
    shard = [q for q in shard if q["id"] not in done]
    print(f"[{args.tag} shard {args.shard_id}/{args.num_shards}] "
          f"{len(shard)} to run ({len(done)} already done)", flush=True)
    if not shard:
        return model

    if model is None:
        model = _load_model()
    items = [(q, f"{adir}/{q['id']}.wav") for q in shard]
    items = [(q, w) for q, w in items if os.path.exists(w)]
    items.sort(key=lambda t: os.path.getsize(t[1]))   # length buckets
    # adaptive batches: budget = batch_size x largest wav in the window
    BUDGET = 8 * 1024 * 1024                          # ~ 8 x 32 s wavs
    batches, cur, cur_max = [], [], 0
    for it in items:
        size = os.path.getsize(it[1])
        if cur and (len(cur) + 1) * max(size, cur_max) > BUDGET:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(it)
        cur_max = max(cur_max, size)
        if len(cur) == 8:
            batches.append(cur)
            cur, cur_max = [], 0
    if cur:
        batches.append(cur)

    ids, E, M, O, P, R, ONS, rows = [], [], [], [], [], [], [], []
    k = 0
    for chunk in batches:
        try:
            rs = _infer_batch(model, [w for _, w in chunk])
        except Exception as e:
            print(f"  !! batch@{k}: {type(e).__name__}: {str(e)[:150]}",
                  flush=True)
            k += len(chunk)
            torch.cuda.empty_cache()
            continue
        for (q, _), r in zip(chunk, rs):
            if r["eot"] is None:
                continue
            ids.append(q["id"])
            E.append(r["eot"])
            M.append(r["mean"])
            O.append(r["onset_h"])
            P.append(r["pre_h"]); R.append(r["run_h"])
            ONS.append(r["onset_frame"])
            rows.append({"id": q["id"], "answer": r["text"],
                         "answer_raw": r["text_raw"],
                         "secs": round(r["secs"], 2),
                         "n_frames_query": r["n_frames_query"],
                         "prompt_len": r["prompt_len"],
                         "onset_frame": r["onset_frame"]})
        if k % 32 == 0:
            r0 = rs[0]
            print(f"  [{k}/{len(items)}] B={len(chunk)} "
                  f"batch {r0['batch_secs']:.0f}s "
                  f"({r0['secs']:.1f}s/q) {repr(r0['text'])[:70]}",
                  flush=True)
        k += len(chunk)

    if not ids:
        print(f">>> nvda_{args.tag} shard {args.shard_id}: nothing "
              f"captured, no file written", flush=True)
        return model
    # never overwrite an existing shard file: a resume run writes to a
    # free recovery id instead (mirrors run_missing's 100+i convention)
    out_id = args.shard_id
    while os.path.exists(f"{OUT}/nvda_h_{args.tag}.shard{out_id}.npz"):
        out_id += 100
    final = f"{OUT}/nvda_h_{args.tag}.shard{out_id}.npz"
    np.savez_compressed(
        final + ".tmp.npz",
        ids=np.array(ids), H_eot=np.stack(E), H_mean=np.stack(M),
        H_onset=np.stack(O), H_pre=np.stack(P), H_run=np.stack(R),
        onset_frame=np.array(ONS, dtype=np.int32),
        layers=np.array(NVDA_LAYERS))
    os.replace(final + ".tmp.npz", final)          # atomic: readers never see a partial file
    with open(f"{OUT}/nvda_answers_{args.tag}.shard{out_id}.jsonl",
              "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    print(f">>> wrote nvda_{args.tag} shard {out_id} "
          f"({len(ids)})", flush=True)
    return model


if __name__ == "__main__":
    main()
