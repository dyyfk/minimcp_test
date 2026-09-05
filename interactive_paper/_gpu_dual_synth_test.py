"""8ck isolation: can a second model instance synthesize TTS
concurrently with the duplex loop on one H100?

The 8ck-2 live attempt (dual instance + synth worker) broke the relay
chain in ways the demo's moving parts obscured. This strips it to the
bone, with full tracebacks:

  stage 1  load model A (duplex) + model B (synth)  -> VRAM, load_s
  stage 2  standalone synth on B (the load-time stall path)
  stage 3  duplex chunks on A alone                 -> latency baseline
  stage 4  duplex chunks on A WHILE B synthesizes in a thread
           -> exceptions, latency inflation, synth outputs
  stage 5  duplex still answers a question after all that

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _gpu_dual_synth_test.py::run
"""
import os

import modal

app = modal.App("dual-synth-test")
weights = modal.Volume.from_name("minicpm-o45-weights")
gate_data = modal.Volume.from_name("gate-data")
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"
DATA = "/data"

_HERE = os.path.dirname(os.path.abspath(__file__))

gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]", "transformers==4.51.0",
        "accelerate==1.12.0", "setuptools<81", "pydantic>=2.11",
        "PyYAML", "soundfile", "opencv-python-headless",
        "huggingface_hub[hf_transfer]", "sentencepiece", "librosa",
    )
    .add_local_dir(os.path.join(_HERE, "_model_src"),
                   "/workspace/model_src"))

STALL = "Hmm, let me double-check that — one moment."
PIECES = ["Nvidia (NVDA) is currently trading at $230.36 USD,",
          "up $1.77 (+0.77%) from the previous close.",
          "The intraday range so far is $216.33 to $221.25.",
          "Volume is roughly in line with the 30-day average."]


@app.function(image=gpu_image, gpu="H100", timeout=60 * 30,
              volumes={"/workspace/models": weights, DATA: gate_data})
def run():
    import glob as _glob
    import inspect
    import shutil
    import threading
    import time
    import traceback

    import librosa
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)
    for f in _glob.glob("/workspace/model_src/*.py"):
        shutil.copy(f, cache)

    def vram(tag):
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        print(f">>> [{tag}] vram alloc {a:.1f}G reserved {r:.1f}G",
              flush=True)

    def _call_def(fn, /, **kw):
        p = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in p})

    # ---- stage 1: loads ------------------------------------------------
    t0 = time.time()
    mA = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
        init_tts=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR,
                                        trust_remote_code=True)
    duplex = mA.as_duplex()
    print(f">>> model A + duplex in {time.time() - t0:.0f}s", flush=True)
    vram("A")
    t0 = time.time()
    try:
        mB = AutoModel.from_pretrained(
            MODEL_DIR, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
            init_vision=False, init_audio=True,
            init_tts=True).eval().cuda()
        print(f">>> model B in {time.time() - t0:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
        return {"stage": 1, "ok": False}
    vram("A+B")

    # ---- stage 2: standalone synth on B --------------------------------
    def synth(text):
        mB.reset_session(reset_token2wav_cache=False)
        sys_msg = _call_def(mB.get_sys_prompt, mode="omni",
                            language="en")
        _call_def(mB.streaming_prefill, session_id="s1",
                  msgs=[sys_msg], tokenizer=tok)
        _call_def(mB.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user",
                         "content": [np.zeros(16000, dtype="float32")]}],
                  tokenizer=tok, is_last_chunk=True)
        res = _call_def(mB.streaming_generate, tokenizer=tok,
                        temperature=0.1, generate_audio=True,
                        use_tts_template=True, teacher_forcing=True,
                        teacher_forcing_text=text, max_new_tokens=256,
                        session_id="s1")
        parts = []
        for item in res:
            wf = item[0] if isinstance(item, tuple) else None
            if wf is not None:
                parts.append(wf.float().cpu().numpy().reshape(-1))
        return np.concatenate(parts) if parts else None

    ref, _ = librosa.load(PROMPT_WAV, sr=16000, mono=True)
    try:
        # init_tts attaches the Token2wav audio tokenizer — on model A
        # as_duplex() does this implicitly; a bare second instance must
        # call it itself (missing this line was the whole 8ck-2 outage:
        # tts.audio_tokenizer stayed None -> .frontend AttributeError
        # -> load-time stall synth failed -> tts_ok False -> the demo
        # silently fell back to steer mode).
        mB.init_tts()
        mB.init_token2wav_cache(ref)
        t0 = time.time()
        p = synth(STALL)
        print(f">>> stage 2 standalone synth: "
              f"{0 if p is None else len(p) / 24000:.2f}s audio in "
              f"{time.time() - t0:.1f}s", flush=True)
        assert p is not None and len(p) > 24000
    except Exception:
        traceback.print_exc()
        return {"stage": 2, "ok": False}
    vram("post-synth")

    # ---- stage 3: duplex baseline --------------------------------------
    duplex.force_listen_count = 3
    duplex.prepare(prefix_system_prompt="You are a friendly assistant.",
                   ref_audio=ref, prompt_wav_path=PROMPT_WAV)
    rng = np.random.default_rng(3)

    def chunkstep():
        ok = duplex.streaming_prefill(
            audio_waveform=rng.normal(0, 0.003, 16000)
            .astype(np.float32))
        if ok.get("success"):
            duplex.streaming_generate(prompt_wav_path=PROMPT_WAV,
                                      top_k=20)

    lat_base = []
    for _ in range(15):
        t0 = time.time()
        chunkstep()
        lat_base.append(time.time() - t0)
    print(f">>> stage 3 duplex alone: chunk p50 "
          f"{np.median(lat_base):.2f}s p90 "
          f"{np.percentile(lat_base, 90):.2f}s", flush=True)

    # ---- stage 4: concurrent -------------------------------------------
    synth_out, synth_err = [], []

    def worker():
        for pc in PIECES:
            try:
                t0 = time.time()
                w = synth(" " + pc)
                synth_out.append((None if w is None
                                  else round(len(w) / 24000, 1),
                                  round(time.time() - t0, 1)))
            except Exception as e:
                synth_err.append(str(e)[:200])
                traceback.print_exc()

    th = threading.Thread(target=worker, daemon=True)
    lat_conc = []
    th.start()
    while th.is_alive():
        t0 = time.time()
        try:
            chunkstep()
        except Exception:
            traceback.print_exc()
            return {"stage": 4, "ok": False}
        lat_conc.append(time.time() - t0)
    th.join()
    print(f">>> stage 4 concurrent: {len(lat_conc)} chunks p50 "
          f"{np.median(lat_conc):.2f}s p90 "
          f"{np.percentile(lat_conc, 90):.2f}s | synth results "
          f"{synth_out} | errors {len(synth_err)}", flush=True)

    # ---- stage 5: duplex still commits ---------------------------------
    from openai import OpenAI  # noqa: F401  (not needed; use tts on B)
    q = synth(" What is the capital of France?")
    committed, answer = False, []
    if q is not None:
        # 24k -> 16k
        import scipy.signal as sps
        q16 = sps.resample_poly(q, 2, 3).astype(np.float32)
        chunks = [q16[i:i + 16000] for i in range(0, len(q16), 16000)]
        chunks = [np.pad(c, (0, 16000 - len(c)))
                  if len(c) < 16000 else c for c in chunks]
        for ch in chunks + [None] * 20:
            ok = duplex.streaming_prefill(
                audio_waveform=(rng.normal(0, 0.003, 16000)
                                .astype(np.float32) if ch is None
                                else ch))
            if not ok.get("success"):
                continue
            r = duplex.streaming_generate(prompt_wav_path=PROMPT_WAV,
                                          top_k=20)
            if not r["is_listen"]:
                committed = True
                if r.get("text"):
                    answer.append(r["text"])
            if r.get("end_of_turn"):
                break
    print(f">>> stage 5 post-concurrency commit: {committed} "
          f"answer='{(''.join(answer))[:80]}'", flush=True)
    print(">>> ALL STAGES DONE", flush=True)
    return {"ok": True, "lat_base_p50": round(float(np.median(lat_base)), 2),
            "lat_conc_p50": round(float(np.median(lat_conc)), 2),
            "synth": synth_out, "synth_err": synth_err,
            "commit_after": committed}
