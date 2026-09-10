"""TTFA supplement run (8cz-ttfa, 2026-09-09): measure real
time-to-first-audio for the main-table protocol — measured, not
reconstructed from stage costs.

Independent of modal_native_bench.py (v1, main table) and
modal_native_bench2.py (v2, val1): NEW modal app + NEW output tree
/data/ttfa_real/{run_id}/... Nothing here writes into v1/v2 outputs.

Protocol "ttfa-v3" — the v1/v2 session loop with exactly these changes:
1. REAL-TIME INPUT. Chunk i of the query wav becomes *available* at
   feed_start + min((i+1)*1.0s, audio_s) on the worker monotonic
   clock (a chunk can only be consumed after its last sample exists);
   trailing silence keeps the 1 chunk/s schedule. Per-chunk pacing lag
   is logged. input_end := feed_start + audio_s = scheduled arrival of
   the last real input sample (actual feed completion also logged).
2. REAL AUDIO ON BOTH PATHS. The duplex head runs generate_audio=True
   (deployed demo config: as_duplex() defaults +
   prepare(prompt_wav_path=...) + streaming_generate(prompt_wav_path,
   top_k=20)); every speak chunk returns its PCM. The escalated relay
   uses the deployed teacher-forced talker synth iterated as a STREAM:
   the first yielded PCM block is stamped when the generator hands it
   back, not after the full waveform.
3. TTFA_server := first_answer_pcm - input_end, same time.monotonic
   clock, direct subtraction, SIGNED (early responses keep their
   negative value). first_answer_pcm = first PCM block of the FINAL
   delivered answer channel: local speak audio (unfired) or relay
   synth (fired). Stall audio and the onset-chunk local candidate are
   stamped separately (candidate_first_pcm / stall_first_pcm) and roll
   into first_any_pcm; they never count as the answer.
   Granularity note (recorded, not hidden): duplex audio is
   chunk-blocking — a chunk's PCM is deliverable when its
   streaming_generate call returns, and that return instant is the
   stamp; relay synth is generator-streamed (~1 s blocks). Deployed
   release rule (demo_duplex 8cp): the onset chunk's local PCM has
   already been emitted when the gate reads — it is stamped as
   candidate_first_pcm and counted in first_any_pcm.
   These are SERVER pcm-ready times; no client clock exists here and
   nothing in this file may be reported as client-heard latency.
4. CAUSAL EXPERT INPUT (v2 rule): the expert gets exactly the real
   audio prefix consumed at fire time; with real-time pacing consumed
   == arrived. Samples + sha256 logged.
5. EVERY row kept. status: completed / non_commit / expert_timeout /
   expert_error / no_relay_audio / no_local_audio / empty_response /
   error. Missing stamps stay null, never 0. One attempt per query;
   reruns must pass --attempt 2 and land as extra rows, never
   replacing old ones.
6. No judging, no gate refit, no threshold search in this file.

Arms: local (tier "never": gate scored, threshold +inf) /
conservative / balanced / aggressive (nominal 15/30/50% en
thresholds from the frozen artifact) / always.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_ttfa_bench.py::freeze_ep --run-id ttfa1
  modal run modal_ttfa_bench.py::manifest_ep --pool frozen --run-id ttfa1
  modal run modal_ttfa_bench.py::run_bench --pool frozen --arm always --limit 3 --run-id ttfa1
  modal run modal_ttfa_bench.py::run_bench --pool frozen --arm aggressive --workers 6 --run-id ttfa1
"""
import json
import os
import time

import modal

DATA = "/data"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"
LAYER = 22
K3 = 8
ART = f"{DATA}/gate_native.json"
ACT = f"{DATA}/gate_act.json"
OUT_DIR = f"{DATA}/ttfa_real"
MAX_WAIT_S = 150
MAX_ANS = 60
MAX_CHUNKS = 400
SYS_PROMPT = "You are a friendly assistant."       # official serving cfg
GEN_TOP_K = 20
FORCE_LISTEN = 3
SR = 16000
CHUNK_S = 1.0
OUT_SR = 24000
PROTOCOL = "ttfa-v3"
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
ARM2TIER = {"local": "never", "conservative": "conservative",
            "balanced": "balanced", "aggressive": "aggressive",
            "always": "always"}

POOLS = {   # pool -> (queries file, audio dir, lang)
    "frozen":    (f"{DATA}/queries.jsonl",           f"{DATA}/audio_pool", "en"),
    "striviaqa": (f"{DATA}/queries_striviaqa.jsonl", f"{DATA}/bench_audio", "en"),
    "swebq":     (f"{DATA}/queries_swebq.jsonl",     f"{DATA}/bench_audio", "en"),
    "sllama":    (f"{DATA}/queries_sllama.jsonl",    f"{DATA}/bench_audio", "en"),
    "sdqa":      (f"{DATA}/queries_sdqa.jsonl",      f"{DATA}/sdqa_audio",  "en"),
}

STALL = "Hmm, let me double-check that — one moment."
STALL_NOTE = ("[SYSTEM NOTE] Your answer so far is likely wrong. You "
              "just told the user: \"" + STALL + "\" A verified answer "
              "will arrive in a moment.")

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

app = modal.App("ttfa-bench")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")

util_img = (modal.Image.debian_slim(python_version="3.11")
            .apt_install("libsndfile1")
            .pip_install("soundfile", "numpy")
            .add_local_file(_APP_PY, "/root/modal_app.py"))
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]",
        "transformers==4.51.0",
        "accelerate==1.12.0",
        "setuptools<81",
        "pydantic>=2.11",
        "PyYAML",
        "soundfile",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "openai",
        "sentencepiece",
        "fastapi[standard]",   # layer-hash parity with demo_duplex
    )
    .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
    .add_local_file(_APP_PY, "/root/modal_app.py"))


def _load_queries(pool):
    qfile, _, _ = POOLS[pool]
    qs = [json.loads(x) for x in open(qfile, encoding="utf-8") if x.strip()]
    if pool == "frozen":
        qs = [q for q in qs if q.get("split") == "test"]
    return qs


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


@app.function(image=gpu_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60 * 6)
def live_shard(shard: list, pool: str, arm: str, shard_id: int = -1,
               run_id: str = "", attempt: int = 1) -> list:
    import glob as _glob
    import hashlib
    import shutil
    import socket
    import sys
    import threading
    import zlib
    if not run_id:
        raise ValueError("run_id required")
    tier = ARM2TIER[arm]
    mono = time.monotonic

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import escalate
    import relay_fmt

    _, audio_dir, lang = POOLS[pool]
    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=True,
    ).eval().cuda()
    _ = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # deployed demo config: audio ON (as_duplex defaults generate_audio=True)
    duplex = model.as_duplex()
    duplex.force_listen_count = FORCE_LISTEN
    ref, _sr = librosa.load(PROMPT_WAV, sr=SR, mono=True)

    art = json.load(open(ART))
    art_sha = hashlib.sha256(open(ART, "rb").read()).hexdigest()
    act_sha = (hashlib.sha256(open(ACT, "rb").read()).hexdigest()
               if os.path.exists(ACT) else None)
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    thr_tab = (art.get("eot_thresholds_lang", {}).get(lang)
               or art.get("eot_thresholds", {}))
    thr = {"never": 1e9, "always": -1e9}.get(tier, thr_tab.get(tier, 1e9))
    act = json.load(open(ACT)) if os.path.exists(ACT) else None
    aw = np.array(act["w"], dtype=np.float32) if act else None

    cfg = {"protocol": PROTOCOL, "arm": arm, "tier": tier, "pool": pool,
           "gate_art_sha256": art_sha, "gate_act_sha256": act_sha,
           "threshold": None if abs(thr) > 1e8 else thr,
           "layer": LAYER, "k3": K3, "gen_top_k": GEN_TOP_K,
           "force_listen_count": FORCE_LISTEN, "sys_prompt": SYS_PROMPT,
           "chunk_s": CHUNK_S, "max_wait_s": MAX_WAIT_S,
           "max_ans": MAX_ANS, "max_chunks": MAX_CHUNKS,
           "relay": "tts", "relay_fmt": "v2", "expert": "web",
           "generate_audio": True, "out_sr": OUT_SR}
    cfg_sha = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    gpu_name = torch.cuda.get_device_name(0)
    host = socket.gethostname()

    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            sm = h.sum(0).cpu()
            st3["sum"] = sm if st3["sum"] is None else st3["sum"] + sm
            st3["cnt"] += h.shape[0]
    hh = model.llm.model.layers[LAYER].register_forward_hook(hook)

    def feat_now():
        parts = [st3["tail"][-1], st3["tail"].mean(0),
                 st3["sum"] / max(1, st3["cnt"])]
        return torch.cat(parts).numpy()

    rng = np.random.default_rng(9)

    def sil():
        return rng.normal(0, 0.003, SR).astype(np.float32)

    def gen():
        return duplex.streaming_generate(prompt_wav_path=PROMPT_WAV,
                                         top_k=GEN_TOP_K)

    import inspect as _insp

    def _call_def(fn, **kw):
        ps = set(_insp.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in ps})

    clean_expert = relay_fmt.clean_expert_v2

    tts_ready = {"ok": False}
    tok_tf = None
    try:    # relay path: deployed teacher-forced talker synth
        model.init_token2wav_cache(ref)
        tok_tf = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        tts_ready["ok"] = True
    except Exception as e:
        print(f">>> token2wav init failed: {e}", flush=True)

    def synth_stream(text):
        """Teacher-forced talker synth, iterated as a stream. Returns
        (wav float32 24k | None, first_pcm_mono | None, wall_ms).
        first_pcm_mono is stamped the instant the generator yields the
        first non-empty PCM block — never after full-waveform join."""
        t0 = time.time()
        first = None
        parts = []
        model.reset_session(reset_token2wav_cache=False)
        sys_msg = _call_def(model.get_sys_prompt, mode="omni", language="en")
        _call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg],
                  tokenizer=tok_tf)
        _call_def(model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user",
                         "content": [np.zeros(SR, dtype="float32")]}],
                  tokenizer=tok_tf, is_last_chunk=True)
        res = _call_def(model.streaming_generate, tokenizer=tok_tf,
                        temperature=0.1, generate_audio=True,
                        use_tts_template=True, teacher_forcing=True,
                        teacher_forcing_text=text, max_new_tokens=256,
                        session_id="s1")
        for item in res:
            wf = item[0] if isinstance(item, tuple) else None
            if wf is not None:
                a = np.asarray(wf).reshape(-1)
                if a.size:
                    if first is None:
                        first = mono()
                    parts.append(a)
        wav = np.concatenate(parts).astype("float32") if parts else None
        return wav, first, int((time.time() - t0) * 1000)

    base = f"{OUT_DIR}/{run_id}"
    adir = f"{base}/{pool}/audio"
    os.makedirs(adir, exist_ok=True)

    def save_wav(qid, kind, wav):
        if wav is None or not len(wav):
            return None
        p = f"{adir}/{qid}_{arm}_{kind}.wav"
        sf.write(p, np.asarray(wav, dtype="float32"), OUT_SR)
        return {"path": p, "sha256": _sha(p), "sr": OUT_SR,
                "seconds": round(len(wav) / OUT_SR, 3)}

    # ---- warm-up (fixed policy, logged, never in results) -------------
    t_wu = time.time()
    duplex.prepare(prefix_system_prompt=SYS_PROMPT,
                   ref_audio=ref, prompt_wav_path=PROMPT_WAV)
    for _ in range(6):
        duplex.streaming_prefill(audio_waveform=sil())
        gen()
    if tts_ready["ok"]:
        synth_stream("Warm up complete.")
    warmup_s = round(time.time() - t_wu, 1)
    print(f">>> warmup {warmup_s}s on {gpu_name} ({host})", flush=True)

    results = []
    for qi, q in enumerate(shard):
        wav_p = q.get("audio") or f"{audio_dir}/{q['id']}.wav"
        if not os.path.exists(wav_p):
            results.append({"id": q["id"], "pool": pool, "arm": arm,
                            "run_id": run_id, "attempt": attempt,
                            "status": "missing_input", "input_path": wav_p})
            continue
        au, _sr = librosa.load(wav_p, sr=SR, mono=True)
        audio_s = len(au) / SR
        n_real = (len(au) + SR - 1) // SR
        input_sha = hashlib.sha256(au.tobytes()).hexdigest()

        def chunk_at(i):
            c = au[i * SR:(i + 1) * SR]
            return (np.pad(c, (0, SR - len(c))) if len(c) < SR else c)

        duplex.prepare(prefix_system_prompt=SYS_PROMPT,
                       ref_audio=ref, prompt_wav_path=PROMPT_WAV)
        st3.update(tail=None, sum=None, cnt=0, accum=False)
        seed = zlib.crc32(f"{run_id}:{q['id']}".encode())
        torch.manual_seed(seed)

        ts0 = mono()
        ts = {"feed_start": 0.0, "input_end": round(audio_s, 3),
              "last_real_feed_done": None, "onset": None,
              "gate_read_done": None, "fire": None,
              "candidate_first_pcm": None, "stall_note": None,
              "stall_first_pcm": None, "local_first_pcm": None,
              "wait_start": None, "expert_req": None, "expert_resp": None,
              "relay_synth_start": None, "relay_first_pcm": None,
              "relay_synth_end": None, "first_any_pcm": None,
              "first_answer_pcm": None, "session_end": None}
        rec = {"id": q["id"], "pool": pool, "arm": arm, "tier": tier,
               "lang": q.get("lang", lang), "run_id": run_id,
               "attempt": attempt, "seed": int(seed),
               "protocol": PROTOCOL, "config_sha256": cfg_sha,
               "gate_art_sha256": art_sha, "threshold": cfg["threshold"],
               "gpu": gpu_name, "host": host, "shard": shard_id,
               "query": q.get("query"),
               "input_path": wav_p, "input_sha256": input_sha,
               "input_samples": int(len(au)),
               "input_s": round(audio_s, 3), "n_real_chunks": int(n_real),
               "score": None, "act": None, "is_info": None,
               "fired": False, "mode": "local",
               "answer": "", "relay_text": "", "expert_answer": "",
               "transcript": "", "asr_s": None, "expert_s": None,
               "expert_input_samples": None, "expert_input_s": None,
               "expert_input_sha256": None,
               "expert_timed_out": False, "wait_s": None,
               "relay_synth_ms": None, "onset_chunk": None,
               "eot_seen": False, "n_speak_chunks": 0,
               "feed_lag_max_ms": 0, "feed_lag_sum_ms": 0,
               "audio_files": {}, "chunks": [], "status": None,
               "error": None}
        exp = {}
        exp_done = threading.Event()

        def expert_call(snapshot):
            try:
                ts["expert_req"] = mono() - ts0
                t0 = time.time()
                up_p = f"/tmp/up{shard_id}.wav"
                sf.write(up_p, snapshot, SR)
                with open(up_p, "rb") as fh:
                    tr = (escalate._client().audio.transcriptions
                          .create(model="gpt-transcribe", file=fh,
                                  response_format="text"))
                up = (tr if isinstance(tr, str)
                      else getattr(tr, "text", str(tr)))
                exp["uplink"] = str(up)
                exp["asr_s"] = round(time.time() - t0, 2)
                t1 = time.time()
                r = escalate.ask_expert_web(up, effort="low")
                if r.get("error"):
                    r = escalate.ask_expert(up, effort="low")
                exp["answer"] = (r.get("answer")
                                 or f"[error: {r.get('error')}]")
                exp["expert_s"] = round(time.time() - t1, 2)
            except Exception as e:
                exp["answer"] = f"[thinker failed: {str(e)[:100]}]"
            finally:
                ts["expert_resp"] = mono() - ts0
                exp_done.set()

        texts = []
        segments = {"answer": [], "candidate": [], "stall": []}
        phase = "answer"        # pre-fire speak audio; on fire -> stall
        n_ans = 0
        prev_listen = True
        waiting = False
        go_wait = False
        ci = -1
        try:
            while ci < MAX_CHUNKS - 1:
                ci += 1
                # ---- real-time pacing: chunk available only after its
                # last sample has arrived --------------------------------
                if ci < n_real:
                    avail = min((ci + 1) * CHUNK_S, audio_s)
                    ch = chunk_at(ci)
                    is_real = True
                else:
                    avail = audio_s + (ci - n_real + 1) * CHUNK_S
                    ch = sil()
                    is_real = False
                dt = (ts0 + avail) - mono()
                if dt > 0:
                    time.sleep(dt)
                else:
                    lag = int(-dt * 1000)
                    rec["feed_lag_max_ms"] = max(rec["feed_lag_max_ms"], lag)
                    rec["feed_lag_sum_ms"] += lag
                st3["accum"] = True
                ok = duplex.streaming_prefill(audio_waveform=ch)
                st3["accum"] = False
                t_feed = mono() - ts0
                if is_real and ci == n_real - 1:
                    ts["last_real_feed_done"] = t_feed
                if not ok.get("success"):
                    rec["chunks"].append([ci, int(is_real), round(avail, 3),
                                          round(t_feed, 3), None, 1, 0, 0])
                    continue
                r = gen()
                t_ret = mono() - ts0
                wf = r.get("audio_waveform")
                n_wav = (0 if (r["is_listen"] or wf is None)
                         else int(np.asarray(wf).reshape(-1).shape[0]))
                rec["chunks"].append([ci, int(is_real), round(avail, 3),
                                      round(t_feed, 3), round(t_ret, 3),
                                      int(r["is_listen"]), n_wav,
                                      int(bool(r.get("end_of_turn")))])
                if r.get("text"):
                    texts.append(r["text"])
                if n_wav:
                    a = np.asarray(wf, dtype=np.float32).reshape(-1)
                    segments[phase].append(a)
                    if ts["first_any_pcm"] is None:
                        ts["first_any_pcm"] = t_ret
                    key = {"answer": "local_first_pcm",
                           "candidate": "candidate_first_pcm",
                           "stall": "stall_first_pcm"}[phase]
                    if ts[key] is None:
                        ts[key] = t_ret

                if prev_listen and not r["is_listen"] \
                        and rec["onset_chunk"] is None:
                    rec["onset_chunk"] = ci
                    ts["onset"] = t_ret
                    v = feat_now()
                    sc = float(1.0 / (1.0 + np.exp(-(float(v @ w) + b))))
                    rec["score"] = round(sc, 4)
                    if aw is not None:
                        a2 = float(1.0 / (1.0 + np.exp(
                            -(float(v @ aw) + act["b"]))))
                        rec["act"] = round(a2, 4)
                        rec["is_info"] = bool(a2 >= act["act_threshold"])
                    ts["gate_read_done"] = mono() - ts0
                    fire = sc >= thr
                    if tier in RATES and rec["is_info"] is False:
                        fire = False        # 8bh: floor turns never escalate
                    rec["fired"] = bool(fire)
                    if rec["fired"]:
                        rec["mode"] = "escalated"
                        ts["fire"] = mono() - ts0
                        # deployed release rule: this chunk's PCM was
                        # already emitted when the gate read -> candidate
                        if segments["answer"]:
                            segments["candidate"] = segments["answer"]
                            segments["answer"] = []
                            ts["candidate_first_pcm"] = ts["local_first_pcm"]
                            ts["local_first_pcm"] = None
                        phase = "stall"
                        # causal expert input: the arrived real prefix
                        consumed = au[:min(len(au), (ci + 1) * SR)]
                        rec["expert_input_samples"] = int(len(consumed))
                        rec["expert_input_s"] = round(len(consumed) / SR, 3)
                        rec["expert_input_sha256"] = hashlib.sha256(
                            consumed.tobytes()).hexdigest()
                        threading.Thread(target=expert_call,
                                         args=(consumed,),
                                         daemon=True).start()
                        if not r.get("end_of_turn"):
                            ts["stall_note"] = mono() - ts0
                            duplex.streaming_prefill(text_list=[STALL_NOTE])
                            r2 = gen()
                            t2 = mono() - ts0
                            if r2.get("text"):
                                texts.append(r2["text"])
                            wf2 = r2.get("audio_waveform")
                            if not r2["is_listen"] and wf2 is not None:
                                a3 = np.asarray(wf2,
                                                dtype=np.float32).reshape(-1)
                                if a3.size:
                                    segments["stall"].append(a3)
                                    if ts["first_any_pcm"] is None:
                                        ts["first_any_pcm"] = t2
                                    if ts["stall_first_pcm"] is None:
                                        ts["stall_first_pcm"] = t2
                            prev_listen = r2["is_listen"]
                            if r2.get("end_of_turn"):
                                go_wait = True
                                break
                            continue
                        go_wait = True
                        break

                if not r["is_listen"]:
                    n_ans += 1
                if r.get("end_of_turn"):
                    if rec["fired"]:
                        go_wait = True
                        break
                    rec["eot_seen"] = True
                    break
                if n_ans >= MAX_ANS:
                    break
                prev_listen = r["is_listen"]
        except Exception as e:
            rec["error"] = str(e)[:200]
        rec["n_speak_chunks"] = n_ans
        # fired sessions that fell out on the MAX_ANS / MAX_CHUNKS cap
        # still owe the user the relay (deployed intent); flagged via
        # eot_seen=False
        if rec["fired"] and rec["error"] is None:
            go_wait = True

        # ---- wait for expert + relay (fired only) ----------------------
        if rec["fired"] and rec["error"] is None and go_wait:
            ts["wait_start"] = mono() - ts0
            t_w0 = mono()
            while not exp_done.is_set() and mono() - t_w0 < MAX_WAIT_S:
                exp_done.wait(0.05)
            rec["wait_s"] = round(mono() - t_w0, 3)
            if not exp_done.is_set():
                rec["expert_timed_out"] = True
            spoken = ""
            if exp.get("answer") and not rec["expert_timed_out"]:
                spoken = clean_expert(exp["answer"])
            rec["relay_text"] = spoken
            if spoken and tts_ready["ok"]:
                try:
                    ts["relay_synth_start"] = mono() - ts0
                    rwav, rfirst, rms = synth_stream(spoken)
                    ts["relay_synth_end"] = mono() - ts0
                    rec["relay_synth_ms"] = rms
                    if rfirst is not None:
                        ts["relay_first_pcm"] = rfirst - ts0
                    if rwav is not None:
                        rec["audio_files"]["relay"] = save_wav(
                            q["id"], "relay", rwav)
                except Exception as e:
                    rec["error"] = f"relay_synth: {str(e)[:150]}"
                # duplex context note (deployed parity; timing-inert)
                try:
                    duplex.streaming_prefill(text_list=[
                        "[SYSTEM NOTE] You just told the user: "
                        + spoken + " Do not repeat it."])
                    r3 = gen()
                    rec["eot_seen"] = bool(r3.get("end_of_turn"))
                except Exception:
                    pass

        ts["session_end"] = mono() - ts0
        exp_done.wait(timeout=5) if rec["fired"] else None

        full = "".join(texts).strip()
        if rec["fired"]:
            rec["expert_answer"] = exp.get("answer", "")
            rec["transcript"] = exp.get("uplink", "")[:300]
            rec["asr_s"] = exp.get("asr_s")
            rec["expert_s"] = exp.get("expert_s")
        else:
            rec["answer"] = full

        # first_any among candidate/stall/local/relay (min of stamps)
        cand = [ts[k] for k in ("candidate_first_pcm", "stall_first_pcm",
                                "local_first_pcm", "relay_first_pcm")
                if ts[k] is not None]
        ts["first_any_pcm"] = min(cand) if cand else None
        ts["first_answer_pcm"] = (ts["relay_first_pcm"] if rec["fired"]
                                  else ts["local_first_pcm"])

        for kind in ("answer", "candidate", "stall"):
            if segments[kind]:
                wavcat = np.concatenate(segments[kind])
                name = "local" if kind == "answer" else kind
                rec["audio_files"][name] = save_wav(q["id"], name, wavcat)

        deliv_txt = rec["relay_text"] if rec["fired"] else full
        if rec["error"]:
            rec["status"] = "error"
        elif rec["onset_chunk"] is None:
            rec["status"] = "non_commit"
        elif rec["expert_timed_out"]:
            rec["status"] = "expert_timeout"
        elif rec["fired"] and str(rec["expert_answer"]).startswith(
                ("[thinker failed", "[error")):
            rec["status"] = "expert_error"
        elif not deliv_txt:
            rec["status"] = "empty_response"
        elif rec["fired"] and ts["relay_first_pcm"] is None:
            rec["status"] = "no_relay_audio"
        elif not rec["fired"] and ts["local_first_pcm"] is None:
            rec["status"] = "no_local_audio"
        else:
            rec["status"] = "completed"

        rec["ts"] = {k: (None if v is None else round(v, 3))
                     for k, v in ts.items()}
        rec["ttfa_answer_s"] = (
            None if ts["first_answer_pcm"] is None
            else round(ts["first_answer_pcm"] - audio_s, 3))
        rec["ttfa_any_s"] = (
            None if ts["first_any_pcm"] is None
            else round(ts["first_any_pcm"] - audio_s, 3))
        results.append(rec)
        print(f"  [{qi}] {q['id']} arm={arm} score={rec['score']} "
              f"fired={rec['fired']} status={rec['status']} "
              f"ttfa={rec['ttfa_answer_s']} any={rec['ttfa_any_s']} "
              f"lagmax={rec['feed_lag_max_ms']}ms", flush=True)

        # incremental append so crashes lose at most one session
        os.makedirs(f"{base}/{pool}", exist_ok=True)
        sfx = "smoke" if shard_id < 0 else f"shard{shard_id}"
        with open(f"{base}/{pool}/{arm}.jsonl.{sfx}", "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if qi % 5 == 4:
            gate_data.commit()

    hh.remove()
    gate_data.commit()
    return [r.get("status") for r in results]


@app.function(image=util_img, volumes={DATA: gate_data}, timeout=60 * 5)
def _todo(pool: str, arm: str, run_id: str = "",
          include_done: bool = False) -> list:
    import glob as _glob
    if not run_id:
        raise ValueError("run_id required")
    qs = _load_queries(pool)
    if include_done:
        return qs
    done = set()
    for p in _glob.glob(f"{OUT_DIR}/{run_id}/{pool}/{arm}.jsonl.shard*"):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                done.add(json.loads(ln)["id"])
    return [q for q in qs if q["id"] not in done]


@app.function(image=util_img,
              volumes={DATA: gate_data, "/workspace/models": weights},
              timeout=60 * 10)
def freeze(run_id: str) -> dict:
    """Record the frozen run config: artifact hashes, thresholds, query
    file hashes, model identity. Pure read + one new file; refits
    nothing."""
    art = json.load(open(ART))
    out = {"run_id": run_id, "protocol": PROTOCOL,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gate_art_path": ART, "gate_art_sha256": _sha(ART),
           "gate_act_path": ACT,
           "gate_act_sha256": _sha(ACT) if os.path.exists(ACT) else None,
           "eot_thresholds_lang": art.get("eot_thresholds_lang"),
           "eot_thresholds": art.get("eot_thresholds"),
           "gate_layer": art.get("layer"), "gate_modes": art.get("modes"),
           "gate_label_source": art.get("label_source"),
           "read_point": {"layer": LAYER, "k3": K3,
                          "at": "listen->speak commit"},
           "serving": {"sys_prompt": SYS_PROMPT, "gen_top_k": GEN_TOP_K,
                       "force_listen_count": FORCE_LISTEN,
                       "generate_audio": True,
                       "duplex_defaults": "as_duplex() stock",
                       "relay": "tts teacher-forced talker",
                       "relay_fmt": "clean_expert_v2",
                       "expert": "ask_expert_web(effort=low) -> "
                                 "ask_expert fallback",
                       "expert_asr": "gpt-transcribe"},
           "timing": {"chunk_s": CHUNK_S, "max_wait_s": MAX_WAIT_S,
                      "max_ans": MAX_ANS, "max_chunks": MAX_CHUNKS,
                      "clock": "time.monotonic per worker",
                      "ttfa_def": "first_answer_pcm - input_end, signed",
                      "input_end": "feed_start + audio_s (scheduled "
                                   "arrival of last real sample)"},
           "pools": {}, "model": {}}
    for pool, (qf, adir, lng) in POOLS.items():
        qs = _load_queries(pool)
        out["pools"][pool] = {"queries_file": qf,
                              "queries_sha256": _sha(qf),
                              "n_queries": len(qs), "lang": lng,
                              "audio_dir": adir,
                              "note": ("split==test only" if pool == "frozen"
                                       else "all")}
    md = {}
    for f in ("config.json", "model.safetensors.index.json",
              "generation_config.json"):
        p = f"{MODEL_DIR}/{f}"
        if os.path.exists(p):
            md[f] = _sha(p)
    try:
        cfgj = json.load(open(f"{MODEL_DIR}/config.json"))
        md["_name_or_path"] = cfgj.get("_name_or_path")
    except Exception:
        pass
    n_files = sum(len(fs) for _, _, fs in os.walk(MODEL_DIR))
    out["model"] = {"dir": MODEL_DIR, "file_hashes": md,
                    "n_files": n_files}
    base = f"{OUT_DIR}/{run_id}"
    os.makedirs(base, exist_ok=True)
    with open(f"{base}/freeze.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    gate_data.commit()
    return out


@app.function(image=util_img, volumes={DATA: gate_data}, timeout=60 * 15)
def manifest(pool: str, run_id: str) -> int:
    """Input manifest: per query id -> wav sha256, bytes, seconds."""
    import soundfile as sf
    _, audio_dir, _ = POOLS[pool]
    base = f"{OUT_DIR}/{run_id}/{pool}"
    os.makedirs(base, exist_ok=True)
    n = 0
    with open(f"{base}/input_manifest.jsonl", "w", encoding="utf-8") as fh:
        for q in _load_queries(pool):
            p = q.get("audio") or f"{audio_dir}/{q['id']}.wav"
            row = {"id": q["id"], "path": p, "exists": os.path.exists(p)}
            if row["exists"]:
                row["sha256"] = _sha(p)
                row["bytes"] = os.path.getsize(p)
                try:
                    info = sf.info(p)
                    row["seconds"] = round(info.frames / info.samplerate, 3)
                    row["sr"] = info.samplerate
                except Exception as e:
                    row["seconds"] = None
                    row["read_err"] = str(e)[:80]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    gate_data.commit()
    return n


@app.local_entrypoint()
def run_bench(pool: str = "frozen", arm: str = "local",
              workers: int = 6, limit: int = 0, run_id: str = "",
              ids: str = "", attempt: int = 1):
    if not run_id:
        raise SystemExit("--run-id required")
    if arm not in ARM2TIER:
        raise SystemExit(f"arm must be one of {list(ARM2TIER)}")
    qs = _todo.remote(pool, arm, run_id, bool(ids) and attempt > 1)
    if ids:
        want = set(ids.split(","))
        qs = [q for q in qs if q["id"] in want]
        if not limit:
            limit = len(qs)
    smoke = bool(limit) and limit <= 8
    if limit:
        qs = qs[:limit]
        if smoke:
            workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> ttfa bench [{pool}/{arm}/{run_id}]: {len(qs)} queries, "
          f"{workers} workers{' (smoke)' if smoke else ''}", flush=True)
    if not qs:
        print(">>> nothing to do (all ids already recorded)")
        return
    done = list(live_shard.starmap(
        [(shards[i], pool, arm, i if not smoke else -1, run_id, attempt)
         for i in range(workers) if shards[i]]))
    from collections import Counter
    print(f">>> complete: {Counter(s for d in done for s in d)}")


@app.local_entrypoint()
def freeze_ep(run_id: str = ""):
    if not run_id:
        raise SystemExit("--run-id required")
    out = freeze.remote(run_id)
    print(json.dumps(out, ensure_ascii=False, indent=2))


@app.local_entrypoint()
def manifest_ep(pool: str = "frozen", run_id: str = ""):
    if not run_id:
        raise SystemExit("--run-id required")
    n = manifest.remote(pool, run_id)
    print(f">>> manifest {pool}: {n} rows")
