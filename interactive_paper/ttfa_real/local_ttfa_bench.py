#!/usr/bin/env python
"""Standalone ttfa-v3 runner — the exact modal_ttfa_bench.py session
loop with Modal removed, for rebuilding the TTFA measurement on your
own GPU box (RunPod / local). Protocol, timestamps, pacing, and record
schema are identical to interactive_paper/modal_ttfa_bench.py (see its
docstring for the full definition); results verify with
verify_rows.py / summarize_ttfa.py unchanged.

Needs:
  - GPU with bf16 + ~24 GB free (H100/A100 for timing parity with the
    paper runs; TTFA on a slower card measures THAT card, not the
    deployed stack — record the GPU name, it lands in every row).
  - model dir = the frozen MiniCPM-o-4_5 snapshot (verify config.json
    sha against freeze.json).
  - data root with queries*.jsonl + audio_pool/ bench_audio/
    sdqa_audio/ + gate_native.json + gate_act.json
    (fetch_inputs.sh pulls all of it from the gate-data volume).
  - repo src/ on disk (escalate.py, relay_fmt.py) — auto-found from
    this file's location.
  - OPENAI_API_KEY env for any arm that can escalate (conservative/
    balanced/aggressive/always). The local arm runs without it.

Usage:
  python local_ttfa_bench.py --model-dir /workspace/MiniCPM-o-4_5 \
      --data-root ./data_bundle --out-root ./results_local \
      --pool frozen --arm local --run-id ttfa-local1 [--limit 3]
"""
import argparse
import glob
import hashlib
import json
import os
import socket
import sys
import threading
import time
import zlib

LAYER = 22
K3 = 8
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
STALL = "Hmm, let me double-check that — one moment."
STALL_NOTE = ("[SYSTEM NOTE] Your answer so far is likely wrong. You "
              "just told the user: \"" + STALL + "\" A verified answer "
              "will arrive in a moment.")


def pools(data):
    return {
        "frozen":    (f"{data}/queries.jsonl",           f"{data}/audio_pool", "en"),
        "striviaqa": (f"{data}/queries_striviaqa.jsonl", f"{data}/bench_audio", "en"),
        "swebq":     (f"{data}/queries_swebq.jsonl",     f"{data}/bench_audio", "en"),
        "sllama":    (f"{data}/queries_sllama.jsonl",    f"{data}/bench_audio", "en"),
        "sdqa":      (f"{data}/queries_sdqa.jsonl",      f"{data}/sdqa_audio",  "en"),
    }


def _load_queries(data, pool):
    qfile, _, _ = pools(data)[pool]
    qs = [json.loads(x) for x in open(qfile, encoding="utf-8") if x.strip()]
    if pool == "frozen":
        qs = [q for q in qs if q.get("split") == "test"]
    return qs


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="./results_local")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARM2TIER))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--attempt", type=int, default=1)
    args = ap.parse_args()

    data = args.data_root.rstrip("/\\")
    model_dir = args.model_dir.rstrip("/\\")
    prompt_wav = f"{model_dir}/assets/system_ref_audio.wav"
    art_p = f"{data}/gate_native.json"
    act_p = f"{data}/gate_act.json"
    base = f"{args.out_root.rstrip('/')}/{args.run_id}"
    pool, arm, attempt = args.pool, args.arm, args.attempt
    tier = ARM2TIER[arm]
    shard_id = args.shard_id
    run_id = args.run_id
    mono = time.monotonic

    # resume: skip ids already recorded in ANY shard of this pool/arm
    done = set()
    for p in glob.glob(f"{base}/{pool}/{arm}.jsonl.shard*"):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                done.add(json.loads(ln)["id"])
    shard = [q for q in _load_queries(data, pool) if q["id"] not in done]
    if args.ids:
        want = set(args.ids.split(","))
        shard = [q for q in shard if q["id"] in want]
    if args.limit:
        shard = shard[:args.limit]
    if not shard:
        print("nothing to do (all ids recorded)")
        return
    print(f">>> local ttfa bench [{pool}/{arm}/{run_id}]: "
          f"{len(shard)} queries", flush=True)

    if tier != "never" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required for escalating arms")

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoTokenizer
    src_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src")
    sys.path.insert(0, src_dir)
    import escalate
    import relay_fmt

    _, audio_dir, lang = pools(data)[pool]
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=True,
    ).eval().cuda()
    _ = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    duplex = model.as_duplex()          # deployed demo config, audio ON
    duplex.force_listen_count = FORCE_LISTEN
    ref, _sr = librosa.load(prompt_wav, sr=SR, mono=True)

    art = json.load(open(art_p))
    art_sha = _sha(art_p)
    act_sha = _sha(act_p) if os.path.exists(act_p) else None
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    thr_tab = (art.get("eot_thresholds_lang", {}).get(lang)
               or art.get("eot_thresholds", {}))
    thr = {"never": 1e9, "always": -1e9}.get(tier, thr_tab.get(tier, 1e9))
    act = json.load(open(act_p)) if os.path.exists(act_p) else None
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
        return duplex.streaming_generate(prompt_wav_path=prompt_wav,
                                         top_k=GEN_TOP_K)

    import inspect as _insp

    def _call_def(fn, **kw):
        ps = set(_insp.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in ps})

    clean_expert = relay_fmt.clean_expert_v2

    tts_ready = {"ok": False}
    tok_tf = None
    try:
        model.init_token2wav_cache(ref)
        tok_tf = AutoTokenizer.from_pretrained(model_dir,
                                               trust_remote_code=True)
        tts_ready["ok"] = True
    except Exception as e:
        print(f">>> token2wav init failed: {e}", flush=True)

    def synth_stream(text):
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
                   ref_audio=ref, prompt_wav_path=prompt_wav)
    for _ in range(6):
        duplex.streaming_prefill(audio_waveform=sil())
        gen()
    if tts_ready["ok"]:
        synth_stream("Warm up complete.")
    print(f">>> warmup {round(time.time() - t_wu, 1)}s on {gpu_name} "
          f"({host})", flush=True)

    for qi, q in enumerate(shard):
        wav_p = q.get("audio") or f"{audio_dir}/{q['id']}.wav"
        os.makedirs(f"{base}/{pool}", exist_ok=True)
        outp = f"{base}/{pool}/{arm}.jsonl.shard{shard_id}"
        if not os.path.exists(wav_p):
            with open(outp, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"id": q["id"], "pool": pool, "arm": arm,
                     "run_id": run_id, "attempt": attempt,
                     "status": "missing_input", "input_path": wav_p},
                    ensure_ascii=False) + "\n")
            continue
        au, _sr = librosa.load(wav_p, sr=SR, mono=True)
        audio_s = len(au) / SR
        n_real = (len(au) + SR - 1) // SR
        input_sha = hashlib.sha256(au.tobytes()).hexdigest()

        def chunk_at(i):
            c = au[i * SR:(i + 1) * SR]
            return (np.pad(c, (0, SR - len(c))) if len(c) < SR else c)

        duplex.prepare(prefix_system_prompt=SYS_PROMPT,
                       ref_audio=ref, prompt_wav_path=prompt_wav)
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
                up_p = f"/tmp/up{shard_id}.wav" if os.name != "nt" \
                    else os.path.join(os.environ.get("TEMP", "."),
                                      f"up{shard_id}.wav")
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
        phase = "answer"
        n_ans = 0
        prev_listen = True
        go_wait = False
        ci = -1
        try:
            while ci < MAX_CHUNKS - 1:
                ci += 1
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
                        fire = False
                    rec["fired"] = bool(fire)
                    if rec["fired"]:
                        rec["mode"] = "escalated"
                        ts["fire"] = mono() - ts0
                        if segments["answer"]:
                            segments["candidate"] = segments["answer"]
                            segments["answer"] = []
                            ts["candidate_first_pcm"] = ts["local_first_pcm"]
                            ts["local_first_pcm"] = None
                        phase = "stall"
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
                                a3 = np.asarray(
                                    wf2, dtype=np.float32).reshape(-1)
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
        if rec["fired"] and rec["error"] is None:
            go_wait = True

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
        with open(outp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{qi}] {q['id']} arm={arm} score={rec['score']} "
              f"fired={rec['fired']} status={rec['status']} "
              f"ttfa={rec['ttfa_answer_s']} any={rec['ttfa_any_s']} "
              f"lagmax={rec['feed_lag_max_ms']}ms", flush=True)

    hh.remove()


if __name__ == "__main__":
    main()
