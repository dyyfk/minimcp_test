"""Native full-duplex LIVE benchmark on every eval pool (8bu): the
figure-3..14 protocol re-run on the deployed stack, so the gallery /
paper curves come from full-duplex native sessions, not the retired
harness loop.

Per query one fresh MiniCPMODuplex session under the OFFICIAL serving
config (top_k=20, force_listen_count=3, "You are a friendly assistant."),
the deployed 8bq gate read at the head's listen->speak commit
(per-language thresholds + the 8bh dialogue-act gate), real
gpt-transcribe uplink of the raw question audio, real gpt-5.5 (web,
low effort), wait paced 1 chunk / wall-second, real relay through the
talker. Delivered-channel outcome = relay text on fired turns, the
local answer otherwise. Latency components logged per turn in the
field names figures/bench_figures.py already consumes.

Arms: never / conservative / balanced / aggressive / always (same as
the retired conclive sweep). `never` and `always` are the two live
branches every query can take; the mid tiers are separate live runs.

Outputs: /data/native_bench/{pool}/{tier}.jsonl.shard{i} (restartable
by id), /data/native_bench/{pool}_{tier}_judged.parquet after `judge`.
Judges: OpenAudioBench's own gpt-4o judge (striviaqa/swebq/sllama),
VoiceBench gpt-4o-mini 1-5 (valpaca), our ref-anchored gpt-5.4-mini
judge (frozen/sdqa/sreason) — the same assignment as the old figures.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_native_bench.py::run_bench --pool striviaqa --tier always --limit 3
  modal run modal_native_bench.py::run_bench --pool striviaqa --tier never --workers 6
  modal run modal_native_bench.py::judge --pool striviaqa --tier never
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
OUT_DIR = f"{DATA}/native_bench"
SHADOW = {"P9": f"{DATA}/shadow/gate_shadow_distilled_semantic_rtj.json",
          "P16": f"{DATA}/shadow/gate_shadow_robust_ensemble.json"}
BLOCK = {"eot_last": 0, "eot_mean8": 1, "eot_mean": 1, "user_mean": 2}
MAX_WAIT_S = 150
MAX_ANS = 60
SYS_PROMPT = "You are a friendly assistant."       # official serving cfg
GEN_TOP_K = 20
FORCE_LISTEN = 3
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}

POOLS = {   # pool -> (queries file, audio dir, lang, judge)
    "frozen":    (f"{DATA}/queries.jsonl",           f"{DATA}/audio_pool", "en", "ours"),
    "striviaqa": (f"{DATA}/queries_striviaqa.jsonl", f"{DATA}/bench_audio", "en", "oab"),
    "swebq":     (f"{DATA}/queries_swebq.jsonl",     f"{DATA}/bench_audio", "en", "oab"),
    "sllama":    (f"{DATA}/queries_sllama.jsonl",    f"{DATA}/bench_audio", "en", "oab"),
    "sdqa":      (f"{DATA}/queries_sdqa.jsonl",      f"{DATA}/sdqa_audio",  "en", "ours"),
    "sreason":   (f"{DATA}/queries_sreason.jsonl",   f"{DATA}/bench_audio", "zh", "ours"),
    "valpaca":   (f"{DATA}/queries_valpaca.jsonl",   f"{DATA}/bench_audio", "en", "vb"),
}

RELAY_TMPL = ("A verified answer came back: {ans}\n"
              "Relay it to the user in one or two spoken sentences.")
RELAY_NUDGE = "Say the verified answer aloud to the user now."
STALL = "Hmm, let me double-check that — one moment."
STALL_NOTE = ("[SYSTEM NOTE] Your answer so far is likely wrong. You "
              "just told the user: \"" + STALL + "\" A verified answer "
              "will arrive in a moment.")

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")
_BENCH_PY = os.path.join(_HERE, "modal_bench.py")

app = modal.App("native-bench")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")

util_img = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("pandas", "pyarrow")
            .add_local_file(_APP_PY, "/root/modal_app.py"))
judge_img = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("pandas", "pyarrow", "openai", "pydantic>=2.11")
             .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
             .add_local_file(_APP_PY, "/root/modal_app.py")
             .add_local_file(_BENCH_PY, "/root/modal_bench.py"))
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
    qfile, _, _, _ = POOLS[pool]
    qs = [json.loads(x) for x in open(qfile, encoding="utf-8") if x.strip()]
    if pool == "frozen":
        qs = [q for q in qs if q.get("split") == "test"]
    return qs


@app.function(image=gpu_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60 * 5)
def live_shard(shard: list, pool: str, tier: str, shard_id: int = -1,
               relay: str = "steer") -> list:
    import glob as _glob
    import shutil
    import sys
    import threading

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, "/workspace/gate")
    import escalate

    _, audio_dir, lang, _ = POOLS[pool]
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
    duplex = model.as_duplex(generate_audio=False)
    duplex.force_listen_count = FORCE_LISTEN
    ref, _sr = librosa.load(PROMPT_WAV, sr=16000, mono=True)

    art = json.load(open(ART))
    w = np.array(art["w"], dtype=np.float32)
    b = float(art["b"])
    thr_tab = art.get("eot_thresholds_lang", {}).get(lang, art["eot_thresholds"])
    thr = {"never": 1e9, "always": -1e9}.get(tier, thr_tab.get(tier, 1e9))
    act = json.load(open(ACT)) if os.path.exists(ACT) else None
    aw = np.array(act["w"], dtype=np.float32) if act else None
    # shadow candidates (issue #8): scored at the same read, logged only —
    # they never touch the fire decision
    shadow = {}
    for name, path in SHADOW.items():
        if os.path.exists(path):
            a = json.load(open(path))
            sw = np.array(a["w"], dtype=np.float32)
            blocks = (a.get("feature_recipe", {}).get("blocks") or a.get("modes"))
            idx = (np.arange(len(sw)) if len(sw) == 12288 else
                   np.concatenate([np.arange(4096) + 4096 * BLOCK[n] for n in blocks]))
            shadow[name] = (idx, sw, float(a["b"]))
    feats = {}

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
        return rng.normal(0, 0.003, 16000).astype(np.float32)

    def gen():
        return duplex.streaming_generate(top_k=GEN_TOP_K)

    import inspect as _insp
    import re as _re

    def _call_def(fn, **kw):
        ps = set(_insp.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in ps})

    def clean_expert(txt, max_chars=400):
        """Expert markdown -> one spoken paragraph: strip emphasis/links/
        tables, flatten bullets into a comma list, keep whole sentences
        (abbreviation-aware) up to max_chars."""
        t = str(txt)
        t = _re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)          # [text](url)
        t = _re.sub(r"\(\s*https?://[^)]*\)", "", t)              # bare (url)
        t = _re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=_re.M)         # table rows
        t = _re.sub(r"^\s*#{1,6}\s*", "", t, flags=_re.M)            # headings
        t = _re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", ", ", t, flags=_re.M)   # bullets
        t = _re.sub(r"[*_`>]+", "", t)
        t = _re.sub(r"\s*\n+\s*", " ", t)
        t = _re.sub(r"\s*,\s*,+", ", ", t)
        t = _re.sub(r":\s*,\s*", ": ", t)
        t = _re.sub(r"\s+", " ", t).strip(" ,")
        sents = _re.split(r"(?<!\b[A-Z])(?<!\b[A-Z][a-z])(?<!\bU\.S)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bSt)(?<!\bNo)(?<=[.!?])\s+(?=[A-Z0-9\u4e00-\u9fff])", t)
        out = ""
        for se in sents:
            if out and len(out) + 1 + len(se) > max_chars:
                break
            out = (out + " " + se).strip()
        if len(out) > max_chars + 80:
            out = out[:max_chars].rsplit(" ", 1)[0] + "."
        return out or t[:max_chars]

    tts_ready = {"ok": False}
    tok_tf = None
    if relay == "tts":
        try:
            model.init_token2wav_cache(ref)
            tok_tf = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
            tts_ready["ok"] = True
        except Exception as e:
            print(f">>> token2wav init failed: {e}", flush=True)

    def synth(text):
        """Talker's own voice, verbatim text, via the turn-based
        teacher-forcing path (the demo's canned-stall call). Returns
        (seconds of audio, synth wall ms)."""
        t0 = time.time()
        model.reset_session(reset_token2wav_cache=False)
        sys_msg = _call_def(model.get_sys_prompt, mode="omni", language="en")
        _call_def(model.streaming_prefill, session_id="s1", msgs=[sys_msg], tokenizer=tok_tf)
        _call_def(model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user", "content": [np.zeros(16000, dtype="float32")]}],
                  tokenizer=tok_tf, is_last_chunk=True)
        res = _call_def(model.streaming_generate, tokenizer=tok_tf, temperature=0.1,
                        generate_audio=True, use_tts_template=True, teacher_forcing=True,
                        teacher_forcing_text=text, max_new_tokens=256, session_id="s1")
        n = 0
        for item in res:
            wf = item[0] if isinstance(item, tuple) else None
            if wf is not None:
                n += int(wf.reshape(-1).shape[0])
        return n / 24000.0, int((time.time() - t0) * 1000)

    results = []
    for qi, q in enumerate(shard):
        wav_p = f"{audio_dir}/{q['id']}.wav"
        if not os.path.exists(wav_p):
            continue
        au, _sr = librosa.load(wav_p, sr=16000, mono=True)
        chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
        chunks = [np.pad(c, (0, 16000 - len(c)))
                  if len(c) < 16000 else c for c in chunks]

        duplex.prepare(prefix_system_prompt=SYS_PROMPT,
                       ref_audio=ref, prompt_wav_path=None)
        st3.update(tail=None, sum=None, cnt=0, accum=False)

        rec = {"id": q["id"], "pool": pool, "tier": tier, "lang": lang,
               "query": q.get("query"),
               "reference_answer": q.get("reference_answer"),
               "audio_s": round(len(au) / 16000, 2),
               "n_chunks": len(chunks), "score": None, "act": None,
               "is_info": None, "fired": False, "mode": "local",
               "answer": "", "relay": "", "expert_answer": "",
               "transcript": "", "asr_s": None, "expert_latency_s": None,
               "eot_read_ms": None, "stall_ms": None, "relay_ms": None,
               "answer_ms": None, "wait_chunks": 0, "relay_nudged": False,
               "onset_chunk": None, "eot_seen": False}
        exp = {}
        exp_done = threading.Event()

        def expert_call(snapshot):
            try:
                t0 = time.time()
                sf.write(f"/tmp/up{shard_id}.wav", snapshot, 16000)
                with open(f"/tmp/up{shard_id}.wav", "rb") as fh:
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
                exp_done.set()

        texts, n_ans = [], 0
        prev_listen = True
        feed = list(chunks)
        waiting, relay_started = False, False
        t_wait0 = t_onset = t_relay0 = None
        ci = -1
        try:
            while ci < 400:
                ci += 1
                if waiting and not relay_started:
                    time.sleep(max(0.0, 1.0 - 0.15))
                    if exp_done.is_set() or \
                            time.time() - t_wait0 > MAX_WAIT_S:
                        relay_started = True
                        t_relay0 = time.time()
                        if relay == "tts":
                            # verbatim expert text in the talker's own
                            # voice; the duplex context only gets a note
                            spoken = clean_expert(exp.get("answer", ""))
                            rec["relay_text"] = spoken
                            try:
                                if tts_ready["ok"]:
                                    rec["relay_audio_s"], rec["relay_synth_ms"] = synth(spoken)
                            except Exception as e:
                                rec["relay_synth_err"] = str(e)[:120]
                            duplex.streaming_prefill(text_list=[
                                "[SYSTEM NOTE] You just told the user: "
                                + spoken + " Do not repeat it."])
                            r = gen()
                            rec["eot_seen"] = bool(r.get("end_of_turn"))
                            break
                        duplex.streaming_prefill(text_list=[
                            RELAY_TMPL.format(
                                ans=exp.get("answer", "[no answer]"))])
                        r = gen()
                        if not r.get("text"):
                            rec["relay_nudged"] = True
                            duplex.streaming_prefill(
                                text_list=[RELAY_NUDGE])
                            r = gen()
                        if r.get("text"):
                            texts.append(r["text"])
                        prev_listen = r["is_listen"]
                        if r.get("end_of_turn"):
                            rec["eot_seen"] = True
                            break
                        continue
                    rec["wait_chunks"] += 1
                ch = feed.pop(0) if feed else sil()
                st3["accum"] = True
                ok = duplex.streaming_prefill(audio_waveform=ch)
                st3["accum"] = False
                if not ok.get("success"):
                    continue
                r = gen()
                if r.get("text"):
                    texts.append(r["text"])

                if prev_listen and not r["is_listen"] \
                        and rec["onset_chunk"] is None:
                    rec["onset_chunk"] = ci
                    t_onset = time.time()
                    v = feat_now()
                    sc = float(1.0 / (1.0 + np.exp(-(float(v @ w) + b))))
                    rec["score"] = round(sc, 4)
                    if aw is not None:
                        a = float(1.0 / (1.0 + np.exp(-(float(v @ aw) + act["b"]))))
                        rec["act"] = round(a, 4)
                        rec["is_info"] = bool(a >= act["act_threshold"])
                    for name, (idx, sw, sb) in shadow.items():
                        rec[f"shadow_{name}"] = round(float(v[idx] @ sw + sb), 4)
                    feats[q["id"]] = v.astype(np.float16)
                    rec["eot_read_ms"] = int((time.time() - t_onset) * 1000)
                    fire = sc >= thr
                    if tier in RATES and rec["is_info"] is False:
                        fire = False            # 8bh: floor turns never escalate
                    rec["fired"] = bool(fire)
                    if rec["fired"]:
                        rec["mode"] = "escalated"
                        snap = au[-30 * 16000:]
                        threading.Thread(target=expert_call,
                                         args=(snap,), daemon=True).start()
                        t0n = time.time()
                        if not r.get("end_of_turn"):
                            duplex.streaming_prefill(text_list=[STALL_NOTE])
                            r2 = gen()
                            if r2.get("text"):
                                texts.append(r2["text"])
                            rec["stall_ms"] = int((time.time() - t0n) * 1000)
                            prev_listen = r2["is_listen"]
                            if r2.get("end_of_turn"):
                                waiting = True
                                t_wait0 = time.time()
                            continue
                        waiting = True
                        t_wait0 = time.time()
                        continue

                if not r["is_listen"]:
                    n_ans += 1
                if r.get("end_of_turn"):
                    if rec["fired"] and not relay_started:
                        waiting = True
                        t_wait0 = time.time()
                        prev_listen = True
                        continue
                    rec["eot_seen"] = True
                    break
                if not rec["fired"] and n_ans >= MAX_ANS:
                    break
                if relay_started and n_ans >= MAX_ANS:
                    break
                prev_listen = r["is_listen"]
        except Exception as e:
            rec["answer"] = f"[error: {str(e)[:120]}]"

        full = "".join(texts).strip()
        if rec["fired"]:
            exp_done.wait(timeout=5)
            rec["relay"] = (rec.get("relay_text", "") if relay == "tts" else full)
            rec["expert_answer"] = exp.get("answer", "")
            rec["transcript"] = exp.get("uplink", "")[:300]
            rec["asr_s"] = exp.get("asr_s")
            rec["expert_latency_s"] = (
                None if exp.get("expert_s") is None
                else round((exp.get("asr_s") or 0) + exp["expert_s"], 2))
            if t_relay0 is not None:
                rec["relay_ms"] = int((time.time() - t_relay0) * 1000)
        else:
            rec["answer"] = full
            if t_onset is not None:
                rec["answer_ms"] = int((time.time() - t_onset) * 1000)
        results.append(rec)
        print(f"  [{qi}] {q['id']} score={rec['score']} act={rec['act']} "
              f"fired={rec['fired']} wait={rec['wait_chunks']} "
              f"txt={full[:60]!r}", flush=True)

    hh.remove()
    os.makedirs(f"{OUT_DIR}/{pool}", exist_ok=True)
    sfx = "smoke" if shard_id < 0 else f"shard{shard_id}"
    tier_out = tier if relay == "steer" else f"{tier}_{relay}"
    with open(f"{OUT_DIR}/{pool}/{tier_out}.jsonl.{sfx}", "a",
              encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if feats:
        fp = f"{OUT_DIR}/{pool}/{tier_out}_feats.{sfx}.npz"
        old = (dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else None)
        ids = list(feats.keys()); X = np.stack([feats[i] for i in ids])
        if old is not None:
            ids = list(old["ids"]) + ids; X = np.concatenate([old["X"], X])
        np.savez_compressed(fp, ids=np.array(ids), X=X)
    gate_data.commit()
    return results


@app.function(image=util_img, volumes={DATA: gate_data}, timeout=60 * 5)
def _todo(pool: str, tier: str, relay: str = "steer") -> list:
    import glob as _glob
    qs = _load_queries(pool)
    done = set()
    tier_out = tier if relay == "steer" else f"{tier}_{relay}"
    for p in _glob.glob(f"{OUT_DIR}/{pool}/{tier_out}.jsonl.shard*"):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                done.add(json.loads(ln)["id"])
    return [q for q in qs if q["id"] not in done]


@app.function(image=judge_img, volumes={DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 90)
def judge_pool(pool: str, tier: str, field: str = "delivered",
               relay: str = "steer") -> int:
    """Score one (pool, tier) run with that pool's judge ->
    /data/native_bench/{pool}_{tier}_judged.parquet (all trace fields +
    oab_ok / adequate / vb_score). field="delivered" scores what the
    user heard (relay on fired turns, local answer otherwise);
    field="expert" scores the expert's own text on fired turns (the
    relay-channel counterfactual) into *_expert columns of the same
    parquet."""
    import asyncio
    import glob as _glob
    import re as _re
    import sys

    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    sys.path.insert(0, "/root")
    import escalate

    _, _, _, jkind = POOLS[pool]
    rows = {}
    tier_out = tier if relay == "steer" else f"{tier}_{relay}"
    for p in sorted(_glob.glob(f"{OUT_DIR}/{pool}/{tier_out}.jsonl.shard*")):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                rows[r["id"]] = r
    out_p = f"{OUT_DIR}/{pool}_{tier_out}_judged.parquet"
    old = (pd.read_parquet(out_p) if os.path.exists(out_p)
           else pd.DataFrame(columns=["id"]))
    sfx = "" if field == "delivered" else "_expert"
    base_col = {"ours": "adequate", "oab": "oab_ok", "vb": "vb_score"}[jkind]
    col = base_col + sfx
    # rows without a verdict (judge retries exhausted) are re-judged
    have = (set(old.loc[old[col].notna(), "id"]) if col in old.columns else set())
    todo = []
    for r in rows.values():
        if r["id"] in have:
            continue
        r = dict(r)
        if field == "delivered":
            r["delivered"] = (r["relay"] if r["fired"] else r["answer"]) or ""
        else:
            if not r["fired"]:
                continue
            r["delivered"] = r.get("expert_answer") or ""
        todo.append(r)
    print(f">>> judge[{pool}/{tier}/{jkind}]: {len(todo)} to judge", flush=True)
    if not todo:
        return 0

    def _bench_ns():
        """Pull the official judge prompts/models (+ _oab_judge) out of
        modal_bench.py without importing its Modal app definitions."""
        import ast
        src = open("/root/modal_bench.py", encoding="utf-8").read()
        want = {"OAB_JUDGE_MODEL", "OAB_PATTERN", "VB_JUDGE_MODEL",
                "VB_META_PROMPT_OPEN", "_oab_judge"}
        keep = [n for n in ast.parse(src).body
                if (isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) in want for t in n.targets))
                or (isinstance(n, ast.FunctionDef) and n.name in want)]
        ns = {"sys": sys}
        exec(compile(ast.Module(body=keep, type_ignores=[]), "modal_bench", "exec"), ns)
        return ns

    if jkind == "ours":
        jin = [{"query": r["query"], "reference_answer": r["reference_answer"],
                "answer": r["delivered"]} for r in todo]
        jr = asyncio.run(escalate.judge_many(jin, concurrency=8))
        for r, j in zip(todo, jr):
            r["adequate" + sfx] = j.get("adequate")
            r["judge_reason" + sfx] = j.get("judge_reason")
    elif jkind == "oab":
        mb = _bench_ns()
        jin = [{"query": r["query"], "reference_answer": r["reference_answer"],
                "answer": r["delivered"]} for r in todo]
        jr = mb["_oab_judge"](jin, concurrency=3)
        for r, j in zip(todo, jr):
            r["oab_ok" + sfx] = j.get("oab_ok")
    else:   # VoiceBench 1-5 open-ended judge, official prompt
        mb = _bench_ns()
        client = escalate._async_client()
        sem = asyncio.Semaphore(3)

        async def one(r):
            prompt = (mb["VB_META_PROMPT_OPEN"]
                      .replace("{prompt}", str(r["query"]))
                      .replace("{response}", str(r["delivered"])))
            r["vb_score" + sfx] = None
            for _ in range(6):
                async with sem:
                    try:
                        resp = await client.chat.completions.create(
                            model=mb["VB_JUDGE_MODEL"], max_tokens=1024,
                            frequency_penalty=0, presence_penalty=0,
                            messages=[
                                {"role": "system", "content":
                                 "You are a helpful assistant who tries to "
                                 "help answer the user's question."},
                                {"role": "user", "content": prompt}],
                            user=escalate.USER_ID)
                        txt = (resp.choices[0].message.content or "").strip()
                        m = _re.search(r"\d+", txt)
                        if m:
                            r["vb_score" + sfx] = int(m.group(0))
                            return
                    except Exception:
                        pass
                await asyncio.sleep(3)

        async def run():
            await asyncio.gather(*(one(r) for r in todo))
        asyncio.run(run())
    if field == "delivered" or not len(old):
        keep = old[~old["id"].isin({r["id"] for r in todo})] if len(old) else old
        new = pd.concat([keep, pd.DataFrame(todo)], ignore_index=True)
    else:   # merge the *_expert columns onto the existing rows by id
        add = pd.DataFrame(todo).set_index("id")[[c for c in
                 (col, "judge_reason" + sfx) if c in pd.DataFrame(todo).columns]]
        new = old.set_index("id")
        for c in add.columns:
            col_new = add[c].reindex(new.index)
            new[c] = (col_new.combine_first(new[c]) if c in new.columns
                      else col_new)
        new = new.reset_index()
    new.to_parquet(out_p)
    gate_data.commit()
    ok = new[new[col].notna()]
    print(f">>> {pool}/{tier}: n={len(ok)} fire={ok['fired'].mean():.2f} "
          f"{col}={ok[col].astype(float).mean():.3f}", flush=True)
    return len(todo)


@app.local_entrypoint()
def run_bench(pool: str = "striviaqa", tier: str = "never",
              workers: int = 6, limit: int = 0, relay: str = "steer"):
    qs = _todo.remote(pool, tier, relay)
    smoke = bool(limit) and limit <= 8
    if limit:
        qs = qs[:limit]
        if smoke:
            workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> native bench [{pool}/{tier}/{relay}]: {len(qs)} queries, "
          f"{workers} workers{' (smoke)' if smoke else ''}", flush=True)
    done = list(live_shard.starmap(
        [(shards[i], pool, tier, i if not smoke else -1, relay)
         for i in range(workers) if shards[i]]))
    print(f">>> complete: {sum(len(d) for d in done)}")


@app.local_entrypoint()
def judge(pool: str = "striviaqa", tier: str = "never", field: str = "delivered",
          relay: str = "steer"):
    n = judge_pool.remote(pool, tier, field, relay)
    print(f">>> judged {n}")
