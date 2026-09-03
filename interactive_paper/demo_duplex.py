"""Native full-duplex demo — the harness barge-in is GONE.

demo_app.py's live loop wrapped MiniCPM's TURN-BASED streaming API in a
server-side energy VAD + burst-ASR classifier + abort Event ("soft
barge-in"). This app replaces all of that with the model's own duplex
head (MiniCPMODuplex, `model.as_duplex()`):

  mic 16 kHz PCM --continuous--> streaming_prefill (1 s units)
                                 streaming_generate
        <-- per chunk: <|listen|> (stay silent) or spoken audio

Every second of mic audio is prefilled into the SAME context the model
is generating from, so the model hears the user while it speaks and
yields the floor itself (<|turn_eos|>). Barge-in vs backchannel is the
duplex head's decision — no VAD, no burst ASR, no abort Event, no
client-side duck/kill-switch. The only echo protection is browser AEC
(use headphones for a clean run).

The gate lives at the listen->speak transition: the chunk where the
talker first decides to answer is its "starts thinking" moment. The
L22 probe (concurrent-regime weights, gate_conc_frozen.json — closest
calibrated regime; native-duplex token schema is NOT yet calibrated,
scores are exploratory) reads the context right after that chunk's
audio prefill. P(fail) >= tier threshold => the thinker (gpt-5.5, web
search) runs in the background WHILE the duplex loop keeps rolling.
When the thinker returns, its answer is prefilled as a TEXT unit into
the same duplex stream and the talker voices it; the mic never stops
flowing, so the user can interrupt the relay exactly like any other
speech — same native mechanism, zero special-casing.

Deliberately NOT implemented yet (recorded in project memory): aborting
the in-flight thinker when the user speaks during the wait.

Deploy:  modal deploy demo_duplex.py
Page:    https://rhe9527--gate-duplex.modal.run/62dc5cd9/
Voice:   wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws
"""
import json
import os
import time

import modal

TOKEN = "62dc5cd9"

app = modal.App("gate-demo-duplex")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")
DATA = "/data"
TRACK_WARMUP = 20   # onset scores before the windowed quantile takes over
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"
# 8bl: serve with the OFFICIAL duplex config. The A/B control demo
# showed post-stop yield 2.0s (official: top_k=20, force_listen=3,
# assistant prompt) vs 1-13s high-variance under as_duplex defaults
# (top_k=100) — our old config was masking the head's native
# mid-answer turn_eos responsiveness.
GEN_TOP_K = 20
FORCE_LISTEN = 3
SYS_PROMPT = "You are a friendly assistant."
LAYER = 22
TIERS = ("conservative", "balanced", "aggressive")

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")
_SHADOW_ARTIFACT = os.path.join(
    _HERE, "data", "shadow", "gate_shadow_robust_ensemble.json")

web_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]")
             .add_local_file(_APP_PY, "/root/modal_app.py"))
# Same PROVEN spec as demo_app.py (torch 2.8 / transformers 4.51.0 pin),
# plus the local _model_src duplex sources: the volume checkpoint's
# modeling files may predate MiniCPMODuplex, so we overwrite the HF
# modules cache with the shipped copies at @enter.
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
        "fastapi[standard]",
    )
    .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
    .add_local_dir(os.path.join(_HERE, "_model_src"), "/workspace/model_src")
    .add_local_file(
        _SHADOW_ARTIFACT,
        "/workspace/gate_shadow_robust_ensemble.json")
    .add_local_file(_APP_PY, "/root/modal_app.py"))

# proven wording (first escalate smoke: relayed + self-corrected);
# free-text imperatives like "stop speaking and wait" made the head
# swallow the relay, and "say X now" got followed one unit late — the
# canned-stall + factual context note below avoids steering entirely.
RELAY_TMPL = ("A verified answer came back: {ans}\n"
              "Relay it to the user in one or two spoken sentences.")
RELAY_NUDGE = "Say the verified answer aloud to the user now."
# 8bu relay mode. "steer": prefill RELAY_TMPL and let the talker voice the
# answer itself (loses ~20-27 pts of correct expert answers: truncation,
# self-answering, 99% nudges). "tts": speak the cleaned expert text
# verbatim in the talker's own voice via the same teacher-forcing path
# that synthesizes the canned stall, then hand the context a note. The
# chunk loop keeps running, so the relay stays interruptible.
RELAY_MODE = os.environ.get("RELAY_MODE", "tts")
RELAY_NOTE = "[SYSTEM NOTE] You just told the user: \"{ans}\" Do not repeat it."


import re


def clean_expert(txt, max_chars=400):
    """Expert markdown -> one spoken paragraph: strip emphasis/links/
    tables, flatten bullets into a comma list, keep whole sentences
    (abbreviation-aware) up to max_chars."""
    t = str(txt)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)          # [text](url)
    t = re.sub(r"\(\s*https?://[^)]*\)", "", t)              # bare (url)
    t = re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=re.M)         # table rows
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)            # headings
    t = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", ", ", t, flags=re.M)   # bullets
    t = re.sub(r"[*_`>]+", "", t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    t = re.sub(r"\s*,\s*,+", ", ", t)
    t = re.sub(r":\s*,\s*", ": ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    sents = re.split(r"(?<!\b[A-Z])(?<!\b[A-Z][a-z])(?<!\bU\.S)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bSt)(?<!\bNo)(?<=[.!?])\s+(?=[A-Z0-9\u4e00-\u9fff])", t)
    out = ""
    for se in sents:
        if out and len(out) + 1 + len(se) > max_chars:
            break
        out = (out + " " + se).strip()
    if len(out) > max_chars + 80:
        out = out[:max_chars].rsplit(" ", 1)[0] + "."
    return out or t[:max_chars]


def _call_def(fn, /, **kw):
    import inspect
    p = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in kw.items() if k in p})


@app.cls(image=gpu_image, gpu="H100", max_containers=4,
         volumes={"/workspace/models": weights, DATA: gate_data},
         secrets=[OPENAI], timeout=60 * 60, scaledown_window=420)
@modal.concurrent(max_inputs=1)
class DuplexVoice:

    @modal.enter()
    def load(self):
        import glob as _glob
        import shutil
        import sys
        import threading

        import torch
        from transformers import AutoModel, AutoTokenizer
        sys.path.insert(0, "/workspace/gate")
        import gate as gate_mod

        t0 = time.time()
        cache = os.path.expanduser("~/.cache/huggingface/modules/"
                                   "transformers_modules/"
                                   + os.path.basename(MODEL_DIR))
        os.makedirs(cache, exist_ok=True)
        for f in _glob.glob(f"{MODEL_DIR}/*.py"):
            shutil.copy(f, cache)
        for f in _glob.glob("/workspace/model_src/*.py"):
            shutil.copy(f, cache)      # duplex-capable modeling sources win
        self.model = AutoModel.from_pretrained(
            MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False, init_audio=True,
            init_tts=True).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR,
                                                 trust_remote_code=True)
        # as_duplex() runs init_tts itself (default asset path) and owns
        # its token2wav stream cache via prepare(prompt_wav_path=...)
        self.duplex = self.model.as_duplex()

        # in-regime probe: 8be native-duplex refit (2310 rows, same
        # speak-onset read point as this app; scripts/22)
        self.art = json.load(open(f"{DATA}/gate_native.json"))
        # Candidate scoring is observational only. Its artifact explicitly
        # prohibits activation, and no threshold from it enters `fired`.
        self.shadow_art = json.load(open(
            "/workspace/gate_shadow_robust_ensemble.json"))
        if (self.shadow_art.get("status") != "shadow_only" or
                self.shadow_art.get("activation_prohibited") is not True):
            raise RuntimeError("candidate is not shadow-safe")
        self.shadow_probe = gate_mod.Probe(
            self.shadow_art["w"], self.shadow_art["b"])
        self.shadow_modes = self.shadow_art["feature_recipe"]["blocks"]
        self.score_wins = {}       # lang -> deque of recent onset scores
        # 8bh dialogue-act gate: stop words / backchannels hit the same
        # commit as questions and the failure probe is OOD on them —
        # escalate only when the SAME L22 read says "info-seeking"
        try:
            self.act = json.load(open(f"{DATA}/gate_act.json"))
        except Exception:
            self.act = None
        self.probe = gate_mod.Probe(self.art["w"], self.art["b"])
        self.K3 = self.art.get("k_eot", 8)
        shadow_tail = max([
            int(mode.removeprefix("eot_mean"))
            for mode in self.shadow_modes
            if mode.startswith("eot_mean") and mode != "eot_mean"] or [1])
        self.tail_k = max(self.K3, shadow_tail)
        self.st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

        import torch as _t

        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            h = hs[0].detach().float()
            t = h[-self.tail_k:].cpu()
            self.st3["tail"] = (t if self.st3["tail"] is None
                                else _t.cat([self.st3["tail"],
                                             t])[-self.tail_k:])
            if self.st3["accum"]:
                sm = h.sum(0).cpu()
                self.st3["sum"] = (sm if self.st3["sum"] is None
                                   else self.st3["sum"] + sm)
                self.st3["cnt"] += h.shape[0]
        self.model.llm.model.layers[LAYER].register_forward_hook(hook)
        self.lock = threading.Lock()
        # canned stall in the talker's own voice (teacher-forced via the
        # turn-based path; the duplex wrapper reuses the same TTS)
        self.stall_pcm = None
        self.tts_ok = False
        try:
            import librosa as _lb
            ref, _ = _lb.load(PROMPT_WAV, sr=16000, mono=True)
            self.model.init_token2wav_cache(ref)
            self.tts_ok = True
            self.stall_pcm = self._synth_pcm(STALL, max_new_tokens=64)
            if self.stall_pcm is not None:
                print(f">>> canned stall: "
                      f"{len(self.stall_pcm) / 24000:.2f}s", flush=True)
        except Exception as e:
            print(f">>> stall synth failed (no audio stall): {e}",
                  flush=True)
        self.load_s = round(time.time() - t0, 1)
        print(f">>> DuplexVoice ready in {self.load_s}s", flush=True)

    def _synth_pcm(self, text, max_new_tokens=256):
        """Talker's own voice, verbatim `text`, via the turn-based
        teacher-forcing path (24 kHz float32 pcm or None)."""
        import numpy as _np
        self.model.reset_session(reset_token2wav_cache=False)
        sys_msg = _call_def(self.model.get_sys_prompt, mode="omni",
                            language="en")
        _call_def(self.model.streaming_prefill, session_id="s1",
                  msgs=[sys_msg], tokenizer=self.tok)
        _call_def(self.model.streaming_prefill, session_id="s1",
                  msgs=[{"role": "user",
                         "content": [_np.zeros(16000, dtype="float32")]}],
                  tokenizer=self.tok, is_last_chunk=True)
        res = _call_def(self.model.streaming_generate,
                        tokenizer=self.tok, temperature=0.1,
                        generate_audio=True, use_tts_template=True,
                        teacher_forcing=True, teacher_forcing_text=text,
                        max_new_tokens=max_new_tokens, session_id="s1")
        parts = []
        for item in res:
            wf = item[0] if isinstance(item, tuple) else None
            if wf is not None:
                parts.append(wf.float().cpu().numpy().reshape(-1))
        return _np.concatenate(parts) if parts else None

    def _feat_now(self, artifact=None):
        import torch
        if self.st3["tail"] is None or self.st3["cnt"] == 0:
            return None
        artifact = self.art if artifact is None else artifact
        modes = artifact.get("modes", artifact.get("feature_recipe", {}).get(
            "blocks", []))
        parts = []
        for m in modes:
            if m == "eot_last":
                parts.append(self.st3["tail"][-1])
            elif m.startswith("eot_mean"):
                suffix = m.removeprefix("eot_mean")
                k = int(suffix) if suffix else artifact.get("k_eot", self.K3)
                parts.append(self.st3["tail"][-k:].mean(0))
            elif m == "user_mean":
                parts.append(self.st3["sum"] / max(1, self.st3["cnt"]))
            else:
                raise ValueError(f"unknown probe feature mode: {m}")
        return torch.cat(parts).numpy()

    def _score_now(self):
        v = self._feat_now()
        return None if v is None else float(self.probe.score(v))

    def _shadow_score_now(self):
        v = self._feat_now(self.shadow_art)
        return None if v is None else float(self.shadow_probe.score(v))

    def _act_now(self):
        """P(info-seeking) from the same read; None = act gate off."""
        import numpy as np
        if self.act is None:
            return None
        v = self._feat_now()
        if v is None:
            return None
        z = float(v @ np.array(self.act["w"], dtype=v.dtype)) \
            + self.act["b"]
        return float(1.0 / (1.0 + np.exp(-z)))

    def _session_reset(self):
        import librosa
        ref, _ = librosa.load(PROMPT_WAV, sr=16000, mono=True)
        self.duplex.force_listen_count = FORCE_LISTEN
        self.duplex.prepare(
            prefix_system_prompt=SYS_PROMPT,
            ref_audio=ref, prompt_wav_path=PROMPT_WAV)
        self.st3.update(tail=None, sum=None, cnt=0, accum=False)

    # ---- web -------------------------------------------------------------
    @modal.asgi_app(label="gate-duplex-voice")
    def ws_app(self):
        import asyncio
        import base64
        import threading as _th

        import numpy as np
        from fastapi import FastAPI, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse

        wapp = FastAPI()
        wapp.add_middleware(CORSMiddleware, allow_origins=["*"],
                            allow_methods=["*"], allow_headers=["*"])

        @wapp.get(f"/{TOKEN}/ready")
        def ready():
            return JSONResponse({"ready": True, "load_s": self.load_s,
                                 "busy": self.lock.locked(),
                                 "distilled_shadow": True})

        @wapp.websocket(f"/{TOKEN}/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            tier = sock.query_params.get("tier", "balanced")
            probe_on = sock.query_params.get("probe_on", "1") == "1"
            # per-language operating point (review item 3): zh scores
            # sit below the en calib distribution, so the global tier
            # quantile barely fires on zh. gate_native.json may carry
            # eot_thresholds_lang = {en: {...}, zh: {...}} quantiled on
            # the training OOF per language; fall back to global.
            lang = sock.query_params.get("lang", "en")
            thr = (self.art.get("eot_thresholds_lang", {})
                   .get(lang, self.art["eot_thresholds"])
                   .get(tier, 1e9))
            # 8bq: online windowed quantile tracker (8bn simulation,
            # WINDOW=100). The static per-language point depends on the
            # calibration slice's family mix matching the stream; the
            # tracker thresholds at the (1-rate) quantile of the last
            # 100 onset scores THIS process has seen for the language
            # (shared across sessions), no labels, no pool identity.
            # Static threshold until TRACK_WARMUP scores exist.
            tracker_on = sock.query_params.get("tracker", "1") == "1"
            tier_rate = {"conservative": .15, "balanced": .30,
                         "aggressive": .50}.get(tier)
            score_win = self.score_wins.setdefault(
                lang, __import__("collections").deque(maxlen=100))

            def effective_thr():
                if (tracker_on and tier_rate is not None
                        and len(score_win) >= TRACK_WARMUP):
                    return (float(np.quantile(np.array(score_win),
                                              1 - tier_rate)), "window")
                return (thr, "static")
            if not self.lock.acquire(timeout=3):
                await sock.send_json({"type": "error",
                                      "msg": "model busy — try again"})
                await sock.close()
                return
            loop = asyncio.get_event_loop()
            stop = _th.Event()
            inbox, ilock = [], _th.Lock()
            relay_box = []            # thinker -> chunk loop (one slot)

            def emit(m):
                asyncio.run_coroutine_threadsafe(sock.send_json(m), loop)

            def chunk_loop():
                """The whole session: one native duplex stream. No VAD,
                no abort — the model owns the floor."""
                import sys
                sys.path.insert(0, "/workspace/gate")
                import escalate
                import soundfile as sf
                import base64 as _b64

                CH = 16000                       # 1 s @ 16 kHz
                pend = np.zeros(0, dtype=np.float32)
                user_win = []                    # ASR uplink window
                history = []                      # rolling dialogue text
                turn_text = []                    # this turn's spoken text
                turn_fired = False                # did this turn escalate
                active_turn = None                # structured telemetry state
                relay_turn = None                 # expert relay being voiced
                turn_index = 0
                turn_scores = []
                relay_guard = False               # relay being delivered
                muted = 0                         # 8bm: suppressed chunks
                prev_listen = True
                thinking = _th.Event()           # thinker in flight
                n_chunk = 0
                completed_turns = 0               # pre-answer context state
                prior_escalations = 0             # completed/in-flight fires
                history_lock = _th.Lock()
                resolved_history = {}
                next_history_index = 1

                def transcribe_turn(state):
                    """ASR every committed user turn without blocking audio."""
                    path = f"/tmp/duplex_up_{state['index']}.wav"
                    t0 = time.time()
                    try:
                        sf.write(path, state["snapshot"], 16000)
                        with open(path, "rb") as fh:
                            tr = (escalate._client().audio.transcriptions
                                  .create(model="gpt-transcribe", file=fh,
                                          response_format="text"))
                        state["uplink_text"] = (
                            tr if isinstance(tr, str)
                            else getattr(tr, "text", str(tr)))
                        state["asr_s"] = round(time.time() - t0, 2)
                        emit({"type": "log",
                              "msg": f"turn {state['index']} ASR heard: "
                                     f"“{state['uplink_text'][:120]}” "
                                     f"({state['asr_s']:.1f}s)"})
                    except Exception as e:
                        state["asr_error"] = str(e)[:160]
                        emit({"type": "log",
                              "msg": f"turn {state['index']} ASR failed: "
                                     f"{state['asr_error']}"})
                    finally:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                        state["asr_done"].set()

                def add_history(state):
                    """Append completed turns in conversation order."""
                    nonlocal next_history_index
                    answer = (state.get("expert_answer")
                              if state["fired"] else state.get("answer"))
                    with history_lock:
                        resolved_history[state["index"]] = (
                            state.get("uplink_text"), answer)
                        while next_history_index in resolved_history:
                            user, assistant = resolved_history.pop(
                                next_history_index)
                            if user:
                                history.append(f"User: {str(user).strip()}")
                            if assistant:
                                history.append(
                                    f"Assistant: {str(assistant).strip()}")
                            del history[:-8]
                            next_history_index += 1

                def finish_turn(state):
                    """Emit one analysis-ready event after ASR and speech end."""
                    if not state["asr_done"].wait(timeout=90):
                        state["asr_error"] = "timeout"
                    state["answer"] = "".join(
                        state.get("assistant_parts", [])).strip()
                    add_history(state)
                    snapshot = state["snapshot"]
                    pcm16 = (np.clip(snapshot, -1, 1) * 32767).astype("<i2")
                    emit({
                        "type": "turn",
                        "protocol": "duplex_v1",
                        "turn_index": state["index"],
                        "mode": "escalated" if state["fired"] else "local",
                        "fired": state["fired"],
                        "probe_on": probe_on,
                        "eot_score": state.get("score"),
                        "threshold": state.get("threshold"),
                        "scores": state.get("scores", []),
                        "act_score": state.get("act_score"),
                        "is_info": state.get("is_info"),
                        "uplink_text": state.get("uplink_text"),
                        "asr_error": state.get("asr_error"),
                        "answer": state.get("answer", ""),
                        "expert_answer": state.get("expert_answer"),
                        "expert_error": state.get("expert_error"),
                        "asr_s": state.get("asr_s"),
                        "expert_latency_s": state.get("expert_latency_s"),
                        "relay_ms": state.get("relay_ms"),
                        "total_ms": int((time.time() - state["started_at"])
                                        * 1000),
                        "audio_s": round(len(snapshot) / 16000, 2),
                        # 30 seconds of PCM16 is ~960 KB, below Modal's
                        # 2 MiB WebSocket-message limit after base64 encoding.
                        "user_pcm16": _b64.b64encode(
                            pcm16.tobytes()).decode(),
                    })

                def finish_turn_async(state):
                    _th.Thread(target=finish_turn, args=(state,),
                               daemon=True).start()

                def thinker(state):
                    # context: the resolved dialogue so far. The probe
                    # reads L22 WITH this context (it is in the model's
                    # KV cache), but the expert is stateless — a
                    # follow-up like "what about apple" is unanswerable
                    # in isolation, so we uplink the history too and ask
                    # the expert to resolve references against it.
                    try:
                        transcribe_turn(state)
                        up = state.get("uplink_text")
                        if not up:
                            raise RuntimeError(
                                state.get("asr_error") or "empty ASR transcript")
                        with history_lock:
                            context = "\n".join(history[-6:])
                        if context:
                            q = ("Conversation so far:\n" + context
                                 + "\n\nThe user now asks (resolve any "
                                   "references like \"it\"/\"that\"/"
                                   "\"what about X\" against the "
                                   "conversation above): " + str(up))
                        else:
                            q = str(up)
                        t0 = time.time()
                        r = escalate.ask_expert_web(q, effort="low")
                        if r.get("error"):
                            r = escalate.ask_expert(q, effort="low")
                        state["expert_answer"] = (
                            r.get("answer") or f"[error: {r.get('error')}]")
                        state["expert_latency_s"] = round(
                            time.time() - t0, 2)
                        emit({"type": "log",
                              "msg": f"thinker answered in "
                                     f"{state['expert_latency_s']:.1f}s"})
                        relay_box.append(state)
                    except Exception as e:
                        state["expert_error"] = str(e)[:160]
                        emit({"type": "log",
                              "msg": f"thinker failed: {str(e)[:120]}"})
                        finish_turn_async(state)
                    finally:
                        thinking.clear()

                try:
                    while not stop.is_set():
                        try:
                            with ilock:
                                got, inbox[:] = inbox[:], []
                            if got:
                                pend = np.concatenate([pend] + got)
                            if len(pend) < CH:
                                time.sleep(0.02)
                                continue
                            ch, pend = pend[:CH], pend[CH:]
                            if len(pend) > 6 * CH:
                                emit({"type": "log",
                                      "msg": f"falling behind realtime "
                                             f"({len(pend) / CH:.1f}s queued)"})

                            # thinker result: prefill as a TEXT unit into the
                            # SAME stream; the talker voices it in-band and
                            # stays interruptible (native, chunk 3 below)
                            if relay_box:
                                relay_turn = relay_box.pop(0)
                                ans = relay_turn.get("expert_answer", "")
                                relay_turn["relay_started_at"] = time.time()
                                if muted:
                                    emit({"type": "log",
                                          "msg": f"muted {muted - 1} chunks "
                                                 "of local continuation"})
                                    muted = 0
                                relay_guard = True   # no gate fire until
                                #                      this delivery's eot
                                emit({"type": "phase", "v": "relaying"})
                                if RELAY_MODE == "tts" and self.tts_ok:
                                    # 8bu: verbatim expert text in the
                                    # talker's own voice; the duplex
                                    # context only gets a note, and the
                                    # local continuation stays muted to
                                    # end_of_turn so nothing talks over it
                                    spoken = clean_expert(ans)
                                    t_s = time.time()
                                    pcm = None
                                    try:
                                        pcm = self._synth_pcm(spoken)
                                    except Exception as se:
                                        emit({"type": "log",
                                              "msg": "relay synth failed: "
                                                     + str(se)[:100]})
                                    if pcm is not None:
                                        i16r = (np.clip(pcm, -1, 1)
                                                * 32767).astype("<i2")
                                        emit({"type": "audio", "sr": 24000,
                                              "pcm": base64.b64encode(
                                                  i16r.tobytes()).decode()})
                                    emit({"type": "text", "v": " " + spoken,
                                          "relay": True})
                                    relay_turn["assistant_parts"].append(
                                        " " + spoken)
                                    emit({"type": "log",
                                          "msg": f"relay (tts) "
                                                 f"{len(pcm) / 24000 if pcm is not None else 0:.1f}s "
                                                 f"audio, synth "
                                                 f"{time.time() - t_s:.1f}s"})
                                    self.duplex.streaming_prefill(
                                        text_list=[RELAY_NOTE.format(ans=spoken)])
                                    r = self.duplex.streaming_generate(
                                        prompt_wav_path=PROMPT_WAV,
                                        top_k=GEN_TOP_K)
                                    _emit_gen(r, mute=True)
                                    muted = 1
                                    prev_listen = r["is_listen"]
                                else:
                                    self.duplex.streaming_prefill(
                                        text_list=[RELAY_TMPL.format(ans=ans)])
                                    r = self.duplex.streaming_generate(
                                        prompt_wav_path=PROMPT_WAV,
                                        top_k=GEN_TOP_K)
                                    _emit_gen(r, relay=True)
                                    if r.get("text"):
                                        relay_turn["assistant_parts"].append(
                                            r["text"])
                                    if not r.get("text"):
                                        emit({"type": "log",
                                              "msg": "relay swallowed — nudging"})
                                        self.duplex.streaming_prefill(
                                            text_list=[RELAY_NUDGE])
                                        r = self.duplex.streaming_generate(
                                            prompt_wav_path=PROMPT_WAV,
                                            top_k=GEN_TOP_K)
                                        _emit_gen(r, relay=True)
                                        if r.get("text"):
                                            relay_turn["assistant_parts"].append(
                                                r["text"])
                                    prev_listen = r["is_listen"]

                            user_win.append(ch)
                            if len(user_win) > 45:
                                user_win = user_win[-45:]

                            self.st3["accum"] = True
                            ok = self.duplex.streaming_prefill(
                                audio_waveform=ch)
                            self.st3["accum"] = False
                            if not ok.get("success"):
                                emit({"type": "log",
                                      "msg": f"prefill skipped: "
                                             f"{ok.get('reason', '')[:80]}"})
                                continue
                            r = self.duplex.streaming_generate(
                                prompt_wav_path=PROMPT_WAV,
                                top_k=GEN_TOP_K)
                            n_chunk += 1

                            score = self._score_now()
                            shadow_score = self._shadow_score_now()
                            if score is not None:
                                turn_scores.append(round(score, 4))
                                emit({"type": "score", "i": n_chunk,
                                      "v": round(score, 4),
                                      "shadow_v": (None if shadow_score is None
                                                   else round(shadow_score, 4)),
                                      "listen": bool(r["is_listen"])})

                            fired_now = False
                            if prev_listen and not r["is_listen"]:
                                # the talker just decided to answer — the
                                # gate reads exactly here. 8bh: floor-
                                # management commits (stop words,
                                # backchannel replies) must not escalate.
                                act = self._act_now()
                                is_info = (act is None
                                           or act >= self.act[
                                               "act_threshold"])
                                thr_eff, thr_mode = effective_thr()
                                if score is not None and is_info:
                                    score_win.append(float(score))
                                fired = bool(probe_on and score is not None
                                             and score >= thr_eff and is_info
                                             and not thinking.is_set()
                                             and not relay_guard
                                             and len(user_win) > 0)
                                fired_now = fired
                                turn_index += 1
                                snap = (np.concatenate(user_win)
                                        if user_win else
                                        np.zeros(1600,
                                                 np.float32))[-30 * 16000:]
                                active_turn = {
                                    "index": turn_index,
                                    "started_at": time.time(),
                                    "snapshot": snap.copy(),
                                    "fired": fired,
                                    "score": (None if score is None
                                              else round(score, 4)),
                                    "threshold": round(thr, 4),
                                    "scores": list(turn_scores),
                                    "act_score": (None if act is None
                                                  else round(act, 4)),
                                    "is_info": bool(is_info),
                                    "assistant_parts": [],
                                    "asr_done": _th.Event(),
                                }
                                emit({"type": "gate",
                                      "score": (None if score is None
                                                else round(score, 4)),
                                      "shadow_score": (
                                          None if shadow_score is None
                                          else round(shadow_score, 4)),
                                      "thr": round(thr_eff, 4),
                                      "thr_mode": thr_mode,
                                      "thr_static": round(thr, 4),
                                      "n_window": len(score_win),
                                      "fired": fired,
                                      "act": (None if act is None
                                              else round(act, 4)),
                                      "is_info": bool(is_info),
                                      "turn_index": completed_turns + 1,
                                      "has_context": completed_turns > 0,
                                      "prior_escalations": prior_escalations,
                                      "probe_on": probe_on})
                                if fired:
                                    thinking.set()
                                    emit({"type": "phase", "v": "escalating"})
                                    _th.Thread(target=thinker,
                                               args=(active_turn,),
                                               daemon=True).start()
                                else:
                                    _th.Thread(target=transcribe_turn,
                                               args=(active_turn,),
                                               daemon=True).start()

                            _emit_gen(r, mute=muted > 0)
                            if muted:
                                muted += 1
                            if r.get("text") and not muted:
                                turn_text.append(r["text"])
                                target_turn = (relay_turn if relay_guard
                                               and relay_turn is not None
                                               else active_turn)
                                if target_turn is not None:
                                    target_turn["assistant_parts"].append(
                                        r["text"])
                            if fired_now:
                                turn_fired = True
                                prior_escalations += 1
                                if self.stall_pcm is not None:
                                    i16s = (np.clip(self.stall_pcm, -1, 1)
                                            * 32767).astype("<i2")
                                    emit({"type": "audio", "sr": 24000,
                                          "pcm": base64.b64encode(
                                              i16s.tobytes()).decode()})
                                    emit({"type": "text", "v": " " + STALL})
                                    active_turn["assistant_parts"].append(
                                        " " + STALL)
                                emit({"type": "log",
                                      "msg": "canned stall + context note; "
                                             "local continuation muted "
                                             "until relay"})
                                muted = 1
                                self.duplex.streaming_prefill(
                                    text_list=[STALL_NOTE])
                                r = self.duplex.streaming_generate(
                                    prompt_wav_path=PROMPT_WAV,
                                    top_k=GEN_TOP_K)
                                _emit_gen(r, mute=True)
                            if r.get("end_of_turn"):
                                completed_turns += 1
                                if muted:
                                    emit({"type": "log",
                                          "msg": f"muted {muted - 1} chunks "
                                                 "of local continuation"})
                                    muted = 0
                                ans = "".join(turn_text).strip()
                                if relay_guard and relay_turn is not None:
                                    relay_turn["relay_ms"] = int(
                                        (time.time() - relay_turn.get(
                                            "relay_started_at", time.time()))
                                        * 1000)
                                    finish_turn_async(relay_turn)
                                    relay_turn = None
                                elif active_turn is not None:
                                    if not active_turn["fired"]:
                                        if ans and not active_turn[
                                                "assistant_parts"]:
                                            active_turn["assistant_parts"].append(
                                                ans)
                                        finish_turn_async(active_turn)
                                    # Fired local continuation is intentionally
                                    # muted; its state is owned by thinker/relay.
                                    active_turn = None
                                user_win = []
                                turn_text, turn_fired = [], False
                                turn_scores = []
                                relay_guard = False
                                self.st3.update(sum=None, cnt=0)
                            prev_listen = r["is_listen"]
                        except Exception as ie:
                            # 8bn: one bad iteration must not
                            # kill the session (a fire-time
                            # concatenate on an empty uplink
                            # window did exactly that)
                            import traceback
                            traceback.print_exc()
                            emit({"type": "log",
                                  "msg": "loop error "
                                         "(recovered): "
                                         + str(ie)[:100]})
                            time.sleep(0.1)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    emit({"type": "error", "msg": str(e)[:200]})
                finally:
                    emit({"type": "bye"})

            def _emit_gen(r, relay=False, mute=False):
                if mute:
                    # 8bm: thinker engaged — the condemned turn's
                    # continuation is generated (context true,
                    # perception intact, natively interruptible)
                    # but not voiced
                    emit({"type": "chunk",
                          "listen": bool(r["is_listen"]),
                          "eot": bool(r.get("end_of_turn")),
                          "muted": True,
                          "cost": round(r.get("cost_all", 0), 3)})
                    if r.get("end_of_turn"):
                        emit({"type": "phase", "v": "listening"})
                    return
                wf = r.get("audio_waveform")
                if not r["is_listen"] and wf is not None and len(wf):
                    i16 = (np.clip(np.asarray(wf, dtype=np.float32),
                                   -1, 1) * 32767).astype("<i2")
                    emit({"type": "audio", "sr": 24000,
                          "pcm": base64.b64encode(i16.tobytes()).decode()})
                if not r["is_listen"] and r.get("text"):
                    emit({"type": "text", "v": r["text"],
                          "relay": relay})
                emit({"type": "chunk",
                      "listen": bool(r["is_listen"]),
                      "eot": bool(r.get("end_of_turn")),
                      "cost": round(r.get("cost_all", 0), 3)})
                if r.get("end_of_turn"):
                    emit({"type": "phase", "v": "listening"})

            try:
                await sock.send_json(
                    {"type": "hello", "protocol": "duplex_v1",
                     "thr": round(thr, 4), "tier": tier,
                     "lang": lang, "probe_on": probe_on,
                     "tracker": tracker_on, "n_window": len(score_win),
                     "mode": "NATIVE full duplex — the model itself "
                             "decides listen/speak every second; no VAD, "
                             "no soft barge-in harness"})
                await loop.run_in_executor(None, self._session_reset)
                await sock.send_json({"type": "phase", "v": "listening"})
                worker = _th.Thread(target=chunk_loop, daemon=True)
                worker.start()
                while True:
                    msg = await sock.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    txt = msg.get("text")
                    if txt:
                        try:
                            if json.loads(txt).get("type") == "stop":
                                break
                        except Exception:
                            pass
                        continue
                    b = msg.get("bytes")
                    if b:
                        f = (np.frombuffer(b, dtype=np.int16)
                             .astype(np.float32) / 32768.0)
                        with ilock:
                            inbox.append(f)
            except RuntimeError:
                pass
            finally:
                stop.set()
                self.duplex.set_session_stop()
                await loop.run_in_executor(None, worker.join)
                self.duplex.clear_session_stop()
                self.lock.release()

        return wapp


@app.function(image=web_image)
@modal.asgi_app(label="gate-duplex")
def page():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    wapp = FastAPI()

    @wapp.get(f"/{TOKEN}/")
    def root():
        return HTMLResponse(HTML.replace("__TOKEN__", TOKEN))
    return wapp


HTML = r"""<!doctype html><html><head><meta charset=utf-8>
<title>gate — native duplex</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1c2733}
.wrap{max-width:880px;margin:0 auto;padding:1.2rem}
h1{font-size:1.25rem;margin:.2rem 0}.sub{color:#5b6b7c;font-size:.85rem}
.card{background:#fff;border:1px solid #dfe5ec;border-radius:10px;
 padding:1rem;margin:.8rem 0}
button{font:inherit;padding:.5rem 1rem;border-radius:8px;
 border:1px solid #c8d2dd;background:#fff;cursor:pointer}
button.primary{background:#2a78d6;color:#fff;border-color:#2a78d6}
button:disabled{opacity:.45;cursor:default}
select{font:inherit;padding:.35rem}
#vu{height:6px;background:#2a78d6;width:0%;border-radius:3px;
 transition:width .08s}
#log{font:12px/1.55 ui-monospace,monospace;background:#0e1621;color:#cfe3f7;
 border-radius:8px;padding:.7rem;height:270px;overflow-y:auto;
 white-space:pre-wrap}
#log .off{color:#8fa3b8}#log .esc{color:#ffcf6e}#log .txt{color:#9be49b}
.pill{display:inline-block;padding:.1rem .55rem;border-radius:999px;
 font-size:.75rem;background:#e8edf3;margin-right:.4rem}
.pill.on{background:#d9ecdb;color:#1c6b2a}
#swlab{cursor:pointer;user-select:none}
</style></head><body><div class=wrap>
<h1>escalation gate · native full duplex</h1>
<div class=sub>MiniCPMODuplex — the model itself holds the floor: it hears
you while it speaks and yields on its own. No VAD, no soft barge-in
harness, no kill switch. Browser AEC is the only echo control —
<b>use headphones</b>.</div>
<div class=card>
 <span class=pill id=swlab>PROBE ON</span>
 tier <select id=tier><option>conservative</option>
 <option selected>balanced</option><option>aggressive</option></select>
 lang <select id=lang><option selected>en</option><option>zh</option></select>
 <button id=talk class=primary disabled>GPU starting…</button>
 <div style="margin-top:.6rem"><div id=vu></div></div>
 <div class=sub id=state>—</div>
</div>
<div class=card><b>talker</b> <span id=phase class=pill>idle</span>
 <div id=text class=sub style="min-height:2.2rem"></div></div>
<div class=card><div id=log></div></div>
</div><script>
const T="__TOKEN__";
const VOICE="https://rhe9527--gate-duplex-voice.modal.run";
const $=s=>document.querySelector(s);
let probeOn=true,ws=null,ac=null,micStream=null,proc=null,talking=false;
let playCtx=null,playT=0,gpuReady=false;
function log(m,c){const l=$("#log");
 l.innerHTML+=`<div class="${c||''}"><b>${new Date()
 .toLocaleTimeString()}</b> ${m}</div>`;l.scrollTop=l.scrollHeight;}
$("#swlab").onclick=()=>{probeOn=!probeOn;
 $("#swlab").textContent=probeOn?"PROBE ON":"PROBE OFF";
 $("#swlab").classList.toggle("on",probeOn);
 log("probe "+(probeOn?"ON":"OFF")+" for the NEXT session","off");};
function playPCM(b64,sr){
 if(!playCtx)playCtx=new (window.AudioContext||window.webkitAudioContext)();
 if(playCtx.state==="suspended")playCtx.resume();
 const raw=atob(b64),n=raw.length>>1,f=new Float32Array(n);
 for(let i=0;i<n;i++){let v=raw.charCodeAt(2*i)|(raw.charCodeAt(2*i+1)<<8);
  if(v>=32768)v-=65536;f[i]=v/32768;}
 const buf=playCtx.createBuffer(1,n,sr);buf.getChannelData(0).set(f);
 const src=playCtx.createBufferSource();src.buffer=buf;
 src.connect(playCtx.destination);
 playT=Math.max(playT,playCtx.currentTime);
 src.start(playT);playT+=buf.duration;}
async function warm(){
 const t0=Date.now();let j=null;
 const tick=setInterval(()=>{if(!gpuReady)$("#state").textContent=
  `GPU starting + loading MiniCPM… ${((Date.now()-t0)/1000|0)}s`;},1000);
 for(let i=0;i<90&&!j;i++){
  try{const r=await fetch(`${VOICE}/${T}/ready`,
   {signal:AbortSignal.timeout(30000)});if(r.ok)j=await r.json();}
  catch(_){}
  if(!j)await new Promise(s=>setTimeout(s,4000));}
 clearInterval(tick);
 if(j&&j.ready){gpuReady=true;$("#talk").disabled=false;
  $("#talk").textContent="Start duplex session";
  $("#state").textContent=`GPU ready (model loaded in ${j.load_s}s)`;}
 else $("#state").textContent="GPU start timed out — reload to retry.";}
warm();
function stopTalk(){talking=false;
 try{if(ws&&ws.readyState<2){ws.send(JSON.stringify({type:"stop"}));
  ws.close();}}catch(_){}
 try{if(proc)proc.disconnect()}catch(_){}
 try{if(ac)ac.close()}catch(_){}
 try{if(micStream)micStream.getTracks().forEach(t=>t.stop())}catch(_){}
 ws=ac=proc=micStream=null;
 $("#talk").textContent="Start duplex session";$("#vu").style.width="0%";
 $("#phase").textContent="idle";}
async function startTalk(){
 try{micStream=await navigator.mediaDevices.getUserMedia({audio:{
  channelCount:1,echoCancellation:true,noiseSuppression:true,
  autoGainControl:true}});}
 catch(e){log("mic permission denied: "+e,"off");return;}
 ac=new AudioContext();
 const src=ac.createMediaStreamSource(micStream);
 proc=ac.createScriptProcessor(2048,1,1);
 const ratio=ac.sampleRate/16000;
 ws=new WebSocket(`${VOICE.replace("https","wss")}/${T}/ws`
  +`?tier=${$("#tier").value}&lang=${$("#lang").value}`
  +`&probe_on=${probeOn?1:0}`);
 ws.onmessage=ev=>handle(JSON.parse(ev.data));
 ws.onclose=()=>{if(talking){log("session closed","off");stopTalk();}};
 ws.onerror=()=>{log("websocket error","off");stopTalk();};
 ws.onopen=()=>{src.connect(proc);proc.connect(ac.destination);
  talking=true;$("#talk").textContent="■ End session";
  log(`duplex session open (tier ${$("#tier").value}, probe `
   +`${probeOn?"ON":"OFF"}) — just talk; talk over it to interrupt`);};
 proc.onaudioprocess=e=>{
  const f=e.inputBuffer.getChannelData(0);
  let ss=0;for(let i=0;i<f.length;i++)ss+=f[i]*f[i];
  $("#vu").style.width=Math.min(100,Math.sqrt(ss/f.length)*700)+"%";
  if(!ws||ws.readyState!==1)return;
  const n=Math.floor(f.length/ratio),out=new Int16Array(n);
  for(let i=0;i<n;i++){const v=f[Math.floor(i*ratio)];
   out[i]=Math.max(-32768,Math.min(32767,v*32767));}
  ws.send(out.buffer);};}
$("#talk").onclick=()=>{talking?stopTalk():startTalk()};
let turnText="";
function handle(m){
 if(m.type==="hello")log(`session config: thr ${m.thr} (${m.tier}/${m.lang}), `
  +`probe ${m.probe_on?"ON":"OFF"} — ${m.mode}`);
 else if(m.type==="phase")$("#phase").textContent=m.v;
 else if(m.type==="audio")playPCM(m.pcm,m.sr);
 else if(m.type==="text"){turnText+=m.v;
  $("#text").textContent=turnText.slice(-300);
  log((m.relay?"[relay] ":"")+m.v,"txt");}
 else if(m.type==="chunk"){
  if(m.eot){turnText="";log("— turn ended (model yielded the floor) —",
   "off");}}
 else if(m.type==="score"){
  if(m.listen)$("#state").textContent=
   `listening · running P(fail)=${m.v.toFixed(3)}`;}
 else if(m.type==="gate")log(`TALKER COMMITS — P(fail)=${m.score} vs thr `
  +`${m.thr}`+(m.act==null?"":` · P(info)=${m.act}`)
  +` → ${m.fired?"ESCALATE (thinker launched)":
   (m.is_info===false?"floor turn — gate bypassed":"stay local")}`
  +(m.probe_on?"":" [probe off]"),m.fired?"esc":"");
 else if(m.type==="log")log(m.msg,"off");
 else if(m.type==="error")log("ERROR: "+m.msg,"esc");
}
</script></body></html>"""
