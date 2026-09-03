# -*- coding: utf-8 -*-
"""NVDA probe comparison UI (worker-local, 127.0.0.1:8090).
One turn per request, three probes side by side.

Audio in -> NVIDIA's native frame-streaming duplex wrapper (one 80 ms frame
at a time over user audio + a 12 s silent tail; the model's own agent
channel decides when to speak, nothing forces the turn) -> hidden states at
the commit-to-speak moment -> three gates:

  deployed  gate_demo_nvda.json         L30: onset_last | onset_mean8 | user_mean   (+ act head)
  v2        gate_demo_nvda_v2.json      L26/30/34 avg: commit | onset_last | onset_mean8 | user_mean
  causal    gate_demo_nvda_causal.json  L34: commit | mean(8 frames before commit) | mean(all frames before commit)

Also returns the agent-channel timeline (PAD vs text per frame), the model's
own spoken answer (TTS wav, if produced) and, if OPENAI_API_KEY is set, the
gpt-5.5 expert answer for turns the deployed gate fires on. No key -> the
expert step is skipped and the UI says so. Turn-level replay, NOT live
streaming.
"""
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.environ.get("NVDA_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NVDA_DATA", f"{ROOT}/data")
MODEL_DIR = os.environ.get("NVDA_MODEL_DIR", f"{ROOT}/weights")
ARTIFACT_DIR = os.environ.get("NVDA_ARTIFACT_DIR", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.environ.get("NVDA_NEMO_ROOT", f"{ROOT}/nemo-speech"))

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nvda_replay_v2 import SYS_PROMPT, TAIL_SIL_S, NVDA_LAYERS, K_EOT

ART = {k: json.load(open(f"{ARTIFACT_DIR}/{f}")) for k, f in
       (("deployed", "gate_demo_nvda.json"), ("v2", "gate_demo_nvda_v2.json"), ("causal", "gate_demo_nvda_causal.json"))}
LIDX = {L: i for i, L in enumerate(NVDA_LAYERS)}
# the deployed artifact predates these fields; numbers on the same 2,258-row calib-only cohort as v2/causal
ART["deployed"].setdefault("read", "L30: onset_last | onset_mean8 | user_mean (commit + 8 frames) + act head")
ART["deployed"].setdefault("calib_oof_auc", 0.811)
ART["deployed"].setdefault("external_auc_cold", {"striviaqa": 0.837, "swebq": 0.8547, "sllama": 0.7683, "sdqa": 0.7703, "mean": 0.8076})
TEXT_PAD_ID = 12
MODEL = None
STREAMING_MODEL = None
LOCK = threading.Lock()
REQUEST_LOCK = threading.Lock()
STATUS_LOCK = threading.Lock()
STATUS = {"busy": False, "phase": "idle", "started_at": None}
INFERENCE_CACHE = {}
HAVE_KEY = bool(os.environ.get("OPENAI_API_KEY"))


def sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))))


def encode_wav(audio, sample_rate=22050):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def set_status(busy, phase):
    with STATUS_LOCK:
        STATUS.update(
            busy=busy,
            phase=phase,
            started_at=time.time() if busy and not STATUS["busy"] else (
                STATUS["started_at"] if busy else None
            ),
        )


def load_streaming_model():
    """Load NVIDIA's frame-streaming wrapper with its stateful codec path."""
    from omegaconf import OmegaConf
    from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
        NemotronVoicechatInferenceWrapper,
    )

    # NVIDIA's wrapper implements these as supported inference controls: silence
    # codec tokens while the agent is idle, and cap runaway TTS PAD tails.
    os.environ["S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD"] = "true"
    config = OmegaConf.create(
        {
            "model_path": MODEL_DIR,
            "llm_checkpoint_path": MODEL_DIR,
            "speaker_name": "Aria",
            "speaker_reference": None,
            "engine_type": "native",
            "device": "cuda",
            "device_id": 0,
            "compute_dtype": "bfloat16",
            "decode_audio": True,
            "top_p": 1.0,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "use_perception_cache": True,
            "use_perception_cudagraph": False,
            "use_codec_cache": True,
            "force_turn_taking": False,
            "fc_async_enabled": False,
            "fc_async_two_phase": False,
            "enable_builtin_tools": False,
            "system_prompt": SYS_PROMPT,
            "max_agent_response_sec": 0,
            "tts_text_token_ratio_cap": 6.0,
            "tts_text_token_min": 5,
        }
    )
    return NemotronVoicechatInferenceWrapper(config)


def infer_one(wav_path, decode_audio=True):
    """Replay one recording through NVIDIA's native frame-streaming wrapper."""
    import librosa
    import torch

    au, _ = librosa.load(wav_path, sr=16000, mono=True)
    n_frames_q = int(np.ceil(len(au) / int(0.08 * 16000)))
    tokenizer = MODEL.stt_model.tokenizer
    prompt_len = len(
        [tokenizer.bos_id]
        + tokenizer.text_to_ids(SYS_PROMPT)
        + [tokenizer.eos_id]
    )
    store = {}

    def mk(L):
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out; store[L] = hs.detach()
        return hook

    handles = [MODEL.stt_model.llm.layers[L].register_forward_hook(mk(L)) for L in (26, 30, 34)]
    t0 = time.time()
    prior_decode_audio = STREAMING_MODEL.decode_audio
    try:
        STREAMING_MODEL.decode_audio = decode_audio
        STREAMING_MODEL._agent_idle = True
        STREAMING_MODEL._tts_in_turn_content = 0
        STREAMING_MODEL._tts_in_turn_pads = 0
        res = STREAMING_MODEL.inference_realtime_streaming(
            wav_path,
            num_frames_per_chunk=1,
            pad_audio_by_sec=TAIL_SIL_S,
            system_prompt=SYS_PROMPT,
        )
    finally:
        STREAMING_MODEL.decode_audio = prior_decode_audio
        for h in handles:
            h.remove()
    secs = time.time() - t0
    valid_len = int(res["tokens_len"][0])
    tok = res["tokens_text"][0, :valid_len].tolist()
    onset = None
    for i in range(len(tok) - 2):
        if all(t != TEXT_PAD_ID for t in tok[i:i + 3]):
            onset = i; break
    H = {L: store[L][0].float() for L in (26, 30, 34)}
    t_end = prompt_len + n_frames_q
    feats = {}
    for L, h in H.items():
        hi = min(t_end, h.shape[0]); f = {"user_mean": h[prompt_len:hi].mean(0).cpu().numpy()}
        if onset is not None:
            olo = prompt_len + onset; ohi = min(olo + K_EOT, h.shape[0])
            ow = h[olo:ohi].cpu().numpy(); f["onset_last"] = ow[-1]; f["onset_mean8"] = ow.mean(0); f["commit"] = ow[0]
            plo = max(prompt_len, olo - K_EOT)
            f["pre_mean8"] = h[plo:olo].cpu().numpy().mean(0) if olo > plo else np.zeros_like(f["user_mean"])
            f["run_mean"] = h[prompt_len:olo].cpu().numpy().mean(0) if olo > prompt_len else np.zeros_like(f["user_mean"])
        feats[L] = {k: v.astype(np.float32) for k, v in f.items()}
    raw = res.get("text", [""])[0]
    clean = re.sub(r"<[^>]{0,24}>", " ", raw); clean = re.sub(r"  +", " ", clean).strip()
    src = ""
    for k in ("rnnt_asr_text", "asr_text", "src_text"):
        v = res.get(k)
        if v is not None and str(v[0]).strip():
            src = str(v[0]); break
    timeline = [0 if t == TEXT_PAD_ID else 1 for t in tok]      # per 80 ms frame: 0 listen, 1 speak
    audio_b64 = None
    if res.get("audio") is not None:
        try:
            a = res["audio"][0].float().cpu().reshape(-1).numpy()
            audio_b64 = encode_wav(a)
        except Exception:
            audio_b64 = None
    return dict(answer=clean, answer_raw=raw, src_text=re.sub(r"<[^>]{0,24}>", " ", src).strip(), onset=onset,
                n_frames_query=n_frames_q, prompt_len=prompt_len, secs=secs, feats=feats, timeline=timeline,
                audio_b64=audio_b64)


def score_all(feats):
    out = {}
    # deployed: single head L30, 3 blocks + act head
    a = ART["deployed"]; f = feats[30]
    x = np.concatenate([f["onset_last"], f["onset_mean8"], f["user_mean"]])
    pf = sigmoid(x @ np.asarray(a["fail"]["w"], np.float32) + a["fail"]["b"])
    pa = sigmoid(x @ np.asarray(a["act"]["w"], np.float32) + a["act"]["b"])
    out["deployed"] = {"p_fail": pf, "p_act": pa, "tau": a["act"]["tau"], "thresholds": a["fail"]["thresholds"]}
    # v2: three heads, standardised logits averaged
    a = ART["v2"]; zs = []
    for hd in a["fail"]["heads"]:
        f = feats[hd["layer"]]; x = np.concatenate([f["commit"], f["onset_last"], f["onset_mean8"], f["user_mean"]])
        zs.append((x @ np.asarray(hd["w"], np.float32) + hd["b"]) / hd["logit_std"])
    out["v2"] = {"p_fail": sigmoid(np.mean(zs)), "thresholds": a["fail"]["thresholds"]}
    # causal: single head L34
    a = ART["causal"]; f = feats[a["layer"]]
    x = np.concatenate([f["commit"], f["pre_mean8"], f["run_mean"]])
    out["causal"] = {"p_fail": sigmoid(x @ np.asarray(a["fail"]["w"], np.float32) + a["fail"]["b"]), "thresholds": a["fail"]["thresholds"]}
    return out


def expert(query):
    from openai import OpenAI
    r = OpenAI().chat.completions.create(model="gpt-5.5", reasoning_effort="low", messages=[{"role": "user", "content": query}])
    return r.choices[0].message.content


def uplink_asr(wav_path):
    from openai import OpenAI
    cli = OpenAI()
    for m in ("gpt-transcribe", "gpt-4o-transcribe", "whisper-1"):
        try:
            with open(wav_path, "rb") as fh:
                r = cli.audio.transcriptions.create(model=m, file=fh)
            if r.text and r.text.strip():
                return r.text.strip(), m
        except Exception:
            continue
    return "", ""


def infer_question(wav_path, cache_key, decode_audio):
    cached = INFERENCE_CACHE.get(cache_key)
    if cached is not None and (not decode_audio or cached["audio_b64"] is not None):
        set_status(True, "using the cached baseline pass")
        return cached
    with LOCK:
        result = infer_one(wav_path, decode_audio=decode_audio)
    if cache_key:
        if len(INFERENCE_CACHE) >= 4:
            INFERENCE_CACHE.pop(next(iter(INFERENCE_CACHE)))
        INFERENCE_CACHE[cache_key] = result
    return result


def run_turn(wav_path, tier, cache_key):
    set_status(True, "running baseline model pass")
    r = infer_question(wav_path, cache_key, decode_audio=True)
    out = {"answer": r["answer"], "src_text": r["src_text"], "onset_frame": r["onset"], "n_frames_query": r["n_frames_query"],
           "commit_gap_s": None if r["onset"] is None else round((r["onset"] - r["n_frames_query"]) * 0.08, 2),
           "infer_s": round(r["secs"], 1), "timeline": r["timeline"], "audio_b64": r["audio_b64"], "tier": tier,
           "expert_enabled": HAVE_KEY, "probes": {}}
    if r["onset"] is None:
        out["no_commit"] = True; return out
    sc = score_all(r["feats"])
    for name, s in sc.items():
        thr = s["thresholds"][tier]
        fired = s["p_fail"] >= thr
        if name == "deployed":
            floor = s["p_act"] < s["tau"]; fired = fired and not floor
            out["probes"][name] = {"p_fail": round(s["p_fail"], 4), "threshold": round(thr, 4), "fired": bool(fired), "p_act": round(s["p_act"], 4), "tau": round(s["tau"], 4), "floor_turn": bool(floor)}
        else:
            out["probes"][name] = {"p_fail": round(s["p_fail"], 4), "threshold": round(thr, 4), "fired": bool(fired)}
    if HAVE_KEY and out["probes"]["deployed"]["fired"]:
        try:
            q = r["src_text"]
            if not q:
                q, m = uplink_asr(wav_path); out["uplink_asr"] = m; out["src_text"] = q
            out["expert_answer"] = expert(q) if q else "(no transcript available)"
        except Exception as e:
            out["expert_answer"] = f"(expert call failed: {e})"
    return out


STIMS = {}
try:
    for line in open(f"{DATA_DIR}/queries_flooract.jsonl", encoding="utf-8"):
        r = json.loads(line); STIMS[r["id"]] = r["query"]
except Exception:
    pass
STIM_CHOICES = [("fa0000", "Stop."), ("fa0002", "Stop!"), ("fa0010", "Stop talking."), ("fa0012", "Hold on."),
                ("fa0016", "Wait."), ("fa0080", "Okay."), ("fa0074", "Uh-huh."), ("fa0134", "Thank you.")]


WINDOW = 75   # 6 s after the interruption


def _yield_stats(inter_tl, base_tl, stim_frame, stim_len, gap_frames=5):
    """Compare the 6 s after the interruption in the interrupted replay against the
    same window in the baseline. 'Stopped' = the model speaks <= 1 s in that window
    while the baseline spoke >= 2 s more than that. If the model was already
    silent when the stim arrived, we report whether it resumed anyway."""
    post = inter_tl[stim_frame:stim_frame + WINDOW]; base = base_tl[stim_frame:stim_frame + WINDOW]
    speak_i, speak_b = 0.08 * sum(post), 0.08 * sum(base)
    speaking_at_stim = bool(
        sum(inter_tl[max(0, stim_frame - 3):stim_frame]) >= 2
    )
    latency = None
    if speaking_at_stim:
        run = 0
        for i, v in enumerate(post):
            run = run + 1 if v == 0 else 0
            if run >= gap_frames:
                latency = round((i - gap_frames + 1) * 0.08, 2); break
    resumed = None if speaking_at_stim else bool(sum(post) >= 3)
    stopped = speak_i <= 1.0 and (speak_b - speak_i) >= 2.0
    return {"window_s": WINDOW * 0.08, "speak_s_after_stim": round(speak_i, 2), "baseline_speak_s_same_window": round(speak_b, 2),
            "speaking_at_stim": speaking_at_stim, "yield_latency_s": latency, "resumed_after_stim": resumed,
            "stopped": bool(stopped), "stim_s": round(stim_len * 0.08, 2)}


def choose_stim_frame(base_timeline, question_end_frame, requested_frame):
    """Choose a point where the baseline is speaking and would keep speaking.

    A stop test is informative only if the uninterrupted model speaks for at
    least two more seconds in the six-second comparison window.
    """
    minimum_future_speaking_frames = 25
    eligible = []
    for frame in range(question_end_frame, len(base_timeline)):
        future = base_timeline[frame:frame + WINDOW]
        if (
            base_timeline[frame]
            and sum(future) >= minimum_future_speaking_frames
        ):
            eligible.append(frame)
    if not eligible:
        raise ValueError(
            "the baseline has no post-question speaking interval long enough "
            "for a valid stop test"
        )
    return min(eligible, key=lambda frame: abs(frame - requested_frame))


def trim_interruption_wav(wav_path):
    """Trim recorder silence so the injected duration reflects actual speech."""
    import librosa
    import soundfile as sf
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    intervals = librosa.effects.split(
        audio,
        top_db=30,
        frame_length=1024,
        hop_length=256,
    )
    if len(intervals) == 0:
        raise ValueError("the interruption recording contains no audible speech")
    pad = int(0.08 * 16000)
    start = max(0, int(intervals[0, 0]) - pad)
    end = min(len(audio), int(intervals[-1, 1]) + pad)
    trimmed = audio[start:end]
    if len(trimmed) > 4 * 16000:
        raise ValueError(
            "the interruption is longer than 4 seconds after silence trimming"
        )
    sf.write(wav_path, trimmed, 16000)
    return round(len(trimmed) / 16000, 2)


def run_interrupt(q_wav, stim_wav, offset_s, tier, cache_key):
    """Timing-controlled native-duplex barge-in replay.

    A cached baseline is reused when available. Audio is decoded through
    NVIDIA's stateful streaming codec, not the broken bulk offline decoder.
    """
    import librosa
    import soundfile as sf
    started = time.time()
    baseline_cached = cache_key in INFERENCE_CACHE
    set_status(True, "pass 1/2: loading or running the baseline question")
    base = infer_question(q_wav, cache_key, decode_audio=False)
    if base["onset"] is None:
        return {"error": "the model never committed to speak on the question alone; nothing to interrupt", "base": {"answer": base["answer"]}}
    q, _ = librosa.load(q_wav, sr=16000, mono=True); s, _ = librosa.load(stim_wav, sr=16000, mono=True)
    commit_s = base["onset"] * 0.08; q_end_s = len(q) / 16000
    # NVDA often commits before the question ends. Start the requested offset
    # after both events, then clamp it to a baseline speaking frame that leaves
    # enough future speech for a meaningful stop comparison.
    anchor_s = max(commit_s, q_end_s)
    requested_stim_at = anchor_s + float(offset_s)
    requested_frame = int(round(requested_stim_at / 0.08))
    question_end_frame = int(np.ceil(q_end_s / 0.08))
    stim_frame = choose_stim_frame(
        base["timeline"],
        question_end_frame,
        requested_frame,
    )
    stim_at = stim_frame * 0.08
    n_lead = stim_frame * int(0.08 * 16000)
    comp = np.zeros(n_lead + len(s), dtype=np.float32); comp[:len(q)] = q; comp[n_lead:n_lead + len(s)] += s
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, comp, 16000); cwav = f.name
    set_status(True, "pass 2/2: running the timed interruption")
    with LOCK:
        inter = infer_one(cwav, decode_audio=True)
    stim_len = int(np.ceil(len(s) / 16000 / 0.08))
    sc = score_all(base["feats"]); pb = sc["deployed"]
    out = {"question_answer": base["answer"], "commit_frame": base["onset"], "commit_s": round(commit_s, 2), "question_end_s": round(q_end_s, 2),
           "anchor_s": round(anchor_s, 2), "requested_stim_at_s": round(requested_stim_at, 2),
           "stim_at_s": round(stim_at, 2), "timing_adjusted": stim_frame != requested_frame,
           "stim_frame": stim_frame, "stim_text": None, "requested_offset_s": float(offset_s),
           "actual_offset_s": round(stim_at - anchor_s, 2), "n_frames_query": base["n_frames_query"],
           "baseline_timeline": base["timeline"], "interrupted_timeline": inter["timeline"], "interrupted_answer": inter["answer"],
           "interrupted_onset": inter["onset"], "audio_b64": inter["audio_b64"],
           "compute_s": round(time.time() - started, 1),
           "baseline_cached": baseline_cached,
           "deployed_p_fail": round(pb["p_fail"], 4), "deployed_p_act": round(pb["p_act"], 4), "tier": tier,
           "stats": _yield_stats(inter["timeline"], base["timeline"], stim_frame, stim_len)}
    return out


PAGE_PATH = os.environ.get(
    "NVDA_PROBE_PAGE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_compare_page.html"),
)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            with STATUS_LOCK:
                status = dict(STATUS)
            if status["started_at"] is not None:
                status["elapsed_s"] = round(time.time() - status["started_at"], 1)
            body = json.dumps({"ok": MODEL is not None, **status}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        page_source = open(PAGE_PATH, encoding="utf-8").read()
        page = page_source.replace("__ARTINFO__", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("version", "read", "layer", "layers", "n_calib", "calib_oof_auc", "external_auc_cold")} for k, v in ART.items()}))
        page = page.replace("__STIMS__", json.dumps(STIM_CHOICES))
        self.wfile.write(page.encode())

    def do_POST(self):
        if not REQUEST_LOCK.acquire(blocking=False):
            body = json.dumps(
                {"error": "another evaluation is already running"}
            ).encode()
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        set_status(True, "decoding uploaded audio")
        try:
            from urllib.parse import parse_qs, urlparse
            u = urlparse(self.path or ""); qs = {k: v[0] for k, v in parse_qs(u.query).items()}
            n = int(self.headers.get("Content-Length", "0")); raw = self.rfile.read(n)
            cache_key = hashlib.sha256(raw).hexdigest()
            tier = qs.get("tier", "balanced")
            if tier not in ART["deployed"]["fail"]["thresholds"]:
                tier = "balanced"
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                f.write(raw); src = f.name
            wav = src + ".wav"
            subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav], capture_output=True)
            if u.path == "/interrupt":
                # body = question audio; stim = a flooract id (TTS'd) or "custom" (second upload cached via /stim)
                stim = qs.get("stim", "fa0000"); offset = float(qs.get("offset", "2.0"))
                if stim == "custom":
                    stim_wav = "/tmp/custom_stim.wav"
                else:
                    stim_wav = f"{DATA_DIR}/flooract_audio/{stim}.wav"
                out = run_interrupt(wav, stim_wav, offset, tier, cache_key)
                out["stim_text"] = STIMS.get(stim, "custom recording") if stim != "custom" else "custom recording"
            elif u.path == "/stim":
                subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "/tmp/custom_stim.wav"], capture_output=True)
                out = {
                    "ok": True,
                    "duration_s": trim_interruption_wav(
                        "/tmp/custom_stim.wav"
                    ),
                }
            else:
                out = run_turn(wav, tier, cache_key)
            body = json.dumps(out, ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers(); self.wfile.write(body)
        except Exception as e:
            self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())
        finally:
            set_status(False, "idle")
            REQUEST_LOCK.release()


if __name__ == "__main__":
    print("loading NVIDIA native streaming model…", flush=True)
    STREAMING_MODEL = load_streaming_model()
    MODEL = STREAMING_MODEL.model
    print(f"model ready; expert {'ENABLED' if HAVE_KEY else 'disabled (no OPENAI_API_KEY)'}; serving on 127.0.0.1:8090", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8090), H).serve_forever()
