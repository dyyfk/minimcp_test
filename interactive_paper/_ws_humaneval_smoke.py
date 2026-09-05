"""Participant-path smoke for the human-eval study (8cb follow-up).

Drives the REAL participant pipeline — study-session assignment, the
/api/conversations/{id}/stream proxy, both blinded arms — with one
TTS'd realtime question per conversation. Checks, per arm:
  - ready arrives, and (debug mode) reports the expected tier
  - the model answers at all: first audio event latency ("不理人")
  - a `turn` event lands (server-side transcript finalized)

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_humaneval_smoke.py::run
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("humaneval-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_dir("src", "/workspace/gate")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

BASE = "https://rhe9527--minicpm-human-eval.modal.run"
QUESTION = "What is the weather in Seattle right now?"


@app.function(image=img, secrets=[OPENAI], timeout=60 * 20)
async def run():
    import asyncio
    import time as _t
    import urllib.request

    import librosa
    import numpy as np
    import websockets
    import sys
    sys.path.insert(0, "/workspace/gate")
    import escalate

    cli = escalate._client()
    r = cli.audio.speech.create(model="tts-1", voice="alloy",
                                input=QUESTION, response_format="wav")
    open("/tmp/q.wav", "wb").write(r.content)
    au, _ = librosa.load("/tmp/q.wav", sr=16000, mono=True)
    au = au.astype(np.float32)

    # readiness first, like the real page does
    t0 = _t.time()
    while _t.time() - t0 < 480:
        try:
            rd = json.load(urllib.request.urlopen(
                f"{BASE}/api/model/readiness", timeout=30))
            if rd.get("ready"):
                break
        except Exception:
            pass
        await asyncio.sleep(4)
    print(f"readiness ok in {_t.time() - t0:.0f}s")

    sess = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/api/study-sessions", data=b"{}",
        headers={"Content-Type": "application/json"})))
    convs = [c["conversation_id"]
             for c in sess["tasks"][0]["conversations"]]
    print(f"session {sess['session_id']}, task-1 conversations: {convs}")

    FR = 2048
    results = []
    for cid in convs:
        events = []
        t0s = _t.time()
        ws_url = (BASE.replace("https", "wss")
                  + f"/api/conversations/{cid}/stream")
        async with websockets.connect(ws_url, max_size=2 ** 24,
                                      open_timeout=120) as sock:
            async def reader():
                async for m in sock:
                    if isinstance(m, bytes):
                        continue
                    e = json.loads(m)
                    events.append((round(_t.time() - t0s, 1), e))
            rt = asyncio.create_task(reader())

            async def send_pcm(x):
                i16 = (x * 32767).clip(-32767, 32767).astype(np.int16)
                for i in range(0, len(i16), FR):
                    await sock.send(i16[i:i + FR].tobytes())
                    await asyncio.sleep(FR / 16000)

            async def silence(sec):
                rng = np.random.default_rng(1)
                for _ in range(int(sec * 16000 / FR)):
                    await sock.send(
                        (rng.normal(0, 0.003, FR) * 32767)
                        .clip(-32767, 32767).astype(np.int16).tobytes())
                    await asyncio.sleep(FR / 16000)

            # wait for ready (proxy connects upstream first)
            for _ in range(120):
                if any(e.get("type") == "ready" for _, e in events):
                    break
                await asyncio.sleep(1)
            ready = next((e for _, e in events
                          if e.get("type") == "ready"), None)
            t_ask = _t.time() - t0s
            await send_pcm(au)
            # wait event-driven: first audio after the question, then a
            # turn event (transcript finalized server-side); 90s cap.
            for _ in range(90):
                await silence(1)
                if any(e.get("type") == "turn" for t, e in events
                       if t > t_ask):
                    break
            try:
                await sock.send(json.dumps(
                    {"type": "finish_conversation"}))
            except Exception:
                pass
            await asyncio.sleep(1)
            rt.cancel()

        audio_ts = [t for t, e in events
                    if e.get("type") == "audio" and t > t_ask]
        turn_ok = any(e.get("type") == "turn" for _, e in events)
        dbg = (ready or {}).get("debug") or {}
        res = {
            "conversation": cid,
            "arm": dbg.get("model", "hidden"),
            "tier": dbg.get("tier"),
            "ready": ready is not None,
            "first_audio_after_ask_s": (round(audio_ts[0] - t_ask, 1)
                                        if audio_ts else None),
            "n_audio_events": len(audio_ts),
            "turn_event": turn_ok,
        }
        results.append(res)
        print("ARM RESULT:", json.dumps(res))

    print("\n===== HUMAN-EVAL SMOKE =====")
    ok = all(r["ready"] and r["first_audio_after_ask_s"] is not None
             and r["turn_event"] for r in results)
    for r in results:
        print(f"  {r['arm']:<14} tier={r['tier']} "
              f"first_audio={r['first_audio_after_ask_s']}s "
              f"audio_events={r['n_audio_events']} "
              f"turn={r['turn_event']}")
    print("PASS" if ok else "CHECK MANUALLY")
    return results
