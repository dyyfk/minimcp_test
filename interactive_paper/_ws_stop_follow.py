"""8ci: the user's exact stop-then-ask script, verified end to end.

  User:  "What's the stock price of Nvidia today?"
  Model: stall ... relays NVDA quote
  User:  "Stop."                 (single word, mid-relay)
  Model: stops
  User:  "What's the current stock price of Google?"
  Model: stall ... relays GOOGL quote        <- THE ASSERTION

PASS iff after the stop the follow-up question (a) gets a fired gate
read, and (b) a relay text mentioning Google/Alphabet/GOOG arrives.
Everything else (cut fired or relay simply finished, act reads, listen
flags) is logged for diagnosis.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_stop_follow.py::run
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("stop-follow-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_dir("src", "/workspace/gate")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

WS = "wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"
BASE = "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"

Q1 = "What's the stock price of Nvidia today?"
BARGE = "Stop."
FOLLOWUP = "What's the current stock price of Google?"


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
    wavs = []
    for i, txt in enumerate((Q1, BARGE, FOLLOWUP)):
        r = cli.audio.speech.create(model="tts-1", voice="alloy",
                                    input=txt, response_format="wav")
        open(f"/tmp/t{i}.wav", "wb").write(r.content)
        au, _ = librosa.load(f"/tmp/t{i}.wav", sr=16000, mono=True)
        wavs.append(au.astype(np.float32))

    t0 = _t.time()
    while _t.time() - t0 < 480:
        try:
            rd = json.load(urllib.request.urlopen(f"{BASE}/ready",
                                                  timeout=25))
            if rd.get("ready"):
                break
        except Exception:
            pass
        await asyncio.sleep(4)

    events = []
    FR = 2048
    async with websockets.connect(f"{WS}?tier=aggressive&probe_on=1&tracker=0",
                                  max_size=2 ** 24,
                                  open_timeout=60) as sock:
        t0s = _t.time()

        async def reader():
            async for m in sock:
                e = json.loads(m)
                t = round(_t.time() - t0s, 1)
                events.append((t, e))
                if e.get("type") in ("gate", "log", "text"):
                    tag = e["type"].upper()
                    val = (e.get("msg") or e.get("v")
                           or f"fired={e.get('fired')} "
                              f"is_info={e.get('is_info')} "
                              f"score={e.get('score')} "
                              f"act={e.get('act')}")
                    print(f"[{t}s] {tag}: {str(val)[:120]}")

        rt = asyncio.create_task(reader())

        async def send_pcm(x):
            i16 = (x * 32767).clip(-32767, 32767).astype(np.int16)
            for i in range(0, len(i16), FR):
                await sock.send(i16[i:i + FR].tobytes())
                await asyncio.sleep(FR / 16000)

        async def silence(sec):
            rng = np.random.default_rng(1)
            for _ in range(int(sec * 16000 / FR)):
                await sock.send((rng.normal(0, 0.003, FR) * 32767)
                                .clip(-32767, 32767)
                                .astype(np.int16).tobytes())
                await asyncio.sleep(FR / 16000)

        await send_pcm(wavs[0])
        print(f"--- sent: {Q1}")

        def relay_started():
            rts = [t for t, e in events
                   if e.get("type") == "text" and e.get("relay")]
            if not rts:
                return False
            return sum(1 for t, e in events
                       if e.get("type") == "audio" and t > rts[0]) >= 2
        for _ in range(60):
            await silence(1)
            if relay_started():
                break
        t_stop = _t.time() - t0s
        await send_pcm(wavs[1])
        print(f"--- said at {t_stop:.1f}s: {BARGE}")
        await silence(4)
        t_follow = _t.time() - t0s
        await send_pcm(wavs[2])
        print(f"--- follow-up at {t_follow:.1f}s: {FOLLOWUP}")
        await silence(30)
        try:
            await sock.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        await asyncio.sleep(1)
        rt.cancel()

    # ---- analysis ----------------------------------------------------
    def sel(typ, after=0.0):
        return [(t, e) for t, e in events
                if e.get("type") == typ and t > after]

    gates_follow = [(t, e.get("fired"), e.get("is_info"),
                     e.get("score"), e.get("act"))
                    for t, e in sel("gate", t_follow)]
    relay_follow = [e.get("v", "") for t, e in sel("text", t_follow)
                    if e.get("relay")]
    texts_follow = [e.get("v", "") for t, e in sel("text", t_follow)]
    audio_follow = len(sel("audio", t_follow))
    cut_logs = [t for t, e in sel("log")
                if "takes the floor" in e.get("msg", "")]
    close_logs = [t for t, e in sel("log")
                  if "playback done" in e.get("msg", "")]
    scores_follow = [(t, e.get("listen")) for t, e in
                     sel("score", t_follow)]
    n_listen = sum(1 for _, ls in scores_follow if ls)
    n_speak = sum(1 for _, ls in scores_follow if ls is False)
    google_ok = any(("Google" in r or "Alphabet" in r or "GOOG" in r)
                    for r in relay_follow)
    # 8ck: playback continuity — simulate the client cursor (frames
    # schedule back-to-back from arrival); a frame arriving after the
    # cursor means the buffer ran dry = audible stutter.
    aud = [(t, len(e.get("pcm", "")) * 3 // 4 // 2 / 24000.0)
           for t, e in events if e.get("type") == "audio"]
    stalls, cursor = [], None
    for t, d in aud:
        # 0.15-5s late = mid-delivery stutter; >5s = the ordinary
        # silence between turns (not a stall)
        if cursor is not None and 0.15 < t - cursor <= 5.0:
            stalls.append(round(t - cursor, 2))
        cursor = (t if cursor is None else max(cursor, t)) + d
    print(f"playback stalls 0.15-5s: {len(stalls)} "
          f"({sum(stalls):.1f}s total) {stalls[:8]}")
    print("\n===== STOP-FOLLOW SMOKE =====")
    print(f"stop at {t_stop:.1f}s -> cut logs {cut_logs} / "
          f"relay-close logs {close_logs}")
    print(f"follow-up at {t_follow:.1f}s")
    print(f"post-follow-up chunks: listen={n_listen} speak={n_speak}")
    print(f"gate reads after follow-up: {gates_follow[:4]}")
    print(f"model text after follow-up: "
          f"{' | '.join(texts_follow)[:200]}")
    print(f"relay after follow-up: {relay_follow[:1]}")
    fired = any(f for _, f, _, _, _ in gates_follow)
    print(f"follow-up fired: {fired}  google relay: {google_ok}  "
          f"audio events: {audio_follow}")
    # 8ck: PASS requires the relay AUDIO to actually ship (a text-only
    # PASS masked a dead pacer thread once)
    print("PASS" if (fired and google_ok and audio_follow >= 3)
          else "FAIL")
    return {"fired": fired, "google": google_ok,
            "gates": gates_follow[:4], "cut": cut_logs,
            "n_listen": n_listen, "n_speak": n_speak}
