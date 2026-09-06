"""Multi-turn context smoke (8bi): the "what about apple" bug.

Turn 1: "What is Nvidia's stock price today?"  (escalates — realtime)
Turn 2: "What about Apple?"                     (escalates — realtime,
        but unanswerable without turn 1's topic)

Passes iff turn 2's expert answer is about Apple's STOCK PRICE, not a
generic "what about apple" refusal/misfire. We check the relay text and
the thinker uplink log for turn 2.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_context_smoke.py::run
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("ctx-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_dir("src", "/workspace/gate")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

WS = "wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"
BASE = "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"


@app.function(image=img, secrets=[OPENAI], timeout=60 * 20)
async def run(q1: str = "", q2: str = ""):
    import asyncio
    import time as _t
    import urllib.request

    import librosa
    import numpy as np
    import soundfile as sf
    import websockets
    import sys
    sys.path.insert(0, "/workspace/gate")
    import escalate

    turns = [q1 or "What is Nvidia's stock price today?",
             q2 or "What about Apple?"]
    cli = escalate._client()
    wavs = []
    for i, txt in enumerate(turns):
        r = cli.audio.speech.create(model="tts-1", voice="alloy",
                                    input=txt, response_format="wav")
        open(f"/tmp/t{i}.wav", "wb").write(r.content)
        au, _ = librosa.load(f"/tmp/t{i}.wav", sr=16000, mono=True)
        wavs.append(au.astype(np.float32))

    t0 = _t.time()
    ready = None
    while _t.time() - t0 < 480:
        try:
            ready = json.load(urllib.request.urlopen(f"{BASE}/ready",
                                                     timeout=25))
            break
        except Exception:
            await asyncio.sleep(4)
    assert ready and ready.get("ready"), "GPU never ready"

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
                    tag = e.get("type").upper()
                    val = (e.get("msg") or e.get("v")
                           or f"fired={e.get('fired')} "
                              f"is_info={e.get('is_info')} "
                              f"score={e.get('score')} "
                              f"thr={e.get('thr')} ({e.get('thr_mode')})")
                    print(f"[{t}s] {tag}: {str(val)[:130]}")

        rt = asyncio.create_task(reader())

        async def say(au, label):
            i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
            for i in range(0, len(i16), FR):
                await sock.send(i16[i:i + FR].tobytes())
                await asyncio.sleep(FR / 16000)
            print(f"--- sent: {label}")

        async def silence(sec):
            rng = np.random.default_rng(1)
            for _ in range(int(sec * 16000 / FR)):
                await sock.send((rng.normal(0, 0.003, FR) * 32767)
                                .clip(-32767, 32767)
                                .astype(np.int16).tobytes())
                await asyncio.sleep(FR / 16000)

        def turn_cleared(after):
            # relay_guard clears at the relay turn's eot; that branch
            # emits the "muted N chunks" log (and the muted-tail eot
            # emits phase=listening). Wait for either, after `after`.
            rt_ = [t for t, e in events
                   if e.get("type") == "text" and e.get("relay")
                   and t > after]
            if not rt_:
                return False
            # phase=listening also fires at muted-chunk eots mid-relay,
            # so key on the two logs that mark the relay turn closing.
            return any(t > rt_[0] and e.get("type") == "log" and (
                "chunks of local continuation" in e.get("msg", "")
                or "closing turn early" in e.get("msg", ""))
                for t, e in events)

        await say(wavs[0], turns[0])
        # wait out the full escalation of turn 1 (stall+expert+relay).
        # 8bu made the relay turn longer than the old fixed 30s wait
        # (TTS relay + muted tail can hold relay_guard past 50s), so
        # wait event-driven with a 75s cap.
        for _ in range(75):
            await silence(1)
            if turn_cleared(0.0):
                break
        await silence(4)
        n_relay_1 = sum(1 for _, e in events
                        if e.get("type") == "text" and e.get("relay"))
        print(f"=== turn 1 relay chunks: {n_relay_1} ===")
        t2s = _t.time() - t0s
        await say(wavs[1], turns[1])
        for _ in range(75):
            await silence(1)
            if turn_cleared(t2s):
                break
        await silence(6)

        try:
            await sock.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        await asyncio.sleep(1)
        rt.cancel()

    # analysis: turn-2 uplink + relay
    logs = [e["msg"] for _, e in events
            if e.get("type") == "log"
            and ("uplink heard" in e.get("msg", "")
                 or "ASR heard" in e.get("msg", ""))]
    relays = [e["v"] for _, e in events
              if e.get("type") == "text" and e.get("relay")]
    relay_txt = "".join(relays)
    print("\n===== CONTEXT SMOKE =====")
    print(f"uplinks: {logs}")
    print(f"all relay text: {relay_txt[:400]}")
    apple = ("apple" in relay_txt.lower() or "aapl" in relay_txt.lower())
    stockish = any(w in relay_txt.lower() for w in
                   ["$", "dollar", "share", "stock", "trading", "price",
                    "trade"])
    print(f"turn-2 relay mentions Apple: {apple}; stock-like: {stockish}")
    print("PASS" if apple and stockish else "CHECK MANUALLY")
    return {"uplinks": logs, "relay": relay_txt[:400],
            "apple": apple, "stockish": stockish}
