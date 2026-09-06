"""Spontaneous-speech probe: stream mic-like ambient noise, no speech.

User report (2026-09-05, phone session): the model starts lecturing
("Hyperloop test track... SpaceX") with nobody talking — ASR of the
turn snapshot comes back empty. Hypothesis: head-native commit-on-
noise (audio-LM hallucination on an open mic), not our escalation
machinery. Decisive check: same noise into BOTH arms —
  vanilla (stock config, zero gate machinery) vs gate demo.
If both commit spontaneously at similar rates, it is the head.

Streams `SECONDS` of low-RMS noise (mic floor ~0.003 plus occasional
breath-like bumps), counts talker commits + spoken text.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_silence_probe.py::run --arm vanilla
  modal run _ws_silence_probe.py::run --arm gate
"""
import json

import modal

app = modal.App("silence-probe")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "numpy"))

URLS = {
    "vanilla": ("wss://rhe9527--vanilla-duplex-voice.modal.run/62dc5cd9/ws",
                "https://rhe9527--vanilla-duplex-voice.modal.run/62dc5cd9"),
    "gate": ("wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"
             "?tier=aggressive&probe_on=1",
             "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"),
}
SECONDS = 100


@app.function(image=img, timeout=60 * 20)
async def run(arm: str = "vanilla"):
    import asyncio
    import time as _t
    import urllib.request

    import numpy as np
    import websockets

    ws_url, base = URLS[arm]
    t0 = _t.time()
    while _t.time() - t0 < 480:
        try:
            rd = json.load(urllib.request.urlopen(f"{base}/ready",
                                                  timeout=25))
            if rd.get("ready"):
                break
        except Exception:
            pass
        await asyncio.sleep(4)

    events = []
    FR = 2048
    async with websockets.connect(ws_url, max_size=2 ** 24,
                                  open_timeout=60) as sock:
        t0s = _t.time()

        async def reader():
            async for m in sock:
                e = json.loads(m)
                t = round(_t.time() - t0s, 1)
                events.append((t, e))
                if e.get("type") == "text":
                    print(f"[{t}s] TEXT: {str(e.get('v'))[:80]}")
                elif e.get("type") in ("gate", "log"):
                    print(f"[{t}s] {e['type'].upper()}: "
                          f"{str(e.get('msg') or e)[:110]}")

        rt = asyncio.create_task(reader())
        rng = np.random.default_rng(7)
        for i in range(int(SECONDS * 16000 / FR)):
            # mic floor noise; every ~15s a slightly louder breath-like
            # bump (still far below speech RMS ~0.03)
            amp = 0.003 if (i % 120) else 0.008
            x = rng.normal(0, amp, FR)
            await sock.send((x * 32767).clip(-32767, 32767)
                            .astype(np.int16).tobytes())
            await asyncio.sleep(FR / 16000)
        try:
            await sock.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        await asyncio.sleep(1)
        rt.cancel()

    # ---- analysis ----------------------------------------------------
    chunks = [(t, e) for t, e in events if e.get("type") == "chunk"]
    commits = []
    prev = True
    for t, e in chunks:
        if prev and not e.get("listen"):
            commits.append(t)
        prev = e.get("listen")
    texts = "".join(e.get("v", "") for _, e in events
                    if e.get("type") == "text")
    n_audio = sum(1 for _, e in events if e.get("type") == "audio")
    print(f"\n===== SILENCE PROBE ({arm}, {SECONDS}s noise) =====")
    print(f"spontaneous commits: {len(commits)} at {commits}")
    print(f"audio events: {n_audio}")
    print(f"spoken text: {texts[:300]}")
    return {"arm": arm, "commits": commits, "n_audio": n_audio,
            "text": texts[:300]}
