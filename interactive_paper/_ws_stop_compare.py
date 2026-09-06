"""8cj: three-arm stop-then-ask comparison — is the post-stop lock ours?

Same flow on every arm:
  Q1 "What's the stock price of Nvidia today?" -> model speaks
  mid-speech: "Stop."  -> model yields
  4s silence -> Q2 "What's the current stock price of Google?"
  PRIMARY METRIC: does the head COMMIT (a non-listen chunk) within
  15s of Q2 ending?

Arms:
  vanilla — stock checkpoint sources, official cfg, bare loop
  off     — our demo, probe_on=0 (must equal vanilla; 8bl discipline)
  on      — our demo, probe_on=1 (escalation + relay path)

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_stop_compare.py::run --arm vanilla --n 6
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("stop-compare")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_dir("src", "/workspace/gate")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

ARMS = {
    "vanilla": ("wss://rhe9527--vanilla-duplex-voice.modal.run/"
                "62dc5cd9/ws",
                "https://rhe9527--vanilla-duplex-voice.modal.run/"
                "62dc5cd9"),
    "off": ("wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"
            "?tier=aggressive&probe_on=0&tracker=0",
            "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"),
    "on": ("wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"
           "?tier=aggressive&probe_on=1&tracker=0",
           "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"),
}
Q1 = "What's the stock price of Nvidia today?"
BARGE = "Stop."
Q2 = "What's the current stock price of Google?"


@app.function(image=img, secrets=[OPENAI], timeout=60 * 40)
async def run(arm: str = "vanilla", n: int = 6):
    import asyncio
    import time as _t
    import urllib.request

    import librosa
    import numpy as np
    import websockets
    import sys
    sys.path.insert(0, "/workspace/gate")
    import escalate

    ws_url, base = ARMS[arm]
    cli = escalate._client()
    wavs = []
    for i, txt in enumerate((Q1, BARGE, Q2)):
        r = cli.audio.speech.create(model="tts-1", voice="alloy",
                                    input=txt, response_format="wav")
        open(f"/tmp/t{i}.wav", "wb").write(r.content)
        au, _ = librosa.load(f"/tmp/t{i}.wav", sr=16000, mono=True)
        wavs.append(au.astype(np.float32))

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

    results = []
    for run_i in range(n):
        events = []
        FR = 2048
        try:
            async with websockets.connect(ws_url, max_size=2 ** 24,
                                          open_timeout=90) as sock:
                t0s = _t.time()

                async def reader():
                    async for m in sock:
                        e = json.loads(m)
                        events.append((round(_t.time() - t0s, 1), e))

                rt = asyncio.create_task(reader())

                async def send_pcm(x):
                    i16 = (x * 32767).clip(-32767, 32767).astype(
                        np.int16)
                    for i in range(0, len(i16), FR):
                        await sock.send(i16[i:i + FR].tobytes())
                        await asyncio.sleep(FR / 16000)

                async def silence(sec):
                    rng = np.random.default_rng(1)
                    for _ in range(int(sec * 16000 / FR)):
                        await sock.send(
                            (rng.normal(0, 0.003, FR) * 32767)
                            .clip(-32767, 32767)
                            .astype(np.int16).tobytes())
                        await asyncio.sleep(FR / 16000)

                def speaking():
                    """model audibly speaking: >=2 audio events."""
                    return sum(1 for _, e in events
                               if e.get("type") == "audio") >= 2

                await send_pcm(wavs[0])
                for _ in range(40):
                    await silence(1)
                    if speaking():
                        break
                t_stop = _t.time() - t0s
                await send_pcm(wavs[1])
                await silence(4)
                await send_pcm(wavs[2])
                t_q2 = _t.time() - t0s
                await silence(25)
                try:
                    await sock.send(json.dumps({"type": "stop"}))
                except Exception:
                    pass
                await asyncio.sleep(1)
                rt.cancel()
        except Exception as e:
            print(f"[{arm} run {run_i}] session error: {str(e)[:80]}")
            results.append({"error": str(e)[:80]})
            continue

        spoke_q1 = sum(1 for t, e in events
                       if e.get("type") == "audio" and t < t_stop)
        commits = [t for t, e in events if e.get("type") == "chunk"
                   and not e.get("listen") and t > t_q2]
        texts = "".join(e.get("v", "") for t, e in events
                        if e.get("type") == "text" and t > t_q2
                        and not e.get("relay"))
        relays = "".join(e.get("v", "") for t, e in events
                         if e.get("type") == "text" and t > t_q2
                         and e.get("relay"))
        commit_ok = bool(commits) and commits[0] <= t_q2 + 15
        lat = round(commits[0] - t_q2, 1) if commits else None
        print(f"[{arm} run {run_i}] q1_spoke={spoke_q1 > 0} "
              f"stop@{t_stop:.0f}s q2@{t_q2:.0f}s "
              f"commit={'%.1fs' % lat if lat is not None else 'NEVER'} "
              f"ok={commit_ok} text='{(texts + relays)[:90]}'")
        results.append({"commit_s": lat, "ok": commit_ok,
                        "q1_spoke": spoke_q1 > 0,
                        "text": (texts + relays)[:200]})

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\n===== STOP-COMPARE [{arm}] =====")
    print(f"post-stop follow-up commit (<=15s): {ok}/{len(results)}")
    print(f"latencies: {[r.get('commit_s') for r in results]}")
    return {"arm": arm, "ok": ok, "n": len(results),
            "lat": [r.get("commit_s") for r in results]}
