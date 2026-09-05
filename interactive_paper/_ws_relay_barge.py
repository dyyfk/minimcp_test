"""8ce barge-in smoke: interrupt the relay mid-delivery.

Turn 1 escalates (realtime stock question). While the paced relay is
being delivered, the user barges in with a new info question. Expect:
  - relay audio frames arrive paced (~1/s), not as one burst
  - on the barge-in commit: "user takes the floor mid-relay" log,
    remaining frames dropped (audio events stop within ~2s)
  - the interrupting question gets its own turn (gate read, escalation)

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_relay_barge.py::run
"""
import json

import modal

from modal_app import OPENAI

app = modal.App("relay-barge-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "openai")
       .add_local_dir("src", "/workspace/gate")
       .add_local_file("modal_app.py", "/root/modal_app.py"))

WS = "wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"
BASE = "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"

Q1 = "What is Nvidia's stock price today?"
# short interjection to trigger the cut, then a PAUSE, then a fresh
# question — reproduces the user's "deaf after barge" flow (a long
# continuous barge masks the bug because the head eventually commits).
BARGE = "Stop, stop."
FOLLOWUP = "What is Apple's stock price today?"


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
                if e.get("type") == "audio":
                    n = len(e.get("pcm", "")) * 3 // 4 // 2
                    print(f"[{t}s] AUDIO {n / 24000:.1f}s")
                elif e.get("type") in ("gate", "log", "text"):
                    tag = e["type"].upper()
                    val = (e.get("msg") or e.get("v")
                           or f"fired={e.get('fired')} "
                              f"is_info={e.get('is_info')} "
                              f"score={e.get('score')}")
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
        # wait for the paced relay to start delivering (>=2 audio
        # events after the relay text), then barge in mid-delivery
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
        t_barge = _t.time() - t0s
        await send_pcm(wavs[1])
        print(f"--- barged at {t_barge:.1f}s: {BARGE}")
        # pause after the short interjection, THEN the fresh question —
        # this is the flow that goes deaf without the cut-path reset.
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
    rts = [t for t, e in events
           if e.get("type") == "text" and e.get("relay")]
    audio1 = [t for t, e in events if e.get("type") == "audio"
              and rts and rts[0] <= t < t_barge]
    cut_logs = [t for t, e in events if e.get("type") == "log"
                and "takes the floor mid-relay" in e.get("msg", "")]
    # recovery: after the CUT (not the fragile follow-up timestamp —
    # the head can commit before it), did the model produce a fresh
    # turn for the follow-up? a new gate read + relay audio.
    t_cut = cut_logs[0] if cut_logs else t_barge
    gates_follow = [(t, e.get("fired"), e.get("is_info"), e.get("score"))
                    for t, e in events if e.get("type") == "gate"
                    and t > t_cut]
    relay_follow = [e.get("v") for t, e in events
                    if e.get("type") == "text" and e.get("relay")
                    and t > t_cut]
    audio_follow = [t for t, e in events if e.get("type") == "audio"
                    and t > t_cut + 1.0]
    asr_follow = [e.get("msg") for t, e in events
                  if e.get("type") == "log" and t > t_cut
                  and "ASR heard" in e.get("msg", "")]
    print("\n===== RELAY BARGE SMOKE =====")
    print(f"relay text at: {rts[:1]}, paced audio events pre-barge: "
          f"{len(audio1)} at {audio1}")
    print(f"cut log at: {cut_logs}")
    # is the chunk loop even alive after the cut? score events carry a
    # per-chunk listen flag. all-listen => head stuck listening (the
    # intrinsic listen-lock, same as turn 1's ~1/6); none => loop dead;
    # mixed/speak => head committed.
    scores_after = [(t, e.get("listen")) for t, e in events
                    if e.get("type") == "score" and t > t_cut]
    n_listen = sum(1 for _, ls in scores_after if ls)
    n_speak = sum(1 for _, ls in scores_after if ls is False)
    print(f"--- post-follow-up recovery ---")
    print(f"post-cut score events: {len(scores_after)} "
          f"(listen={n_listen}, speak={n_speak})")
    print(f"ASR after follow-up: {asr_follow}")
    print(f"gate reads after follow-up: {gates_follow[:4]}")
    print(f"relay text after follow-up: {relay_follow[:1]}")
    print(f"audio events after follow-up: {len(audio_follow)}")
    cut_ok = bool(cut_logs) and len(audio1) >= 2
    # "not deaf" = after the cut the head committed to a turn (any gate
    # read = a listen->speak flip) AND produced audio — whether it
    # escalates (relay) or answers locally is the gate's call and
    # subject to is_info variance on the "stop"-prefixed utterance.
    recover_ok = (len(gates_follow) >= 1 and len(audio_follow) >= 2)
    print(f"cut PASS: {cut_ok} | recovery PASS: {recover_ok}")
    print("PASS" if cut_ok and recover_ok else "CHECK MANUALLY")
    return {"cut": cut_logs, "gates_follow": gates_follow,
            "relay_follow": relay_follow[:1],
            "audio_follow": len(audio_follow)}
