"""Smoke-test the NATIVE duplex demo (demo_duplex.py) end to end.

Arms:
  local    — easy pool question at realtime pace, then silence. Expect:
             listen chunks while audio streams, a gate event (not fired),
             spoken answer chunks, end_of_turn.
  barge    — easy question; once the talker's audio starts, a SECOND
             question is spoken over it at realtime pace. Measures how
             many speak-chunks the native head emits before yielding the
             floor (no harness involved — this is the model's decision).
  escalate — hard question (escalated in the frozen traces). Expect:
             gate fired, thinker logs, [relay] text + audio later.

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run _ws_native_smoke.py::smoke --arm local
  modal run _ws_native_smoke.py::smoke --arm barge
  modal run _ws_native_smoke.py::smoke --arm escalate
"""
import json

import modal

app = modal.App("ws-native-smoke")
vol = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("websockets", "librosa", "soundfile", "numpy",
                    "pandas", "pyarrow"))

BASE = "https://rhe9527--gate-duplex-voice.modal.run/62dc5cd9"
WS = "wss://rhe9527--gate-duplex-voice.modal.run/62dc5cd9/ws"


@app.function(image=img, volumes={"/data": vol}, timeout=60 * 30)
async def smoke(arm: str = "local", tier: str = "balanced",
                qid: str = "", qid2: str = ""):
    import asyncio
    import time as _time
    import urllib.request

    import librosa
    import numpy as np
    import pandas as pd
    import websockets

    assert arm in ("local", "barge", "escalate", "stopword",
                   "phatic"), arm
    if arm in ("stopword", "phatic"):
        # speak ONLY floor-management utterances from silence — with the
        # 8bh act gate none of them may escalate (gate events must show
        # is_info=false or stay local; zero "escalating" phases)
        import glob as _glob
        if arm == "phatic":
            # 8cp: the user-hit false escalations — social openers,
            # channel checks, meta (en+zh); zero may escalate
            stims = [f"/data/phatic_audio/ph{i:04d}.wav"
                     for i in (0, 4, 18, 30, 36, 66)]
        else:
            stims = sorted(_glob.glob("/data/flooract_audio/fa0*.wav"))
            stims = stims[::37][:6] or stims[:6]
        t0 = _time.time()
        r = None
        while _time.time() - t0 < 480:
            try:
                r = json.load(urllib.request.urlopen(
                    f"{BASE}/ready", timeout=25))
                break
            except Exception:
                await asyncio.sleep(4)
        assert r and r.get("ready"), "GPU never became ready"
        gates, esc_phases = [], 0
        FRB = 2048
        rngb = np.random.default_rng(3)
        async with websockets.connect(
                f"{WS}?tier={tier}&probe_on=1", max_size=2 ** 24,
                open_timeout=60) as sock:
            t0s = _time.time()

            async def rd():
                nonlocal esc_phases
                async for m in sock:
                    e = json.loads(m)
                    t = round(_time.time() - t0s, 1)
                    if e.get("type") == "gate":
                        gates.append(e)
                        print(f"[{t}s] GATE score={e.get('score')} "
                              f"act={e.get('act')} "
                              f"is_info={e.get('is_info')} "
                              f"fired={e.get('fired')}")
                    elif e.get("type") == "phase" and \
                            e.get("v") == "escalating":
                        esc_phases += 1
                        print(f"[{t}s] !! ESCALATING")
                    elif e.get("type") == "text":
                        print(f"[{t}s] TEXT: {e['v'][:60]}")
            rt = asyncio.create_task(rd())
            for sp in stims:
                au, _sr = librosa.load(sp, sr=16000, mono=True)
                i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
                for i in range(0, len(i16), FRB):
                    await sock.send(i16[i:i + FRB].tobytes())
                    await asyncio.sleep(FRB / 16000)
                for _ in range(int(6 * 16000 / FRB)):   # 6 s silence
                    await sock.send((rngb.normal(0, 0.003, FRB) * 32767)
                                    .clip(-32767, 32767)
                                    .astype(np.int16).tobytes())
                    await asyncio.sleep(FRB / 16000)
                print(f"--- stim {sp.split('/')[-1]} done")
            try:
                await sock.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            await asyncio.sleep(1)
            rt.cancel()
        fired = sum(1 for g in gates if g.get("fired"))
        print(f"\n===== {arm.upper()} SUMMARY =====\n{len(stims)} stims, "
              f"{len(gates)} gate reads, fired={fired}, "
              f"escalating-phases={esc_phases}")
        return {"arm": arm, "n_stims": len(stims),
                "gates": len(gates), "fired": fired,
                "esc_phases": esc_phases}

    tr = pd.read_parquet("/data/frozen_v3_traces.parquet")
    import os

    def pick(df):
        for i in df["id"]:
            if os.path.exists(f"/data/audio_pool/{i}.wav"):
                return i
        raise RuntimeError("no wav for candidates")

    loc = (tr[tr["mode"] == "local"].groupby("id")
           .agg(score=("eot_score", "mean"), ok=("heard_ok", "mean"),
                aud=("audio_s", "mean")).reset_index())
    easy = qid or pick(loc[(loc["score"] < 0.2) & (loc["ok"] > 0.5)]
                       .sort_values("score"))
    esc = (tr[tr["mode"] == "escalated"].groupby("id")
           .agg(score=("eot_score", "mean")).reset_index())
    hard = qid2 or pick(esc.sort_values("score", ascending=False))
    main_id = hard if arm == "escalate" else easy
    over_id = easy if arm == "escalate" else hard  # barge: interrupt w/ hard
    print(f">>> arm={arm} main={main_id} overlap={over_id}")

    t0 = _time.time()
    r = None
    while _time.time() - t0 < 480:
        try:
            r = json.load(urllib.request.urlopen(f"{BASE}/ready",
                                                 timeout=25))
            break
        except Exception as e:
            print(f"  warm poll: {type(e).__name__}")
            await asyncio.sleep(4)
    assert r and r.get("ready"), "GPU never became ready"
    print(f"ready: {r}")

    def load(i):
        au, _ = librosa.load(f"/data/audio_pool/{i}.wav", sr=16000,
                             mono=True)
        return au.astype(np.float32)

    main_au = load(main_id)
    over_au = load(over_id)
    FR = 2048
    rng = np.random.default_rng(7)

    def to_frames(au):
        i16 = (au * 32767).clip(-32767, 32767).astype(np.int16)
        return [i16[i:i + FR].tobytes() for i in range(0, len(i16), FR)]

    def sil_frame():
        return ((rng.normal(0, 0.003, FR) * 32767)
                .clip(-32767, 32767).astype(np.int16).tobytes())

    events = []
    speak_chunks_after_overlap = []
    state = {"audio_started": None, "overlap_started": None,
             "turn_ends": 0, "fired": None, "relay_seen": False,
             "err": None}

    async with websockets.connect(
            f"{WS}?tier={tier}&probe_on=1", max_size=2 ** 24,
            open_timeout=60) as sock:

        async def reader():
            async for m in sock:
                e = json.loads(m)
                t = round(_time.time() - t0s, 1)
                events.append((t, e))
                k = e.get("type")
                if k == "audio" and state["audio_started"] is None:
                    state["audio_started"] = t
                    print(f"[{t}s] FIRST AUDIO")
                elif k == "chunk":
                    tag = "L" if e["listen"] else "S"
                    if (state["overlap_started"] is not None
                            and not e["listen"]):
                        speak_chunks_after_overlap.append(t)
                    if e.get("eot"):
                        state["turn_ends"] += 1
                        print(f"[{t}s] END OF TURN "
                              f"#{state['turn_ends']}")
                    print(f"[{t}s] chunk {tag} cost={e['cost']}s")
                elif k == "gate":
                    state["fired"] = e["fired"]
                    print(f"[{t}s] GATE score={e['score']} "
                          f"thr={e['thr']} fired={e['fired']}")
                elif k == "text":
                    if e.get("relay"):
                        state["relay_seen"] = True
                    print(f"[{t}s] TEXT{' [relay]' if e.get('relay') else ''}: "
                          f"{e['v'][:80]}")
                elif k == "log":
                    print(f"[{t}s] log: {e['msg'][:110]}")
                elif k == "error":
                    state["err"] = e["msg"]
                    print(f"[{t}s] ERROR: {e['msg']}")
                elif k in ("hello", "phase", "score"):
                    if k != "score":
                        print(f"[{t}s] {k}: "
                              f"{e.get('v') or e.get('mode', '')[:60]}")

        t0s = _time.time()
        rt = asyncio.create_task(reader())

        frames = to_frames(main_au)
        deadline = _time.time() + (150 if arm == "escalate" else 75)
        fi = 0
        overlap_frames = None
        while _time.time() < deadline:
            if overlap_frames:
                f = overlap_frames.pop(0)
                if not overlap_frames:
                    print(f"[{round(_time.time() - t0s, 1)}s] "
                          "overlap speech finished")
            elif fi < len(frames):
                f = frames[fi]
                fi += 1
            else:
                f = sil_frame()
            await sock.send(f)
            await asyncio.sleep(FR / 16000)

            if (arm == "barge" and overlap_frames is None
                    and state["audio_started"] is not None
                    and _time.time() - t0s > state["audio_started"] + 1.5):
                overlap_frames = to_frames(over_au)
                state["overlap_started"] = round(_time.time() - t0s, 1)
                print(f"[{state['overlap_started']}s] >>> BARGING IN "
                      f"({over_id}) over the talker's speech")

            # end conditions
            if arm == "local" and state["turn_ends"] >= 1 and \
                    fi >= len(frames):
                await asyncio.sleep(2)
                break
            if arm == "escalate" and state["relay_seen"] and \
                    state["turn_ends"] >= 1:
                await asyncio.sleep(2)
                break
            if arm == "barge" and state["overlap_started"] and \
                    state["turn_ends"] >= 1:
                await asyncio.sleep(2)
                break
            if state["err"]:
                break

        try:
            await sock.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        await asyncio.sleep(1)
        rt.cancel()

    chunks = [(t, e) for t, e in events if e.get("type") == "chunk"]
    pattern = "".join("L" if e["listen"] else "S" for _, e in chunks)
    costs = [e["cost"] for _, e in chunks]
    print("\n===== SUMMARY =====")
    print(f"arm={arm} main={main_id}")
    print(f"chunk pattern: {pattern}")
    print(f"n_chunks={len(chunks)} mean_cost="
          f"{np.mean(costs) if costs else -1:.2f}s "
          f"max={max(costs) if costs else -1:.2f}s")
    print(f"first_audio={state['audio_started']}s "
          f"turn_ends={state['turn_ends']} fired={state['fired']} "
          f"relay_seen={state['relay_seen']} err={state['err']}")
    if arm == "barge" and state["overlap_started"]:
        after = [t for t in speak_chunks_after_overlap
                 if t >= state["overlap_started"]]
        print(f"overlap at {state['overlap_started']}s; speak-chunks "
              f"after overlap: {len(after)} "
              f"(yield latency ≈ {after[-1] - state['overlap_started']:.1f}s"
              f" to last speak)" if after else "(model yielded instantly)")
    return {"arm": arm, "pattern": pattern, "fired": state["fired"],
            "turn_ends": state["turn_ends"],
            "relay_seen": state["relay_seen"], "err": state["err"]}
