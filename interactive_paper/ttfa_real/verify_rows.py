"""Spot-verify ttfa-v3 rows: timestamp ordering, pacing, audio presence.

Usage: python verify_rows.py <one-or-more .jsonl files>
Prints one line per row + a PASS/FAIL summary. Read-only.
"""
import json
import sys


def check(r):
    probs = []
    ts = r.get("ts") or {}

    def g(k):
        return ts.get(k)

    # ordering
    order = [("feed_start", "onset"), ("onset", "gate_read_done"),
             ("gate_read_done", "fire"), ("fire", "expert_req"),
             ("expert_req", "expert_resp"),
             ("wait_start", "relay_synth_start"),
             ("relay_synth_start", "relay_first_pcm"),
             ("relay_first_pcm", "relay_synth_end"),
             ("relay_synth_end", "session_end")]
    for a, b in order:
        if g(a) is not None and g(b) is not None and g(a) > g(b) + 1e-6:
            probs.append(f"order:{a}>{b}")
    # pacing: chunk gen_ret must be >= avail (can't answer before samples)
    for c in r.get("chunks", []):
        ci, is_real, avail, feed_done, gen_ret, listen, n_wav, eot = c
        if feed_done is not None and feed_done < avail - 0.005:
            probs.append(f"pacing:chunk{ci} fed {feed_done} < avail {avail}")
            break
    # expert prefix causal
    if r.get("fired") and r.get("expert_input_s") is not None:
        if r["expert_input_s"] > r["input_s"] + 1e-6:
            probs.append("expert_prefix>input")
        if g("fire") is not None and r["expert_input_s"] > (g("fire")) + 1.0:
            probs.append("expert_prefix_beyond_fire_arrival")
    # ttfa consistency
    if r.get("ttfa_answer_s") is not None:
        want = round(g("first_answer_pcm") - ts["input_end"], 3)
        if abs(want - r["ttfa_answer_s"]) > 0.002:
            probs.append("ttfa_mismatch")
    # audio bookkeeping
    if r.get("status") == "completed":
        af = r.get("audio_files") or {}
        if r.get("fired") and "relay" not in af:
            probs.append("completed_fired_no_relay_wav")
        if not r.get("fired") and "local" not in af:
            probs.append("completed_local_no_wav")
    return probs


n = bad = 0
for p in sys.argv[1:]:
    for ln in open(p, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        n += 1
        probs = check(r)
        ts = r.get("ts") or {}
        af = {k: (v or {}).get("seconds") for k, v in
              (r.get("audio_files") or {}).items()}
        print(f"{r['id']} {r.get('arm')} status={r.get('status')} "
              f"fired={r.get('fired')} score={r.get('score')} "
              f"input_s={r.get('input_s')} ttfa={r.get('ttfa_answer_s')} "
              f"any={r.get('ttfa_any_s')} lagmax={r.get('feed_lag_max_ms')}ms "
              f"audio={af} "
              f"{'!! ' + ';'.join(probs) if probs else 'OK'}")
        if probs:
            bad += 1
print(f"\n{n} rows, {bad} with problems")
