"""v1 nominal-clock TTFA RECONSTRUCTION, calibrated + validated against
the ttfa-v3 measured sample. Output is an ESTIMATE and must always be
labeled "recon", never "measured".

Basis: the duplex head is chunk-synchronous — its listen/speak
trajectory depends on the audio content and chunk index, not on wall
clock — so v1's onset_chunk maps onto the real-time (nominal) axis:
chunk i's samples arrive at min((i+1)*1s, audio_s) (silence chunks
continue the 1/s schedule). Real wall-clock components that v1 DID
record (expert_latency_s = ASR + gpt-5.5) are used as-is.

Model per v1 row:
  onset_nom   = avail(onset_chunk) + d_proc
  local  TTFA = onset_nom - audio_s
  fired  TTFA = onset_nom + max(wait_off, expert_latency_s) + c_relay
                - audio_s
where
  d_proc   = onset-chunk processing latency  (calibrated, measured)
  wait_off = fire->EOT offset: max(stall_ms/1000, c_wait)  (calibrated)
  c_relay  = expert-ready -> first relay PCM  (calibrated, measured)

Calibration + validation data: results/<run_id>/{pool}/{arm}.jsonl.shard*
(the ttfa-v3 measured sample; same frozen artifacts, same query pools).

Usage:
  python recon_ttfa.py --v1 v1_logs --measured results/ttfa1 \
      --samples sample_ids.json --out recon_ttfa1
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

POOLS = ["frozen", "striviaqa", "swebq", "sllama", "sdqa"]
ARMS = ["local", "conservative", "balanced", "aggressive", "always"]
V1FILE = {"local": "never", "conservative": "conservative_tts",
          "balanced": "balanced_tts", "aggressive": "aggressive_tts",
          "always": "always_tts"}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def dist(xs):
    if not xs:
        return None
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 3),
            "p50": round(pct(xs, 50), 3), "p95": round(pct(xs, 95), 3),
            "p99": round(pct(xs, 99), 3),
            "min": round(min(xs), 3), "max": round(max(xs), 3)}


def med(xs):
    return pct(xs, 50)


def avail(onset_chunk, audio_s, n_real):
    if onset_chunk < n_real:
        return min((onset_chunk + 1) * 1.0, audio_s)
    return audio_s + (onset_chunk - n_real + 1) * 1.0


def load_measured(root):
    rows = {}
    for pool in POOLS:
        for arm in ARMS:
            for p in glob.glob(os.path.join(root, pool,
                                            f"{arm}.jsonl.shard*")):
                for ln in open(p, encoding="utf-8"):
                    if ln.strip():
                        r = json.loads(ln)
                        if r.get("attempt", 1) == 1:
                            rows[(pool, arm, r["id"])] = r
    return rows


def load_v1(root):
    rows = {}
    for pool in POOLS:
        for arm in ARMS:
            for p in sorted(glob.glob(os.path.join(
                    root, pool, f"{V1FILE[arm]}.jsonl.shard*"))):
                for ln in open(p, encoding="utf-8"):
                    if ln.strip():
                        r = json.loads(ln)
                        rows[(pool, arm, r["id"])] = r   # last wins (dedupe)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--measured", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meas = load_measured(args.measured)
    v1 = load_v1(args.v1)
    samples = json.load(open(args.samples, encoding="utf-8"))

    # ---- calibration from measured rows -------------------------------
    d_proc, waitoffs, crelays = [], [], []
    for r in meas.values():
        ts = r.get("ts") or {}
        if r.get("onset_chunk") is not None and ts.get("onset") is not None:
            n_real = r.get("n_real_chunks") or math.ceil(r["input_s"])
            d_proc.append(ts["onset"] -
                          avail(r["onset_chunk"], r["input_s"], n_real))
        if r.get("fired"):
            if ts.get("fire") is not None and ts.get("wait_start") is not None:
                waitoffs.append(ts["wait_start"] - ts["fire"])
            if ts.get("relay_first_pcm") is not None \
                    and ts.get("wait_start") is not None:
                trigger = max(ts.get("expert_resp") or 0, ts["wait_start"])
                crelays.append(ts["relay_first_pcm"] - trigger)
    calib = {"d_proc_med": round(med(d_proc), 3) if d_proc else None,
             "d_proc": dist(d_proc), "c_wait_med": (round(med(waitoffs), 3)
                                                    if waitoffs else None),
             "c_wait": dist(waitoffs),
             "c_relay_med": round(med(crelays), 3) if crelays else None,
             "c_relay": dist(crelays)}
    dp = calib["d_proc_med"] or 0.3
    cw = calib["c_wait_med"] or 3.0
    cr = calib["c_relay_med"] or 0.8

    # ---- reconstruct every v1 row --------------------------------------
    recon = {}
    counts = defaultdict(lambda: defaultdict(int))
    for (pool, arm, qid), r in v1.items():
        c = counts[(pool, arm)]
        c["n"] += 1
        if r.get("onset_chunk") is None:
            c["non_commit"] += 1
            continue
        n_real = r.get("n_chunks") or math.ceil(r["audio_s"])
        onset_nom = avail(r["onset_chunk"], r["audio_s"], n_real) + dp
        if not r.get("fired"):
            t = onset_nom - r["audio_s"]
        else:
            if r.get("expert_latency_s") is None:
                c["expert_missing"] += 1
                continue
            wait_off = max((r.get("stall_ms") or 0) / 1000.0, cw)
            t = (onset_nom + max(wait_off, r["expert_latency_s"]) + cr
                 - r["audio_s"])
        recon[(pool, arm, qid)] = {"ttfa_recon_s": round(t, 3),
                                   "fired": bool(r.get("fired")),
                                   "onset_chunk": r["onset_chunk"],
                                   "audio_s": r["audio_s"]}
        c["recon_ok"] += 1

    # ---- validation: measured sample vs recon (same query ids) ---------
    validation = {}
    for pool in POOLS:
        for arm in ARMS:
            pairs = []
            for qid in samples.get(pool, {}).get("ids", []):
                m = meas.get((pool, arm, qid))
                rr = recon.get((pool, arm, qid))
                if m and rr and m.get("status") == "completed" \
                        and m.get("ttfa_answer_s") is not None:
                    pairs.append((m["ttfa_answer_s"], rr["ttfa_recon_s"],
                                  m["fired"], rr["fired"], qid))
            if not pairs:
                continue
            diffs = [r_ - m_ for m_, r_, *_ in pairs]
            same_route = [p for p in pairs if p[2] == p[3]]
            diffs_same = [r_ - m_ for m_, r_, mf, rf, _ in same_route]
            validation[f"{pool}/{arm}"] = {
                "n_pairs": len(pairs),
                "n_same_route": len(same_route),
                "bias_s": round(sum(diffs) / len(diffs), 3),
                "mae_s": round(sum(abs(d) for d in diffs) / len(diffs), 3),
                "bias_same_route_s": (round(sum(diffs_same) /
                                            len(diffs_same), 3)
                                      if diffs_same else None),
                "mae_same_route_s": (round(sum(abs(d) for d in diffs_same) /
                                           len(diffs_same), 3)
                                     if diffs_same else None),
                "measured": dist([m_ for m_, *_ in pairs]),
                "recon_same_ids": dist([r_ for _, r_, *_ in pairs]),
            }

    # ---- full-pool recon tables ----------------------------------------
    tables = {}
    for pool in POOLS:
        tables[pool] = {}
        for arm in ARMS:
            rs = [v for (p2, a2, _), v in recon.items()
                  if p2 == pool and a2 == arm]
            if not rs:
                continue
            c = counts[(pool, arm)]
            all_t = [v["ttfa_recon_s"] for v in rs]
            tables[pool][arm] = {
                "n_v1_rows": c["n"], "n_recon": c["recon_ok"],
                "n_non_commit": c["non_commit"],
                "n_expert_missing": c["expert_missing"],
                "escalation_rate": round(
                    sum(1 for v in rs if v["fired"]) / len(rs), 4),
                "n_early_response": sum(1 for t in all_t if t < 0),
                "ttfa_recon": dist(all_t),
                "ttfa_recon_local_path": dist(
                    [v["ttfa_recon_s"] for v in rs if not v["fired"]]),
                "ttfa_recon_escalated_path": dist(
                    [v["ttfa_recon_s"] for v in rs if v["fired"]]),
            }

    out = {"label": "RECONSTRUCTION (nominal chunk clock, calibrated); "
                    "NOT measured TTFA",
           "calibration": calib,
           "constants_used": {"d_proc": dp, "c_wait": cw, "c_relay": cr},
           "validation_vs_measured_sample": validation,
           "recon_tables": tables}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    md = ["| pool | arm | n | esc% | recon mean | P50 | P95 | bias vs meas | MAE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pool in POOLS:
        for arm in ARMS:
            t = tables.get(pool, {}).get(arm)
            if not t:
                continue
            v = validation.get(f"{pool}/{arm}", {})
            d = t["ttfa_recon"] or {}
            md.append(f"| {pool} | {arm} | {t['n_recon']} "
                      f"| {round(t['escalation_rate'] * 100, 1)} "
                      f"| {d.get('mean')} | {d.get('p50')} | {d.get('p95')} "
                      f"| {v.get('bias_s', '')} | {v.get('mae_s', '')} |")
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"calib: d_proc={dp} c_wait={cw} c_relay={cr} "
          f"(from {len(d_proc)}/{len(waitoffs)}/{len(crelays)} measured rows)")
    print(f"wrote {args.out}.json / .md")


if __name__ == "__main__":
    main()
