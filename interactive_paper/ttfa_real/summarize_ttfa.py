"""Aggregate the ttfa-v3 run logs into per pool x arm TTFA tables.

Usage (cwd = interactive_paper/ttfa_real):
    python summarize_ttfa.py --results results/ttfa1 --out summary_ttfa1.json

Reads every {pool}/{arm}.jsonl.shard* under --results (smoke files are
ignored). Emits JSON + a markdown table. Never touches the raw logs.

Definitions (must match modal_ttfa_bench.py):
  TTFA_server = ts.first_answer_pcm - ts.input_end   (signed seconds)
  valid TTFA  = status == "completed" and ttfa_answer_s is not None
  early       = valid and ttfa_answer_s < 0
Also verifies per-row timestamp ordering and reports violations.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

ARMS = ["local", "conservative", "balanced", "aggressive", "always"]
POOLS = ["frozen", "striviaqa", "swebq", "sllama", "sdqa"]


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
    return {"n": len(xs),
            "mean": round(sum(xs) / len(xs), 3),
            "p50": round(pct(xs, 50), 3),
            "p95": round(pct(xs, 95), 3),
            "p99": round(pct(xs, 99), 3),
            "min": round(min(xs), 3), "max": round(max(xs), 3)}


ORDER = [  # (a, b): ts[a] <= ts[b] whenever both present
    ("feed_start", "onset"), ("onset", "gate_read_done"),
    ("gate_read_done", "fire"), ("fire", "expert_req"),
    ("expert_req", "expert_resp"), ("wait_start", "relay_synth_start"),
    ("relay_synth_start", "relay_first_pcm"),
    ("relay_first_pcm", "relay_synth_end"),
    ("relay_synth_end", "session_end")]


def check_row(r):
    bad = []
    ts = r.get("ts") or {}
    for a, b in ORDER:
        if ts.get(a) is not None and ts.get(b) is not None \
                and ts[a] > ts[b] + 1e-6:
            bad.append(f"{a}>{b}")
    if r.get("status") == "completed" and r.get("ttfa_answer_s") is None:
        bad.append("completed_without_ttfa")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = defaultdict(dict)   # (pool, arm) -> id -> row (last wins per attempt)
    for pool in POOLS:
        for arm in ARMS:
            for p in sorted(glob.glob(
                    os.path.join(args.results, pool, f"{arm}.jsonl.shard*"))):
                for ln in open(p, encoding="utf-8"):
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    key = (r["id"], r.get("attempt", 1))
                    rows[(pool, arm)][key] = r

    summary = {"pools": {}, "order_violations": [], "pacing": {}}
    for pool in POOLS:
        pool_out = {"arms": {}, "paired": {}}
        valid_ids = {}
        any_data = False
        for arm in ARMS:
            rs = [r for (qid, att), r in rows[(pool, arm)].items()
                  if att == 1]
            if not rs:
                continue
            any_data = True
            st = Counter(r["status"] for r in rs)
            for r in rs:
                v = check_row(r)
                if v:
                    summary["order_violations"].append(
                        {"pool": pool, "arm": arm, "id": r["id"],
                         "bad": v})
            ok = [r for r in rs if r["status"] == "completed"
                  and r.get("ttfa_answer_s") is not None]
            ttfa = [r["ttfa_answer_s"] for r in ok]
            ttfa_any = [r["ttfa_any_s"] for r in rs
                        if r.get("ttfa_any_s") is not None]
            onset = [r for r in rs if r.get("onset_chunk") is not None]
            fired = [r for r in rs if r.get("fired")]
            lagmax = [r.get("feed_lag_max_ms", 0) for r in rs]
            arm_out = {
                "n_sessions": len(rs),
                "status_counts": dict(st),
                "n_valid_ttfa": len(ok),
                "n_missing_or_failed": len(rs) - len(ok),
                "n_early_response": sum(1 for t in ttfa if t < 0),
                "escalation_rate": (round(len(fired) / len(onset), 4)
                                    if onset else None),
                "ttfa_answer": dist(ttfa),
                "ttfa_first_any": dist(ttfa_any),
                "feed_lag_max_ms": dist([float(x) for x in lagmax]),
            }
            if arm not in ("local", "always"):
                loc = [r["ttfa_answer_s"] for r in ok if not r["fired"]]
                esc = [r["ttfa_answer_s"] for r in ok if r["fired"]]
                arm_out["ttfa_local_path"] = dist(loc)
                arm_out["ttfa_escalated_path"] = dist(esc)
            pool_out["arms"][arm] = arm_out
            valid_ids[arm] = {r["id"] for r in ok}
        if any_data and len(valid_ids) >= 2:
            common = set.intersection(*valid_ids.values())
            pool_out["paired"]["n_common_valid_ids"] = len(common)
            for arm in valid_ids:
                ttfa = [r["ttfa_answer_s"]
                        for (qid, att), r in rows[(pool, arm)].items()
                        if att == 1 and qid in common
                        and r.get("ttfa_answer_s") is not None]
                pool_out["paired"][arm] = dist(ttfa)
        if any_data:
            summary["pools"][pool] = pool_out

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # markdown table
    md = ["| pool | arm | n | valid | fail | early | esc% | mean | P50 | P95 | P99 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pool, po in summary["pools"].items():
        for arm in ARMS:
            a = po["arms"].get(arm)
            if not a:
                continue
            d = a["ttfa_answer"] or {}
            er = a["escalation_rate"]
            md.append(
                f"| {pool} | {arm} | {a['n_sessions']} | {a['n_valid_ttfa']} "
                f"| {a['n_missing_or_failed']} | {a['n_early_response']} "
                f"| {'' if er is None else round(er * 100, 1)} "
                f"| {d.get('mean', '')} | {d.get('p50', '')} "
                f"| {d.get('p95', '')} | {d.get('p99', '')} |")
    mdp = os.path.splitext(args.out)[0] + ".md"
    with open(mdp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.out} and {mdp}; "
          f"{len(summary['order_violations'])} order violations")


if __name__ == "__main__":
    main()
