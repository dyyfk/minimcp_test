"""Internal repeat-run variance report (r0 = 2026-09-02 native_v1 run,
r1/r2 = 2026-09-07 repeats under the identical v1 protocol,
run_id-keyed outputs).

Per (run, arm): n, escalation, accuracy over scored rows,
reconstructed timing diagnostic (same endpoints as manifest rev2).
Across runs: mean +/- sd/min/max per arm, per-query pairwise flip
rates, per-category accuracy, GSM8K unfired check, and the ANALYTIC
matched-rate random mixture (per run, from that run's never/always
outcomes at the aggressive arm's realized rate - NOT a new random arm).

Usage (from interactive_paper/):
  set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\51_repeat_variance.py
Output: data/paper_revision_pr0907/repeat_variance.json
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
OUT = D / "paper_revision_pr0907"
ARMS = ["never", "conservative", "balanced", "aggressive", "always"]
RUNS = ["r0", "r1", "r2"]
CATS = ["easy-chat", "easy-fact", "hard-knowledge", "hard-math", "trap"]

qmeta = {q["id"]: q for q in (json.loads(l) for l in open(
    D / "queries.jsonl", encoding="utf-8") if l.strip())
    if q.get("split") == "test"}


def load(run, arm):
    fn = f"frozen_{arm}_judged.parquet" if arm == "never" \
        else f"frozen_{arm}_tts_judged.parquet"
    base = D / "native_bench" if run == "r0" \
        else D / "native_bench_repeats" / run
    df = pd.read_parquet(base / fn).drop_duplicates("id", keep="last")
    df = df.set_index("id")
    df["cat"] = pd.Series({i: q["pool"] for i, q in qmeta.items()})
    df["src"] = pd.Series({i: q.get("source") for i, q in qmeta.items()})
    on = (df["onset_chunk"].fillna(df["n_chunks"]) + 1
          - df["n_chunks"]).clip(lower=0)
    local_t = on + df["answer_ms"].fillna(0) / 1000
    if arm == "never":
        df["t_s"] = local_t
    else:
        esc = df["mode"] == "escalated"
        relay_s = (df["relay_synth_ms"].fillna(0) / 1000
                   + df["relay_audio_s"].fillna(0))
        df["t_s"] = np.where(
            esc, on + df["stall_ms"].fillna(0) / 1000
            + df["wait_chunks"].fillna(0) + relay_s, local_t)
    return df


def arm_stats(df):
    sc = df["adequate"].notna()
    return {
        "n_total": int(len(df)), "n_scored": int(sc.sum()),
        "escalation_rate": float(df["fired"].astype(bool).mean()),
        "acc_over_scored": float(df.loc[sc, "adequate"].astype(float).mean()),
        "t_mean_s": float(df["t_s"].mean()),
        "t_p50_s": float(np.percentile(df["t_s"], 50)),
        "t_p95_s": float(np.percentile(df["t_s"], 95)),
        "timeout_proxy_rows_wait_ge_145": int(
            (df["fired"].astype(bool)
             & (df.get("wait_chunks", pd.Series(0, index=df.index))
                .fillna(0) >= 145)).sum()),
        "by_cat_acc": {c: float(df.loc[sc & (df["cat"] == c), "adequate"]
                                .astype(float).mean())
                       for c in CATS},
        "by_cat_esc": {c: float(df.loc[df["cat"] == c, "fired"]
                                .astype(bool).mean()) for c in CATS},
    }


data, missing = {}, []
for run in RUNS:
    for arm in ARMS:
        try:
            data[(run, arm)] = load(run, arm)
        except FileNotFoundError as e:
            missing.append(f"{run}/{arm}: {e}")
if missing:
    print("MISSING (report will be partial):", missing)

out = {"protocol": "identical v1 (commit 51fdb4e + run_id param; gate "
                   "8bq sha 0e6494c2; same serving/judge config); "
                   "decoder stochastic (top_k=20), no remote seed "
                   "control - run-to-run variation includes decoding + "
                   "judge nondeterminism",
       "runs": {}, "across_runs": {}, "pairwise_flips": {},
       "gsm8k_unfired": {}, "analytic_random": {}, "missing": missing}

for (run, arm), df in data.items():
    out["runs"].setdefault(run, {})[arm] = arm_stats(df)

for arm in ARMS:
    have = [r for r in RUNS if (r, arm) in data]
    if len(have) < 2:
        continue
    accs = [out["runs"][r][arm]["acc_over_scored"] for r in have]
    escs = [out["runs"][r][arm]["escalation_rate"] for r in have]
    ts = [out["runs"][r][arm]["t_mean_s"] for r in have]
    out["across_runs"][arm] = {
        "runs": have, "acc_mean": float(np.mean(accs)),
        "acc_sd": float(np.std(accs, ddof=1)),
        "acc_min": float(min(accs)), "acc_max": float(max(accs)),
        "esc_mean": float(np.mean(escs)),
        "esc_sd": float(np.std(escs, ddof=1)),
        "t_mean_mean": float(np.mean(ts)),
        "t_mean_sd": float(np.std(ts, ddof=1)),
    }
    flips = {}
    for a, b in itertools.combinations(have, 2):
        da, db = data[(a, arm)], data[(b, arm)]
        ids = da.index.intersection(db.index)
        ya = da.loc[ids, "adequate"].astype(float)
        yb = db.loc[ids, "adequate"].astype(float)
        k = ya.notna() & yb.notna()
        flips[f"{a}-{b}"] = {
            "n": int(k.sum()),
            "flip_rate": float((ya[k] != yb[k]).mean()),
            "acc_delta": float(yb[k].mean() - ya[k].mean()),
            "fired_agree": float(
                (da.loc[ids, "fired"].astype(bool)
                 == db.loc[ids, "fired"].astype(bool)).mean()),
        }
    out["pairwise_flips"][arm] = flips

# GSM8K zero-escalation check: hard-math rows unfired in EVERY loaded
# arm of a run pair, accuracy spread
for run in RUNS:
    if (run, "never") not in data or (run, "aggressive") not in data:
        continue
    nv, ag = data[(run, "never")], data[(run, "aggressive")]
    m = (ag["cat"] == "hard-math") & (~ag["fired"].astype(bool))
    ids = m[m].index
    out["gsm8k_unfired"][run] = {
        "n_unfired_hard_math": int(len(ids)),
        "never_acc": float(nv.loc[ids, "adequate"].astype(float).mean()),
        "aggressive_acc": float(ag.loc[ids, "adequate"].astype(float).mean()),
        "identical_answer_text": int(
            (nv.loc[ids, "answer"].fillna("")
             == ag.loc[ids, "answer"].fillna("")).sum()),
    }

# analytic matched-rate random mixture (per run; expectation over
# uniform random escalation at the aggressive arm's REALIZED rate,
# using the same run's never/always query-level outcomes)
for run in RUNS:
    trio = [(run, a) for a in ("never", "aggressive", "always")]
    if not all(t in data for t in trio):
        continue
    nv, ag, al = (data[t] for t in trio)
    ids = nv.index.intersection(al.index)
    r = float(ag["fired"].astype(bool).mean())
    ynv = nv.loc[ids, "adequate"].astype(float)
    yal = al.loc[ids, "adequate"].astype(float)
    out["analytic_random"][run] = {
        "matched_rate": r,
        "acc_expectation": float(r * yal.mean() + (1 - r) * ynv.mean()),
        "note": "analytic mixture E[acc] = r*always + (1-r)*never at "
                "the aggressive arm's realized rate; NOT a newly run "
                "random arm; per-query mixture weights = uniform r",
    }

OUT.mkdir(exist_ok=True)
(OUT / "repeat_variance.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["across_runs"], indent=1))
print(json.dumps(out["gsm8k_unfired"], indent=1))
print(json.dumps(out["analytic_random"], indent=1))
