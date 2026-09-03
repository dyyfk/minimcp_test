"""8bv: probe lift by FAILURE TYPE on the native benchmark.

For every never-arm failure (classified by modal_failure_taxonomy.py) and
every never-arm correct row: was it escalated at the deployed balanced /
aggressive thresholds (never-arm onset score, per-language threshold, act
gate), and would the expert have fixed it (always-arm TTS-relay outcome on
the same id)? Reports per type: share of failures, recall at each tier,
the false-fire rate on correct rows for reference, expert-fixable rate,
and the realized accuracy points recovered per 100 questions.

Also: AUC of the never-arm score for "this failure type vs correct",
pooled over pools (raw score; thresholds are per-language so recall is
the deployment-faithful number, AUC the ranking one).

Usage: .venv_boot\\Scripts\\python.exe scripts\\37_failure_taxonomy_lift.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path("data/native_bench")
POOLS = {"frozen": ("adequate", "en"), "striviaqa": ("oab_ok", "en"), "swebq": ("oab_ok", "en"),
         "sllama": ("oab_ok", "en"), "sdqa": ("adequate", "en"), "sreason": ("adequate", "zh")}
TYPES = ["perception", "knowledge_gap", "confident_wrong", "execution", "quality_other"]
NICE = {"perception": "perception (misheard)", "knowledge_gap": "knowledge gap (hedges)",
        "confident_wrong": "confident wrong", "execution": "execution (working slips)",
        "quality_other": "quality / other"}
gate = json.load(open("data/gate_native.json"))
ft = pd.read_parquet(D / "failure_types.parquet").dropna(subset=["ftype"])

rows = []
for pool, (col, lang) in POOLS.items():
    n = pd.read_parquet(D / f"{pool}_never_judged.parquet").drop_duplicates("id", keep="last").set_index("id")
    a = pd.read_parquet(D / f"{pool}_always_tts_judged.parquet").drop_duplicates("id", keep="last").set_index("id")
    n = n[n[col].notna()]
    thr = gate["eot_thresholds_lang"][lang]
    types = ft[ft.pool == pool].set_index("id")["ftype"]
    for i, r in n.iterrows():
        info = True if pd.isna(r.get("is_info")) else bool(r["is_info"])
        rows.append({"pool": pool, "id": i, "correct": int(r[col]),
                     "ftype": ("correct" if r[col] == 1 else types.get(i, "unclassified")),
                     "score": float(r["score"]) if pd.notna(r["score"]) else np.nan,
                     "fire_bal": bool(pd.notna(r["score"]) and r["score"] >= thr["balanced"] and info),
                     "fire_agg": bool(pd.notna(r["score"]) and r["score"] >= thr["aggressive"] and info),
                     "fire_con": bool(pd.notna(r["score"]) and r["score"] >= thr["conservative"] and info),
                     "fixable": (float(a[col].get(i)) if i in a.index and pd.notna(a[col].get(i)) else np.nan)})
df = pd.DataFrame(rows)
N = len(df); corr = df[df.correct == 1]; fail = df[df.correct == 0]
print(f"rows {N}  correct {len(corr)}  failures {len(fail)}")
print(f"false-fire on correct rows: cons {corr.fire_con.mean():.2f}  bal {corr.fire_bal.mean():.2f}  agg {corr.fire_agg.mean():.2f}")
out = {}
print(f"\n{'type':26s} {'n':>4s} {'share':>6s} {'AUC':>5s} {'rec@cons':>8s} {'rec@bal':>7s} {'rec@agg':>7s} {'fixable':>7s} {'pts@bal':>7s} {'pts@agg':>7s} {'ceil pts':>8s}")
for t in TYPES:
    g = fail[fail.ftype == t]
    if len(g) == 0:
        continue
    both = pd.concat([g.assign(y=1), corr.assign(y=0)]).dropna(subset=["score"])
    auc = roc_auc_score(both.y, both.score)
    fx = g.fixable.mean()
    pts_bal = 100 * (g.fire_bal & (g.fixable == 1)).sum() / N
    pts_agg = 100 * (g.fire_agg & (g.fixable == 1)).sum() / N
    ceil = 100 * (g.fixable == 1).sum() / N
    out[t] = dict(n=int(len(g)), share=len(g) / len(fail), auc=auc, rec_con=g.fire_con.mean(),
                  rec_bal=g.fire_bal.mean(), rec_agg=g.fire_agg.mean(), fixable=fx,
                  pts_bal=pts_bal, pts_agg=pts_agg, ceil_pts=ceil)
    print(f"{NICE[t]:26s} {len(g):4d} {len(g)/len(fail):6.2f} {auc:5.3f} {g.fire_con.mean():8.2f} {g.fire_bal.mean():7.2f} {g.fire_agg.mean():7.2f} {fx:7.2f} {pts_bal:7.2f} {pts_agg:7.2f} {ceil:8.2f}")
# per pool x type counts for the write-up
print("\nfailure type by pool:")
print(fail.pivot_table(index="pool", columns="ftype", values="id", aggfunc="count", fill_value=0).to_string())
Path("figures/failure_taxonomy_lift.json").write_text(json.dumps({
    "n_rows": N, "n_fail": int(len(fail)), "false_fire_correct": {"cons": corr.fire_con.mean(), "bal": corr.fire_bal.mean(), "agg": corr.fire_agg.mean()},
    "types": out}, indent=1))

# ---- figure: recall at each tier + fixable + AUC per type ----
BLUE, GREY, TEAL, ORANGE = "#2a78d6", "#8a97a5", "#1baf7a", "#eb6834"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
ts = [t for t in TYPES if t in out]
x = np.arange(len(ts)); w = .2
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ax.bar(x - 1.5 * w, [out[t]["rec_con"] for t in ts], w, color="#9fc3e8", label="escalated @conservative")
ax.bar(x - .5 * w, [out[t]["rec_bal"] for t in ts], w, color=BLUE, label="escalated @balanced")
ax.bar(x + .5 * w, [out[t]["rec_agg"] for t in ts], w, color="#1d4f80", label="escalated @aggressive")
ax.bar(x + 1.5 * w, [out[t]["fixable"] for t in ts], w, color=TEAL, label="expert would fix")
for tier, ls in (("bal", "--"), ("agg", ":")):
    ax.axhline(corr[f"fire_{tier}"].mean(), color=GREY, ls=ls, lw=1.2)
ax.text(x[-1] + .55, corr.fire_bal.mean() + .01, "false-fire on correct @bal", fontsize=8, color=GREY, ha="right")
ax.text(x[-1] + .55, corr.fire_agg.mean() + .01, "false-fire on correct @agg", fontsize=8, color=GREY, ha="right")
SHORT = {"perception": "perception", "knowledge_gap": "knowledge\ngap", "confident_wrong": "confident\nwrong",
         "execution": "execution", "quality_other": "quality /\nother"}
ax.set_xticks(x, [f"{SHORT[t]}\nn={out[t]['n']} ({out[t]['share']:.0%})\nAUC {out[t]['auc']:.2f}" for t in ts], fontsize=8.5)
ax.set_ylim(0, 1.02); ax.set_ylabel("fraction of this failure type")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
ax.set_title("What the gate catches, by failure type\n(never-arm onset score at the deployed thresholds; six QA pools, %d questions)" % N, fontsize=10, loc="left")
ax = axes[1]
ax.bar(x - w, [out[t]["pts_bal"] for t in ts], w * 1.0, color=BLUE, label="recovered @balanced")
ax.bar(x, [out[t]["pts_agg"] for t in ts], w * 1.0, color="#1d4f80", label="recovered @aggressive")
ax.bar(x + w, [out[t]["ceil_pts"] for t in ts], w * 1.0, color=TEAL, alpha=.55, label="recoverable if all escalated")
ax.set_xticks(x, [SHORT[t] for t in ts], fontsize=8.5)
ax.set_ylabel("accuracy points per 100 questions")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_title("Where the routing gain comes from\n(points recovered = escalated and expert fixed it)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("figures/failure_taxonomy_lift.png", dpi=170)
print("\nwrote figures/failure_taxonomy_lift.png / .json")
