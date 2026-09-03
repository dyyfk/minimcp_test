"""NVDA probe read points: what each read costs/buys (paper figure).

(a) cold external AUC by read point on the pass-3 capture (same 2,481-row
    calibration, same 4 external pools; data/gate_pull/new/probe_doc/results_round4.json)
(b) gate accuracy at fixed escalation budgets, 4-pool mean under the official
    judges, deployed vs v2 vs matched-rate random (remix_eval.json)

Run from interactive_paper/figures:  python nvda_probe_reads.py
Writes nvda_probe_reads.{png,pdf}; copy to ../paper/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN = "#2a78d6", "#1baf7a"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e4de"
ORANGE, RED = "#f28e2b", "#d62728"
HERE = Path(__file__).resolve().parent
PD = HERE.parent / "data" / "gate_pull" / "new" / "probe_doc"
POOLS = ["striviaqa", "swebq", "sllama", "sdqa"]
R4 = {r["name"]: r for r in json.load(open(PD / "results_round4.json"))["results"]}
RM = json.load(open(PD / "remix_eval.json"))

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

reads = [
    ("end of\nuser audio\n(L34)", "E0 old eot read (pass-3): eot_last|eot_mean8|user_mean @L34", MUT),
    ("causal:\ncommit frame\n(L34)", "S2 @L34", ORANGE),
    ("causal:\ncommit frame\n(L30)", "S2 at-commit: commit|pre_mean8|run_mean @L30", ORANGE),
    ("deployed:\ncommit + 8\nframes (L30)", "A0' deployed read (pass-3): onset_last|onset_mean8|user_mean @L30", BLUE),
    ("deployed,\ncausal\nuser_mean", "F1 deployed read, causal user_mean: onset_last|onset_mean8|run_mean @L30", BLUE),
    ("v2: 3-layer\naverage +\ncommit frame", "V2c layer-avg ×3 (26,30,34): commit|onset_last|onset_mean8|run_mean", GREEN),
]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.45, 1]})
x = np.arange(len(reads))
for i, (lab, key, c) in enumerate(reads):
    r = R4[key]; per = [r["ext"][p] for p in POOLS]; m = r["ext"]["mean"]
    ax.bar(i, m, color=c, alpha=.85, width=.62, zorder=2)
    ax.scatter([i - .18, i - .06, i + .06, i + .18], per, s=11, color=INK, zorder=3, alpha=.8)
    ax.text(i, m + .004, f"{m:.3f}", ha="center", fontsize=7.5, color=INK)
ticklabels = []
for lab, key, c in reads:
    d = R4[key].get("delta_ext")
    ticklabels.append(lab + (f"\nΔ {d['mean']:+.3f}\n[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else "\n(reference)\n"))
ax.set_xticks(x); ax.set_xticklabels(ticklabels, fontsize=6.2)
ax.set_ylim(.74, .87); ax.set_ylabel("cold external AUC (bar = 4-pool mean; dots = pools)")
ax.axhline(R4[reads[3][1]]["ext"]["mean"], color=BLUE, ls=":", lw=1, zorder=1)
ax.set_title("(a) Cold external AUC by read point (pass-3 capture; Δ vs deployed, 95% paired CI)", fontsize=8, loc="left")
ax.grid(axis="y", color=GRID, lw=.6, zorder=0)

rates = ["0.15", "0.3", "0.5"]; xs = np.array([0, 15, 30, 50, 100])
def mean_curve(name, lab="loc_official"):
    d = RM[name][lab]; return [np.mean([d[p]["local"] for p in POOLS])] + [np.mean([d[p][f"gate@{r}"] for p in POOLS]) for r in rates] + [np.mean([d[p]["expert"] for p in POOLS])]
dep = mean_curve("deployed LR L30 3-block"); v2 = mean_curve("v2 layer-avg LR x3 4-block")
bx.plot([0, 100], [dep[0], dep[-1]], color=MUT, ls="--", lw=1.2, label="matched-rate random")
bx.plot(xs, dep, "-o", color=BLUE, ms=4, lw=1.6, label="deployed probe")
bx.plot(xs, v2, "-s", color=GREEN, ms=3.5, lw=1.2, label="v2 probe")
for xi, r in zip(xs[1:4], rates):
    pmax = max(RM["deployed LR L30 3-block"]["loc_official"][p][f"p@{r}"] for p in POOLS)
    bx.text(xi, dep[int(np.where(xs == xi)[0][0])] + .012, f"all pools p≤{pmax:.3f}" if pmax >= .001 else "all pools p<.001", ha="center", fontsize=6.3, color=BLUE)
bx.set_xticks([0, 15, 30, 50, 100]); bx.set_xlabel("escalation budget (%)"); bx.set_ylabel("accuracy, 4-pool mean (official judges)")
bx.set_title("(b) Accuracy at fixed escalation budgets (official judges)", fontsize=8, loc="left"); bx.legend(frameon=False, fontsize=7, loc="lower right")
bx.grid(color=GRID, lw=.6, zorder=0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"nvda_probe_reads.{ext}", dpi=200, bbox_inches="tight")
print("wrote nvda_probe_reads.png/pdf")
