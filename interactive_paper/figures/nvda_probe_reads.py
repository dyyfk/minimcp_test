"""Latest pass-3 NVDA method-transfer results (paper figure).

(a) external AUC by read point on the pass-3 capture (same 2,481-row
    calibration, same 4 external pools; data/gate_pull/new/probe_doc/results_round4.json)
(b) gate accuracy at fixed escalation budgets, 4-pool mean under the official
    judges, reported 3-layer probe vs causal ablation vs random (remix_eval3.json)

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
RM = json.load(open(PD / "remix_eval3.json"))

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

reads = [
    ("single-layer\nanswer onset\n(L30)", "A0' deployed read (pass-3): onset_last|onset_mean8|user_mean @L30", BLUE),
    ("reported 3-layer\nanswer onset\n(L26/30/34)", "V2c layer-avg ×3 (26,30,34): commit|onset_last|onset_mean8|run_mean", GREEN),
    ("strictly causal\ncommit frame\n(L34)", "S2 @L34", ORANGE),
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
ax.set_ylim(.74, .87); ax.set_ylabel("external AUC (bar = 4-pool mean; dots = pools)")
ax.axhline(R4[reads[0][1]]["ext"]["mean"], color=BLUE, ls=":", lw=1, zorder=1)
ax.set_title("(a) External AUC by read point (pass-3; Δ vs single-layer reference)", fontsize=8, loc="left")
ax.grid(axis="y", color=GRID, lw=.6, zorder=0)

rates = ["0.15", "0.3", "0.5"]; xs = np.array([0, 15, 30, 50, 100])
def mean_curve(name, lab="loc_official"):
    d = RM[name][lab]; return [np.mean([d[p]["local"] for p in POOLS])] + [np.mean([d[p][f"gate@{r}"] for p in POOLS]) for r in rates] + [np.mean([d[p]["expert"] for p in POOLS])]
latest_name = "v2: layer-avg x3, commit|onset_last|onset_mean8|run_mean"
causal_name = "strictly causal: commit|pre_mean8|run_mean @L34"
latest = mean_curve(latest_name); causal = mean_curve(causal_name)
bx.plot([0, 100], [latest[0], latest[-1]], color=MUT, ls="--", lw=1.2, label="matched-rate random")
bx.plot(xs, latest, "-s", color=GREEN, ms=4, lw=1.6, label="reported 3-layer answer-onset probe")
bx.plot(xs, causal, "-o", color=ORANGE, ms=3.5, lw=1.2, label="strictly causal ablation")
for xi, r in zip(xs[1:4], rates):
    pmax = max(RM[latest_name]["loc_official"][p][f"p@{r}"] for p in POOLS)
    bx.text(xi, latest[int(np.where(xs == xi)[0][0])] + .012, f"max pool p={pmax:.3f}" if pmax >= .001 else "all pools p<.001", ha="center", fontsize=6.3, color=GREEN)
bx.set_xticks([0, 15, 30, 50, 100]); bx.set_xlabel("escalation budget (%)"); bx.set_ylabel("accuracy, 4-pool mean (official judges)")
bx.set_title("(b) Accuracy at fixed escalation budgets (official judges)", fontsize=8, loc="left"); bx.legend(frameon=False, fontsize=7, loc="lower right")
bx.grid(color=GRID, lw=.6, zorder=0)
fig.tight_layout()
for target in (HERE, HERE.parent / "paper" / "figures"):
    for ext in ("png", "pdf"):
        fig.savefig(target / f"nvda_probe_reads.{ext}", dpi=200, bbox_inches="tight")
print("wrote nvda_probe_reads.png/pdf to figures/ and paper/figures/")
