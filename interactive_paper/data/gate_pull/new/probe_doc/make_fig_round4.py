"""fig12: causal re-capture (pass 3) — what each read costs/buys (reads results_round4.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "results_round4.json"))
res = {r["name"]: r for r in R["results"]}
rows = [
    ("deployed read (pass-3 replication)\nonset_last | onset_mean8 | user_mean @L30", "A0' deployed read (pass-3): onset_last|onset_mean8|user_mean @L30", "#d62728"),
    ("same, causal user_mean (run_mean)\n= what the streaming code computes", "F1 deployed read, causal user_mean: onset_last|onset_mean8|run_mean @L30", "#2a78d6"),
    ("v2 architecture, causal user_mean\nlayer-avg ×3, commit | onset_last | onset_mean8 | run_mean", "V2c layer-avg ×3 (26,30,34): commit|onset_last|onset_mean8|run_mean", "#1baf7a"),
    ("everything @L30 (onset + pre + run)", "X1 everything @L30", "#2a78d6"),
    ("old eot read @L34 (pass-3)", "E0 old eot read (pass-3): eot_last|eot_mean8|user_mean @L34", "#8a8a8a"),
    ("STRICTLY CAUSAL, best layer:\ncommit | pre_mean8 | run_mean @L34", "S2 @L34", "#f28e2b"),
    ("strictly causal @L30", "S2 at-commit: commit|pre_mean8|run_mean @L30", "#f28e2b"),
    ("strictly causal, layer-avg ×3", "V2s layer-avg ×3 (26,30,34): commit|pre_last|pre_mean8|run_mean [causal]", "#f28e2b"),
]
labels = [r[0] for r in rows]; keys = [r[1] for r in rows]; cols = [r[2] for r in rows]
oof = [res[k]["oof"] for k in keys]; lopo = [res[k]["lopo"] for k in keys]; lopoM = [res[k]["lopo_macro"] for k in keys]; ext = [res[k]["ext"]["mean"] for k in keys]
ci = [res[k].get("delta_ext", {}).get("ci95") for k in keys]; base = ext[0]

fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.2), sharey=True)
y = np.arange(len(rows))[::-1]
for ax, vals, title, xlim in ((axes[0], oof, "pooled 5-fold OOF AUC", (.78, .84)), (axes[1], lopo, "LOPO AUC (pooled)", (.72, .81)),
                              (axes[2], lopoM, "LOPO: mean within-held-pool AUC", (.64, .72)), (axes[3], ext, "cold external AUC, 4-pool mean", (.74, .84))):
    ax.barh(y, vals, color=cols, alpha=.85)
    for yi, v in zip(y, vals):
        ax.text(v + (xlim[1] - xlim[0]) * .01, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(vals[0], color="#d62728", ls=":", lw=1); ax.set_xlim(*xlim); ax.set_title(title, fontsize=9.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for yi, c, v in zip(y, ci, ext):
    if c:
        axes[3].plot([base + c[0], base + c[1]], [yi - .3, yi - .3], color="black", lw=1.2); axes[3].plot([v], [yi - .3], "k|", ms=6)
axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8)
d = R["drift"]
fig.suptitle(f"Causal re-capture (pass 3, n=2,481 calib, pass-2 labels; answer text drifted for {d['answer_text_changed']:.0%} of queries, commit frame moved for {d['commit_frame_changed']:.1%}). "
             "Orange = nothing read after the commit frame. Black bars = 95 % paired-bootstrap CI vs the deployed read.", fontsize=8.8)
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(HERE / "figs" / "fig12_causal_recapture.png", dpi=130, bbox_inches="tight")
print("wrote fig12_causal_recapture.png")
