"""fig10: summary of the review-round experiments (reads results_round{2,3}.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
r1 = {r["name"]: r for r in json.load(open(HERE / "results_batch1.json"))}
r2 = {r["name"]: r for r in json.load(open(HERE / "results_round2.json"))}
r3 = {r["name"]: r for r in json.load(open(HERE / "results_round3.json"))}
ELIG = {'easy-chat', 'easy-fact', 'easy-mathword', 'hard-knowledge', 'hard-math', 'hard-multihop', 'know-arc',
        'know-commonsense', 'know-longtail', 'know-mmlu', 'know-open', 'know-openbook', 'trap-truthful'}
a8 = dict(r1["onset@L30 all8frames+umean"])
a8["lopo_macro"] = np.mean([v for k, v in a8["lopo_within"].items() if k in ELIG])
rows = [
    ("deployed: LR, L30, 3-block", r3["A0 deployed LR L30 3-block"]),
    ("LR, L26, 3-block", r3["A1 LR L26 3-block"]),
    ("LR, L30, 4-block (+commit frame)", r3["B1 LR L30 4-block (+commit frame)"]),
    ("LR, L30, all 8 onset frames + user_mean (lopoM #1)", a8),
    ("layer-avg LR ×3 (26/30/34), 4-block  ← v2", r3["D2 layer-avg LR ×3 (26,30,34) 4-block"]),
    ("layer-avg LDA λ=30 ×3, 4-block", r3["E2 layer-avg LDA λ=30 ×3 (26,30,34) 4-block"]),
    ("layer-avg LDA λ=30 ×5, 4-block", r3["E1 layer-avg LDA λ=30 ×5 4-block  [Fable combo]"]),
    ("shrink-LDA λ=30, L26, 3-block (best external)", r3["C2 LDA λ=30 L26 3-block"]),
    ("strictly-at-commit: commit frame + user_mean, L34", r2["S commit-frame@L34 + umean"]),
    ("remove 2 pool-centroid dirs, L26 3-block", r2["P L26 stack, remove 2 pool-centroid dirs"]),
]
labels = [r[0] for r in rows]
oof = [r[1]["oof"] for r in rows]; lopoM = [r[1]["lopo_macro"] for r in rows]; lopo = [r[1]["lopo"] for r in rows]
ext = [r[1]["ext"]["mean"] for r in rows]
ci = [r[1].get("delta_ext", {}).get("ci95") for r in rows]
base_ext = ext[0]

fig, axes = plt.subplots(1, 4, figsize=(15, 5.0), sharey=True)
y = np.arange(len(rows))[::-1]
for ax, vals, title, xlim in ((axes[0], oof, "pooled 5-fold OOF AUC", (.78, .84)),
                              (axes[1], lopo, "leave-one-pool-out AUC (pooled)", (.70, .81)),
                              (axes[2], lopoM, "LOPO: mean within-held-pool AUC", (.64, .72)),
                              (axes[3], ext, "cold external AUC, 4-pool mean", (.76, .84))):
    cols = ["#d62728" if i == 0 else ("#1baf7a" if "v2" in labels[i] else "#2a78d6") for i in range(len(rows))]
    ax.barh(y, vals, color=cols, alpha=.85)
    for yi, v in zip(y, vals):
        ax.text(v + (xlim[1] - xlim[0]) * .01, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(vals[0], color="#d62728", ls=":", lw=1)
    ax.set_xlim(*xlim); ax.set_title(title, fontsize=9.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
# paired bootstrap CI on the external panel
for yi, c, v in zip(y, ci, ext):
    if c:
        axes[3].plot([base_ext + c[0], base_ext + c[1]], [yi - .28, yi - .28], color="black", lw=1.2)
        axes[3].plot([v], [yi - .28], "k|", ms=6)
axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8.5)
fig.suptitle("Review-round experiments (calibration n=2,481; externals cold). Black bars on the right = 95 % paired-bootstrap CI of the "
             "difference vs deployed, drawn around the deployed value.", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(HERE / "figs" / "fig10_review_round.png", dpi=130, bbox_inches="tight")
print("wrote fig10_review_round.png")
