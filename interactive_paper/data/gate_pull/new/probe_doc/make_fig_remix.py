"""fig11: gate accuracy at fixed escalation budgets vs matched-rate random (reads remix_eval.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "remix_eval.json"))
POOLS = ["striviaqa", "swebq", "sllama", "sdqa"]
RATES = ["0.15", "0.3", "0.5"]
probes = [("deployed LR L30 3-block", "#d62728", "deployed probe"), ("v2 layer-avg LR x3 4-block", "#1baf7a", "v2 probe")]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=False)
for ax, pool in zip(axes, POOLS):
    x = np.array([0, 15, 30, 50, 100])
    d0 = R[probes[0][0]]["loc_official"][pool]
    ax.plot([0, 100], [d0["local"], d0["expert"]], color="#8a8a8a", ls="--", lw=1.2, label="random escalation (matched rate)")
    for name, c, lab in probes:
        d = R[name]["loc_official"][pool]
        ys = [d["local"]] + [d[f"gate@{r}"] for r in RATES] + [d["expert"]]
        ax.plot(x, ys, "-o", color=c, ms=4, label=lab)
        for xi, r in zip(x[1:4], RATES):
            p = d[f"p@{r}"]
            if name == probes[0][0]:
                ax.text(xi, d[f"gate@{r}"] + .012, f"p={p:.3f}" if p >= .001 else "p<.001", ha="center", fontsize=6.5, color=c)
    fr = R[probes[0][0]]["loc_official"][pool]["fire_at_calib_thr"]
    ax.set_title(f"{pool}  (n={d0['n']}; official judge)\nfire at calib thresholds: {fr['0.15']:.0%}/{fr['0.3']:.0%}/{fr['0.5']:.0%} for nominal 15/30/50 %", fontsize=8.5)
    ax.set_xlabel("escalation budget (%)"); ax.set_xticks([0, 15, 30, 50, 100])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("accuracy (local answers kept, top-r by score sent to gpt-5.5)")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("Gate benefit (re-mix): both probes beat random at every budget on every pool; v2 ≈ deployed. Fixed calibration thresholds do not deliver the nominal budgets.", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(HERE / "figs" / "fig11_remix.png", dpi=130, bbox_inches="tight")
print("wrote fig11_remix.png")
