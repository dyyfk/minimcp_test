"""Gallery figures for 8bq (official-config deployed gate, per-language
thresholds) and 8bt (shadow candidates vs live).
Run: .venv_boot\Scripts\python.exe figures\native_gallery2.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1baf7a"
GREY, AMBER, SLATE = "#8a97a5", "#b5651d", "#3d5a73"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---- fig N4: official-config validity (8bq deployed gate) ---------------
valid = json.load(open("figures/native_validity_official.json"))
ORDER = ["frozen", "striviaqa", "swebq", "sllama", "sdqa", "sreason"]
NICE = {"frozen": "our pool", "striviaqa": "TriviaQA", "swebq": "WebQ",
        "sllama": "LlamaQ", "sdqa": "SD-QA", "sreason": "Reasoning-zh"}
TIERS = ["never", "conservative", "balanced", "aggressive", "always"]
fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
for ax, pool in zip(axes.flat, ORDER):
    d = valid[pool]
    xs = [d["tiers"][t]["esc_rate"] for t in TIERS]
    ys = [d["tiers"][t]["acc"] for t in TIERS]
    rs = [d["tiers"][t]["random_matched"] for t in TIERS]
    fl, ce = d["local_floor"], d["expert_ceiling"]
    pb_lo, pb_hi = max(0.0, ce - fl), min(1 - fl, ce)
    rg = np.linspace(0, 1, 201)

    def oracle(pb):
        acc = fl + np.minimum(rg, pb)
        tail = rg >= pb
        acc[tail] = (fl + pb) + (ce - fl - pb) * (rg[tail] - pb) / (1 - pb)
        return acc

    ax.fill_between(rg, oracle(pb_lo), oracle(pb_hi), color=TEAL, alpha=.13, lw=0)
    ax.plot(rg, oracle(pb_lo), color=TEAL, lw=1.3, alpha=.8, label="oracle selector")
    ax.plot(xs, rs, "--", color=GREY, lw=1.6, label="matched random")
    ax.plot(xs, ys, "o-", color=BLUE, lw=2, label="gated (8bq, deployed)")
    lang = "zh" if pool == "sreason" else "en"
    ax.set_title(f"{NICE[pool]}  (n={d['n']}, {lang} thresholds)", fontsize=11)
    ax.set_xlim(-.04, 1.04)
    for t in ("conservative", "balanced", "aggressive"):
        p = d["tiers"][t]["perm_p"]; i = TIERS.index(t)
        if p is not None and p < .05:
            ax.annotate("*", (xs[i], ys[i]), textcoords="offset points",
                        xytext=(0, 4), ha="center", color=ORANGE, fontsize=15)
axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
for ax in axes[1]:
    ax.set_xlabel("escalation rate")
for r in range(2):
    axes[r][0].set_ylabel("delivered accuracy")
fig.suptitle("Deployed gate (8bq: official serving config, native labels, per-language thresholds)\n"
             "gated accuracy vs matched-random vs the oracle bound  (* = permutation p<.05)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("figures/native_validity_official.png", dpi=170)
plt.close(fig)

# ---- fig N5: shadow candidates vs live -----------------------------------
R = json.load(open("figures/shadow_compare.json"))
POOLS = ["frozen", "striviaqa", "swebq", "sllama", "sdqa", "sreason"]
C = [("P9_distilled", "P9 distilled student (8192-d)", AMBER, -0.17),
     ("P16_alpha1", "P16 alpha-1 ensemble (12288-d)", TEAL, 0.17)]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
y = np.arange(len(POOLS))[::-1]
for k, lab, col, off in C:
    d = [R[p]["scorers"][k]["delta_native_auc"] for p in POOLS]
    lo = [R[p]["scorers"][k]["delta_native_ci"][0] for p in POOLS]
    hi = [R[p]["scorers"][k]["delta_native_ci"][1] for p in POOLS]
    ax.hlines(y + off, lo, hi, color=col, lw=2.2, alpha=.75)
    ax.plot(d, y + off, "o", color=col, ms=7, label=lab)
ax.axvline(0, color=GREY, ls=":", lw=1.2)
ax.axvline(.015, color=ORANGE, ls="--", lw=1, alpha=.7)
ax.text(.0155, y[-1] - .55, "replacement gate +.015", color=ORANGE, fontsize=9)
ax.set_yticks(y, [NICE[p] for p in POOLS])
ax.set_xlabel("native-failure AUC delta vs live 8bq  (95% paired bootstrap)")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_ylim(y[-1] - .9, y[0] + .7)
ax.set_title("Ranking: both candidates beat live on every external pool", fontsize=11)

ax = axes[1]
m = R["_external_mean"]
keys = [("live_8bq", "live 8bq", SLATE), ("P9_distilled", "P9", AMBER), ("P16_alpha1", "P16", TEAL)]
metrics = [("native AUC", lambda v: v["native_auc"]), ("benefit AUC", lambda v: v["benefit_auc"]),
           ("cascade acc\n@30% exact", lambda v: v["cascade_exact"]["balanced"])]
x = np.arange(len(metrics)); w = .26
for j, (k, lab, col) in enumerate(keys):
    vals = [f(m[k]) for _, f in metrics]
    ax.bar(x + (j - 1) * w, vals, w, color=col, label=lab)
    for xi, v in zip(x + (j - 1) * w, vals):
        ax.text(xi, v + .004, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_xticks(x, [n for n, _ in metrics])
ax.set_ylim(.6, .8)
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
ax.set_title("External-5 mean: AUC +.02, cascade +0.8 / +0.3 pt", fontsize=11)
fig.suptitle("Shadow candidates from issue #8 rescored on our six official-native pools (8bt, $0)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figures/shadow_compare.png", dpi=170)
plt.close(fig)
print("wrote figures/native_validity_official.png, figures/shadow_compare.png")
