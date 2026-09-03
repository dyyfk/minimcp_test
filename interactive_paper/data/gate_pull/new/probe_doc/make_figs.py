"""Figures for NVDA_PROBE_TRAINING.md — everything is computed from the
checked-in replay data under data/gate_pull/ (CPU only).

    cd interactive_paper/data/gate_pull/new/probe_doc
    uv run --with numpy --with pandas --with pyarrow --with scikit-learn \
           --with matplotlib python make_figs.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
GP = HERE.parents[1]                      # data/gate_pull
FIG = HERE / "figs"
FIG.mkdir(exist_ok=True)
LAYERS = list(range(2, 56, 4))
J30 = LAYERS.index(30)
BLUE, GREEN, ORANGE, RED, GREY = "#2a78d6", "#1baf7a", "#f28e2b", "#d62728", "#8a8a8a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- data ---
def load_calib():
    """Onset-read shards (pass 2) + pass-2 labels + pool per id."""
    ids, E_on, E_eot, M, ONS = [], [], [], [], []
    for tag in ("frozen", "expansion", "expansion2"):
        for sh in sorted(glob.glob(str(GP / "onset" / f"nvda_h_{tag}.shard*.npz"))):
            z = np.load(sh, allow_pickle=True)
            ids += [(tag, str(x)) for x in z["ids"]]
            E_on.append(z["H_onset"]); E_eot.append(z["H_eot"])
            M.append(z["H_mean"]); ONS.append(z["onset_frame"])
    E_on, E_eot = np.concatenate(E_on), np.concatenate(E_eot)
    M, ONS = np.concatenate(M), np.concatenate(ONS)
    lab = {}
    for tag in ("frozen", "expansion", "expansion2"):
        df = pd.read_parquet(GP / "onset_fit" / f"nvda_{tag}.parquet")
        for _, r in df.iterrows():
            if pd.notna(r["escalate_label"]):
                lab[(tag, str(r["id"]))] = int(r["escalate_label"])
    q = {}
    for f in ("queries.jsonl", "queries_expansion.jsonl", "queries_expansion2.jsonl"):
        for line in (GP / f).open(encoding="utf-8"):
            r = json.loads(line)
            q[str(r["id"])] = r
    nfr = {}
    for f in glob.glob(str(GP / "onset" / "nvda_answers_*.shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            nfr[str(r["id"])] = r["n_frames_query"]
    keep = [i for i, k in enumerate(ids) if k in lab]
    meta = pd.DataFrame({
        "tag": [ids[i][0] for i in keep], "id": [ids[i][1] for i in keep],
        "y": [lab[ids[i]] for i in keep], "onset": ONS[keep],
        "pool": [q[ids[i][1]]["pool"] for i in keep],
        "split": [q[ids[i][1]].get("split") for i in keep],
        "n_frames": [nfr.get(ids[i][1], np.nan) for i in keep]})
    return meta, E_on[keep], E_eot[keep], M[keep]


def stack(E, M, j):
    return np.concatenate([E[:, j, -1], E[:, j].mean(1), M[:, j]], axis=1).astype(np.float32)


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1e-4, max_iter=5000))


meta, E_on, E_eot, M = load_calib()
committed = (meta.onset >= 0).values
print(f"calib rows {len(meta)}, committed {committed.sum()}, fail {meta.y[committed].mean():.3f}")

# ------------------------------------------------- fig1: read-point schematic
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(-2, 102); ax.set_ylim(0, 11); ax.axis("off")
def box(x0, x1, y, h, color, text, tcolor="black", alpha=1.0, fs=8.5):
    ax.add_patch(Rectangle((x0, y), x1 - x0, h, color=color, alpha=alpha, ec="none"))
    if text:
        ax.text((x0 + x1) / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tcolor)
box(0, 14, 6, 2.2, "#dddddd", "system\nprompt", fs=8)
box(14, 58, 6, 2.2, "#cfe3fb", "user audio\n(agent channel emits PAD every 80 ms frame)")
box(62, 100, 6, 2.2, "#d9f3e6", "agent speaks\n(non-PAD text tokens, 1 per frame)")
ax.plot([58, 58], [4.9, 6], color="black", lw=1)
ax.text(58, 4.5, "end of user audio (t_end)", ha="center", fontsize=8)
ax.plot([62, 62], [6, 8.6], color=RED, lw=2)
ax.text(62, 8.9, "commit frame = first run of ≥3 non-PAD agent tokens", ha="center", fontsize=8, color=RED)
# windows
ax.add_patch(Rectangle((50, 3.1), 8, 0.9, color=BLUE, alpha=.8)); ax.text(49, 3.55, "H_eot: last 8 frames of user audio", ha="right", va="center", fontsize=8, color=BLUE)
ax.add_patch(Rectangle((14, 1.9), 44, 0.9, color=GREY, alpha=.6)); ax.text(13, 2.35, "H_mean: mean over all\nuser-audio frames", ha="right", va="center", fontsize=8, color=GREY)
ax.add_patch(Rectangle((62, 0.6), 8, 0.9, color=GREEN, alpha=.9)); ax.text(71, 1.05, "H_onset: 8 frames from the commit frame\n(deployed read; 'last' = 8th frame, 'mean8' = window mean)", ha="left", va="center", fontsize=8, color=GREEN)
ax.text(-2, 10.4, "One query = one cacheless forward per 80 ms frame (the last frame's forward contains every position). "
        "Hooks on 14 layers (L2, L6, …, L54) store three windows per layer.", fontsize=8.5)
save(fig, "fig1_read_points.png")

# ------------------------------------------------- fig2: calibration composition
g = meta[committed].groupby(["tag", "pool"]).agg(n=("y", "size"), fail=("y", "mean")).reset_index()
order = {"frozen": 0, "expansion": 1, "expansion2": 2}
g["o"] = g.tag.map(order); g = g.sort_values(["o", "pool"])
fig, ax = plt.subplots(figsize=(10, 3.6))
cols = {"frozen": BLUE, "expansion": ORANGE, "expansion2": GREEN}
x = np.arange(len(g))
ax.bar(x, g.n, color=[cols[t] for t in g.tag])
for xi, (n, f) in enumerate(zip(g.n, g.fail)):
    ax.text(xi, n + 4, f"fail {f:.2f}", ha="center", fontsize=7.5, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels(g.pool, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("committed queries"); ax.set_ylim(0, g.n.max() * 1.55)
for t, c in cols.items():
    n = int(g[g.tag == t].n.sum()); ax.bar([0], [0], color=c, label=f"{t} (n={n})")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Calibration set: 2,481 committed English queries, pass-2 labels (fail = escalate_label mean)", fontsize=9.5)
save(fig, "fig2_calib_composition.png")

# ------------------------------------------------- fig3: layer sweeps
sw = {r: json.load(open(GP / "onset" / f"sweep_{r}.json")) for r in ("eot", "onset", "commit")}
v1 = json.load(open(GP / "nvda_probe_sweep.json"))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for r, c, lab in (("eot", BLUE, "eot read: last frame of user audio (n=2,481)"),
                  ("onset", GREEN, "onset read: 8th frame after commit (n=2,481)"),
                  ("commit", ORANGE, "commit read: the commit frame itself (n=2,481)")):
    L = sw[r]["layers"]; ax.plot([int(k) for k in L], list(L.values()), "-o", ms=3.5, color=c, label=lab)
ax.plot([int(k) for k in v1["layers"]], list(v1["layers"].values()), "--", color=GREY, label="8ac original: eot_last, frozen-600 calib only")
ax.axvline(30, color=GREEN, ls=":", lw=1); ax.text(30.4, .70, "deployed L30", color=GREEN, fontsize=8)
ax.set_xlabel("layer (of 56 NemotronH blocks)"); ax.set_ylabel("5-fold OOF AUC, single 'last' feature")
ax.set_xticks(LAYERS); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Layer sweep — usable signal peaks mid-network on every read", fontsize=9.5)
save(fig, "fig3_layer_sweep.png")

# ------------------------------------------------- fig4: feature stacking
fig, ax = plt.subplots(figsize=(7.5, 3.4))
groups = [("eot @L34", sw["eot"]["combos"], ["eot_last", "eot_last+eot_mean8", "eot_last+eot_mean8+user_mean"], BLUE),
          ("onset @L30", sw["onset"]["combos"], ["eot_last", "eot_last+eot_mean8", "eot_last+eot_mean8+user_mean"], GREEN),
          ("commit @L38", sw["commit"]["combos"], ["commit_last", "commit_last+user_mean"], ORANGE)]
labels3 = ["last", "+ mean8", "+ user_mean"]
w = 0.25
for gi, (name, combos, keys, c) in enumerate(groups):
    vals = [combos[k]["oof"] for k in keys]
    xs = gi + np.arange(len(vals)) * w - w * (len(vals) - 1) / 2
    ax.bar(xs, vals, width=w * .9, color=c, alpha=[.45, .7, 1.0][:len(vals)][-1])
    for xi, v, k in zip(xs, vals, (labels3 if len(vals) == 3 else ["last", "+ user_mean"])):
        ax.text(xi, v + .003, f"{v:.3f}\n{k}", ha="center", fontsize=7.5)
ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylim(.74, .85); ax.set_ylabel("OOF AUC (n=2,481)")
ax.set_title("Feature stacking: the 8-frame mean and the user-audio mean each add signal", fontsize=9.5)
save(fig, "fig4_feature_stacking.png")

# ------------------------------------------------- fig5: hillclimb
hc = json.load(open(GP / "onset" / "hillclimb.json"))
hc = sorted(hc, key=lambda r: r["auc"])
fig, ax = plt.subplots(figsize=(8, 5))
names = [r["name"] for r in hc]; aucs = [r["auc"] for r in hc]
colors = [GREEN if n.startswith("A stack@30 C=0.0001 scaler=True") else (RED if n.startswith("E") else BLUE) for n in names]
ax.barh(range(len(hc)), aucs, color=colors)
for i, a in enumerate(aucs):
    ax.text(a + .001, i, f"{a:.4f}", va="center", fontsize=7.5)
ax.set_yticks(range(len(hc))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlim(.76, .83); ax.set_xlabel("OOF AUC (n=2,481)")
ax.set_title("Recipe hill-climb (18 variants). Green = deployed recipe; red = MLP-64 head.\nA: C / scaler sweep on the L30 stack; B: masked mean; C: multi-layer stacks; D: dual read (eot+onset)", fontsize=9)
save(fig, "fig5_hillclimb.png")

# ------------------------------------------------- fig6: external transfer
v2 = json.load(open(GP / "new" / "nvda_probe_sweep_v2.json"))
ext_by_read = {"onset@L30": {"striviaqa": 0.8375, "swebq": 0.8508, "sllama": 0.7712, "sdqa": 0.7741},
               "eot@L34": {"striviaqa": 0.8012, "swebq": 0.8077, "sllama": 0.7548, "sdqa": 0.7605}}
minicpm_native = {"striviaqa": .711, "swebq": .736, "sllama": .757, "sdqa": .736}   # RESULTS §8be, native refit
pools = ["striviaqa", "swebq", "sllama", "sdqa"]
series = [("8ac: 600-row calib, eot@L34 (Aug 19)", v1["combos"]["eot_last+eot_mean8+user_mean"]["ext"], "#9ecae1"),
          ("v2: 2,550-row calib, eot@L34, pass-1 labels", v2["combos"]["eot_last+eot_mean8+user_mean"]["ext"], BLUE),
          ("pass-2: 2,481-row calib, eot@L34", ext_by_read["eot@L34"], "#08519c"),
          ("pass-2: 2,481-row calib, onset@L30 (deployed)", ext_by_read["onset@L30"], GREEN)]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
w = 0.2
for si, (name, d, c) in enumerate(series):
    xs = np.arange(len(pools)) + (si - 1.5) * w
    ax.bar(xs, [d[p] for p in pools], width=w * .92, color=c, label=name)
    for xi, p in zip(xs, pools):
        ax.text(xi, d[p] + .004, f"{d[p]:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
ax.scatter(np.arange(len(pools)), [minicpm_native[p] for p in pools], marker="_", s=600, color=RED, lw=2, label="MiniCPM-o 4.5 native in-regime refit (RESULTS §8be)", zorder=5)
ax.set_xticks(range(len(pools))); ax.set_xticklabels(pools)
ax.set_ylim(.6, .93); ax.set_ylabel("cold external AUC (probe fit on calib only)")
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
ax.set_title("External transfer, n = 250/250/250/200 (single-pool SE ≈ ±.03–.04)", fontsize=9.5)
save(fig, "fig6_external_transfer.png")

# ------------------------------------------------- fig7: commit gap
mc = meta[committed & meta.n_frames.notna()]
gap = (mc.onset - mc.n_frames) * 0.08
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(gap.clip(-6, 6), bins=60, color=GREEN, alpha=.85)
ax.axvline(0, color="black", lw=1); ax.axvline(gap.median(), color=RED, ls="--", lw=1)
ax.text(gap.median() + .1, ax.get_ylim()[1] * .9, f"median {gap.median():+.2f} s", color=RED, fontsize=8)
ax.set_xlabel("commit frame − end of user audio  (s; clipped to ±6)"); ax.set_ylabel("queries")
ax.set_title(f"When does the model commit to speak?  {(gap < 0).mean():.0%} before the audio ends, {(gap.abs() <= 1).mean():.0%} within ±1 s, "
             f"{(~committed).sum()} never commit (dropped)", fontsize=9)
save(fig, "fig7_commit_gap.png")

# ------------------------------------------------- fig8/9: OOF scores, within-pool AUC, thresholds
X = stack(E_on[committed], M[committed], J30); y = meta.y[committed].values
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(probe(), X, y, cv=cv, method="predict_proba")[:, 1]
print("pooled OOF AUC", round(roc_auc_score(y, oof), 4))
pools_c = meta.pool[committed].values
rows = []
for p in sorted(set(pools_c)):
    m = pools_c == p
    if len(set(y[m])) > 1:
        rows.append((p, m.sum(), y[m].mean(), roc_auc_score(y[m], oof[m])))
wp = pd.DataFrame(rows, columns=["pool", "n", "fail", "auc"]).sort_values("auc")
fig, ax = plt.subplots(figsize=(7.5, 4))
sc = ax.scatter(wp.fail, wp.auc, s=wp.n * 1.2, c=BLUE, alpha=.6, edgecolor="k")
for _, r in wp.iterrows():
    ax.annotate(f"{r.pool} (n={r.n})", (r.fail, r.auc), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
ax.axhline(roc_auc_score(y, oof), color=GREEN, ls="--", lw=1); ax.text(.27, roc_auc_score(y, oof) + .008, f"pooled OOF AUC {roc_auc_score(y, oof):.3f}", color=GREEN, fontsize=8)
ax.axhline(.5, color=GREY, lw=.8)
ax.set_xlabel("pool fail rate"); ax.set_ylabel("within-pool AUC of the pooled OOF scores")
ax.set_title("Diagnostic: the pooled AUC is partly pool identity / base rate — within-pool ranking is weaker", fontsize=9.5)
save(fig, "fig8_within_pool_auc.png")

thr = json.load(open(GP / "new" / "demo" / "gate_demo_nvda.json"))["fail"]["thresholds"]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.hist(oof[y == 0], bins=40, alpha=.6, color=GREEN, label=f"model correct (n={int((y==0).sum())})", density=True)
ax.hist(oof[y == 1], bins=40, alpha=.6, color=RED, label=f"model fails (n={int((y==1).sum())})", density=True)
for k, v in thr.items():
    ax.axvline(v, color="black", ls=":" , lw=1); ax.text(v, ax.get_ylim()[1] * .95, f"{k}\n{v:.3f}", ha="center", fontsize=7)
ax.set_xlabel("OOF P(fail) at the deployed read (onset @L30)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Scores and the three deployed tiers (thresholds = OOF quantiles for 15 / 30 / 50 % escalation; demo artifact)", fontsize=9)
save(fig, "fig9_score_distribution.png")

json.dump({"pooled_oof_auc": float(roc_auc_score(y, oof)),
           "within_pool": wp.to_dict(orient="records"),
           "commit_gap": {"median_s": float(gap.median()), "frac_before_end": float((gap < 0).mean()),
                          "frac_within_1s": float((gap.abs() <= 1).mean()), "no_commit": int((~committed).sum())},
           "calib_composition": g.drop(columns="o").to_dict(orient="records")},
          open(HERE / "doc_numbers.json", "w"), indent=1)
print("wrote doc_numbers.json")
