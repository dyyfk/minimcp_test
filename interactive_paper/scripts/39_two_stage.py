"""8bx: two-stage gate — commit-point probe first, k2 re-score only for a
gray band. Same dumps/probes as scripts/38.

Policy P(r, d, f) on a pool of N rows, exact budget r (fraction escalated):
  A = top a*N rows by ONSET score, a = r*(1-f)       -> fire at the commit (no delay)
  B = next d*N rows by ONSET score (the gray band)   -> decision deferred to k2
  within B, fire the top (r*N - |A|) rows by K2 score
  everything else answers locally with no delay.
d = 0 is the deployed one-stage gate; f = 1, d = 1 is "k2 only".
Delivered accuracy = expert outcome (native bench TTS-relay always arm) on
fired rows, local outcome (this dump) otherwise. Reported per pool and as
the external-5 mean, with the deferred fraction d as the latency cost
(each deferred turn waits ~2 answer chunks before its decision).

Usage: .venv_boot\\Scripts\\python.exe scripts\\39_two_stage.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

D = Path("data"); NB = Path("data/native_bench")
TRAIN = ["calibk", "expk", "exp2k", "exp3k", "exp3zhk", "freshk"]
EVAL = [("testk", "frozen", "adequate", "internal"), ("striviaqak", "striviaqa", "oab_ok", "TriviaQA"),
        ("swebqk", "swebq", "oab_ok", "WebQ"), ("sllamak", "sllama", "oab_ok", "LlamaQ"),
        ("sdqak", "sdqa", "adequate", "SD-QA"), ("sreasonk", "sreason", "adequate", "Reasoning-zh")]
C = 3e-4
BUDGETS = [.15, .30, .50]
BANDS = [0.0, .1, .2, .3, .5, 1.0]
FRACS = [.5, 1.0]


def load(tag, keys=("X", "X_k2")):
    ids, arrs = [], {k: [] for k in keys}
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True); ids += list(z["ids"])
        for k in keys:
            arrs[k].append(z[k])
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids))).drop_duplicates("id", keep="last")
    sel = df["row"].to_numpy()
    lab = pd.read_parquet(D / f"frozen_native_{tag}_judged.parquet").drop_duplicates("id", keep="last").set_index("id")
    y = lab["escalate_label"].reindex(df["id"]).to_numpy(float); keep = ~np.isnan(y)
    return df["id"].to_numpy()[keep], {k: np.concatenate(v)[sel][keep] for k, v in arrs.items()}, y[keep].astype(int)


Xtr = {"X": [], "X_k2": []}; ytr = []
for t in TRAIN:
    ids, X, y = load(t)
    if t == "freshk":
        fl = pd.read_parquet(D / "fresh_labels.parquet").set_index("id")["split"]
        m = np.array([fl.get(i) == "train" for i in ids]); X = {k: v[m] for k, v in X.items()}; y = y[m]
    for k in Xtr:
        Xtr[k].append(X[k])
    ytr.append(y)
ytr = np.concatenate(ytr)
probe = {k: LogisticRegression(C=C, max_iter=5000).fit(np.concatenate(v), ytr) for k, v in Xtr.items()}
print("train rows", len(ytr))

pools = {}
for tag, pool, col, nice in EVAL:
    ids, X, y = load(tag)
    a = pd.read_parquet(NB / f"{pool}_always_tts_judged.parquet").drop_duplicates("id", keep="last").set_index("id")[col]
    e = a.reindex(ids).to_numpy(float); ok = ~np.isnan(e)
    pools[tag] = {"nice": nice, "lo": (1 - y[ok]).astype(float), "eo": e[ok],
                  "s_on": probe["X"].decision_function(X["X"][ok]), "s_k2": probe["X_k2"].decision_function(X["X_k2"][ok])}
    print(f"{nice:13s} n={ok.sum()} floor={pools[tag]['lo'].mean():.3f} ceil={pools[tag]['eo'].mean():.3f}")


def policy(P, r, d, f):
    N = len(P["lo"]); k = int(round(r * N))
    order_on = np.argsort(-P["s_on"])
    a = int(round(k * (1 - f))) if d > 0 else k
    a = min(a, k)
    A = order_on[:a]
    nb = int(round(d * N)) if d < 1 else N - a
    B = order_on[a:a + nb]
    need = k - a
    fire = np.zeros(N, bool); fire[A] = True
    if need > 0 and len(B):
        kb = np.argsort(-P["s_k2"][B])[:need]
        fire[B[kb]] = True
    acc = float(np.where(fire, P["eo"], P["lo"]).mean())
    return acc, float(fire.mean()), len(B) / N


res = {}
ext = [t for t, _, _, _ in EVAL if t != "testk"]
print("\nexternal-5 mean delivered accuracy (rows: budget; cols: band d / fraction f of the budget decided at k2)")
for r in BUDGETS:
    res[r] = {}
    line = []
    for d in BANDS:
        for f in FRACS:
            if d == 0 and f != 1.0:
                continue
            if d == 1.0 and f != 1.0:
                continue
            if 0 < d < 1 and d < r * f:      # band too small to hold the k2-decided fires
                continue
            accs = {t: policy(pools[t], r, d, f) for t in pools}
            m = float(np.mean([accs[t][0] for t in ext]))
            res[r][f"d{d}_f{f}"] = {"ext5": m, "pools": {t: accs[t][0] for t in pools}, "deferred": float(np.mean([accs[t][2] for t in ext]))}
            line.append(f"d={d:.1f}{'' if d in (0, 1) else f'/f={f}'}:{m:.3f}")
    print(f"r={r:.2f}  " + "  ".join(line))
print("\nper pool at r=.30: one-stage (onset) | band .3 all-at-k2 | band .5 half-at-k2 | k2-only:")
for t in pools:
    o = res[.3]["d0.0_f1.0"]["pools"][t]; b3 = res[.3]["d0.3_f1.0"]["pools"][t]; b5 = res[.3]["d0.5_f0.5"]["pools"][t]; k = res[.3]["d1.0_f1.0"]["pools"][t]
    print(f"  {pools[t]['nice']:13s} onset {o:.3f}  band.3 {b3:.3f}  band.5/half {b5:.3f}  k2-only {k:.3f}")
Path("figures/two_stage.json").write_text(json.dumps(res, indent=1))

# ---- figure: ext-5 accuracy vs deferred fraction, one line per budget ----
BLUE, TEAL, ORANGE, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#8a97a5"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
for r, col in zip(BUDGETS, (TEAL, BLUE, ORANGE)):
    xs, ys = [], []
    for d in BANDS:
        key = f"d{d}_f1.0" if f"d{d}_f1.0" in res[r] else f"d{d}_f0.5"
        if key not in res[r]:
            continue
        xs.append(res[r][key]["deferred"] if d < 1 else 1.0); ys.append(res[r][key]["ext5"])
    ax.plot(xs, ys, "o-", color=col, label=f"budget {int(r*100)}%")
    ax.annotate("one-stage (deployed)", (xs[0], ys[0]), textcoords="offset points", xytext=(6, -12), fontsize=8, color=col)
ax.set_xlabel("fraction of turns whose decision is deferred to k2 (~2 s later)")
ax.set_ylabel("external-5 mean delivered accuracy")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Two-stage gate: commit-point probe + k2 re-score on a gray band", fontsize=10, loc="left")
ax = axes[1]
names = [pools[t]["nice"] for t in pools]; x = np.arange(len(names)); w = .2
for j, (key, lab, col) in enumerate([("d0.0_f1.0", "one-stage (onset)", GREY), ("d0.3_f1.0", "band 30% -> k2", BLUE),
                                     ("d0.5_f0.5", "band 50%, half the budget at k2", "#1d4f80"), ("d1.0_f1.0", "k2 only (all deferred)", TEAL)]):
    ax.bar(x + (j - 1.5) * w, [res[.3][key]["pools"][t] for t in pools], w, color=col, label=lab)
ax.set_xticks(x, names, fontsize=8.5); ax.set_ylabel("delivered accuracy @30% budget")
ax.set_ylim(.4, 1.0); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Per pool at the balanced (30%) budget", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("figures/two_stage.png", dpi=170)
print("wrote figures/two_stage.{json,png}")
