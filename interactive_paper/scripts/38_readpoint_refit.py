"""8bw: does a LATER read point help? Same 8bq recipe (L2 logistic,
C=3e-4, native labels), trained and evaluated on the same fresh official-
config dumps (`*k` tags), one probe per read point:

  pre   : L22 before the onset chunk's generate (no answer token)
  onset : the deployed read (after the onset chunk's generate)
  k1/k2/k3 : after the 1st / 2nd / 3rd answer chunk following onset
  onset+k1 : mean of the two probes' logits (complementarity check)

Labels: escalate_label from frozen_native_{tag}_judged.parquet of the SAME
dump, so features and labels come from one generation. Train = calibk +
expk + exp2k + exp3k + exp3zhk + freshk(train split); eval = testk,
striviaqak, swebqk, sllamak, sdqak, sreasonk. Reports row-random 5-fold
OOF AUC, per-pool eval AUC, En-4 / external means, cascade accuracy at
exact 30% budget (native local vs expert bound from the TTS-relay always
arm of the native bench where the ids match), and the share of rows
whose k-th read had to be padded (turn ended earlier).

Usage: .venv_boot\\Scripts\\python.exe scripts\\38_readpoint_refit.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
TRAIN = ["calibk", "expk", "exp2k", "exp3k", "exp3zhk", "freshk"]
EVAL = [("testk", "internal test"), ("striviaqak", "TriviaQA"), ("swebqk", "WebQ"),
        ("sllamak", "LlamaQ"), ("sdqak", "SD-QA"), ("sreasonk", "Reasoning-zh")]
READS = [("X_pre", "pre"), ("X", "onset"), ("X_k1", "k1"), ("X_k2", "k2"), ("X_k3", "k3")]
C = 3e-4


def load(tag):
    ids, arrs, npost = [], {k: [] for k, _ in READS}, []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        for k, _ in READS:
            arrs[k].append(z[k])
        npost.append(z["n_post"])
    if not ids or not (D / f"frozen_native_{tag}_judged.parquet").exists():
        return None
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids))).drop_duplicates("id", keep="last")
    sel = df["row"].to_numpy()
    X = {k: np.concatenate(v)[sel] for k, v in arrs.items()}
    lab = pd.read_parquet(D / f"frozen_native_{tag}_judged.parquet").drop_duplicates("id", keep="last").set_index("id")
    y = lab["escalate_label"].reindex(df["id"]).to_numpy(float)
    keep = ~np.isnan(y)
    return {"ids": df["id"].to_numpy()[keep], "X": {k: v[keep] for k, v in X.items()},
            "y": y[keep].astype(int), "n_post": np.concatenate(npost)[sel][keep]}


tr = {t: load(t) for t in TRAIN}
missing = [t for t, v in tr.items() if v is None]
if missing:
    print("missing training tags:", missing)
tr = {t: v for t, v in tr.items() if v is not None}
if "freshk" in tr:      # fresh: train split only (heldout stays out, as in 8bq)
    fl = pd.read_parquet(D / "fresh_labels.parquet").set_index("id")["split"]
    m = np.array([fl.get(i) == "train" for i in tr["freshk"]["ids"]])
    tr["freshk"] = {"ids": tr["freshk"]["ids"][m], "X": {k: v[m] for k, v in tr["freshk"]["X"].items()},
                    "y": tr["freshk"]["y"][m], "n_post": tr["freshk"]["n_post"][m]}
y_tr = np.concatenate([v["y"] for v in tr.values()])
npost_tr = np.concatenate([v["n_post"] for v in tr.values()])
print(f"train rows {len(y_tr)} from {list(tr)}  fail rate {y_tr.mean():.3f}")
print("share of train rows with n_post >= k:", {k: round(float((npost_tr >= k).mean()), 3) for k in (1, 2, 3)})
ev = {t: load(t) for t, _ in EVAL}
ev = {t: v for t, v in ev.items() if v is not None}

# optional expert bound per id from the native bench always arm (TTS relay)
NB = Path("data/native_bench")
POOLMAP = {"testk": ("frozen", "adequate"), "striviaqak": ("striviaqa", "oab_ok"), "swebqk": ("swebq", "oab_ok"),
           "sllamak": ("sllama", "oab_ok"), "sdqak": ("sdqa", "adequate"), "sreasonk": ("sreason", "adequate")}
expert = {}
for t, (pool, col) in POOLMAP.items():
    p = NB / f"{pool}_always_tts_judged.parquet"
    if p.exists():
        a = pd.read_parquet(p).drop_duplicates("id", keep="last").set_index("id")[col]
        expert[t] = a

res = {}
scores = {}
for key, name in READS + [(None, "onset+k1")]:
    if key is None:
        if "onset" not in scores or "k1" not in scores:
            continue
        s_tr = (scores["onset"]["oof"] + scores["k1"]["oof"]) / 2
        s_ev = {t: (scores["onset"]["ev"][t] + scores["k1"]["ev"][t]) / 2 for t in ev}
    else:
        X_tr = np.concatenate([v["X"][key] for v in tr.values()])
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        s_tr = cross_val_predict(LogisticRegression(C=C, max_iter=5000), X_tr, y_tr, cv=cv,
                                 method="decision_function")
        clf = LogisticRegression(C=C, max_iter=5000).fit(X_tr, y_tr)
        s_ev = {t: clf.decision_function(v["X"][key]) for t, v in ev.items()}
        scores[name] = {"oof": s_tr, "ev": s_ev}
    r = {"oof_auc": round(roc_auc_score(y_tr, s_tr), 4), "pools": {}}
    for t, nice in EVAL:
        if t not in ev:
            continue
        y = ev[t]["y"]; s = s_ev[t]
        auc = roc_auc_score(y, s)
        entry = {"auc": round(auc, 4), "n": int(len(y))}
        if t in expert:
            e = expert[t].reindex(ev[t]["ids"]).to_numpy(float)
            ok = ~np.isnan(e)
            if ok.sum() > 50:
                lo, eo, ss = 1 - y[ok], e[ok], s[ok]
                k30 = int(round(.3 * ok.sum())); order = np.argsort(-ss)
                esc = np.zeros(ok.sum(), bool); esc[order[:k30]] = True
                entry["casc30"] = round(float(np.where(esc, eo, lo).mean()), 4)
                entry["floor"] = round(float(lo.mean()), 4); entry["ceil"] = round(float(eo.mean()), 4)
        r["pools"][t] = entry
    en4 = [r["pools"][t]["auc"] for t in ("striviaqak", "swebqk", "sllamak", "sdqak") if t in r["pools"]]
    ext5 = en4 + ([r["pools"]["sreasonk"]["auc"] if "sreasonk" in r["pools"] else None] if "sreasonk" in r["pools"] else [])
    r["en4_mean"] = round(float(np.mean(en4)), 4) if en4 else None
    r["ext5_mean"] = round(float(np.mean([a for a in ext5 if a is not None])), 4) if ext5 else None
    res[name] = r
    print(f"{name:9s} OOF {r['oof_auc']:.3f} | " + "  ".join(f"{t[:-1]}={r['pools'][t]['auc']:.3f}" for t in r["pools"])
          + f" | En-4 {r['en4_mean']}  ext-5 {r['ext5_mean']}")
Path("figures/readpoint_refit.json").write_text(json.dumps(res, indent=1))

# ---- figure ----
names = [n for n in ["pre", "onset", "k1", "k2", "k3", "onset+k1"] if n in res]
pools = [t for t, _ in EVAL if t in ev]
BLUE, GREY, TEAL = "#2a78d6", "#8a97a5", "#1baf7a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
x = np.arange(len(pools)); w = .8 / len(names)
cols = plt.cm.Blues(np.linspace(.35, .95, len(names)))
for j, n in enumerate(names):
    ax.bar(x + (j - (len(names) - 1) / 2) * w, [res[n]["pools"][t]["auc"] for t in pools], w, color=cols[j], label=n)
ax.set_xticks(x, [dict(EVAL)[t] for t in pools], fontsize=9)
ax.set_ylim(.5, 1.0); ax.set_ylabel("native-failure AUC")
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", title="read point")
ax.set_title("Where the probe reads: before onset, at onset (deployed), after 1-3 answer chunks", fontsize=10, loc="left")
ax = axes[1]
ax.plot(names, [res[n]["oof_auc"] for n in names], "o-", color=GREY, label="train OOF")
ax.plot(names, [res[n]["en4_mean"] for n in names], "s-", color=BLUE, label="En-4 external mean")
if all(res[n]["ext5_mean"] is not None for n in names):
    ax.plot(names, [res[n]["ext5_mean"] for n in names], "^-", color=TEAL, label="ext-5 mean (with zh)")
ax.set_ylabel("AUC"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Aggregate by read point", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("figures/readpoint_refit.png", dpi=170)
print("wrote figures/readpoint_refit.{json,png}")
