"""Stop-word negatives: quantify the false-fire bug on NVDA, then test
both fixes head-to-head.

  A (8bh design): second linear head, question vs floor-move, on
    the same deployed read; escalate iff P(info)>=tau_act AND
    P(fail)>=tier.
  B (negative-mixing design): mix floor stims into the FAILURE probe training
    as label=0 ("should not escalate").

All probes: v3-style stack (window-last + window-mean8 + user_mean) at
L30 on the onset (deployed) read; StandardScaler + LR C=1e-4;
StratifiedKFold(5, seed 42). Question labels = pass-2 parquets.
Floor stims scored out-of-fold wherever they inform training.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.environ.get(
    "ACT_BASE",
    str(Path(__file__).resolve().parents[3]),
)
SHARDS = os.environ.get("ACT_SHARDS", f"{BASE}/onset")
LABELS_DIR = os.environ.get("ACT_LABELS", f"{BASE}/onset_fit")
QFILE = os.environ.get("ACT_QFILE", f"{BASE}/queries_flooract.jsonl")
LAYERS = list(range(2, 56, 4))
J30 = LAYERS.index(30)


def stack(E, M, j):
    return np.concatenate([E[:, j, -1], E[:, j].mean(1), M[:, j]], axis=1)


def load_questions():
    ids, E, M, ONS = [], [], [], []
    for tag in ("frozen", "expansion", "expansion2"):
        for sh in sorted(glob.glob(f"{SHARDS}/nvda_h_{tag}.shard*.npz")):
            z = np.load(sh, allow_pickle=True)
            ids += [(tag, str(x)) for x in z["ids"]]
            E.append(z["H_onset"]); M.append(z["H_mean"])
            ONS.append(z["onset_frame"])
    E, M, ONS = np.concatenate(E), np.concatenate(M), np.concatenate(ONS)
    lab = {}
    for tag in ("frozen", "expansion", "expansion2"):
        df = pd.read_parquet(f"{LABELS_DIR}/nvda_{tag}.parquet")
        for _, r in df.iterrows():
            if pd.notna(r["escalate_label"]):
                lab[(tag, r["id"])] = int(r["escalate_label"])
    keep = [i for i, k in enumerate(ids) if k in lab and ONS[i] >= 0]
    y = np.array([lab[ids[i]] for i in keep])
    return (E[keep].astype(np.float32), M[keep].astype(np.float32), y)


def load_floor():
    z = np.load(f"{SHARDS}/nvda_h_flooract.shard0.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    E = z["H_onset"].astype(np.float32)
    M = z["H_mean"].astype(np.float32)
    ONS = z["onset_frame"]
    cat = {}
    for l in open(QFILE):
        q = json.loads(l)
        cat[q["id"]] = q["pool"].split("-")[1]
    cats = np.array([cat[i] for i in ids])
    return ids, E, M, np.array(ONS), cats


EQ, MQ, yq = load_questions()
print(f"questions n={len(yq)} fail={yq.mean():.3f}")
fids, EF, MF, ONF, cats = load_floor()
commit = ONF >= 0
print(f"floor stims n={len(fids)}; committed {int(commit.sum())} "
      f"({100 * commit.mean():.0f}%) — no-commit = gate never reads (ideal)")
for c in sorted(set(cats)):
    m = cats == c
    print(f"  {c:8s} n={int(m.sum()):3d} commit-rate "
          f"{100 * commit[m].mean():.0f}%")

Xq = stack(EQ, MQ, J30)
Xf = stack(EF, MF, J30)[commit]
fcats = cats[commit]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
mk = lambda: make_pipeline(StandardScaler(),
                           LogisticRegression(C=1e-4, max_iter=5000))

# ---- bug quantification: current failure probe on floor stims -------
oof = cross_val_predict(mk(), Xq, yq, cv=cv, method="predict_proba")[:, 1]
print(f"\nfailure probe question OOF AUC: {roc_auc_score(yq, oof):.4f}")
probe = mk().fit(Xq, yq)
sf = probe.predict_proba(Xf)[:, 1]
print("\n== BUG: false-fire rate of the CURRENT probe on committed floor stims ==")
thr = {t: float(np.quantile(oof, 1 - b))
       for t, b in (("cons@15%", .15), ("bal@30%", .30), ("agg@50%", .50))}
for t, v in thr.items():
    ff = sf >= v
    per = {c: f"{100 * ff[fcats == c].mean():.0f}%" for c in sorted(set(fcats))}
    print(f"  {t}: {100 * ff.mean():.1f}% overall | {per}")

# ---- Design A: act head --------------------------------------------
# q-side metrics: stratified OOF over questions (tau + the REAL lost
# metric = fraction of true escalations, yq==1, blocked by the gate).
# floor-side metrics: LEAVE-ONE-CATEGORY-OUT — the model never sees
# any stim of the held-out category's pool (verifier finding 2: the
# 196 stims are 4 templated pools; stratified folds leak siblings).
Xa = np.concatenate([Xq, Xf])
ya = np.concatenate([np.ones(len(Xq)), np.zeros(len(Xf))])
oof_a = cross_val_predict(mk(), Xa, ya, cv=cv, method="predict_proba")[:, 1]
auc_a = roc_auc_score(ya, oof_a)
q_scores = oof_a[:len(Xq)]
tau = float(np.quantile(q_scores, 0.005))          # question-side 0.5th pct
lost_all = 100 * np.mean(q_scores < tau)           # ~0.5 by construction
lost_true = 100 * np.mean(q_scores[yq == 1] < tau)  # the real cost
f_lopo = np.zeros(len(Xf))
for c in sorted(set(fcats)):
    m = fcats == c
    Xtr = np.concatenate([Xq, Xf[~m]])
    ytr = np.concatenate([np.ones(len(Xq)), np.zeros(int((~m).sum()))])
    f_lopo[m] = mk().fit(Xtr, ytr).predict_proba(Xf[m])[:, 1]
passed = 100 * np.mean(f_lopo >= tau)
print(f"\n== A: act head (question vs floor) ==")
print(f"  OOF AUC {auc_a:.4f} | tau@q0.5pct={tau:.4f}")
print(f"  q-side blocked: all {lost_all:.2f}% | TRUE escalations "
      f"(yq=1) blocked {lost_true:.2f}%  <- the real cost")
print(f"  floor stims passing act gate (leave-one-pool-out): "
      f"{passed:.2f}%")
for t, v in thr.items():
    ff = (sf >= v) & (f_lopo >= tau)
    per = {c: f"{100 * ff[fcats == c].mean():.0f}%"
           for c in sorted(set(fcats))}
    print(f"  after fix {t}: {100 * ff.mean():.1f}% false-fire | {per}")

# ---- Design B: mix negatives into the failure probe ----------------
# question AUC via stratified OOF (valid, verifier-clean); floor
# false-fire via the same leave-one-category-out discipline.
Xb = np.concatenate([Xq, Xf])
yb = np.concatenate([yq, np.zeros(len(Xf), dtype=int)])
oof_b = cross_val_predict(mk(), Xb, yb, cv=cv, method="predict_proba")[:, 1]
qb = oof_b[:len(Xq)]
auc_q_after = roc_auc_score(yq, qb)
fb_lopo = np.zeros(len(Xf))
for c in sorted(set(fcats)):
    m = fcats == c
    Xtr = np.concatenate([Xq, Xf[~m]])
    ytr = np.concatenate([yq, np.zeros(int((~m).sum()), dtype=int)])
    fb_lopo[m] = mk().fit(Xtr, ytr).predict_proba(Xf[m])[:, 1]
print(f"\n== B: negatives mixed into failure probe ==")
print(f"  question-only OOF AUC after mixing: {auc_q_after:.4f} "
      f"(before {roc_auc_score(yq, oof):.4f})")
thr_b = {t: float(np.quantile(qb, 1 - b))
         for t, b in (("cons@15%", .15), ("bal@30%", .30), ("agg@50%", .50))}
for t, v in thr_b.items():
    ff = fb_lopo >= v
    print(f"  {t} (leave-one-pool-out): {100 * ff.mean():.1f}% false-fire")

json.dump({"bug_false_fire": {t: float(np.mean(sf >= v)) for t, v in thr.items()},
           "act_auc": float(auc_a),
           "act_lost_true_pct": float(lost_true),
           "act_lost_all_pct": float(lost_all),
           "act_pass_lopo_pct": float(passed),
           "mixed_q_auc": float(auc_q_after)},
          open(f"{SHARDS}/act_analysis.json", "w"), indent=1)
print("\n>>> wrote act_analysis.json")
