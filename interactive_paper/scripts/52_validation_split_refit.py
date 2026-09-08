"""Independent validation split + gate refit (P1 step 3, approved).

Carves a stratified validation set OUT of the no-leak fit (core rows
only; the 7 leak-excluded ids stay excluded), refits the gate on the
remaining rows, and writes:
  data/queries_validation.jsonl        - runnable by modal_native_bench2
                                         ("validation" pool; per-row
                                         audio path + lang + reference)
  data/gate_native_v2.json             - refit artifact: w/b, per-lang
                                         quantile grid 10-50% from the
                                         REFIT OOF (validation rows are
                                         out-of-fit; their scores come
                                         from the fitted model)
  data/paper_revision_pr0907/validation_split_manifest.jsonl
  data/paper_revision_pr0907/validation_scores.parquet

Stratification: seed 20260907, per (part, native label), targets
~240 en + ~60 zh. No validation row enters the fit, the OOF, or the
quantile base. Guard: testoff AUC of the refit vs deployed.

Usage: set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\52_validation_split_refit.py
"""
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
OUT = D / "paper_revision_pr0907"
SEED = 20260907
N_EN, N_ZH = 240, 60
AUDIO_DIR = {"caliboff": "audio_pool", "expoff": "audio_expansion",
             "exp2off": "audio_expansion2", "exp3off": "audio_expansion3",
             "exp3zhoff": "audio_expansion3zh"}

m = types.ModuleType("sw")
exec(open("scripts/40_threshold_sweep.py").read().split("if __name__")[0], m.__dict__)
la = types.ModuleType("la")
exec(open("scripts/41_leak_audit.py").read().split("def main")[0], la.__dict__)
EXCLUDE = json.loads((D / "leak_exclude.json").read_text())

# rebuild the no-leak merge exactly as scripts/43 / 48
ids, Xs, ys, parts = [], [], [], []
for tag in la.DEPLOYED_PARTS:
    i, X = m.load_feats(tag)
    y = pd.read_parquet(D / f"frozen_native_{tag}_judged.parquet").set_index(
        "id")["escalate_label"].reindex(i).to_numpy(float)
    k = ~np.isnan(y)
    kept = [q for q, kk in zip(i, k) if kk]
    ids += kept
    Xs.append(X[k])
    ys.append(y[k].astype(int))
    parts += [tag] * len(kept)
fl = pd.read_parquet(D / "fresh_labels.parquet")
fl = fl[fl["escalate_label"].notna()]
lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
nf = pd.read_parquet(D / "frozen_native_freshoff_judged.parquet").set_index(
    "id")["escalate_label"]
for i, p in zip(fl["id"], fl["pool"]):
    if p != "fresh_fast" and i in nf.index and pd.notna(nf[i]):
        lab_f[i] = int(nf[i])
split_f = dict(zip(fl["id"], fl["split"]))
ids_fr, X_fr = m.load_feats("freshoff")
tr_j = [j for j, i in enumerate(ids_fr)
        if i in lab_f and split_f.get(i) == "train"]
n0 = len(ids)
ids += [ids_fr[j] for j in tr_j]
X = np.concatenate(Xs + [X_fr[tr_j]])
y = np.concatenate(ys + [[lab_f[ids_fr[j]] for j in tr_j]])
parts += ["freshoff"] * len(tr_j)
lang = np.array(["zh" if t == "exp3zhoff" else "en" for t in parts])
core = np.arange(len(y)) < n0
keep = np.array([i not in EXCLUDE for i in ids])
print(f"no-leak merge {keep.sum()} rows")

# query text + reference per id (for the runnable validation file)
src = {}
for f in ["queries.jsonl"] + la.CALIB_FILES:
    p = D / f
    if not p.exists():
        continue
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            q = json.loads(ln)
            if q.get("id"):
                src.setdefault(q["id"], q)

# stratified validation draw: core, no-leak, has query+reference
rng = np.random.default_rng(SEED)
elig = [j for j in range(len(y))
        if core[j] and keep[j] and ids[j] in src
        and src[ids[j]].get("reference_answer")]
df_e = pd.DataFrame({"j": elig,
                     "part": [parts[j] for j in elig],
                     "lang": [lang[j] for j in elig],
                     "label": [y[j] for j in elig]})
val_j = []
for lg, target in (("en", N_EN), ("zh", N_ZH)):
    sub = df_e[df_e.lang == lg]
    for (pt, lb), g in sub.groupby(["part", "label"]):
        take = int(round(target * len(g) / len(sub)))
        take = min(take, len(g))
        val_j += list(rng.choice(g["j"].to_numpy(), size=take,
                                 replace=False))
val_j = sorted(set(val_j))
val_ids = {ids[j] for j in val_j}
print(f"validation {len(val_j)} rows "
      f"(en {sum(lang[j] == 'en' for j in val_j)}, "
      f"zh {sum(lang[j] == 'zh' for j in val_j)})")

fitmask = keep & ~np.isin(np.arange(len(y)), val_j)
Xf, yf = X[fitmask], y[fitmask]
core_f, lang_f = core[fitmask], lang[fitmask]
ids_f = [i for i, kk in zip(ids, fitmask) if kk]
print(f"fit {len(yf)} rows")

cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(LogisticRegression(C=3e-4, max_iter=5000),
                        Xf, yf, cv=cv, method="predict_proba")[:, 1]
print(f"refit OOF AUC {roc_auc_score(yf, oof):.4f}")
clf = LogisticRegression(C=3e-4, max_iter=5000).fit(Xf, yf)

# guard: internal frozen-test AUC vs deployed
dep = json.loads((D / "gate_native.json").read_text())
tst_ids, Xt = m.load_feats("testoff")
yt = pd.read_parquet(D / "frozen_native_testoff_judged.parquet").set_index(
    "id")["escalate_label"].reindex(tst_ids).to_numpy(float)
k = ~np.isnan(yt)
a_dep = roc_auc_score(yt[k].astype(int), Xt[k] @ np.array(dep["w"]) + dep["b"])
a_new = roc_auc_score(yt[k].astype(int), Xt[k] @ clf.coef_[0] + clf.intercept_[0])
print(f"guard testoff AUC: deployed {a_dep:.4f} -> v2 refit {a_new:.4f}")

GRID = [.10, .15, .20, .25, .30, .40, .50]
grid = {}
for lg in ("en", "zh"):
    mm = core_f & (lang_f == lg)
    grid[lg] = {str(r): float(np.quantile(oof[mm], 1 - r)) for r in GRID}
def tiers(rates):
    return {lg: {t: grid[lg][str(r)] for t, r in rates.items()}
            for lg in grid}

art = {
    "w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
    "layer": dep["layer"], "k_eot": dep["k_eot"], "modes": dep["modes"],
    "C": 0.0003, "train_n": int(len(yf)), "label_source": "native",
    "recipe": "scripts/52 v2 refit: no-leak merge minus the "
              f"{len(val_j)}-row validation split (seed {SEED})",
    "excluded_leak": sorted(EXCLUDE),
    "validation_ids_sha1": hashlib.sha1(
        "".join(sorted(val_ids)).encode()).hexdigest(),
    "train_ids_sha1": hashlib.sha1(
        "".join(sorted(ids_f)).encode()).hexdigest(),
    "oof_auc": round(float(roc_auc_score(yf, oof)), 4),
    "guard_testoff_auc": {"deployed": round(float(a_dep), 4),
                          "v2_refit": round(float(a_new), 4)},
    "eot_thresholds_lang": tiers(
        {"conservative": .15, "balanced": .30, "aggressive": .50}),
    "eot_thresholds_8bz_lang": tiers(
        {"conservative": .15, "balanced": .25, "aggressive": .40}),
    "nominal_grid_lang": grid,
    "quantile_base": "core fit rows (no fresh, no leak, no validation)",
}
(D / "gate_native_v2.json").write_text(json.dumps(art))
print("wrote data/gate_native_v2.json")

# runnable validation pool + manifests + out-of-fit fitted scores
scores = 1.0 / (1.0 + np.exp(-(X[val_j] @ clf.coef_[0]
                               + clf.intercept_[0])))
with open(D / "queries_validation.jsonl", "w", encoding="utf-8") as fh, \
     open(OUT / "validation_split_manifest.jsonl", "w",
          encoding="utf-8") as mh:
    for j, sc in zip(val_j, scores):
        qid = ids[j]
        q = src[qid]
        row = {"id": qid, "query": q.get("query"),
               "reference_answer": q.get("reference_answer"),
               "lang": str(lang[j]), "pool": q.get("pool"),
               "part": parts[j], "split": "validation",
               "audio": f"/data/{AUDIO_DIR[parts[j]]}/{qid}.wav"}
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        mh.write(json.dumps({**row, "label_native": int(y[j]),
                             "fitted_score_v2": float(sc)},
                            ensure_ascii=False) + "\n")
pd.DataFrame({"id": [ids[j] for j in val_j],
              "part": [parts[j] for j in val_j],
              "lang": [str(lang[j]) for j in val_j],
              "label_native": [int(y[j]) for j in val_j],
              "fitted_score_v2": scores}).to_parquet(
    OUT / "validation_scores.parquet", index=False)
print(f"validation AUC (fitted, out-of-fit): "
      f"{roc_auc_score([int(y[j]) for j in val_j], scores):.4f}")
print("wrote queries_validation.jsonl + manifests")
