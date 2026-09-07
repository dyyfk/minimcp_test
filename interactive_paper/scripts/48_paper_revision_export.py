"""Paper-revision P0 export (pr0907): per-sample OOF records for the
deployed 5,228-row gate recipe and the scripts/43 no-leak refit.

Reconstructs the merge EXACTLY as scripts/43 (DEPLOYED_PARTS + fresh
train rows, seed-42 StratifiedKFold, C=3e-4), assigns fold ids, and
dumps per-sample id / part / source file / lang / label / fold / OOF
score for both fits, plus a thresholds bundle. Validates against the
shipped artifacts (gate_native.json, gate_native_noleak.json) before
writing.

Usage (from interactive_paper/):
  set PYTHONUTF8=1 && .venv_boot\Scripts\python.exe scripts\48_paper_revision_export.py
Outputs: data/paper_revision_pr0907/{calibration_oof.parquet,
  calibration_manifest.jsonl,thresholds.json}
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
m = types.ModuleType("sw")
exec(open("scripts/40_threshold_sweep.py").read().split("if __name__")[0], m.__dict__)
la = types.ModuleType("la")
exec(open("scripts/41_leak_audit.py").read().split("def main")[0], la.__dict__)

EXCLUDE = json.loads((D / "leak_exclude.json").read_text())
ZH_PARTS = {"exp3zhoff"}
PART_SRC = {"caliboff": ["queries_public.jsonl", "queries_gen.jsonl"],
            "expoff": ["queries_expansion.jsonl"],
            "exp2off": ["queries_expansion2.jsonl"],
            "exp3off": ["queries_expansion3.jsonl"],
            "exp3zhoff": ["queries_expansion3zh.jsonl"],
            "freshoff": ["queries_fresh.jsonl"]}

# id -> (source file, raw row) over every calibration query file
src = {}
for f in la.CALIB_FILES:
    p = D / f
    if not p.exists():
        continue
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            q = json.loads(ln)
            if q.get("id"):
                src.setdefault(q["id"], (f, q))

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
lang = np.array(["zh" if t in ZH_PARTS else "en" for t in parts])
core = np.arange(len(y)) < n0
print(f"deployed merge {len(y)} rows (core {n0} + fresh {len(tr_j)})")

def fit_oof(Xa, ya):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(LogisticRegression(C=3e-4, max_iter=5000),
                            Xa, ya, cv=cv, method="predict_proba")[:, 1]
    fold = np.full(len(ya), -1)
    for fi, (_, te) in enumerate(cv.split(Xa, ya)):
        fold[te] = fi
    return oof, fold

oof_dep, fold_dep = fit_oof(X, y)
print(f"deployed OOF AUC {roc_auc_score(y, oof_dep):.4f}")
keep = np.array([i not in EXCLUDE for i in ids])
oof_nl_k, fold_nl_k = fit_oof(X[keep], y[keep])
print(f"noleak merge {keep.sum()} rows, OOF AUC "
      f"{roc_auc_score(y[keep], oof_nl_k):.4f}")
oof_nl = np.full(len(y), np.nan)
fold_nl = np.full(len(y), -1)
oof_nl[keep] = oof_nl_k
fold_nl[keep] = fold_nl_k

# validate deployed threshold construction against the shipped artifacts
dep = json.loads((D / "gate_native.json").read_text())
nl = json.loads((D / "gate_native_noleak.json").read_text())
report = {}
for name, oo, cc, art, rates in (
        ("deployed", oof_dep, core, dep["eot_thresholds_lang"],
         {"conservative": .15, "balanced": .30, "aggressive": .50}),
        ("noleak", oof_nl_k, core[keep], nl["eot_thresholds_lang"],
         {"conservative": .15, "balanced": .30, "aggressive": .50})):
    lg_arr = lang if name == "deployed" else lang[keep]
    rep = {}
    for lg in ("en", "zh"):
        mm = cc & (lg_arr == lg)
        rep[lg] = {t: {"recomputed": float(np.quantile(oo[mm], 1 - r)),
                       "artifact": art[lg][t]}
                   for t, r in rates.items()}
    report[name] = rep
    for lg, d in rep.items():
        for t, v in d.items():
            dv = abs(v["recomputed"] - v["artifact"])
            flag = "" if dv < 1e-3 else "  <-- MISMATCH"
            print(f"{name} {lg} {t}: {v['recomputed']:.4f} vs "
                  f"artifact {v['artifact']:.4f}{flag}")

rows = []
for j, qid in enumerate(ids):
    sf, raw = src.get(qid, (None, {}))
    rows.append({
        "id": qid, "part": parts[j], "source_file": sf,
        "pool": raw.get("pool"), "split": split_f.get(qid),
        "lang": lang[j], "query": raw.get("query"),
        "label_native": int(y[j]), "core": bool(core[j]),
        "fold_deployed": int(fold_dep[j]),
        "oof_deployed": float(oof_dep[j]),
        "fold_noleak": int(fold_nl[j]),
        "oof_noleak": None if np.isnan(oof_nl[j]) else float(oof_nl[j]),
        "leak_excluded": qid in EXCLUDE,
        "leak_match": EXCLUDE.get(qid),
    })
df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT / "calibration_oof.parquet", index=False)
with open(OUT / "calibration_manifest.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps({k: r[k] for k in (
            "id", "part", "source_file", "pool", "split", "lang",
            "query", "label_native", "leak_excluded")},
            ensure_ascii=False) + "\n")

sweep = json.loads((D / "threshold_sweep.json").read_text())
(OUT / "thresholds.json").write_text(json.dumps({
    "rule": "per-language quantile of core-calibration OOF scores at "
            "1-nominal_rate; core = DEPLOYED_PARTS rows (fresh train rows "
            "in the fit but excluded from the quantile base)",
    "deployed_gate_sha256": hashlib.sha256(
        (D / "gate_native.json").read_bytes()).hexdigest(),
    "noleak_gate_sha256": hashlib.sha256(
        (D / "gate_native_noleak.json").read_bytes()).hexdigest(),
    "deployed_tiers_nominal": {"conservative": .15, "balanced": .30,
                               "aggressive": .50},
    "proposed_8bz_tiers_nominal": {"conservative": .15, "balanced": .25,
                                   "aggressive": .40},
    "deployed_eot_thresholds_lang": dep["eot_thresholds_lang"],
    "noleak_eot_thresholds_lang": nl["eot_thresholds_lang"],
    "noleak_eot_thresholds_8bz_lang": nl["eot_thresholds_8bz_lang"],
    "sweep_nominal_thresholds": sweep["nominal_thresholds"],
    "validation": report,
    "manifest_sha1_noleak_ids": nl.get("manifest_sha1"),
}, indent=1))
n_missing_src = int(df["source_file"].isna().sum())
print(f"wrote {len(df)} rows; {n_missing_src} ids without a source-file match")
