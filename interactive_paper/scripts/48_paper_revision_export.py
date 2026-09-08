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
# (queries.jsonl holds the frozen calib/test pool the caliboff rows
# come from; the la.CALIB_FILES expansions cover the rest)
src = {}
for f in ["queries.jsonl"] + la.CALIB_FILES:
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
    bad = []
    for lg, d in rep.items():
        for t, v in d.items():
            dv = abs(v["recomputed"] - v["artifact"])
            flag = "" if dv < 1e-3 else "  <-- MISMATCH"
            if flag:
                bad.append((name, lg, t, dv))
            print(f"{name} {lg} {t}: {v['recomputed']:.4f} vs "
                  f"artifact {v['artifact']:.4f}{flag}")
    if bad:
        raise SystemExit(
            f"threshold validation FAILED, refusing to export: {bad}")

rows = []
for j, qid in enumerate(ids):
    sf, raw = src.get(qid, (None, {}))
    # source_split: the split field carried by the ORIGINAL source file
    # (only queries.jsonl / queries_fresh rows have one); "unknown"
    # where the source never recorded a split. experiment_role: every
    # row in this artifact was in the fit; core rows additionally form
    # the threshold quantile base. NO row here has a validation or
    # held-out-test role.
    ssp = raw.get("split") or split_f.get(qid) or "unknown"
    rows.append({
        "id": qid, "part": parts[j], "source_file": sf,
        "pool": raw.get("pool"), "source_split": ssp,
        "experiment_role": ("fit_core+quantile_base" if core[j]
                            else "fit_fresh_train"),
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
            "id", "part", "source_file", "pool", "source_split",
            "experiment_role", "lang", "query", "label_native",
            "leak_excluded")},
            ensure_ascii=False) + "\n")

sweep = json.loads((D / "threshold_sweep.json").read_text())
dep_sha = hashlib.sha256((D / "gate_native.json").read_bytes()).hexdigest()
nl_sha = hashlib.sha256(
    (D / "gate_native_noleak.json").read_bytes()).hexdigest()
ids_k = [i for i, kk in zip(ids, keep) if kk]
train_ids_sha1 = hashlib.sha1("".join(sorted(ids_k)).encode()).hexdigest()
qbase_ids = sorted(i for i, kk, cc in zip(ids, keep, core) if kk and cc)
qbase_sha1 = hashlib.sha1("".join(qbase_ids).encode()).hexdigest()
GRID_RATES = [.10, .15, .20, .25, .30, .40, .50]
lg_k, core_k = lang[keep], core[keep]
noleak_grid = {}
for lg in ("en", "zh"):
    mm = core_k & (lg_k == lg)
    noleak_grid[lg] = {str(r): float(np.quantile(oof_nl_k[mm], 1 - r))
                       for r in GRID_RATES}
(OUT / "thresholds.json").write_text(json.dumps({
    "rule": "per-language quantile of core-calibration OOF scores at "
            "1-nominal_rate; core = DEPLOYED_PARTS rows (fresh train rows "
            "in the fit but excluded from the quantile base)",
    "deployed_gate_sha256": dep_sha,
    "noleak_gate_sha256": nl_sha,
    "deployed_tiers_nominal": {"conservative": .15, "balanced": .30,
                               "aggressive": .50},
    "proposed_8bz_tiers_nominal": {"conservative": .15, "balanced": .25,
                                   "aggressive": .40},
    "deployed_eot_thresholds_lang": dep["eot_thresholds_lang"],
    "noleak_eot_thresholds_lang": nl["eot_thresholds_lang"],
    "noleak_eot_thresholds_8bz_lang": nl["eot_thresholds_8bz_lang"],
    "noleak_nominal_grid_lang": {
        "grid": noleak_grid,
        "provenance": {
            "fit_gate_sha256": nl_sha,
            "train_ids_sha1": train_ids_sha1,
            "quantile_base": "core rows of the no-leak fit "
                             "(DEPLOYED_PARTS minus the 7 excluded ids; "
                             "fresh train rows in the fit but not in the "
                             "quantile base)",
            "quantile_base_ids_sha1": qbase_sha1,
            "lang_rule": "part tag: exp3zhoff -> zh, all other parts en",
            "score_kind": "OOF, StratifiedKFold(5, shuffle, seed=42), "
                          "LogisticRegression C=3e-4",
            "script": "scripts/48_paper_revision_export.py"}},
    "historical_remix_grid": {
        "grid": sweep["nominal_thresholds"],
        "provenance": {
            "fit_gate_sha256": None, "train_ids_sha1": None,
            "quantile_base_ids_sha1": None,
            "script": "scripts/40_threshold_sweep.py (8bz, 2026-09-04)",
            "limitation": "produced by script 40's own merge (its PARTS "
                "list loads more tags than the deployed 5-part recipe "
                "when the extra feature dumps are present) with its own "
                "zh handling; NOT the same quantile construction as the "
                "deployed/no-leak artifacts and NOT re-derived here - "
                "the fit identity metadata was not recorded at run time. "
                "Its internal remix used already-inspected test "
                "outcomes: exploration only, never independent "
                "validation."}},
    "validation": report,
    "manifest_sha1_noleak_ids": nl.get("manifest_sha1"),
}, indent=1))
n_missing_src = int(df["source_file"].isna().sum())
print(f"wrote {len(df)} rows; {n_missing_src} ids without a source-file match")
