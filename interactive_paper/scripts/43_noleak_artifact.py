"""No-leak deployment candidate (8cb): the deployed 5,228-row recipe
refit with the 7 benchmark-colliding calibration rows removed
(scripts/41 audit), packaged as gate_native_noleak.json.

Does NOT overwrite gate_native.json: the live arms in the paper were
run with the current artifact, so the swap happens together with the
tier re-run (new live arms read this file). Thresholds: per-language
OOF quantiles at both the current rates (15/30/50) and the proposed
8bz tiers (15/25/40).

Usage (from interactive_paper/):
  set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\43_noleak_artifact.py
Outputs: data/leak_exclude.json, data/gate_native_noleak.json.
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
m = types.ModuleType("sw")
exec(open("scripts/40_threshold_sweep.py").read().split(
    "if __name__")[0], m.__dict__)
la = types.ModuleType("la")
exec(open("scripts/41_leak_audit.py").read().split(
    "def main")[0], la.__dict__)

# scripts/41 result 2026-09-04: calibration rows with an exact or
# J>=0.8 near-duplicate in Speech TriviaQA / Llama Questions
EXCLUDE = {
    "x0054": "striviaqa0132 exact", "x0137": "striviaqa0156 exact",
    "x0114": "striviaqa0172 exact", "x0113": "striviaqa0181 exact",
    "x0173": "sllama0170 near .80", "y0205": "sllama0130 near .86",
    "z0601": "sllama0206 exact",
}
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
RATES_8BZ = {"conservative": .15, "balanced": .25, "aggressive": .40}
ZH_PARTS = {"exp3zhoff"}


def main():
    (D / "leak_exclude.json").write_text(
        json.dumps(EXCLUDE, indent=1))

    ids, Xs, ys, lang = [], [], [], []
    for tag in la.DEPLOYED_PARTS:
        i, X = m.load_feats(tag)
        y = pd.read_parquet(
            D / f"frozen_native_{tag}_judged.parquet").set_index(
            "id")["escalate_label"].reindex(i).to_numpy(float)
        k = ~np.isnan(y)
        kept = [q for q, kk in zip(i, k) if kk]
        ids += kept
        Xs.append(X[k])
        ys.append(y[k].astype(int))
        lang += ["zh" if tag in ZH_PARTS else "en"] * len(kept)
    fl = pd.read_parquet(D / "fresh_labels.parquet")
    fl = fl[fl["escalate_label"].notna()]
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    nf = pd.read_parquet(
        D / "frozen_native_freshoff_judged.parquet").set_index(
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
    lang = np.array(lang + ["en"] * len(tr_j))
    core = np.arange(len(y)) < n0

    keep = np.array([i not in EXCLUDE for i in ids])
    print(f"merge {len(y)} rows -> {keep.sum()} after exclusion "
          f"({(~keep).sum()} dropped)")
    ids_k = [i for i, kk in zip(ids, keep) if kk]
    X, y = X[keep], y[keep]
    core, lang = core[keep], lang[keep]

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(LogisticRegression(C=3e-4, max_iter=5000),
                            X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"OOF AUC {roc_auc_score(y, oof):.4f}")
    clf = LogisticRegression(C=3e-4, max_iter=5000).fit(X, y)

    def qthr(rates):
        g = {t: float(np.quantile(oof[core], 1 - b))
             for t, b in rates.items()}
        by_lang = {}
        for lg in ("en", "zh"):
            mm = core & (lang == lg)
            by_lang[lg] = {t: float(np.quantile(oof[mm], 1 - b))
                           for t, b in rates.items()}
        return g, by_lang

    thr, thr_lang = qthr(RATES)
    thr8, thr8_lang = qthr(RATES_8BZ)

    # guard: internal test AUC vs the deployed artifact
    dep = json.loads((D / "gate_native.json").read_text())
    tst_ids, Xt = m.load_feats("testoff")
    yt = pd.read_parquet(
        D / "frozen_native_testoff_judged.parquet").set_index(
        "id")["escalate_label"].reindex(tst_ids).to_numpy(float)
    k = ~np.isnan(yt)
    a_dep = roc_auc_score(yt[k].astype(int),
                          Xt[k] @ np.array(dep["w"]) + dep["b"])
    a_new = roc_auc_score(yt[k].astype(int),
                          Xt[k] @ clf.coef_[0] + clf.intercept_[0])
    print(f"guard testoff AUC: deployed {a_dep:.4f} -> noleak {a_new:.4f}")

    art = dict(dep)
    art.update(
        w=clf.coef_[0].tolist(), b=float(clf.intercept_[0]),
        train_n=int(len(y)), eot_thresholds=thr,
        eot_thresholds_lang=thr_lang,
        eot_thresholds_8bz=thr8, eot_thresholds_8bz_lang=thr8_lang,
        excluded_rows=sorted(EXCLUDE),
        manifest_sha1=hashlib.sha1(
            "".join(sorted(ids_k)).encode()).hexdigest(),
        recipe="scripts/43 no-leak refit (deployed 5,228 recipe minus "
               "the 7 scripts/41 colliding rows)")
    (D / "gate_native_noleak.json").write_text(json.dumps(art))
    print("wrote data/gate_native_noleak.json (deployment candidate; "
          "swap at the tier re-run)")
    for name, tl in (("15/30/50", thr_lang), ("15/25/40", thr8_lang)):
        print(f"  thresholds {name}: " + "  ".join(
            f"{lg}:" + "/".join(f"{v:.3f}" for v in d.values())
            for lg, d in tl.items()))


if __name__ == "__main__":
    main()
