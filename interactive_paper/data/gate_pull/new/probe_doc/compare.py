"""Paired comparison of candidate probes against the deployed baseline.

For each candidate: multi-seed OOF (5 seeds), LOPO, and a PAIRED bootstrap
over external queries (same resample for both models, stratified by pool)
of the external-mean AUC difference.  Also computes the 8-pool matched-random
style "accuracy at budget" is NOT done here (needs expert outcomes).

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import variants, fit_w  # noqa: E402
from probe_lab import Calib, Ext, EXT_POOLS  # noqa: E402

HERE = Path(__file__).resolve().parent
CANDIDATES = [
    "baseline onset@L30 C=1e-4",
    "onset@L26 stack",
    "onset@L26 early4+late4+umean",
    "onset@L26 first+last+mean8+umean",
    "onset@L26 all8frames+umean",
    "onset@L26 C=3e-4",
    "score-ens onset L26,L30",
    "concat onset L26+L30+L34",
    "onset@L30 early4+late4+umean",
]
SEEDS = (0, 1, 2, 3, 42)
B = 2000


def ext_scores(model, fe, ext):
    return {tag: model.predict_proba(fe(d))[:, 1] for tag, d in ext.pools.items()}


def paired_boot(sa, sb, ext, rng):
    """Bootstrap of mean-over-pools AUC(b) - AUC(a), same resample per pool."""
    deltas = np.zeros(B); a_m = np.zeros(B); b_m = np.zeros(B)
    idx = {tag: [rng.integers(0, len(ext.pools[tag]["y"]), len(ext.pools[tag]["y"])) for _ in range(B)] for tag in EXT_POOLS}
    for b in range(B):
        aa, bb = [], []
        for tag in EXT_POOLS:
            y = ext.pools[tag]["y"]; ii = idx[tag][b]
            if len(np.unique(y[ii])) < 2:
                continue
            aa.append(roc_auc_score(y[ii], sa[tag][ii])); bb.append(roc_auc_score(y[ii], sb[tag][ii]))
        a_m[b], b_m[b] = np.mean(aa), np.mean(bb); deltas[b] = b_m[b] - a_m[b]
    return deltas


if __name__ == "__main__":
    cal, ext = Calib(), Ext()
    V = variants(); rng = np.random.default_rng(0)
    out = {}; base_scores = None
    for name in CANDIDATES:
        make_clf, fc, fe, wfun = V[name]
        X = fc(cal); y = cal.y; w = None if wfun is None else wfun(cal.meta, y)
        oofs = []
        for s in SEEDS:
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                p[te] = fit_w(make_clf, X[tr], y[tr], None if w is None else w[tr]).predict_proba(X[te])[:, 1]
            oofs.append(roc_auc_score(y, p))
        full = fit_w(make_clf, X, y, w)
        sc = ext_scores(full, fe, ext)
        ext_auc = {tag: float(roc_auc_score(ext.pools[tag]["y"], sc[tag])) for tag in EXT_POOLS}
        ext_auc["mean"] = float(np.mean(list(ext_auc.values())))
        rep = dict(oof_mean=float(np.mean(oofs)), oof_sd=float(np.std(oofs)), oof_seeds=[float(a) for a in oofs], ext=ext_auc)
        if base_scores is None:
            base_scores = sc; rep["delta_ext_mean"] = 0.0
        else:
            d = paired_boot(base_scores, sc, ext, rng)
            rep["delta_ext_mean"] = float(np.mean(d)); rep["delta_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            rep["p_delta_le_0"] = float((d <= 0).mean())
        out[name] = rep
        ci = rep.get("delta_ci95", [0, 0])
        print(f"{name:40s} OOF {rep['oof_mean']:.4f}±{rep['oof_sd']:.4f}  ext {ext_auc['mean']:.4f}  Δext {rep['delta_ext_mean']:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]  P(Δ≤0)={rep.get('p_delta_le_0', float('nan')):.3f}", flush=True)
        json.dump(out, open(HERE / "compare.json", "w"), indent=1)
