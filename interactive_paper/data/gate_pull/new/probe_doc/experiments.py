"""Probe-improvement experiments (CPU, local data only).

Selection rule, fixed before running: variants are ranked by calibration-only
metrics — leave-one-pool-out AUC (LOPO, primary; the harsh transfer proxy)
and 5-fold OOF (secondary). Cold external AUC is computed for every variant
for transparency but is NOT used to pick the winner (constraint 6).

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn \
           python experiments.py [--only name,name] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_lab import Calib, Ext, EXT_POOLS, stack, within_pool_auc  # noqa: E402

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------ helpers -----
def lr(C=1e-4, scaler="std", **kw):
    sc = {"std": StandardScaler(), "robust": RobustScaler(), None: None}[scaler]
    steps = ([sc] if sc is not None else []) + [LogisticRegression(C=C, max_iter=5000, **kw)]
    return make_pipeline(*steps)


class ScoreEnsemble(BaseEstimator, ClassifierMixin):
    """Fit one probe per feature block (list of column slices), average logits."""

    def __init__(self, blocks, C=1e-4):
        self.blocks = blocks; self.C = C

    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        for sl in self.blocks:
            m = lr(self.C)
            if sample_weight is not None:
                m.fit(X[:, sl], y, logisticregression__sample_weight=sample_weight)
            else:
                m.fit(X[:, sl], y)
            self.models_.append(m)
        self.classes_ = np.array([0, 1]); return self

    def decision_function(self, X):
        return np.mean([m.decision_function(X[:, sl]) for m, sl in zip(self.models_, self.blocks)], axis=0)

    def predict_proba(self, X):
        z = self.decision_function(X); p = 1 / (1 + np.exp(-z))
        return np.c_[1 - p, p]


class RidgeProba(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        self.m_ = make_pipeline(StandardScaler(), RidgeClassifier(alpha=self.alpha))
        self.m_.fit(X, y, ridgeclassifier__sample_weight=sample_weight) if sample_weight is not None else self.m_.fit(X, y)
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        z = self.m_.decision_function(X); p = 1 / (1 + np.exp(-z))
        return np.c_[1 - p, p]


class LDAProba(BaseEstimator, ClassifierMixin):
    """Shrinkage LDA on PCA-reduced standardized features (full-cov LDA on 13k dims is too slow)."""

    def __init__(self, k=512, shrinkage="auto"):
        self.k = k; self.shrinkage = shrinkage

    def fit(self, X, y, sample_weight=None):
        self.m_ = make_pipeline(StandardScaler(), PCA(self.k, random_state=0),
                                LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage))
        self.m_.fit(X, y); self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)


def weights_pool_balanced(meta, y):
    """Each pool gets equal total weight AND each class within a pool equal weight
    (kills the pool-identity / base-rate shortcut in the objective)."""
    w = np.ones(len(y))
    pools = meta.pool.values
    for g in np.unique(pools):
        m = pools == g
        for c in (0, 1):
            mc = m & (y == c)
            if mc.sum():
                w[mc] = 1.0 / (len(np.unique(pools)) * 2 * mc.sum())
    return w * len(y) / w.sum()


def weights_class_balanced(y):
    w = np.where(y == 1, 0.5 / (y == 1).mean(), 0.5 / (y == 0).mean()); return w


def fit_w(make_clf, X, y, w):
    m = make_clf()
    if w is None:
        return m.fit(X, y)
    if hasattr(m, "named_steps"):
        step = list(m.named_steps)[-1]
        return m.fit(X, y, **{f"{step}__sample_weight": w})
    return m.fit(X, y, sample_weight=w)


def run_variant(name, make_clf, fc, fe, cal, ext, weights=None, seed=42):
    t = time.time()
    X = fc(cal); y = cal.y; pools = cal.meta.pool.values
    w = None if weights is None else weights(cal.meta, y)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = fit_w(make_clf, X[tr], y[tr], None if w is None else w[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    lopo = np.zeros(len(y))
    for g in np.unique(pools):
        te = pools == g
        if len(np.unique(y[~te])) < 2:
            continue
        m = fit_w(make_clf, X[~te], y[~te], None if w is None else w[~te]); lopo[te] = m.predict_proba(X[te])[:, 1]
    full = fit_w(make_clf, X, y, w)
    ext_row = {tag: float(roc_auc_score(d["y"], full.predict_proba(fe(d))[:, 1])) for tag, d in ext.pools.items()}
    ext_row["mean"] = float(np.mean([ext_row[t] for t in EXT_POOLS]))
    rep = dict(name=name, dims=int(X.shape[1]), oof=float(roc_auc_score(y, oof)), lopo=float(roc_auc_score(y, lopo)),
               oof_within_mean=float(np.mean(list(within_pool_auc(oof, y, pools).values()))),
               lopo_within=within_pool_auc(lopo, y, pools), ext=ext_row, secs=round(time.time() - t, 1))
    print(f"{name:48s} d={rep['dims']:6d} OOF {rep['oof']:.4f}  LOPO {rep['lopo']:.4f}  wOOF {rep['oof_within_mean']:.4f}  "
          f"ext {ext_row['striviaqa']:.3f}/{ext_row['swebq']:.3f}/{ext_row['sllama']:.3f}/{ext_row['sdqa']:.3f} = {ext_row['mean']:.4f}  ({rep['secs']}s)", flush=True)
    return rep


# ------------------------------------------------------------ features ----
def F_stack(L):
    return (lambda c: stack(c.E_on, c.M, c.j(L)), lambda d: stack(d["E_on"], d["M"], EXT_J[L]))


def F_stack_eot(L):
    return (lambda c: stack(c.E_eot, c.M, c.j(L)), lambda d: stack(d["E_eot"], d["M"], EXT_J[L]))


def _frames(E, M, j, which):
    parts = []
    for wch in which:
        if wch == "last": parts.append(E[:, j, -1])
        elif wch == "first": parts.append(E[:, j, 0])
        elif wch == "mean8": parts.append(E[:, j].mean(1))
        elif wch == "umean": parts.append(M[:, j])
        elif wch == "allframes": parts.append(E[:, j].reshape(len(E), -1))
        elif wch == "last-first": parts.append(E[:, j, -1] - E[:, j, 0])
        elif wch == "mean8-umean": parts.append(E[:, j].mean(1) - M[:, j])
        elif wch == "late4": parts.append(E[:, j, 4:].mean(1))
        elif wch == "early4": parts.append(E[:, j, :4].mean(1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def F_frames(L, which):
    return (lambda c: _frames(c.E_on, c.M, c.j(L), which), lambda d: _frames(d["E_on"], d["M"], EXT_J[L], which))


def F_multi(Ls):
    return (lambda c: np.concatenate([stack(c.E_on, c.M, c.j(L)) for L in Ls], 1),
            lambda d: np.concatenate([stack(d["E_on"], d["M"], EXT_J[L]) for L in Ls], 1))


def F_dual(L_on, L_eot):
    return (lambda c: np.concatenate([stack(c.E_on, c.M, c.j(L_on)), stack(c.E_eot, c.M, c.j(L_eot))], 1),
            lambda d: np.concatenate([stack(d["E_on"], d["M"], EXT_J[L_on]), stack(d["E_eot"], d["M"], EXT_J[L_eot])], 1))


EXT_J = {L: i for i, L in enumerate([22, 26, 30, 34, 38])}
D = 4480


def variants():
    V = {}
    V["baseline onset@L30 C=1e-4"] = (lambda: lr(1e-4), *F_stack(30), None)
    # --- regularisation / scaler ---
    for C in (1e-5, 3e-5, 3e-4):
        V[f"onset@L30 C={C:g}"] = (lambda C=C: lr(C), *F_stack(30), None)
    V["onset@L30 robust-scaler"] = (lambda: lr(1e-4, "robust"), *F_stack(30), None)
    V["onset@L30 ridge a=1e4"] = (lambda: RidgeProba(1e4), *F_stack(30), None)
    V["onset@L30 ridge a=1e5"] = (lambda: RidgeProba(1e5), *F_stack(30), None)
    V["onset@L30 PCA256+LR C=1e-2"] = (lambda: make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=1e-2, max_iter=5000)), *F_stack(30), None)
    V["onset@L30 PCA512+LR C=1e-2"] = (lambda: make_pipeline(StandardScaler(), PCA(512, random_state=0), LogisticRegression(C=1e-2, max_iter=5000)), *F_stack(30), None)
    V["onset@L30 PCA512+shrinkLDA"] = (lambda: LDAProba(512), *F_stack(30), None)
    # --- sample weighting (pool shortcut) ---
    V["onset@L30 pool+class balanced weights"] = (lambda: lr(1e-4), *F_stack(30), weights_pool_balanced)
    V["onset@L30 class balanced weights"] = (lambda: lr(1e-4), *F_stack(30), lambda meta, y: weights_class_balanced(y))
    # --- window usage ---
    V["onset@L30 first+last+mean8+umean"] = (lambda: lr(1e-4), *F_frames(30, ["first", "last", "mean8", "umean"]), None)
    V["onset@L30 all8frames+umean"] = (lambda: lr(1e-4), *F_frames(30, ["allframes", "umean"]), None)
    V["onset@L30 last+mean8+umean+(last-first)"] = (lambda: lr(1e-4), *F_frames(30, ["last", "mean8", "umean", "last-first"]), None)
    V["onset@L30 early4+late4+umean"] = (lambda: lr(1e-4), *F_frames(30, ["early4", "late4", "umean"]), None)
    V["onset@L30 mean8+umean (no last)"] = (lambda: lr(1e-4), *F_frames(30, ["mean8", "umean"]), None)
    V["onset@L30 first+mean8+umean (commit frame as last)"] = (lambda: lr(1e-4), *F_frames(30, ["first", "mean8", "umean"]), None)
    # --- layers / reads ---
    for L in (22, 26, 34, 38):
        V[f"onset@L{L} stack"] = (lambda: lr(1e-4), *F_stack(L), None)
    V["eot@L34 stack"] = (lambda: lr(1e-4), *F_stack_eot(34), None)
    V["concat onset L26+L30+L34"] = (lambda: lr(1e-4), *F_multi([26, 30, 34]), None)
    V["score-ens onset L26,L30,L34"] = (lambda: ScoreEnsemble([slice(i * 3 * D, (i + 1) * 3 * D) for i in range(3)]), *F_multi([26, 30, 34]), None)
    V["score-ens onset L22..L38 (5)"] = (lambda: ScoreEnsemble([slice(i * 3 * D, (i + 1) * 3 * D) for i in range(5)]), *F_multi([22, 26, 30, 34, 38]), None)
    V["concat dual onset@L30+eot@L34"] = (lambda: lr(1e-4), *F_dual(30, 34), None)
    V["score-ens dual onset@L30+eot@L34"] = (lambda: ScoreEnsemble([slice(0, 3 * D), slice(3 * D, 6 * D)]), *F_dual(30, 34), None)
    V["score-ens per-block (last,mean8,umean)@L30"] = (lambda: ScoreEnsemble([slice(0, D), slice(D, 2 * D), slice(2 * D, 3 * D)]), *F_stack(30), None)
    # --- batch 2: combos around L26 and the richer window ---
    V["onset@L26 first+last+mean8+umean"] = (lambda: lr(1e-4), *F_frames(26, ["first", "last", "mean8", "umean"]), None)
    V["onset@L26 all8frames+umean"] = (lambda: lr(1e-4), *F_frames(26, ["allframes", "umean"]), None)
    V["onset@L26 early4+late4+umean"] = (lambda: lr(1e-4), *F_frames(26, ["early4", "late4", "umean"]), None)
    V["onset@L26 pool+class balanced weights"] = (lambda: lr(1e-4), *F_stack(26), weights_pool_balanced)
    V["onset@L26 C=3e-4"] = (lambda: lr(3e-4), *F_stack(26), None)
    V["onset@L26 C=3e-5"] = (lambda: lr(3e-5), *F_stack(26), None)
    V["score-ens onset L22,L26,L30"] = (lambda: ScoreEnsemble([slice(i * 3 * D, (i + 1) * 3 * D) for i in range(3)]), *F_multi([22, 26, 30]), None)
    V["score-ens onset L26,L30"] = (lambda: ScoreEnsemble([slice(i * 3 * D, (i + 1) * 3 * D) for i in range(2)]), *F_multi([26, 30]), None)
    V["all8frames+umean pool+class balanced @L30"] = (lambda: lr(1e-4), *F_frames(30, ["allframes", "umean"]), weights_pool_balanced)
    # --- batch 3: label noise, sparsity ---
    V["onset@L26 drop pass-flipped labels"] = (lambda: lr(1e-4), *F_stack(26), weights_drop_flipped)
    V["onset@L26 elasticnet l1=0.5 C=1e-3"] = (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=2000, tol=1e-3)), *F_stack(26), None)
    return V


def weights_drop_flipped(meta, y):
    """Zero weight for the 34 rows whose label flipped between the two replay passes."""
    import pandas as pd
    flipped = set()
    for tag, pa in (("frozen", GP / "nvda_frozen.parquet"), ("expansion", GP / "new" / "nvda_expansion.parquet"),
                    ("expansion2", GP / "new" / "nvda_expansion2.parquet")):
        A = pd.read_parquet(pa).set_index("id")["escalate_label"]
        Bp = pd.read_parquet(GP / "onset_fit" / f"nvda_{tag}.parquet").set_index("id")["escalate_label"]
        ids = A.index.intersection(Bp.index)
        flipped |= {(tag, str(i)) for i in ids if int(A[i]) != int(Bp[i])}
    w = np.array([0.0 if (t, i) in flipped else 1.0 for t, i in zip(meta.tag, meta.id)])
    print(f"   dropping {int((w == 0).sum())} flipped-label rows")
    return w


from probe_lab import GP  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "results_batch1.json"))
    a = ap.parse_args()
    t = time.time(); cal, ext = Calib(), Ext(); print(f"loaded in {time.time()-t:.0f}s", flush=True)
    V = variants()
    names = [n for n in V if not a.only or any(s in n for s in a.only.split(","))]
    res = []
    for n in names:
        make_clf, fc, fe, w = V[n]
        try:
            res.append(run_variant(n, make_clf, fc, fe, cal, ext, weights=w))
        except Exception as e:  # keep going
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("\n== ranked by LOPO (selection metric) ==")
    for r in sorted(res, key=lambda r: -r["lopo"]):
        print(f"  LOPO {r['lopo']:.4f}  OOF {r['oof']:.4f}  ext {r['ext']['mean']:.4f}  {r['name']}")
