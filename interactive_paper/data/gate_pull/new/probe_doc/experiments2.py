"""Round-2 experiments: reviewer proposals that need no new GPU capture.

Metrics (all calibration-only except `ext`):
  oof        pooled 5-fold OOF AUC (the paper's number)
  oof_macro  mean within-pool AUC of the OOF scores (eligible pools)
  lopo       pooled leave-one-pool-out AUC (inflated by base rates — kept for continuity)
  lopo_macro mean within-held-pool AUC under LOPO  <-- primary selection metric
  lopo_worst worst eligible held pool
  ext        cold external AUC per pool + mean (reported, never used to select)
Eligible pool = at least 10 examples of each class.

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python experiments2.py [--only a,b] [--out f]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import D, EXT_J, F_frames, F_stack, ScoreEnsemble, _frames, fit_w, lr  # noqa: E402
from probe_lab import Calib, Ext, EXT_POOLS, stack  # noqa: E402

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------ metrics -----
def eligible_pools(y, pools, min_minority=10):
    out = []
    for g in np.unique(pools):
        m = pools == g
        if min((y[m] == 1).sum(), (y[m] == 0).sum()) >= min_minority:
            out.append(g)
    return out


def macro_auc(p, y, pools, elig):
    per = {str(g): float(roc_auc_score(y[pools == g], p[pools == g])) for g in elig}
    return float(np.mean(list(per.values()))), float(min(per.values())), per


# ------------------------------------------------------------ transforms --
class PoolCentroidRemover(BaseEstimator, TransformerMixin):
    """Project out the top-k directions spanned by (standardized) pool centroids.
    Needs pool ids at fit time -> passed via fit(X, y, groups=...) through the wrapper below."""

    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y=None, groups=None):
        if self.k == 0 or groups is None:
            self.P_ = None; return self
        C = np.stack([X[groups == g].mean(0) for g in np.unique(groups)])
        C = C - C.mean(0)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        self.P_ = Vt[: self.k]                      # (k, d) orthonormal
        return self

    def transform(self, X):
        if self.P_ is None:
            return X
        return X - (X @ self.P_.T) @ self.P_


class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, lo=-5.0, hi=5.0):
        self.lo = lo; self.hi = hi

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, self.lo, self.hi)


class Winsor(BaseEstimator, TransformerMixin):
    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        self.lo_ = np.quantile(X, self.q, axis=0); self.hi_ = np.quantile(X, 1 - self.q, axis=0); return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


class GroupAwareLR(BaseEstimator, ClassifierMixin):
    """StandardScaler -> PoolCentroidRemover(k) -> LR(C). Groups are read from a column
    appended to X (last column) so the sklearn fit signature stays simple."""

    def __init__(self, k=2, C=1e-4):
        self.k = k; self.C = C

    def fit(self, X, y, sample_weight=None):
        g, Xf = X[:, -1].astype(int), X[:, :-1]
        self.sc_ = StandardScaler().fit(Xf); Z = self.sc_.transform(Xf)
        self.pr_ = PoolCentroidRemover(self.k).fit(Z, y, groups=g); Z = self.pr_.transform(Z)
        self.lr_ = LogisticRegression(C=self.C, max_iter=5000).fit(Z, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        Z = self.pr_.transform(self.sc_.transform(X[:, :-1])); return self.lr_.predict_proba(Z)


class PLSProba(BaseEstimator, ClassifierMixin):
    def __init__(self, n=8):
        self.n = n

    def fit(self, X, y, sample_weight=None):
        self.sc_ = StandardScaler().fit(X)
        self.pls_ = PLSRegression(n_components=self.n, scale=False).fit(self.sc_.transform(X), y.astype(float))
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        z = self.pls_.predict(self.sc_.transform(X)).ravel(); p = 1 / (1 + np.exp(-(z - 0.5) * 6))
        return np.c_[1 - p, p]


def mild_weights(alpha, beta):
    def f(meta, y):
        pools = meta.pool.values; n = len(y); w = np.ones(n)
        for g in np.unique(pools):
            m = pools == g; ng = m.sum()
            for c in (0, 1):
                mc = m & (y == c); ngc = max(mc.sum(), 1)
                w[mc] = (n / ng) ** alpha * (ng / (2 * ngc)) ** beta
        return w * n / w.sum()
    return f


# ------------------------------------------------------------ runner ------
def run(name, make_clf, fc, fe, cal, ext, weights=None, needs_groups=False, seed=42):
    t = time.time(); X = fc(cal); y = cal.y; pools = cal.meta.pool.values
    gid = np.unique(pools, return_inverse=True)[1].astype(np.float32)
    if needs_groups:
        X = np.c_[X, gid]
    w = None if weights is None else weights(cal.meta, y)
    elig = eligible_pools(y, pools)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        oof[te] = fit_w(make_clf, X[tr], y[tr], None if w is None else w[tr]).predict_proba(X[te])[:, 1]
    lopo = np.zeros(len(y)); held = {}
    for g in np.unique(pools):
        te = pools == g
        m = fit_w(make_clf, X[~te], y[~te], None if w is None else w[~te]); lopo[te] = m.predict_proba(X[te])[:, 1]
    lopo_macro, lopo_worst, lopo_per = macro_auc(lopo, y, pools, elig)
    oof_macro, _, _ = macro_auc(oof, y, pools, elig)
    full = fit_w(make_clf, X, y, w)
    def fe2(d):
        Xe = fe(d); return np.c_[Xe, np.zeros(len(Xe), np.float32)] if needs_groups else Xe
    ext_row = {tag: float(roc_auc_score(d["y"], full.predict_proba(fe2(d))[:, 1])) for tag, d in ext.pools.items()}
    ext_row["mean"] = float(np.mean([ext_row[t] for t in EXT_POOLS]))
    rep = dict(name=name, oof=float(roc_auc_score(y, oof)), oof_macro=oof_macro, lopo=float(roc_auc_score(y, lopo)),
               lopo_macro=lopo_macro, lopo_worst=lopo_worst, lopo_per=lopo_per, ext=ext_row, secs=round(time.time() - t, 1))
    print(f"{name:46s} OOF {rep['oof']:.4f} oofM {oof_macro:.4f} | LOPO {rep['lopo']:.4f} lopoM {lopo_macro:.4f} worst {lopo_worst:.3f} | "
          f"ext {ext_row['striviaqa']:.3f}/{ext_row['swebq']:.3f}/{ext_row['sllama']:.3f}/{ext_row['sdqa']:.3f}={ext_row['mean']:.4f} ({rep['secs']}s)", flush=True)
    return rep


def F_commit(L, with_umean=True):
    which = ["first", "umean"] if with_umean else ["first"]
    return F_frames(L, which)


def variants():
    V = {}
    # baselines
    V["A0 deployed: onset@L30 last+mean8+umean"] = dict(clf=lambda: lr(1e-4), F=F_stack(30))
    V["A1 onset@L26 last+mean8+umean"] = dict(clf=lambda: lr(1e-4), F=F_stack(26))
    # strict-at-commit track (GPT proposal 3): the commit frame itself (+ user mean)
    for L in (22, 26, 30, 34, 38):
        V[f"S commit-frame@L{L} + umean"] = dict(clf=lambda: lr(1e-4), F=F_commit(L))
    V["S commit-frame@L38 alone"] = dict(clf=lambda: lr(1e-4), F=F_commit(38, False))
    V["S commit-frame@L26 + umean + eot_last"] = dict(
        clf=lambda: lr(1e-4),
        F=(lambda c: np.concatenate([c.E_on[:, c.j(26), 0], c.M[:, c.j(26)], c.E_eot[:, c.j(26), -1]], 1).astype(np.float32),
           lambda d: np.concatenate([d["E_on"][:, EXT_J[26], 0], d["M"][:, EXT_J[26]], d["E_eot"][:, EXT_J[26], -1]], 1).astype(np.float32)))
    V["S score-ens commit-frame+umean L22..L38"] = dict(
        clf=lambda: ScoreEnsemble([slice(i * 2 * D, (i + 1) * 2 * D) for i in range(5)]),
        F=(lambda c: np.concatenate([_frames(c.E_on, c.M, c.j(L), ["first", "umean"]) for L in (22, 26, 30, 34, 38)], 1),
           lambda d: np.concatenate([_frames(d["E_on"], d["M"], EXT_J[L], ["first", "umean"]) for L in (22, 26, 30, 34, 38)], 1)))
    # pool-centroid subspace removal (GPT 4)
    for k in (1, 2, 4, 8):
        V[f"P L26 stack, remove {k} pool-centroid dirs"] = dict(clf=lambda k=k: GroupAwareLR(k, 1e-4), F=F_stack(26), groups=True)
    V["P commit@L26+umean, remove 2 dirs"] = dict(clf=lambda: GroupAwareLR(2, 1e-4), F=F_commit(26), groups=True)
    # supervised low-rank heads (GPT 5)
    for n in (4, 8, 16, 32):
        V[f"L L26 stack PLS n={n}"] = dict(clf=lambda n=n: PLSProba(n), F=F_stack(26))
    for k in (64, 128):
        V[f"L L26 stack PCA{k}-whiten + LR C=1"] = dict(clf=lambda k=k: make_pipeline(StandardScaler(), PCA(k, whiten=True, random_state=0), LogisticRegression(C=1.0, max_iter=5000)), F=F_stack(26))
    # mild reweighting (GPT 6)
    for a, b in ((0.25, 0), (0.5, 0), (0, 0.25), (0, 0.5), (0.25, 0.25), (0.5, 0.5)):
        V[f"W L26 stack weights a={a} b={b}"] = dict(clf=lambda: lr(1e-4), F=F_stack(26), w=mild_weights(a, b))
    # robust standardisation (GPT 7)
    V["R L26 stack robust+clip5"] = dict(clf=lambda: make_pipeline(RobustScaler(), Clipper(), LogisticRegression(C=1e-4, max_iter=5000)), F=F_stack(26))
    V["R L26 stack winsor1%+std"] = dict(clf=lambda: make_pipeline(Winsor(0.01), StandardScaler(), LogisticRegression(C=1e-4, max_iter=5000)), F=F_stack(26))
    V["R L26 stack std+clip5"] = dict(clf=lambda: make_pipeline(StandardScaler(), Clipper(), LogisticRegression(C=1e-4, max_iter=5000)), F=F_stack(26))
    return V


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--out", default=str(HERE / "results_round2.json"))
    a = ap.parse_args()
    cal, ext = Calib(), Ext()
    elig = eligible_pools(cal.y, cal.meta.pool.values)
    print("eligible pools for macro metrics:", elig, flush=True)
    V = variants(); res = []
    for n, spec in V.items():
        if a.only and not any(s in n for s in a.only.split(",")):
            continue
        try:
            res.append(run(n, spec["clf"], *spec["F"], cal, ext, weights=spec.get("w"), needs_groups=spec.get("groups", False)))
        except Exception as e:
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("\n== ranked by LOPO macro (selection metric) ==")
    for r in sorted(res, key=lambda r: -r["lopo_macro"]):
        print(f"  lopoM {r['lopo_macro']:.4f} worst {r['lopo_worst']:.3f}  OOF {r['oof']:.4f}  ext {r['ext']['mean']:.4f}  {r['name']}")
