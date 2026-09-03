"""Round-3: shrinkage-LDA heads + multi-layer logit averaging + 4-block stack
(Fable 5.1 proposals 1-4), evaluated with the full metric set and a paired
bootstrap against the deployed probe.

Pre-declared selection protocol (calibration only):
  * LDA shrinkage λ chosen by pooled 5-fold OOF over {10, 20, 30, 50, 100}
    on the L30 3-block stack; that λ is then frozen for every other variant.
  * Layer set for averaging fixed a priori to the plateau layers on disk
    {22, 26, 30, 34, 38} (and the 3-layer core {26, 30, 34}).
  * Finalists ranked by LOPO (pooled and macro); external reported after.

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python experiments3.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import D, EXT_J, F_frames, F_stack, _frames, fit_w, lr  # noqa: E402
from experiments2 import eligible_pools, macro_auc  # noqa: E402
from probe_lab import Calib, Ext, EXT_POOLS  # noqa: E402

HERE = Path(__file__).resolve().parent
LAYERS5 = (22, 26, 30, 34, 38)


class ShrinkLDA(BaseEstimator, ClassifierMixin):
    """Fisher direction with ridge-shrunk within-class covariance, dual (n×n) form.
    w = (S + λI)^{-1}(μ1 − μ0) on z-scored features; score standardized on the
    training set so heads can be averaged across layers."""

    def __init__(self, lam=30.0):
        self.lam = lam

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, np.float64)
        self.mu_ = X.mean(0); self.sd_ = X.std(0) + 1e-6
        Z = (X - self.mu_) / self.sd_
        m1, m0 = Z[y == 1].mean(0), Z[y == 0].mean(0)
        Zc = Z.copy(); Zc[y == 1] -= m1; Zc[y == 0] -= m0
        n = len(y); v = m1 - m0
        A = Zc @ Zc.T                                     # n×n
        rhs = Zc @ v
        alpha = np.linalg.solve(self.lam * (n - 2) * np.eye(n) + A, rhs)
        self.w_ = (v - Zc.T @ alpha) / self.lam           # (S+λI)^{-1} v via Woodbury
        s = Z @ self.w_
        self.b_ = -float(self.w_ @ (m0 + m1) / 2)
        self.scale_ = float(s.std() + 1e-9)
        self.classes_ = np.array([0, 1]); return self

    def decision_function(self, X):
        Z = (np.asarray(X, np.float64) - self.mu_) / self.sd_
        return (Z @ self.w_ + self.b_) / self.scale_

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X))); return np.c_[1 - p, p]


class LayerAvg(BaseEstimator, ClassifierMixin):
    """One head per equal-width block (one block per layer); average standardized logits."""

    def __init__(self, make_head, n_blocks):
        self.make_head = make_head; self.n_blocks = n_blocks

    def fit(self, X, y, sample_weight=None):
        w = X.shape[1] // self.n_blocks
        self.heads_ = [self.make_head().fit(X[:, i * w:(i + 1) * w], y) for i in range(self.n_blocks)]
        self.classes_ = np.array([0, 1]); return self

    def decision_function(self, X):
        w = X.shape[1] // self.n_blocks
        zs = []
        for i, h in enumerate(self.heads_):
            z = h.decision_function(X[:, i * w:(i + 1) * w])
            zs.append(z if isinstance(h, ShrinkLDA) else z / (np.std(z) + 1e-9))
        return np.mean(zs, axis=0)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X))); return np.c_[1 - p, p]


BLOCK4 = ["first", "last", "mean8", "umean"]
BLOCK3 = ["last", "mean8", "umean"]


def F_layers(Ls, which):
    return (lambda c: np.concatenate([_frames(c.E_on, c.M, c.j(L), which) for L in Ls], 1),
            lambda d: np.concatenate([_frames(d["E_on"], d["M"], EXT_J[L], which) for L in Ls], 1))


def run(name, make_clf, fc, fe, cal, ext, seeds=(42,)):
    t = time.time(); X = fc(cal); y = cal.y; pools = cal.meta.pool.values; elig = eligible_pools(y, pools)
    oofs = []; oof = None
    for s in seeds:
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            p[te] = make_clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oofs.append(float(roc_auc_score(y, p))); oof = p if oof is None else oof
    lopo = np.zeros(len(y))
    for g in np.unique(pools):
        te = pools == g; lopo[te] = make_clf().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    lopo_macro, lopo_worst, lopo_per = macro_auc(lopo, y, pools, elig); oof_macro, _, _ = macro_auc(oof, y, pools, elig)
    full = make_clf().fit(X, y)
    scores = {tag: full.predict_proba(fe(d))[:, 1] for tag, d in ext.pools.items()}
    ext_row = {tag: float(roc_auc_score(ext.pools[tag]["y"], scores[tag])) for tag in EXT_POOLS}
    ext_row["mean"] = float(np.mean(list(ext_row.values())))
    rep = dict(name=name, dims=int(X.shape[1]), oof=float(np.mean(oofs)), oof_sd=float(np.std(oofs)), oof_macro=oof_macro,
               lopo=float(roc_auc_score(y, lopo)), lopo_macro=lopo_macro, lopo_worst=lopo_worst, lopo_per=lopo_per, ext=ext_row,
               secs=round(time.time() - t, 1))
    print(f"{name:50s} OOF {rep['oof']:.4f} oofM {oof_macro:.4f} | LOPO {rep['lopo']:.4f} lopoM {lopo_macro:.4f} worst {lopo_worst:.3f} | "
          f"ext {ext_row['striviaqa']:.3f}/{ext_row['swebq']:.3f}/{ext_row['sllama']:.3f}/{ext_row['sdqa']:.3f}={ext_row['mean']:.4f} ({rep['secs']}s)", flush=True)
    return rep, scores


def paired_boot(sa, sb, ext, rng, B=2000):
    d = np.zeros(B)
    for b in range(B):
        aa, bb = [], []
        for tag in EXT_POOLS:
            y = ext.pools[tag]["y"]; ii = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ii])) < 2:
                continue
            aa.append(roc_auc_score(y[ii], sa[tag][ii])); bb.append(roc_auc_score(y[ii], sb[tag][ii]))
        d[b] = np.mean(bb) - np.mean(aa)
    return d


if __name__ == "__main__":
    cal, ext = Calib(), Ext(); rng = np.random.default_rng(0)
    out = []; S = {}

    # --- step 1: λ by calibration OOF on the L30 3-block stack -----------------
    print("== λ selection (OOF only) ==", flush=True)
    lam_res = {}
    for lam in (10, 20, 30, 50, 100):
        rep, sc = run(f"lda λ={lam} L30 3-block", lambda lam=lam: ShrinkLDA(lam), *F_stack(30), cal, ext)
        lam_res[lam] = rep["oof"]; out.append(rep); S[rep["name"]] = sc
    LAM = max(lam_res, key=lam_res.get)
    print(f"-> selected λ={LAM} by OOF (not by external)", flush=True)

    # --- step 2: variants with the frozen λ --------------------------------------
    V = {
        "A0 deployed LR L30 3-block": (lambda: lr(1e-4), *F_stack(30)),
        "A1 LR L26 3-block": (lambda: lr(1e-4), *F_stack(26)),
        "B1 LR L30 4-block (+commit frame)": (lambda: lr(1e-4), *F_frames(30, BLOCK4)),
        "B2 LR L26 4-block": (lambda: lr(1e-4), *F_frames(26, BLOCK4)),
        f"C1 LDA λ={LAM} L30 4-block": (lambda: ShrinkLDA(LAM), *F_frames(30, BLOCK4)),
        f"C2 LDA λ={LAM} L26 3-block": (lambda: ShrinkLDA(LAM), *F_stack(26)),
        f"C3 LDA λ={LAM} L26 4-block": (lambda: ShrinkLDA(LAM), *F_frames(26, BLOCK4)),
        "D1 layer-avg LR ×5 4-block": (lambda: LayerAvg(lambda: lr(1e-4), 5), *F_layers(LAYERS5, BLOCK4)),
        "D2 layer-avg LR ×3 (26,30,34) 4-block": (lambda: LayerAvg(lambda: lr(1e-4), 3), *F_layers((26, 30, 34), BLOCK4)),
        f"E1 layer-avg LDA λ={LAM} ×5 4-block  [Fable combo]": (lambda: LayerAvg(lambda: ShrinkLDA(LAM), 5), *F_layers(LAYERS5, BLOCK4)),
        f"E2 layer-avg LDA λ={LAM} ×3 (26,30,34) 4-block": (lambda: LayerAvg(lambda: ShrinkLDA(LAM), 3), *F_layers((26, 30, 34), BLOCK4)),
        f"E3 layer-avg LDA λ={LAM} ×5 3-block": (lambda: LayerAvg(lambda: ShrinkLDA(LAM), 5), *F_layers(LAYERS5, BLOCK3)),
        f"E4 layer-avg LDA λ={LAM} ×3 (22,26,30) 4-block": (lambda: LayerAvg(lambda: ShrinkLDA(LAM), 3), *F_layers((22, 26, 30), BLOCK4)),
    }
    for n, (mk, fc, fe) in V.items():
        rep, sc = run(n, mk, fc, fe, cal, ext, seeds=(0, 1, 42)); out.append(rep); S[n] = sc
        json.dump(out, open(HERE / "results_round3.json", "w"), indent=1)

    # --- step 3: paired bootstrap vs deployed --------------------------------------
    print("\n== paired bootstrap of external-mean Δ vs deployed (2000 resamples) ==", flush=True)
    base = S["A0 deployed LR L30 3-block"]
    for r in out:
        if r["name"] not in S or r["name"].startswith("A0"):
            continue
        d = paired_boot(base, S[r["name"]], ext, rng)
        r["delta_ext"] = dict(mean=float(d.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))
        print(f"  {r['name']:50s} Δext {d.mean():+.4f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  P(Δ≤0)={(d <= 0).mean():.3f}", flush=True)
    json.dump(out, open(HERE / "results_round3.json", "w"), indent=1)

    print("\n== ranked by LOPO macro, then LOPO pooled (calibration-only selection) ==")
    for r in sorted(out, key=lambda r: (-r["lopo_macro"], -r["lopo"])):
        print(f"  lopoM {r['lopo_macro']:.4f}  LOPO {r['lopo']:.4f}  OOF {r['oof']:.4f}  ext {r['ext']['mean']:.4f}  {r['name']}")
