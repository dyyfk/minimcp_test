"""Export the round-3 improved failure probe as a demo-compatible artifact.

Probe v2 = average of three per-layer linear heads (L26, L30, L34), each
StandardScaler -> LogisticRegression(C=1e-4) on the 4-block onset stack
[commit frame | 8th frame | 8-frame mean | user-audio mean] (17,920 dims per
layer).  Scalers are folded into raw-space (w, b) exactly as
fit_demo_artifacts.py does; each head's logit is divided by its training
logit std before averaging, so the exported score is

    z = mean_L ( (w_L . x_L + b_L) / s_L ),   P(fail) = sigmoid(z)

Fit cohort mirrors the deployed artifact: frozen `calib` split + both
expansions, no-commit dropped (n = 2,258); the frozen test-240 rows are
excluded.  Thresholds = OOF quantiles at 15 / 30 / 50 % as before.
The act head is NOT re-exported (unchanged, still reads L30 3-block).

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python export_probe_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import _frames  # noqa: E402
from probe_lab import Calib, Ext, EXT_POOLS, GP  # noqa: E402

HERE = Path(__file__).resolve().parent
LAYERS = (26, 30, 34)
BLOCKS = ["first", "last", "mean8", "umean"]
C = 1e-4

with (GP / "queries.jsonl").open(encoding="utf-8") as fh:
    FROZEN_CALIB_IDS = {str(r["id"]) for r in map(json.loads, fh) if r.get("split") == "calib"}


def feats(E, M, j):
    return _frames(E, M, j, BLOCKS)


def head():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


class V2:
    def fit(self, Xs, y):
        self.heads, self.scale = [], []
        for X in Xs:
            h = head().fit(X, y); self.heads.append(h); self.scale.append(float(h.decision_function(X).std() + 1e-9))
        return self

    def logit(self, Xs):
        return np.mean([h.decision_function(X) / s for h, s, X in zip(self.heads, self.scale, Xs)], axis=0)


def fold(pipe):
    sc, lr = pipe.named_steps.values()
    w = lr.coef_[0] / sc.scale_
    return w.astype(float).tolist(), float(lr.intercept_[0] - (w * sc.mean_).sum())


if __name__ == "__main__":
    cal, ext = Calib(), Ext()
    keep = np.array([(t != "frozen") or (i in FROZEN_CALIB_IDS) for t, i in zip(cal.meta.tag, cal.meta.id)])
    y = cal.y[keep]
    Xs = [feats(cal.E_on[keep], cal.M[keep], cal.j(L)) for L in LAYERS]
    print(f"fit cohort n={len(y)} (frozen test-240 excluded), fail={y.mean():.3f}")

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xs[0], y):
        m = V2().fit([X[tr] for X in Xs], y[tr]); oof[te] = m.logit([X[te] for X in Xs])
    print(f"OOF AUC (calib-only cohort) {roc_auc_score(y, oof):.4f}")
    p_oof = 1 / (1 + np.exp(-oof))
    thr = {t: float(np.quantile(p_oof, 1 - b)) for t, b in (("conservative", .15), ("balanced", .30), ("aggressive", .50))}

    full = V2().fit(Xs, y)
    ext_auc = {}
    for tag in EXT_POOLS:
        d = ext.pools[tag]; z = full.logit([feats(d["E_on"], d["M"], ext.j(L)) for L in LAYERS])
        ext_auc[tag] = float(roc_auc_score(d["y"], z))
        fire = {t: float((1 / (1 + np.exp(-z)) >= v).mean()) for t, v in thr.items()}
        print(f"  {tag:10s} cold AUC {ext_auc[tag]:.4f}   fire-rate at calib thresholds cons/bal/agg = "
              f"{fire['conservative']:.2f}/{fire['balanced']:.2f}/{fire['aggressive']:.2f}")
    ext_auc["mean"] = float(np.mean([ext_auc[t] for t in EXT_POOLS])); print(f"  external mean {ext_auc['mean']:.4f}")

    heads = []
    for L, h, s in zip(LAYERS, full.heads, full.scale):
        w, b = fold(h); heads.append({"layer": L, "w": w, "b": b, "logit_std": s})
    artifact = {
        "version": "v2-2026-09-01", "read": "onset: first-8-frames-from-sustained-nonpad-onset",
        "blocks": BLOCKS, "block_dim": 4480, "k_eot": 8, "layers": list(LAYERS),
        "score": "sigmoid(mean_L((w_L.x_L + b_L)/logit_std_L)); x_L = [onset[0] | onset[-1] | onset.mean(0) | user_mean] at layer L",
        "fail": {"heads": heads, "thresholds": thr},
        "n_calib": int(len(y)), "internal_test_excluded": 240,
        "calib_oof_auc": float(roc_auc_score(y, oof)), "external_auc_cold": ext_auc,
        "source": "official-nemo-native-protocol-replay-calib-only; selected by LOPO on calibration (probe_doc/experiments3.py)",
        "note": "act head unchanged (gate_demo_nvda.json); online readout must hook L26/L30/L34 and keep onset[0]",
    }
    out = HERE / "gate_demo_nvda_v2.json"
    out.write_text(json.dumps(artifact), encoding="utf-8")
    print(f"exported {out} ({out.stat().st_size/2**20:.1f} MB) thr={ {k: round(v, 4) for k, v in thr.items()} }")
