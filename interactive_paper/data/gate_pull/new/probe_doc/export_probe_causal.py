"""Export the strictly causal probe (pass-3 capture) as a demo artifact.

Read: commit frame (onset[0]) | mean of the 8 frames BEFORE the commit (H_pre)
| running mean of all frames before the commit (H_run), at L34 — nothing after
the commit frame is used.  StandardScaler -> LogisticRegression(C=1e-4),
scaler folded into raw-space (w, b).  Fit cohort mirrors the deployed artifact
(frozen `calib` split + expansions, no-commit dropped, n=2,258; frozen
test-240 excluded).  Thresholds = OOF quantiles at 15 / 30 / 50 %.

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python export_probe_causal.py
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
from experiments4 import Pass3  # noqa: E402
from probe_lab import CALIB_TAGS, EXT_POOLS, GP  # noqa: E402

HERE = Path(__file__).resolve().parent
L = 34
BLOCKS = ["commit", "pre_mean8", "run_mean"]
with (GP / "queries.jsonl").open(encoding="utf-8") as fh:
    FROZEN_CALIB_IDS = {str(r["id"]) for r in map(json.loads, fh) if r.get("split") == "calib"}


def fold(pipe):
    sc, lr = pipe.named_steps.values(); w = lr.coef_[0] / sc.scale_
    return w.astype(float).tolist(), float(lr.intercept_[0] - (w * sc.mean_).sum())


if __name__ == "__main__":
    cal = Pass3(CALIB_TAGS, lambda tag: GP / "onset_fit" / f"nvda_{tag}.parquet")
    ext = {tag: Pass3([tag], lambda t: GP / "onset" / f"nvda_{t}_ext2.parquet") for tag in EXT_POOLS}
    keep = np.array([(t != "frozen") or (i in FROZEN_CALIB_IDS) for t, i in zip(cal.tag, cal.ids)])
    X = cal.feats(L, BLOCKS)[keep]; y = cal.y[keep]
    print(f"fit cohort n={len(y)} fail={y.mean():.3f}")
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e-4, max_iter=5000))
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    thr = {t: float(np.quantile(oof, 1 - b)) for t, b in (("conservative", .15), ("balanced", .30), ("aggressive", .50))}
    full = mk().fit(X, y)
    ea = {tag: float(roc_auc_score(e.y, full.predict_proba(e.feats(L, BLOCKS))[:, 1])) for tag, e in ext.items()}
    ea["mean"] = float(np.mean([ea[t] for t in EXT_POOLS]))
    print(f"OOF {roc_auc_score(y, oof):.4f}  ext {ea}")
    w, b = fold(full)
    art = {"version": "causal-2026-09-01", "layer": L, "blocks": BLOCKS, "block_dim": 4480, "k_eot": 8,
           "read": "strictly causal: x = [h(commit frame) | mean(8 frames before commit) | mean(all frames before commit)] at L34; nothing after the commit frame",
           "fail": {"w": w, "b": b, "thresholds": thr}, "n_calib": int(len(y)), "internal_test_excluded": 240,
           "calib_oof_auc": float(roc_auc_score(y, oof)), "external_auc_cold": ea,
           "source": "pass-3 capture (nvda_replay_v2.py), official NeMo offline path, calib-only fit"}
    out = HERE / "gate_demo_nvda_causal.json"; out.write_text(json.dumps(art), encoding="utf-8")
    print(f"exported {out} ({out.stat().st_size/2**20:.2f} MB) thr={ {k: round(v, 4) for k, v in thr.items()} }")
