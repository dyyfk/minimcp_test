"""Local CPU harness for NVDA probe experiments.

Loads the pass-2 calibration shards (onset read, 14 layers) and the L22–38
external slices, exposes the deployed recipe, 5-fold OOF, leave-one-pool-out
(LOPO) and cold external AUC.  Nothing here touches the model or the pod.

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python probe_lab.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
GP = HERE.parents[1]                                  # data/gate_pull
CALIB_LAYERS = list(range(2, 56, 4))                  # 14 layers in calib shards
EXT_LAYERS = [22, 26, 30, 34, 38]                     # layers in the external slices
EXT_POOLS = ("striviaqa", "swebq", "sllama", "sdqa")
CALIB_TAGS = ("frozen", "expansion", "expansion2")


class Calib:
    """Calibration cohort: committed rows only (onset_frame >= 0), pass-2 labels."""

    def __init__(self, layers=EXT_LAYERS):
        ids, E_on, E_eot, M, ONS = [], [], [], [], []
        jj = [CALIB_LAYERS.index(L) for L in layers]
        for tag in CALIB_TAGS:
            for sh in sorted(glob.glob(str(GP / "onset" / f"nvda_h_{tag}.shard*.npz"))):
                z = np.load(sh, allow_pickle=True)
                ids += [(tag, str(x)) for x in z["ids"]]
                E_on.append(z["H_onset"][:, jj]); E_eot.append(z["H_eot"][:, jj])
                M.append(z["H_mean"][:, jj]); ONS.append(z["onset_frame"])
        E_on, E_eot = np.concatenate(E_on), np.concatenate(E_eot)
        M, ONS = np.concatenate(M), np.concatenate(ONS)
        lab = {}
        for tag in CALIB_TAGS:
            df = pd.read_parquet(GP / "onset_fit" / f"nvda_{tag}.parquet")
            for _, r in df.iterrows():
                if pd.notna(r["escalate_label"]):
                    lab[(tag, str(r["id"]))] = int(r["escalate_label"])
        q = {}
        for f in ("queries.jsonl", "queries_expansion.jsonl", "queries_expansion2.jsonl"):
            for line in (GP / f).open(encoding="utf-8"):
                r = json.loads(line); q[str(r["id"])] = r
        nfr = {}
        for f in glob.glob(str(GP / "onset" / "nvda_answers_*.shard*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line); nfr[str(r["id"])] = r["n_frames_query"]
        keep = [i for i, k in enumerate(ids) if k in lab and ONS[i] >= 0]
        self.layers = list(layers)
        self.E_on = E_on[keep]; self.E_eot = E_eot[keep]; self.M = M[keep]
        self.y = np.array([lab[ids[i]] for i in keep])
        self.meta = pd.DataFrame({
            "tag": [ids[i][0] for i in keep], "id": [ids[i][1] for i in keep],
            "pool": [q[ids[i][1]]["pool"] for i in keep],
            "split": [q[ids[i][1]].get("split") for i in keep],
            "onset": ONS[keep], "n_frames": [nfr.get(ids[i][1], -1) for i in keep]})

    def j(self, L):
        return self.layers.index(L)


class Ext:
    def __init__(self, layers=EXT_LAYERS):
        self.layers = list(layers); self.pools = {}
        for tag in EXT_POOLS:
            z = np.load(GP / "onset_ext" / f"nvda_h_{tag}.L22-38.npz", allow_pickle=True)
            zl = [int(x) for x in z["layers"]]; jj = [zl.index(L) for L in layers]
            ids = [str(x) for x in z["ids"]]
            lab = pd.read_parquet(GP / "onset" / f"nvda_{tag}_ext2.parquet").set_index("id")["escalate_label"]
            ons = z["onset_frame"]
            keep = [i for i, q in enumerate(ids) if q in lab.index and pd.notna(lab.get(q)) and ons[i] >= 0]
            self.pools[tag] = dict(
                E_on=z["H_onset"][keep][:, jj], E_eot=z["H_eot"][keep][:, jj], M=z["H_mean"][keep][:, jj],
                y=np.array([int(lab[ids[i]]) for i in keep]), ids=[ids[i] for i in keep])

    def j(self, L):
        return self.layers.index(L)


# ------------------------------------------------------------ recipe ------
def stack(E, M, j):
    """Deployed feature stack: last frame ‖ 8-frame mean ‖ user-audio mean (fp32)."""
    return np.concatenate([E[:, j, -1], E[:, j].mean(1), M[:, j]], axis=1).astype(np.float32)


def deployed_clf(C=1e-4):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def oof_scores(make_clf, X, y, seed=42, n_splits=5):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m = make_clf().fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def lopo_scores(make_clf, X, y, groups):
    """Leave-one-calibration-pool-out scores (a harsher proxy for cold transfer)."""
    p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        p[te] = make_clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def within_pool_auc(p, y, groups):
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(y[m])) > 1:
            out[str(g)] = float(roc_auc_score(y[m], p[m]))
    return out


def ext_auc(model, featurize, ext: Ext):
    """featurize(pool_dict) -> X ; returns per-pool AUC + mean."""
    row = {}
    for tag, d in ext.pools.items():
        row[tag] = float(roc_auc_score(d["y"], model.predict_proba(featurize(d))[:, 1]))
    row["mean"] = float(np.mean([row[t] for t in EXT_POOLS]))
    return row


def evaluate(name, make_clf, feat_calib, feat_ext, calib: Calib, ext: Ext, seeds=(42,), lopo=True):
    """Full report for one variant: OOF (mean over seeds), LOPO, cold external."""
    X = feat_calib(calib); y = calib.y
    oofs = [roc_auc_score(y, oof_scores(make_clf, X, y, seed=s)) for s in seeds]
    rep = {"name": name, "oof": float(np.mean(oofs)), "oof_seeds": [float(a) for a in oofs]}
    if lopo:
        pl = lopo_scores(make_clf, X, y, calib.meta.pool.values)
        rep["lopo"] = float(roc_auc_score(y, pl))
        rep["lopo_within"] = within_pool_auc(pl, y, calib.meta.pool.values)
    m = make_clf().fit(X, y)
    rep["ext"] = ext_auc(m, feat_ext, ext)
    return rep


if __name__ == "__main__":
    import time
    t = time.time()
    cal, ext = Calib(), Ext()
    print(f"calib n={len(cal.y)} fail={cal.y.mean():.3f}; ext " +
          ", ".join(f"{k} n={len(v['y'])} fail={v['y'].mean():.2f}" for k, v in ext.pools.items()),
          f"({time.time()-t:.0f}s load)")
    j = cal.j(30)
    rep = evaluate("deployed onset@L30", deployed_clf,
                   lambda c: stack(c.E_on, c.M, j), lambda d: stack(d["E_on"], d["M"], j), cal, ext)
    print(json.dumps({k: v for k, v in rep.items() if k != "lopo_within"}, indent=1))
    print("expected: OOF .8201, ext .8375/.8508/.7712/.7741 mean .8084")
