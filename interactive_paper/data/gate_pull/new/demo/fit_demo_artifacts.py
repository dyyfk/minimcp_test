"""Fit + export the two demo gate heads from the checked-in replay data.

Failure probe: onset-read stack @L30 (scaler+LR C=1e-4, nominal
2,310-row calibration split; no-commit rows dropped; thresholds = OOF
quantiles). The frozen pool contains both the 360 calibration rows and
the 240 internal-test rows, so the latter are explicitly excluded here.
Act head: questions
vs the 196 floor stims; tau = question-side OOF 0.5th percentile.
Export as plain w/b on the standardized space folded into raw-space
coefficients, so the server needs numpy only."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
GATE_PULL = HERE.parents[1]
SH = GATE_PULL / "onset"
LB = GATE_PULL / "onset_fit"
LAYERS = list(range(2, 56, 4))
J30 = LAYERS.index(30)
with (GATE_PULL / "queries.jsonl").open(encoding="utf-8") as fh:
    FROZEN_CALIB_IDS = {
        str(row["id"])
        for row in map(json.loads, fh)
        if row.get("split") == "calib"
    }

def stack(E, M):
    return np.concatenate([E[:, J30, -1], E[:, J30].mean(1), M[:, J30]],
                          axis=1)

ids, E, M, ONS = [], [], [], []
for tag in ("frozen", "expansion", "expansion2"):
    for sh in sorted(glob.glob(str(SH / f"nvda_h_{tag}.shard*.npz"))):
        z = np.load(sh, allow_pickle=True)
        ids += [(tag, str(x)) for x in z["ids"]]
        E.append(z["H_onset"]); M.append(z["H_mean"])
        ONS.append(z["onset_frame"])
E, M, ONS = np.concatenate(E), np.concatenate(M), np.concatenate(ONS)
lab = {}
for tag in ("frozen", "expansion", "expansion2"):
    df = pd.read_parquet(LB / f"nvda_{tag}.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["escalate_label"]):
            lab[(tag, r["id"])] = int(r["escalate_label"])
keep = [
    i for i, k in enumerate(ids)
    if k in lab and ONS[i] >= 0
    and (k[0] != "frozen" or k[1] in FROZEN_CALIB_IDS)
]
Xq = stack(E[keep].astype(np.float32), M[keep].astype(np.float32))
yq = np.array([lab[ids[i]] for i in keep])

zf = np.load(SH / "nvda_h_flooract.shard0.npz", allow_pickle=True)
Xf = stack(zf["H_onset"].astype(np.float32), zf["H_mean"].astype(np.float32))

cv = StratifiedKFold(5, shuffle=True, random_state=42)
mk = lambda: make_pipeline(StandardScaler(),
                           LogisticRegression(C=1e-4, max_iter=5000))

def fold(pipe):
    """fold scaler+LR into raw-space w, b"""
    sc, lr = pipe.named_steps.values()
    w = lr.coef_[0] / sc.scale_
    b = float(lr.intercept_[0] - (w * sc.mean_).sum())
    return w.astype(float).tolist(), b

oof = cross_val_predict(mk(), Xq, yq, cv=cv, method="predict_proba")[:, 1]
fail = mk().fit(Xq, yq)
thr = {t: float(np.quantile(oof, 1 - b))
       for t, b in (("conservative", .15), ("balanced", .30),
                    ("aggressive", .50))}

Xa = np.concatenate([Xq, Xf])
ya = np.concatenate([np.ones(len(Xq)), np.zeros(len(Xf))])
oof_a = cross_val_predict(mk(), Xa, ya, cv=cv, method="predict_proba")[:, 1]
act = mk().fit(Xa, ya)
tau = float(np.quantile(oof_a[:len(Xq)], 0.005))

wf, bf = fold(fail)
wa, ba = fold(act)
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path,
                    default=HERE / "gate_demo_nvda.json")
args = parser.parse_args()
artifact = {"layer": 30, "j": J30, "k_eot": 8,
            "fail": {"w": wf, "b": bf, "thresholds": thr},
            "act": {"w": wa, "b": ba, "tau": tau},
            "n_calib": int(len(yq)),
            "nominal_calib": 2310,
            "no_commit_dropped": 2310 - int(len(yq)),
            "internal_test_excluded": 240,
            "read": "first-8-frames-from-sustained-nonpad-onset",
            "source": "official-nemo-native-protocol-replay-calib-only"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(artifact), encoding="utf-8")
print(f"exported {args.output}: n={len(yq)} thr={thr} tau={tau:.4f}")
