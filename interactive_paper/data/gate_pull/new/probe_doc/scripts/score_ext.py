"""External transfer AUC at BOTH read points. Probe trained
on the full calib (frozen+exp+exp2, pass-2 labels, no-commit dropped),
externals scored cold with their pass-2 labels."""
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SH = os.environ.get("NVDA_CAPTURE_DIR", "nvda-captures")
LB = os.environ.get("NVDA_LABEL_DIR", "nvda-labels")
LAYERS = list(range(2, 56, 4))

def load(tag, key, labfile):
    ids, E, M, ONS = [], [], [], []
    for sh in sorted(glob.glob(f"{SH}/nvda_h_{tag}.shard*.npz")):
        z = np.load(sh, allow_pickle=True)
        ids += [str(x) for x in z["ids"]]
        E.append(z[key]); M.append(z["H_mean"]); ONS.append(z["onset_frame"])
    E, M, ONS = np.concatenate(E), np.concatenate(M), np.concatenate(ONS)
    lab = pd.read_parquet(labfile).set_index("id")["escalate_label"]
    keep = [i for i, q in enumerate(ids)
            if q in lab.index and pd.notna(lab.get(q)) and ONS[i] >= 0]
    y = np.array([int(lab[ids[i]]) for i in keep])
    return E[keep].astype(np.float32), M[keep].astype(np.float32), y

def stack(E, M, j):
    return np.concatenate([E[:, j, -1], E[:, j].mean(1), M[:, j]], axis=1)

out = {}
for read, key, L in (("onset", "H_onset", 30), ("eot", "H_eot", 34)):
    j = LAYERS.index(L)
    Es, Ms, ys = [], [], []
    for tag in ("frozen", "expansion", "expansion2"):
        E, M, y = load(tag, key, f"{LB}/nvda_{tag}.parquet")
        Es.append(E); Ms.append(M); ys.append(y)
    Xc = stack(np.concatenate(Es), np.concatenate(Ms), j)
    yc = np.concatenate(ys)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1e-4, max_iter=5000))
    clf.fit(Xc, yc)
    row = {}
    for tag in ("striviaqa", "swebq", "sllama", "sdqa"):
        E, M, y = load(tag, key, f"{LB}/nvda_{tag}_ext2.parquet")
        p = clf.predict_proba(stack(E, M, j))[:, 1]
        row[tag] = round(float(roc_auc_score(y, p)), 4)
    row["mean"] = round(float(np.mean(list(row.values()))), 4)
    out[f"{read}@L{L}"] = row
    print(read, row, flush=True)
json.dump(out, open(f"{SH}/ext_transfer_by_read.json", "w"), indent=1)
print("DONE")
