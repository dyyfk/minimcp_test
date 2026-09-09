"""Commit-time next-token entropy baseline + post-norm read cell (8cs).

From the `*e` dumps (modal_native_dump 8cs fields, official config,
same-generation native labels):

  A. Step-0 baseline cells: AUC of next-token entropy / top-1 prob /
     top1-top2 margin against escalate_label, at two moments:
       ENT_pre : before the onset chunk's generate — strictly causal,
                 zero answer tokens (the honest baseline cell)
       ENT     : the deployed read moment (after the onset chunk's
                 generate) — expected degenerate: the last token there
                 is a control token with a near-deterministic successor
     Reference on identical rows: the deployed L22 gate score
     (gate_native.json w/b applied to X).

  B. The untested post-norm cell: the review noted every sweep read is
     a block output PRE final RMSNorm while the LM head sees post-norm.
     H35 (pre-norm) vs H35n (post-norm) last-token probes, evaluated
     LOPO across the five external pools (train LR C=3e-4 on four,
     test the held-out fifth) and pooled 5-fold OOF.

Usage (from interactive_paper/):
  .venv_boot\\Scripts\\python.exe scripts\\47_commit_entropy.py
Outputs: figures/commit_entropy.json, printed tables.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
TAGS = {"teste": "frozen", "striviaqae": "striviaqa", "swebqe": "swebq",
        "sllamae": "sllama", "sdqae": "sdqa", "sreasone": "sreason"}
EXT5 = ["striviaqae", "swebqe", "sllamae", "sdqae", "sreasone"]
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def load(tag):
    ids, arrs = [], {k: [] for k in ("X", "ENT", "ENT_pre", "H35", "H35n")}
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        for k in arrs:
            arrs[k].append(z[k])
    if not ids:
        return None
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    sel = df["row"].to_numpy()
    A = {k: np.concatenate(v)[sel] for k, v in arrs.items()}
    lab = pd.read_parquet(D / f"frozen_native_{tag}_judged.parquet")
    lab = lab.drop_duplicates("id", keep="last").set_index("id")
    y = lab["escalate_label"].reindex(df["id"]).to_numpy(float)
    keep = ~np.isnan(y)
    return ({k: v[keep] for k, v in A.items()}, y[keep].astype(int),
            df["id"].to_numpy()[keep])


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def main():
    gate = json.loads((D / "gate_native.json").read_text())
    w, b = np.array(gate["w"], dtype=np.float32), gate["b"]

    data = {}
    for tag in TAGS:
        got = load(tag)
        if got is None:
            print(f"{tag}: MISSING (dump or judge not done)")
            continue
        data[tag] = got

    out = {"A_baselines": {}, "B_postnorm": {}}
    cells = [("ent_pre", lambda A: A["ENT_pre"][:, 0]),
             ("neg_top1_pre", lambda A: -A["ENT_pre"][:, 1]),
             ("neg_margin_pre", lambda A: -A["ENT_pre"][:, 2]),
             ("ent_onset", lambda A: A["ENT"][:, 0]),
             ("neg_top1_onset", lambda A: -A["ENT"][:, 1]),
             ("probe_L22", lambda A: A["X"] @ w + b)]
    hdr = f"{'pool':12s}" + "".join(f"{n:>16s}" for n, _ in cells) + "     n  fail"
    print(hdr)
    for tag, (A, y, _) in data.items():
        row = {}
        for name, fn in cells:
            row[name] = round(auc(y, fn(A)), 3)
        row["n"], row["fail"] = int(len(y)), round(float(y.mean()), 3)
        out["A_baselines"][TAGS[tag]] = row
        print(f"{TAGS[tag]:12s}" + "".join(f"{row[n]:16.3f}" for n, _ in cells)
              + f"  {row['n']:4d}  {row['fail']:.3f}")
    ext = {n: round(float(np.mean(
        [out["A_baselines"][TAGS[t]][n] for t in EXT5 if TAGS[t] in
         out["A_baselines"]])), 3) for n, _ in cells}
    out["A_baselines"]["ext5_mean"] = ext
    print(f"{'ext5-mean':12s}" + "".join(f"{ext[n]:16.3f}" for n, _ in cells))

    # B: LOPO over the five external pools, pre- vs post-norm last token
    for key in ("H35", "H35n", "X"):
        res = {}
        for held in EXT5:
            if held not in data:
                continue
            tr = [t for t in EXT5 if t != held and t in data]
            Xt = np.concatenate([data[t][0][key] for t in tr])
            yt = np.concatenate([data[t][1] for t in tr])
            m = LogisticRegression(C=3e-4, max_iter=5000).fit(Xt, yt)
            Ah, yh, _ = data[held]
            res[TAGS[held]] = round(
                auc(yh, m.decision_function(Ah[key])), 3)
        if data:
            Xa = np.concatenate([data[t][0][key] for t in EXT5 if t in data])
            ya = np.concatenate([data[t][1] for t in EXT5 if t in data])
            oof = cross_val_predict(
                LogisticRegression(C=3e-4, max_iter=5000), Xa, ya,
                cv=CV, method="decision_function")
            res["pooled_oof"] = round(auc(ya, oof), 3)
            res["lopo_mean"] = round(float(np.mean(
                [v for k, v in res.items() if k != "pooled_oof"])), 3)
        out["B_postnorm"][key] = res
        print(f"\nLOPO {key}: {res}")

    Path("figures/commit_entropy.json").write_text(json.dumps(out, indent=1))
    print("\nwrote figures/commit_entropy.json")


if __name__ == "__main__":
    main()
