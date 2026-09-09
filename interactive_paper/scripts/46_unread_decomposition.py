"""Decompose the unread 2/3 (G0 follow-up to 45_oracle_utilization).

At the 30% budget the probe escalates top-k rows; the oracle spends
the budget on fixable rows (L=0, E=1).  This script takes the MISSED
fixable rows (fixable but not escalated by the probe) and asks what
they are:

  a. failure taxonomy (8bv `failure_types.parquet`, covers every
     never-arm failure): confident_wrong / execution / perception /
     quality_other / knowledge_gap.  Execution failures unfold during
     the answer -- a commit-time router plausibly cannot see them.
  b. would the k2 read (two answer chunks in, 8bw recipe retrained
     here on the same `*k` train dumps) have fired on them at ITS own
     matched per-pool 30% quantile?  Splits "readable later, committed
     too early" from "unread even two chunks in".

Caveat reported alongside: the k2 scores come from the 8bw dump
generation, not the never-arm generation (same query ids, different
sampled answers); the never-vs-kdump label agreement bounds how far
that mapping can be trusted.

Usage (from interactive_paper/):
  .venv_boot\\Scripts\\python.exe scripts\\46_unread_decomposition.py
Outputs: figures/unread_decomposition.json, printed tables.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

spec = importlib.util.spec_from_file_location(
    "s40", Path(__file__).with_name("40_threshold_sweep.py"))
s40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s40)

D = Path("data")
BUDGET = .30
TRAIN_K = ["calibk", "expk", "exp2k", "exp3k", "exp3zhk", "freshk"]
KEY = "X_k2"
POOL2K = {"frozen": "testk", "striviaqa": "striviaqak", "swebq": "swebqk",
          "sllama": "sllamak", "sdqa": "sdqak", "sreason": "sreasonk"}
FTYPES = ["confident_wrong", "knowledge_gap", "perception", "execution",
          "quality_other"]


def load_k(tag, key=KEY):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z[key])
    if not X:
        return None
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    X = np.concatenate(X)[df["row"].to_numpy()]
    lab = pd.read_parquet(D / f"frozen_native_{tag}_judged.parquet")
    lab = lab.drop_duplicates("id", keep="last").set_index("id")
    y = lab["escalate_label"].reindex(df["id"]).to_numpy(float)
    keep = ~np.isnan(y)
    return (df["id"].to_numpy()[keep], X[keep], y[keep].astype(int))


def main():
    # --- k2 probe, 8bw recipe (LR C=3e-4 on X_k2 of the *k train dumps,
    # fresh restricted to its train split)
    fl = pd.read_parquet(D / "fresh_labels.parquet").set_index("id")["split"]
    Xs, ys = [], []
    for tag in TRAIN_K:
        got = load_k(tag)
        if got is None:
            print(f"{tag}: no feats, skipped")
            continue
        ids, X, y = got
        if tag == "freshk":
            m = np.array([fl.get(i) == "train" for i in ids])
            X, y = X[m], y[m]
        Xs.append(X)
        ys.append(y)
    X_tr, y_tr = np.concatenate(Xs), np.concatenate(ys)
    print(f"k2 probe train rows {len(y_tr)}  fail rate {y_tr.mean():.3f}")
    k2 = LogisticRegression(C=3e-4, max_iter=5000).fit(X_tr, y_tr)

    tax = pd.read_parquet(
        D / "native_bench" / "failure_types.parquet").set_index("id")["ftype"]

    out = {"budget": BUDGET, "pools": {}}
    agg = {"missed": [], "caught": []}
    for pool in s40.POOLS:
        never = s40.load_arm(pool, "never")
        always = s40.load_arm(pool, "always")
        ids = never.index.intersection(always.index)
        never, always = never.loc[ids], always.loc[ids]
        L = never["y"].to_numpy()
        E = always["y"].to_numpy()
        info = never["is_info"].fillna(True).astype(bool).to_numpy()
        score = np.where(info, never["score"].to_numpy(float), -np.inf)
        n = len(ids)
        k = int(round(BUDGET * n))
        fire = np.zeros(n, bool)
        fire[np.argsort(-score, kind="stable")[:k]] = True
        fire &= info
        fixable = (L == 0) & (E == 1)
        missed = fixable & ~fire
        caught = fixable & fire

        # k2 scores mapped by id onto this pool's rows
        kt = load_k(POOL2K[pool])
        rec = {}
        agree = None
        if kt is not None:
            kids, kX, ky = kt
            ks = pd.Series(k2.predict_proba(kX)[:, 1], index=kids)
            shared = ids.intersection(ks.index)
            thr_k2 = float(np.quantile(ks.loc[shared], 1 - BUDGET))
            k2s = ks.reindex(ids)
            # label agreement never-arm vs k-dump generation
            kyl = pd.Series(ky, index=kids).reindex(ids)
            m = kyl.notna().to_numpy()
            agree = float(((1 - L[m]) == kyl[m]).mean())
            for name, mask in (("missed", missed), ("caught", caught)):
                v = k2s.to_numpy(float)[mask]
                ok = ~np.isnan(v)
                rec[name] = {
                    "n_with_k2": int(ok.sum()),
                    "k2_fire_share": round(float((v[ok] >= thr_k2).mean()), 3)
                    if ok.any() else None,
                    "k2_score_mean": round(float(v[ok].mean()), 3)
                    if ok.any() else None}

        def ft_counts(mask):
            t = tax.reindex(ids[mask]).fillna("unlabeled")
            return t.value_counts().to_dict()

        out["pools"][pool] = {
            "n": n, "k": k,
            "fixable": int(fixable.sum()),
            "caught": int(caught.sum()), "missed": int(missed.sum()),
            "ftype_missed": ft_counts(missed),
            "ftype_caught": ft_counts(caught),
            "k2": rec, "label_agree_never_vs_kdump": agree}
        for nm, mask in (("missed", missed), ("caught", caught)):
            t = tax.reindex(ids[mask]).fillna("unlabeled")
            agg[nm].append(t)
        p = out["pools"][pool]
        print(f"\n=== {pool}: fixable {p['fixable']}, caught {p['caught']},"
              f" missed {p['missed']}  (agree {agree})")
        print("  missed ftypes:", p["ftype_missed"])
        if rec:
            print(f"  k2 would fire on {rec['missed']['k2_fire_share']}"
                  f" of missed (caught: {rec['caught']['k2_fire_share']})")

    for nm in ("missed", "caught"):
        t = pd.concat(agg[nm])
        out[f"ftype_{nm}_total"] = t.value_counts().to_dict()
        print(f"\nALL POOLS {nm} ({len(t)}):",
              (t.value_counts(normalize=True).round(3)).to_dict())

    Path("figures/unread_decomposition.json").write_text(
        json.dumps(out, indent=1))
    print("\nwrote figures/unread_decomposition.json")


if __name__ == "__main__":
    main()
