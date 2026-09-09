"""Native feature ablation 2x2 (8db): layer x readout-position, one batch.

Does the deployed native probe's value come from the mid-layer (L22)
or from the three-position aggregation ([last token, tail-8 mean,
user-audio mean])? The text-condition layer sweep (fig:layersweep)
says mid-layer; this recomputes the audio-commit answer on ONE batch
under ONE protocol -- the 8cu beta=0 startline dumps (official config,
deployed read moment, X{22,33,34,35} all captured in the same session,
own-generation gpt-5.4-mini labels).

Cells (per layer L in {22, 33, 34, 35}):
  last : X{L}[:, :4096]   single-token read at the commit position
  agg  : X{L}             the deployed 12,288-d three-part aggregate

Protocol (identical for every cell, = scripts/47 head recipe):
  LOPO over the five external pools, LR C=3e-4; per-pool failure AUC
  + ext5 mean.  Routing: per-pool top-30% budget escalation mixing
  native local outcome with the cached always-arm expert outcome
  (scripts/23 remix arithmetic), matched-random control = 2000 joint
  draws (same per-pool k), permutation p on the pooled accuracy.

Regression anchors: agg0_L22 ext5-mean AUC must reproduce
figures/agg_startline.json (.6952); deployed gate on these rows ~.748
(RESULTS 8cu-2).

Usage: .venv_boot\Scripts\python.exe scripts\56_native_feature_ablation.py
Output: figures/native_feature_ablation.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = Path("data")
POOLS = [("striviaqa", "oab_ok"), ("swebq", "oab_ok"),
         ("sllama", "oab_ok"), ("sdqa", "heard_ok"),
         ("sreason", "heard_ok")]
LAYERS = [22, 33, 34, 35]
BUDGET = 0.30
NDRAW = 2000
RNG = np.random.default_rng(42)


def load(pool, ecol):
    ids, arrs = [], {f"X{L}": [] for L in LAYERS}
    for p in sorted(D.glob(f"frozen_native_{pool}agg0_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        for k in arrs:
            arrs[k].append(z[k])
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    sel = df["row"].to_numpy()
    A = {k: np.concatenate(v)[sel] for k, v in arrs.items()}
    lab = pd.read_parquet(D / f"frozen_native_{pool}agg0_judged.parquet")
    lab = (lab.dropna(subset=["adequate"])
           .drop_duplicates("id", keep="last").set_index("id"))
    y = lab["escalate_label"].reindex(df["id"]).to_numpy(float)
    lo = lab["adequate"].reindex(df["id"]).to_numpy(float)
    cl = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
    a = (cl[cl.tier == "always"].dropna(subset=[ecol])
         .drop_duplicates("id", keep="last").set_index("id"))
    eo = a[ecol].reindex(df["id"]).to_numpy(float)
    keep = ~np.isnan(y) & ~np.isnan(eo)
    return ({k: v[keep] for k, v in A.items()}, y[keep].astype(int),
            lo[keep].astype(int), eo[keep].astype(int))


def lopo_scores(data, key, slc):
    sc = {}
    for held, _ in POOLS:
        tr = [p for p, _ in POOLS if p != held]
        Xt = np.concatenate([data[t][0][key][:, slc] for t in tr])
        yt = np.concatenate([data[t][1] for t in tr])
        m = LogisticRegression(C=3e-4, max_iter=5000).fit(Xt, yt)
        sc[held] = m.decision_function(data[held][0][key][:, slc])
    return sc


def evaluate(data, sc):
    out = {"auc": {}, "acc30": {}, "rand30": {}, "delta_pts": {}}
    esc_all, lo_all, eo_all, ks = [], [], [], []
    for p, _ in POOLS:
        _, y, lo, eo = data[p]
        s = sc[p]
        out["auc"][p] = round(roc_auc_score(y, s), 4)
        n = len(y)
        k = int(round(BUDGET * n))
        esc = np.zeros(n, bool)
        esc[np.argsort(-s)[:k]] = True
        acc = float(np.where(esc, eo, lo).mean())
        out["acc30"][p] = round(acc, 4)
        esc_all.append(esc)
        lo_all.append(lo)
        eo_all.append(eo)
        ks.append(k)
    la, ea, es = map(np.concatenate, (lo_all, eo_all, esc_all))
    acc_pool = float(np.where(es, ea, la).mean())
    # matched random: joint draws, same per-pool k
    rnd = np.zeros(NDRAW)
    for _ in range(1):
        pass
    draws = []
    for (p, _), k in zip(POOLS, ks):
        n = len(data[p][1])
        m = np.zeros((NDRAW, n), bool)
        for d in range(NDRAW):
            m[d, RNG.choice(n, k, replace=False)] = True
        draws.append(m)
    M = np.concatenate(draws, axis=1)
    rnd = np.where(M, ea, la).mean(axis=1)
    out["auc"]["ext5_mean"] = round(float(np.mean(
        [out["auc"][p] for p, _ in POOLS])), 4)
    out["acc30"]["pooled"] = round(acc_pool, 4)
    out["rand30"]["pooled"] = round(float(rnd.mean()), 4)
    out["delta_pts"]["pooled"] = round(100 * (acc_pool - rnd.mean()), 2)
    out["perm_p"] = round(float((rnd >= acc_pool).mean()), 4)
    # per-pool random expectation (analytic) for the dots
    for p, _ in POOLS:
        _, y, lo, eo = data[p]
        r = float(((1 - BUDGET) * lo + BUDGET * eo).mean())
        out["rand30"][p] = round(r, 4)
        out["delta_pts"][p] = round(100 * (out["acc30"][p] - r), 2)
    return out


def main():
    data = {}
    for pool, ecol in POOLS:
        data[pool] = load(pool, ecol)
        _, y, lo, eo = data[pool]
        print(f"{pool:10s} n={len(y):4d} fail={y.mean():.3f} "
              f"local={lo.mean():.3f} expert={eo.mean():.3f}")

    out = {"protocol": {
        "batch": "agg0 (8cu beta=0 startline, official config)",
        "head": "LR C=3e-4, LOPO ext-5", "budget": BUDGET,
        "labels": "own-generation gpt-5.4-mini escalate_label",
        "expert": "cached always-arm (oab_ok / heard_ok)"}}
    for L in LAYERS:
        for view, slc in (("last", slice(0, 4096)),
                          ("agg", slice(0, 12288))):
            sc = lopo_scores(data, f"X{L}", slc)
            cell = evaluate(data, sc)
            out[f"L{L}_{view}"] = cell
            print(f"L{L:2d} {view:4s}  AUC {cell['auc']['ext5_mean']:.4f}"
                  f"  acc@30 {cell['acc30']['pooled']:.4f}"
                  f"  rand {cell['rand30']['pooled']:.4f}"
                  f"  delta {cell['delta_pts']['pooled']:+.2f}pt"
                  f"  p={cell['perm_p']:.4f}")

    # deployed-gate reference on the same rows (no refit)
    art = json.loads((D / "gate_native.json").read_text())
    w, b = np.array(art["w"], dtype=np.float32), art["b"]
    sc = {p: data[p][0]["X22"] @ w + b for p, _ in POOLS}
    cell = evaluate(data, sc)
    out["deployed_gate_L22_agg"] = cell
    print(f"deployed    AUC {cell['auc']['ext5_mean']:.4f}"
          f"  acc@30 {cell['acc30']['pooled']:.4f}"
          f"  delta {cell['delta_pts']['pooled']:+.2f}pt")

    Path("figures/native_feature_ablation.json").write_text(
        json.dumps(out, indent=1))
    print("wrote figures/native_feature_ablation.json")


if __name__ == "__main__":
    main()
