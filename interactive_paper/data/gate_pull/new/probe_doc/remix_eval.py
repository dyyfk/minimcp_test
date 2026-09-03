"""Gate-benefit ("re-mix") eval for NVDA probes on the four external pools —
the operating-point numbers a reviewer grades, not AUC.

For a probe's external scores: send the top-r fraction (r = 15/30/50 %) to the
expert (gpt-5.5 measured outcomes, nvda_expert_outcomes.parquet), keep NVDA's
own answer for the rest; compare with matched-rate random escalation
(permutation test, same outcomes).  Two local-outcome label sets:
  official  oab_ok (OpenAudioBench judge) for striviaqa/swebq/sllama, `adequate` for sdqa
            — scripts/18 protocol, judged on the pass-0 answers
  ours      1 - escalate_label (gpt-5.4-mini, pass-2 answers) — matches the hidden states scored
Also reports fire rates at the calibration (global) thresholds; by
construction the fixed-budget rows equal what per-pool quantile thresholds
would deliver.

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python remix_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import _frames, lr  # noqa: E402
from experiments3 import LayerAvg, ShrinkLDA  # noqa: E402
from probe_lab import Calib, Ext, EXT_POOLS, GP, stack  # noqa: E402

HERE = Path(__file__).resolve().parent
RATES = (0.15, 0.30, 0.50)
NPERM = 5000
BLOCK4 = ["first", "last", "mean8", "umean"]


def probes(cal, ext):
    """name -> (calib OOF-free full fit, dict pool -> scores, thresholds from calib OOF quantiles)"""
    from sklearn.model_selection import StratifiedKFold
    out = {}
    specs = {
        "deployed LR L30 3-block": (lambda: lr(1e-4), lambda E, M, j: stack(E, M, j), (30,)),
        "v2 layer-avg LR x3 4-block": (lambda: LayerAvg(lambda: lr(1e-4), 3), lambda E, M, j: _frames(E, M, j, BLOCK4), (26, 30, 34)),
        "layer-avg LDA30 x3 4-block": (lambda: LayerAvg(lambda: ShrinkLDA(30), 3), lambda E, M, j: _frames(E, M, j, BLOCK4), (26, 30, 34)),
    }
    for name, (mk, feat, Ls) in specs.items():
        X = np.concatenate([feat(cal.E_on, cal.M, cal.j(L)) for L in Ls], 1); y = cal.y
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        thr = {r: float(np.quantile(oof, 1 - r)) for r in RATES}
        m = mk().fit(X, y)
        sc = {t: m.predict_proba(np.concatenate([feat(d["E_on"], d["M"], ext.j(L)) for L in Ls], 1))[:, 1] for t, d in ext.pools.items()}
        out[name] = (sc, thr)
        print(f"{name}: calib OOF thresholds {[round(v, 3) for v in thr.values()]}", flush=True)
    return out


def outcomes(ext):
    EXP = pd.read_parquet(GP / "nvda_expert_outcomes.parquet").set_index("id")["expert_ok"].astype(float)
    tab = {}
    for pool in EXT_POOLS:
        nv = pd.read_parquet(GP / f"nvda_{pool}.parquet").set_index("id")
        lab2 = pd.read_parquet(GP / "onset" / f"nvda_{pool}_ext2.parquet").set_index("id")["escalate_label"]
        ids = ext.pools[pool]["ids"]
        off_col = "oab_ok" if ("oab_ok" in nv.columns and pool != "sdqa") else "adequate"
        tab[pool] = pd.DataFrame({
            "id": ids,
            "loc_official": [float(nv[off_col].get(i, np.nan)) for i in ids],
            "loc_ours": [1.0 - float(lab2.get(i, np.nan)) for i in ids],
            "exp": [float(EXP.get(i, np.nan)) for i in ids]})
    return tab


def remix(d, score, loc_col, rng):
    d = d.assign(score=score).dropna(subset=[loc_col, "exp"]).sort_values("score", ascending=False).reset_index(drop=True)
    n = len(d); loc = d[loc_col].to_numpy(); ex = d["exp"].to_numpy()
    rows = {"n": n, "local": float(loc.mean()), "expert": float(ex.mean())}
    for r in RATES:
        k = int(round(r * n))
        gate = float(np.concatenate([ex[:k], loc[k:]]).mean())
        rnd_mean = (1 - r) * loc.mean() + r * ex.mean()
        # permutation: random k-subsets escalated
        perm = np.empty(NPERM)
        for b in range(NPERM):
            idx = rng.permutation(n)[:k]; m = np.zeros(n, bool); m[idx] = True
            perm[b] = np.where(m, ex, loc).mean()
        rows[f"gate@{r}"] = gate; rows[f"random@{r}"] = float(rnd_mean); rows[f"p@{r}"] = float((perm >= gate).mean())
    return rows


if __name__ == "__main__":
    cal, ext = Calib(), Ext(); rng = np.random.default_rng(8)
    P = probes(cal, ext); T = outcomes(ext)
    res = {}
    for name, (sc, thr) in P.items():
        res[name] = {}
        print(f"\n=== {name} ===")
        for lab in ("loc_official", "loc_ours"):
            res[name][lab] = {}
            print(f"  -- local labels: {lab}")
            for pool in EXT_POOLS:
                r = remix(T[pool], sc[pool], lab, rng); res[name][lab][pool] = r
                fire = {k: float((sc[pool] >= v).mean()) for k, v in thr.items()}
                r["fire_at_calib_thr"] = fire
                print(f"  {pool:10s} n={r['n']:3d} local {r['local']:.3f} expert {r['expert']:.3f} | "
                      + " ".join(f"@{int(x*100)}%: gate {r[f'gate@{x}']:.3f} rnd {r[f'random@{x}']:.3f} p={r[f'p@{x}']:.4f}" for x in RATES)
                      + f" | fire@calib-thr {fire[0.15]:.2f}/{fire[0.3]:.2f}/{fire[0.5]:.2f}")
            # 4-pool mean of gate - random
            for x in RATES:
                g = np.mean([res[name][lab][p][f"gate@{x}"] for p in EXT_POOLS]); rn = np.mean([res[name][lab][p][f"random@{x}"] for p in EXT_POOLS])
                print(f"    4-pool mean @{int(x*100)}%: gate {g:.3f} random {rn:.3f} margin {g - rn:+.3f}")
    json.dump(res, open(HERE / "remix_eval.json", "w"), indent=1)
    print("wrote remix_eval.json")
