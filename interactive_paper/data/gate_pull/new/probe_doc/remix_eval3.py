"""Re-mix (accuracy at fixed escalation budgets vs matched-rate random) for the
pass-3 probes: deployed read, v2 architecture, and the strictly causal probe.
Same protocol as remix_eval.py (official OAB labels / `adequate` for SD-QA as
the local outcome; gpt-5.5 measured outcomes as the expert outcome).

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python remix_eval3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import lr  # noqa: E402
from experiments3 import LayerAvg  # noqa: E402
from experiments4 import Pass3  # noqa: E402
from probe_lab import CALIB_TAGS, EXT_POOLS, GP  # noqa: E402
from remix_eval import RATES, remix  # noqa: E402

HERE = Path(__file__).resolve().parent

SPECS = {
    "deployed read (pass-3): onset_last|onset_mean8|user_mean @L30": (lambda: lr(1e-4), [(30, ["onset_last", "onset_mean8", "user_mean"])]),
    "v2: layer-avg x3, commit|onset_last|onset_mean8|run_mean": (lambda: LayerAvg(lambda: lr(1e-4), 3), [(L, ["commit", "onset_last", "onset_mean8", "run_mean"]) for L in (26, 30, 34)]),
    "strictly causal: commit|pre_mean8|run_mean @L34": (lambda: lr(1e-4), [(34, ["commit", "pre_mean8", "run_mean"])]),
    "strictly causal: commit|pre_mean8|run_mean @L30": (lambda: lr(1e-4), [(30, ["commit", "pre_mean8", "run_mean"])]),
}


def outcomes(ext):
    EXP = pd.read_parquet(GP / "nvda_expert_outcomes.parquet").set_index("id")["expert_ok"].astype(float)
    tab = {}
    for pool in EXT_POOLS:
        nv = pd.read_parquet(GP / f"nvda_{pool}.parquet").set_index("id")
        lab2 = pd.read_parquet(GP / "onset" / f"nvda_{pool}_ext2.parquet").set_index("id")["escalate_label"]
        ids = ext[pool].ids
        off_col = "oab_ok" if ("oab_ok" in nv.columns and pool != "sdqa") else "adequate"
        tab[pool] = pd.DataFrame({"id": ids, "loc_official": [float(nv[off_col].get(i, np.nan)) for i in ids],
                                  "loc_ours": [1.0 - float(lab2.get(i, np.nan)) for i in ids], "exp": [float(EXP.get(i, np.nan)) for i in ids]})
    return tab


if __name__ == "__main__":
    cal = Pass3(CALIB_TAGS, lambda tag: GP / "onset_fit" / f"nvda_{tag}.parquet")
    ext = {tag: Pass3([tag], lambda t: GP / "onset" / f"nvda_{t}_ext2.parquet") for tag in EXT_POOLS}
    T = outcomes(ext); rng = np.random.default_rng(8); res = {}
    for name, (mk, parts) in SPECS.items():
        X = np.concatenate([cal.feats(L, b) for L, b in parts], 1); y = cal.y
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        thr = {r: float(np.quantile(oof, 1 - r)) for r in RATES}
        m = mk().fit(X, y)
        res[name] = {}
        print(f"\n=== {name} ===")
        for lab in ("loc_official", "loc_ours"):
            res[name][lab] = {}
            for pool in EXT_POOLS:
                sc = m.predict_proba(np.concatenate([ext[pool].feats(L, b) for L, b in parts], 1))[:, 1]
                r = remix(T[pool], sc, lab, rng); r["fire_at_calib_thr"] = {str(k): float((sc >= v).mean()) for k, v in thr.items()}
                res[name][lab][pool] = r
                if lab == "loc_official":
                    print(f"  {pool:10s} local {r['local']:.3f} | " + " ".join(f"@{int(x*100)}% gate {r[f'gate@{x}']:.3f} rnd {r[f'random@{x}']:.3f} p={r[f'p@{x}']:.3f}" for x in RATES))
            for x in RATES:
                g = np.mean([res[name][lab][p][f"gate@{x}"] for p in EXT_POOLS]); rn = np.mean([res[name][lab][p][f"random@{x}"] for p in EXT_POOLS])
                if lab == "loc_official":
                    print(f"    4-pool mean @{int(x*100)}%: gate {g:.3f} random {rn:.3f} margin {g - rn:+.3f}")
    json.dump(res, open(HERE / "remix_eval3.json", "w"), indent=1); print("wrote remix_eval3.json")
