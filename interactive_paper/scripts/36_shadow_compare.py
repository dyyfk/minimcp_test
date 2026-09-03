"""Issue #8 follow-up: live 8bq gate vs ChangyiYang's two shadow candidates
(P9 distilled 8192-d student, P16 alpha-1 one-pass ensemble) on the six
official-native pools, same remix arithmetic as scripts/23.

Per pool: native-failure AUC, expert-benefit AUC (expert_ok > local_ok),
cascade accuracy at exact per-pool 15/30/50% budgets (all scorers), plus
the live gate at its deployed per-language tier thresholds. Paired
bootstrap (2000) CI on AUC deltas vs live. $0, CPU.

Usage: .venv_boot\Scripts\python.exe scripts\36_shadow_compare.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path("data")
RNG = np.random.default_rng(42)
POOLS = [("frozen", "testoff", None), ("striviaqa", "striviaqaoff", "oab_ok"),
         ("swebq", "swebqoff", "oab_ok"), ("sllama", "sllamaoff", "oab_ok"),
         ("sdqa", "sdqaoff", "heard_ok"), ("sreason", "sreasonoff", "heard_ok")]
BUDGETS = (("conservative", .15), ("balanced", .30), ("aggressive", .50))
BLOCK = {"eot_last": 0, "eot_mean8": 1, "eot_mean": 1, "user_mean": 2}


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"]); X.append(z["X"])
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def scorer(art):
    w = np.asarray(art["w"], dtype=np.float64); b = float(art["b"])
    blocks = art.get("feature_recipe", {}).get("blocks") or art.get("modes")
    if len(w) == 12288:
        return lambda X: X @ w + b
    idx = np.concatenate([np.arange(4096) + 4096 * BLOCK[n] for n in blocks])
    assert len(idx) == len(w), (blocks, len(w))
    return lambda X: X[:, idx] @ w + b


def sig(z):
    return 1 / (1 + np.exp(-z))


live = json.load(open(D / "gate_native.json"))
cands = {"live_8bq": live,
         "P9_distilled": json.load(open(D / "shadow/gate_shadow_distilled_semantic_rtj.json")),
         "P16_alpha1": json.load(open(D / "shadow/gate_shadow_robust_ensemble.json"))}
S = {k: scorer(v) for k, v in cands.items()}
thr_lang = live.get("eot_thresholds_lang", {})

out = {}
for pool, tag, ecol in POOLS:
    ids, X = load_feats(tag)
    j = pd.read_parquet(D / f"frozen_native_{tag}_judged.parquet")
    j = j.dropna(subset=["adequate"]).drop_duplicates("id", keep="last")
    local_ok = dict(zip(j["id"], j["adequate"].astype(int)))
    if pool == "frozen":
        f = pd.read_parquet(D / "frozen_v3_traces.parquet")
        e = f[f["mode"] == "escalated"].groupby("id")["heard_ok"].max()
        exp_ok = e.dropna().astype(int).to_dict()
    else:
        cl = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
        a = cl[cl.tier == "always"].dropna(subset=[ecol]).drop_duplicates("id", keep="last")
        exp_ok = dict(zip(a["id"], a[ecol].astype(int)))
    keep = [k for k, i in enumerate(ids) if i in local_ok and i in exp_ok]
    X = X[keep]; rows = [ids[k] for k in keep]
    lo = np.array([local_ok[i] for i in rows]); eo = np.array([exp_ok[i] for i in rows])
    fail = 1 - lo; ben = (eo > lo).astype(int)
    n = len(rows)
    lang = "zh" if pool == "sreason" else "en"
    thr = thr_lang.get(lang, live["eot_thresholds"])
    sc = {k: S[k](X) for k in S}
    res = {"n": n, "local_floor": round(lo.mean(), 3), "expert_ceiling": round(eo.mean(), 3),
           "scorers": {}}
    for k, s in sc.items():
        r = {"native_auc": round(roc_auc_score(fail, s), 4),
             "benefit_auc": round(roc_auc_score(ben, s), 4), "cascade_exact": {}}
        for tier, rate in BUDGETS:
            kk = int(round(rate * n)); order = np.argsort(-s)
            esc = np.zeros(n, bool); esc[order[:kk]] = True
            r["cascade_exact"][tier] = round(float(np.where(esc, eo, lo).mean()), 4)
        if k == "live_8bq":
            r["cascade_deployed"] = {}
            for tier, _ in BUDGETS:
                esc = sig(s) >= thr[tier]
                r["cascade_deployed"][tier] = {"esc_rate": round(esc.mean(), 3),
                                               "acc": round(float(np.where(esc, eo, lo).mean()), 4)}
        if k != "live_8bq":
            d_n, d_b = [], []
            for _ in range(2000):
                bi = RNG.integers(0, n, n)
                if fail[bi].min() == fail[bi].max() or ben[bi].min() == ben[bi].max():
                    continue
                d_n.append(roc_auc_score(fail[bi], s[bi]) - roc_auc_score(fail[bi], sc["live_8bq"][bi]))
                d_b.append(roc_auc_score(ben[bi], s[bi]) - roc_auc_score(ben[bi], sc["live_8bq"][bi]))
            r["delta_native_auc"] = round(r["native_auc"] - res["scorers"]["live_8bq"]["native_auc"], 4)
            r["delta_native_ci"] = [round(float(np.percentile(d_n, 2.5)), 4), round(float(np.percentile(d_n, 97.5)), 4)]
            r["delta_benefit_auc"] = round(r["benefit_auc"] - res["scorers"]["live_8bq"]["benefit_auc"], 4)
            r["delta_benefit_ci"] = [round(float(np.percentile(d_b, 2.5)), 4), round(float(np.percentile(d_b, 97.5)), 4)]
        res["scorers"][k] = r
    out[pool] = res
    print(f"\n== {pool} n={n} local {lo.mean():.3f} expert {eo.mean():.3f}")
    for k, r in res["scorers"].items():
        c = r["cascade_exact"]
        extra = (f"  dNat={r['delta_native_auc']:+.4f} {r['delta_native_ci']}  dBen={r['delta_benefit_auc']:+.4f}"
                 if k != "live_8bq" else "")
        print(f"  {k:13s} nAUC={r['native_auc']:.4f} bAUC={r['benefit_auc']:.4f} "
              f"casc15/30/50={c['conservative']:.3f}/{c['balanced']:.3f}/{c['aggressive']:.3f}{extra}")

ext = [p for p, _, _ in POOLS if p != "frozen"]
summ = {}
for k in S:
    summ[k] = {m: round(float(np.mean([out[p]["scorers"][k][m] for p in ext])), 4)
               for m in ("native_auc", "benefit_auc")}
    summ[k]["cascade_exact"] = {t: round(float(np.mean([out[p]["scorers"][k]["cascade_exact"][t] for p in ext])), 4)
                                for t, _ in BUDGETS}
out["_external_mean"] = summ
print("\n== external-5 mean")
for k, v in summ.items():
    print(f"  {k:13s} nAUC={v['native_auc']:.4f} bAUC={v['benefit_auc']:.4f} casc={v['cascade_exact']}")
Path("figures/shadow_compare.json").write_text(json.dumps(out, indent=1))
print("wrote figures/shadow_compare.json")
