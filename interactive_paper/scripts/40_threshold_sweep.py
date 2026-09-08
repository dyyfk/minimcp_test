"""Tier re-selection sweep (8bz, Jisen 09-04): remix acc + latency at
candidate nominal rates so the aggressive tier stops paying always-arm
latency.

Part A reconstructs the DEPLOYED calibration OOF scores (script 31,
--source native, C=3e-4, seed-42 folds) and takes label-free
per-language quantile thresholds at nominal rates 5..50%. Part B
branches each pool's never/always live arms on the never-arm onset
score (native_bench_figures branched rule: fire = score >= thr and
is_info) and reports realized rate, delivered accuracy, and
end-to-end latency (mean/P50/P95) per candidate tier. Latency per
query uses the native_bench formula: local = onset gap + answer_ms;
escalated = onset gap + stall + wait_chunks + synth + spoken relay
audio. Offline remix -- no live cost; chosen tiers get live
confirmation arms afterwards.

Usage (from interactive_paper/):
  .venv_boot\\Scripts\\python.exe scripts\\40_threshold_sweep.py
Outputs: data/threshold_sweep.json + printed tables.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
NB = D / "native_bench"
RATES = [.05, .10, .15, .20, .25, .30, .40, .50]
POOLS = {
    "frozen": ("adequate", "en"), "striviaqa": ("oab_ok", "en"),
    "swebq": ("oab_ok", "en"), "sllama": ("oab_ok", "en"),
    "sdqa": ("adequate", "en"), "sreason": ("adequate", "zh"),
}
# script-31 core parts (native labels), fresh rows excluded from the
# quantile base
PARTS = ["caliboff", "expoff", "exp2off", "exp3off", "exp3zhoff",
         "exp4zhoff", "exp5rsoff", "calibctx", "exp3zhctx"]
ZH_TAGS = {"exp3zhoff", "exp4zhoff", "exp3zhctx"}
POOL_FILES = {"exp4zhoff": "queries_expansion4zh.jsonl",
              "exp5rsoff": "queries_expansion5rs.jsonl"}


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z["X"])
    if not X:
        raise FileNotFoundError(tag)
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def row_lang(tag, ids):
    if tag in POOL_FILES:
        pool = {q["id"]: q["pool"] for q in (json.loads(l) for l in open(
            D / POOL_FILES[tag], encoding="utf-8") if l.strip())}
        return ["zh" if pool.get(i, "").startswith("zh-") else "en"
                for i in ids]
    return ["zh" if tag in ZH_TAGS else "en"] * len(ids)


def deployed_oof_thresholds():
    ids, Xs, ys, lang = [], [], [], []
    for tag in PARTS:
        try:
            i, X = load_feats(tag)
            y = pd.read_parquet(
                D / f"frozen_native_{tag}_judged.parquet").set_index(
                "id")["escalate_label"].reindex(i).to_numpy(float)
        except FileNotFoundError as e:
            print(f"{tag}: skipped ({e})")
            continue
        k = ~np.isnan(y)
        kept = [q for q, kk in zip(i, k) if kk]
        ids += kept
        Xs.append(X[k])
        ys.append(y[k].astype(int))
        lang += row_lang(tag, kept)
    n0 = len(ids)

    fl = pd.read_parquet(D / "fresh_labels.parquet")
    fl = fl[fl["escalate_label"].notna()]
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    nf = pd.read_parquet(
        D / "frozen_native_freshoff_judged.parquet").set_index(
        "id")["escalate_label"]
    for i, p in zip(fl["id"], fl["pool"]):
        if p != "fresh_fast" and i in nf.index and pd.notna(nf[i]):
            lab_f[i] = int(nf[i])
    split_f = dict(zip(fl["id"], fl["split"]))
    ids_fr, X_fr = load_feats("freshoff")
    tr_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) == "train"]
    X_tr = np.concatenate(Xs + [X_fr[tr_j]])
    y_tr = np.concatenate(ys + [[lab_f[ids_fr[j]] for j in tr_j]])
    lang = np.array(lang + ["en"] * len(tr_j))
    print(f"train {len(y_tr)} (core {n0} + fresh {len(tr_j)})")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(LogisticRegression(C=3e-4, max_iter=5000),
                            X_tr, y_tr, cv=cv,
                            method="predict_proba")[:, 1]
    core = np.arange(len(y_tr)) < n0
    thr = {}
    for lg in ("en", "zh"):
        m = core & (lang == lg)
        thr[lg] = {r: float(np.quantile(oof[m], 1 - r)) for r in RATES}
    # sanity: reproduce the deployed per-language thresholds at 15/30/50
    dep = json.loads((D / "gate_native.json").read_text()).get(
        "eot_thresholds_lang", {})
    for lg in dep:
        print(f"reproduce deployed {lg}: "
              + "  ".join(
                  f"{t}: {thr[lg][r]:.4f} vs {dep[lg][t]:.4f}"
                  for t, r in (("conservative", .15), ("balanced", .30),
                               ("aggressive", .50))))
    return thr


def load_arm(pool, tier):
    col, _ = POOLS[pool]
    name = tier if tier == "never" else f"{tier}_tts"
    p = NB / f"{pool}_{name}_judged.parquet"
    if not p.exists():
        p = NB / f"{pool}_{tier}_judged.parquet"
    df = pd.read_parquet(p).drop_duplicates("id", keep="last")
    df = df[df[col].notna()].copy()
    df["y"] = df[col].astype(float)
    on = (df["onset_chunk"].fillna(df["n_chunks"]) + 1
          - df["n_chunks"]).clip(lower=0)
    esc = df["mode"] == "escalated"
    if "relay_synth_ms" in df.columns:
        relay_s = (df["relay_synth_ms"].fillna(0) / 1000
                   + df["relay_audio_s"].fillna(0))
    else:
        relay_s = df["relay_ms"].fillna(0) / 1000
    df["t_s"] = np.where(
        esc,
        on + df["stall_ms"].fillna(0) / 1000
        + df["wait_chunks"].fillna(0) + relay_s,
        on + df["answer_ms"].fillna(0) / 1000)
    return df.set_index("id")


def main():
    thr = deployed_oof_thresholds()
    out = {"nominal_thresholds": {lg: {str(r): t for r, t in d.items()}
                                  for lg, d in thr.items()},
           "pools": {}}
    for pool, (col, lg) in POOLS.items():
        never, always = load_arm(pool, "never"), load_arm(pool, "always")
        ids = never.index.intersection(always.index)
        never, always = never.loc[ids], always.loc[ids]
        rows = []
        for r in RATES:
            t = thr[lg][r]
            fire = ((never["score"] >= t)
                    & never["is_info"].fillna(True).astype(bool))
            y = np.where(fire, always["y"], never["y"])
            ts = np.where(fire, always["t_s"], never["t_s"])
            rows.append({
                "nominal": r, "thr": t,
                "realized": float(fire.mean()),
                "acc": float(y.mean()),
                "lat_mean": float(ts.mean()),
                "lat_p50": float(np.percentile(ts, 50)),
                "lat_p95": float(np.percentile(ts, 95)),
            })
        base = {"acc_never": float(never["y"].mean()),
                "acc_always": float(always["y"].mean()),
                "lat_never_mean": float(never["t_s"].mean()),
                "lat_always_mean": float(always["t_s"].mean()),
                "lat_never_p50": float(np.percentile(never["t_s"], 50)),
                "lat_always_p50": float(np.percentile(always["t_s"], 50)),
                "n": int(len(ids))}
        out["pools"][pool] = {"base": base, "sweep": rows}
        print(f"\n=== {pool} (n={len(ids)})  never {base['acc_never']:.3f}"
              f"@{base['lat_never_mean']:.1f}s  always"
              f" {base['acc_always']:.3f}@{base['lat_always_mean']:.1f}s ===")
        print("nom%  real%   acc    lat_mean  lat_p50  lat_p95")
        for w in rows:
            print(f"{w['nominal']*100:4.0f} {w['realized']*100:6.1f}"
                  f" {w['acc']:6.3f} {w['lat_mean']:8.1f}"
                  f" {w['lat_p50']:8.1f} {w['lat_p95']:8.1f}")
    (D / "threshold_sweep.json").write_text(json.dumps(out, indent=1))
    print("\nwrote data/threshold_sweep.json")


if __name__ == "__main__":
    main()
