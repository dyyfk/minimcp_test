"""Tier selection on the independent validation split (P1 step 4).

Inputs: val1 never/always v2 arms (judged: adequate + adequate_audio),
gate_native_v2.json nominal grid. Branch rule = never-arm onset score
(v2 fitted, out-of-fit) >= per-language threshold AND is_info — the
same branched-remix construction as 8bz, but on data no fit ever saw,
under the v2 protocol, with real produced-audio scores.

Latency endpoints (audio-timeline reconstruction; chunks = seconds of
audio timeline, component walls real):
  pcm_ready_s   local: onset_overrun + answer_gen + answer_synth_wall
                escalated: onset_overrun + stall + expert_wait +
                relay_synth_wall   (first delivered-answer PCM ready)
  completion_s  pcm_ready + delivered audio duration

Outputs: data/paper_revision_pr0907/validation_sweep.parquet,
validation_tier_freeze.json, printed curve.

Usage: set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\53_validation_freeze.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
OUT = D / "paper_revision_pr0907"
VD = D / "native_bench_v2" / "val1"

art = json.loads((D / "gate_native_v2.json").read_text())
grid = art["nominal_grid_lang"]

def load(arm):
    df = pd.read_parquet(VD / f"validation_{arm}_tts_judged.parquet")
    df = df.drop_duplicates("id", keep="last").set_index("id")
    on = (df["onset_chunk"].fillna(df["n_chunks"]) + 1
          - df["n_chunks"]).clip(lower=0)
    if arm == "never":
        df["pcm_ready_s"] = (on + df["answer_ms"].fillna(0) / 1000
                             + df["answer_synth_ms"].fillna(0) / 1000)
        df["completion_s"] = (df["pcm_ready_s"]
                              + df["answer_audio_s"].fillna(0))
    else:
        df["pcm_ready_s"] = (on + df["stall_ms"].fillna(0) / 1000
                             + df["wait_chunks"].fillna(0)
                             + df["relay_synth_ms"].fillna(0) / 1000)
        df["completion_s"] = (df["pcm_ready_s"]
                              + df["relay_audio_s"].fillna(0))
    return df

nv, al = load("never"), load("always")
ids = nv.index.intersection(al.index)
nv, al = nv.loc[ids], al.loc[ids]
print(f"paired n={len(ids)} "
      f"(never status: {nv['status'].value_counts().to_dict()}; "
      f"always: {al['status'].value_counts().to_dict()})")

rows = []
for r_str in grid["en"]:
    r = float(r_str)
    thr = nv["lang"].map(lambda lg: grid[lg][r_str])
    fire = ((nv["score"] >= thr)
            & nv["is_info"].fillna(True).astype(bool))
    rec = {"nominal": r, "realized": float(fire.mean())}
    for lg in ("all", "en", "zh"):
        m = slice(None) if lg == "all" else (nv["lang"] == lg)
        f_m = fire if lg == "all" else fire[m]
        for tag, col in (("text", "adequate"), ("audio", "adequate_audio")):
            a = nv.loc[m, col].astype(float)
            b = al.loc[m, col].astype(float)
            k = a.notna() & b.notna()
            y = np.where(f_m[k], b[k], a[k])
            rec[f"acc_{tag}_{lg}"] = float(np.mean(y))
            if lg == "all":
                rec[f"n_scored_{tag}"] = int(k.sum())
        for tag2, col2 in (("pcm", "pcm_ready_s"), ("comp", "completion_s")):
            t = np.where(f_m, al.loc[m, col2], nv.loc[m, col2])
            rec[f"lat_{tag2}_mean_{lg}"] = float(np.mean(t))
            rec[f"lat_{tag2}_p95_{lg}"] = float(np.percentile(t, 95))
    rows.append(rec)
sw = pd.DataFrame(rows)
sw.to_parquet(OUT / "validation_sweep.parquet", index=False)

base = {}
for tag, col in (("text", "adequate"), ("audio", "adequate_audio")):
    base[f"never_acc_{tag}"] = float(nv[col].astype(float).mean())
    base[f"always_acc_{tag}"] = float(al[col].astype(float).mean())
base["never_lat_pcm"] = float(nv["pcm_ready_s"].mean())
base["always_lat_pcm"] = float(al["pcm_ready_s"].mean())

print(f"\nbase: local text {base['never_acc_text']:.3f} / audio "
      f"{base['never_acc_audio']:.3f} @ {base['never_lat_pcm']:.1f}s | "
      f"always text {base['always_acc_text']:.3f} / audio "
      f"{base['always_acc_audio']:.3f} @ {base['always_lat_pcm']:.1f}s")
print("nom%  real%  acc_txt  acc_aud  pcm_mean  pcm_p95  comp_mean")
for _, w in sw.iterrows():
    print(f"{w.nominal*100:4.0f} {w.realized*100:6.1f} "
          f"{w.acc_text_all:7.3f} {w.acc_audio_all:8.3f} "
          f"{w.lat_pcm_mean_all:8.1f} {w.lat_pcm_p95_all:8.1f} "
          f"{w.lat_comp_mean_all:9.1f}")

(OUT / "validation_tier_freeze.json").write_text(json.dumps({
    "gate": "data/gate_native_v2.json",
    "gate_train_ids_sha1": art["train_ids_sha1"],
    "validation_ids_sha1": art["validation_ids_sha1"],
    "base": base,
    "sweep": rows,
    "note": "tier choice + rationale recorded after reading this "
            "curve; 15/25/40 are prior candidates from the "
            "exploration-only 8bz remix, not presumed optimal",
}, indent=1))
print("\nwrote validation_sweep.parquet + validation_tier_freeze.json")
