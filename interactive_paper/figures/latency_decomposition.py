"""Decompose the paper's "reconstructed completion-time diagnostic"
(Table tab:latency / Figure fig:dualview, internal native benchmark) into
its recorded components, per arm.

The paper formula (paper/build_revision_figures.py::load_rows) for an
escalated row is
    onset + stall_ms + wait_chunks + relay_synth_ms + relay_audio_s
and for a local row
    onset + answer_ms.

`relay_audio_s` is the DURATION of the synthesized relay waveform (how long
the answer takes to play), `relay_synth_ms` is the wall time of the blocking,
non-streaming teacher-forced TTS call, and `wait_chunks` is the expert wait
already counted in 1-s chunks (corr with expert_latency_s ~0.96..0.99).
The local path stops at text-generation end and includes neither synthesis
nor playback, so the two paths are not comparable.

This script prints, per arm: the paper number, its component shares, and
three alternative endpoints
    text_ready   = onset + stall + wait_chunks            (expert text back)
    first_audio  = text_ready + relay_synth               (TTS done, playback can start)
    paper        = first_audio + relay_audio_s            (playback finished)

Run from the repo root:
    python interactive_paper/figures/latency_decomposition.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "native_bench"
ARMS = ["never", "conservative", "balanced", "aggressive", "always"]


def load(arm):
    name = f"frozen_{arm}{'' if arm == 'never' else '_tts'}_judged.parquet"
    df = pd.read_parquet(DATA / name).drop_duplicates("id", keep="last")
    df = df[df["adequate"].notna()].set_index("id").copy()
    z = lambda c: (df[c].fillna(0).astype(float) if c in df
                   else pd.Series(0.0, index=df.index))
    esc = df["mode"].eq("escalated")
    onset = (df["onset_chunk"].fillna(df["n_chunks"]) + 1
             - df["n_chunks"]).clip(lower=0).astype(float)
    comp = pd.DataFrame({
        "onset": onset,
        "stall": z("stall_ms") / 1000,
        "wait_chunks": z("wait_chunks"),
        "expert_api": z("expert_latency_s"),
        "relay_synth": z("relay_synth_ms") / 1000,
        "relay_audio_dur": z("relay_audio_s"),
        "answer_local": z("answer_ms") / 1000,
    })
    text_ready = comp.onset + comp.stall + comp.wait_chunks
    first_audio = text_ready + comp.relay_synth
    paper_esc = first_audio + comp.relay_audio_dur
    local = comp.onset + comp.answer_local
    out = pd.DataFrame({
        "esc": esc,
        "paper": np.where(esc, paper_esc, local),
        "first_audio": np.where(esc, first_audio, local),
        "text_ready": np.where(esc, text_ready, local),
    }, index=df.index)
    return comp, out


def q(s):
    s = pd.Series(s).astype(float)
    return {"mean": round(s.mean(), 1), "p50": round(s.quantile(.5), 1),
            "p95": round(s.quantile(.95), 1), "p99": round(s.quantile(.99), 1)}


def main():
    res = {}
    for arm in ARMS:
        comp, out = load(arm)
        esc = out["esc"]
        r = {"n": int(len(out)), "rate": round(float(esc.mean()), 3),
             "paper": q(out["paper"]), "first_audio": q(out["first_audio"]),
             "text_ready": q(out["text_ready"])}
        print(f"\n== {arm:12s} n={r['n']} esc={r['rate']:.1%}")
        for k in ("paper", "first_audio", "text_ready"):
            v = r[k]
            print(f"   {k:12s} mean={v['mean']:6.1f} P50={v['p50']:6.1f} "
                  f"P95={v['p95']:6.1f} P99={v['p99']:6.1f}")
        if esc.any():
            c = comp[esc]
            tot = float(out.loc[esc, "paper"].sum())
            shares = {k: round(float(c[k].sum()) / tot, 3) for k in
                      ("onset", "stall", "wait_chunks", "relay_synth",
                       "relay_audio_dur")}
            means = {k: round(float(c[k].mean()), 2) for k in c.columns
                     if k != "answer_local"}
            r["escalated_component_means_s"] = means
            r["escalated_component_shares_of_paper_total"] = shares
            r["corr_wait_chunks_vs_expert_api"] = round(float(
                np.corrcoef(c.wait_chunks, c.expert_api)[0, 1]), 3)
            print("   escalated-row component means (s):",
                  ", ".join(f"{k}={v}" for k, v in means.items()))
            print("   share of paper total:",
                  ", ".join(f"{k}={v:.0%}" for k, v in shares.items()))
        res[arm] = r
    (HERE / "latency_decomposition.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {HERE / 'latency_decomposition.json'}")


if __name__ == "__main__":
    main()
