"""Native full-duplex benchmark figures (8bu): per pool
  native_{pool}_dualview.png : delivered accuracy vs realized escalation
                               rate, live arms with bootstrap CIs, random
                               reference, official / chat-mode anchors
  native_{pool}_pareto.png   : P50 time-to-answer after the user stops
                               speaking vs accuracy
Data: data/native_bench/{pool}_{tier}_judged.parquet (pull with
scripts/pull_native_bench.sh). Tiers present as live runs are drawn as
live points; if a mid tier has no live run yet, its point is BRANCHED
from the never arm's live onset score (never-arm outcome if the score
is under the tier threshold, always-arm outcome otherwise) and drawn
hollow. Run from interactive_paper/:
  .venv_boot\\Scripts\\python.exe figures\\native_bench_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = Path("data/native_bench")
BLUE, GREEN, GREY, ORANGE = "#2a78d6", "#1baf7a", "#8a97a5", "#eb6834"
INK, GRID = "#0b0b0b", "#e6e4de"
ARMS = ["never", "conservative", "balanced", "aggressive", "always"]
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

SPEC = {
    "frozen": {"title": "our pool (test 240)", "judge": "adequate", "lang": "en"},
    "striviaqa": {"title": "Speech TriviaQA (OpenAudioBench, n=250)", "judge": "oab_ok",
                  "lang": "en", "official": .755, "chatmode": .712,
                  "others": [("Qwen3-Omni-30B", .629), ("Kimi-Audio", .419)]},
    "swebq": {"title": "Speech Web Questions (OpenAudioBench, n=250)", "judge": "oab_ok",
              "lang": "en", "official": .702, "chatmode": .716,
              "others": [("Qwen3-Omni-30B", .749), ("Kimi-Audio", .464)]},
    "sllama": {"title": "Speech Llama Questions (OpenAudioBench, n=250)",
               "judge": "oab_ok", "lang": "en"},
    "sdqa": {"title": "SD-QA — real human speech (VoiceBench, n=200)",
             "judge": "adequate", "lang": "en"},
    "sreason": {"title": "Speech Reasoning QA — Chinese (OpenAudioBench, n=202)",
                "judge": "adequate", "lang": "zh"},
    "valpaca": {"title": "VoiceBench AlpacaEval (n=199) — judge score 1-5",
                "judge": "vb_score", "lang": "en"},
}
gate = json.load(open("data/gate_native.json"))


RELAY = "tts"     # deployed relay since 2026-09-02 (8bu); "steer" = the old talker-steering path


def load(pool, tier, relay=None):
    relay = RELAY if relay is None else relay
    name = tier if (tier == "never" or relay == "steer") else f"{tier}_{relay}"
    p = D / f"{pool}_{name}_judged.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).drop_duplicates("id", keep="last")
    col = SPEC[pool]["judge"]
    df = df[df[col].notna()].copy()
    df["y"] = df[col].astype(float)
    # time-to-answer after the user stops speaking (s): onset may precede
    # the end of the audio in native duplex, so this can be small or 0
    on = (df["onset_chunk"].fillna(df["n_chunks"]) + 1 - df["n_chunks"]).clip(lower=0)
    esc = df["mode"] == "escalated"
    # relay cost: steering relay = talker generation wall; TTS relay =
    # synth wall + the spoken audio itself
    if "relay_synth_ms" in df.columns:
        relay_s = (df["relay_synth_ms"].fillna(0) / 1000
                   + df["relay_audio_s"].fillna(0))
    else:
        relay_s = df["relay_ms"].fillna(0) / 1000
    df["t_s"] = np.where(
        esc, on + df["stall_ms"].fillna(0) / 1000 + df["wait_chunks"].fillna(0) + relay_s,
        on + df["answer_ms"].fillna(0) / 1000)
    return df.set_index("id")


def boot_ci(v, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


summary = {}
for pool, spec in SPEC.items():
    never, always = load(pool, "never"), load(pool, "always")
    if never is None or always is None:
        print(f"{pool}: skipped (need never + always)")
        continue
    ids = never.index.intersection(always.index)
    never, always = never.loc[ids], always.loc[ids]
    thr = gate.get("eot_thresholds_lang", {}).get(spec["lang"], gate["eot_thresholds"])
    pts = {}
    for arm in ARMS:
        live = load(pool, arm) if arm in RATES else None
        if arm == "never":
            y, t, r, kind = never["y"], never["t_s"], np.zeros(len(ids)), "live"
        elif arm == "always":
            y, t, r, kind = always["y"], always["t_s"], np.ones(len(ids)), "live"
        elif live is not None and len(live.index.intersection(ids)) >= .9 * len(ids):
            live = live.reindex(ids).dropna(subset=["y"])
            y, t, r, kind = live["y"], live["t_s"], (live["mode"] == "escalated").astype(float), "live"
        else:
            fire = (never["score"] >= thr[arm]) & (never["is_info"].fillna(True).astype(bool))
            y = np.where(fire, always["y"], never["y"])
            t = np.where(fire, always["t_s"], never["t_s"])
            r, kind = fire.astype(float).to_numpy(), "branched"
        y = np.asarray(y, float); t = np.asarray(t, float)
        lo, hi = boot_ci(y)
        pts[arm] = {"rate": float(np.mean(r)), "acc": float(y.mean()), "ci": [lo, hi],
                    "t50": float(np.median(t)), "kind": kind, "n": int(len(y))}
    summary[pool] = pts
    floor, ceil = pts["never"]["acc"], pts["always"]["acc"]
    xs = [pts[a]["rate"] for a in ARMS]; ys = [pts[a]["acc"] for a in ARMS]
    err = np.array([[pts[a]["acc"] - pts[a]["ci"][0], pts[a]["ci"][1] - pts[a]["acc"]] for a in ARMS]).T
    ylab = "judge score (1-5)" if spec["judge"] == "vb_score" else "delivered-channel answer accuracy"

    # ---- dualview ----
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.grid(color=GRID, lw=.7, zorder=0)
    ax.plot([0, 1], [floor, ceil], "--", color=GREY, lw=1.4, label="random escalation", zorder=2)
    ax.errorbar(xs, ys, yerr=err, fmt="-", color=BLUE, lw=2, capsize=3, zorder=3,
                label="deployed channel (native full duplex, 8bq gate, TTS relay)")
    for a in ARMS:
        p = pts[a]
        ax.plot(p["rate"], p["acc"], "o", ms=8, zorder=4, color=BLUE,
                mfc=(BLUE if p["kind"] == "live" else "white"), mew=2)
        ax.annotate(a + ("" if p["kind"] == "live" else " (branched)"), (p["rate"], p["acc"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=9, color=BLUE)
    st = {}
    for arm in ARMS:
        d = load(pool, arm, relay="steer") if arm != "never" else never
        if d is None or len(d.index.intersection(ids)) < .9 * len(ids):
            continue
        d = d.reindex(ids).dropna(subset=["y"])
        st[arm] = (float((d["mode"] == "escalated").mean()), float(d["y"].mean()))
    if "always" in st and len(st) >= 2:
        arms_st = [a for a in ARMS if a in st]
        ax.plot([st[a][0] for a in arms_st], [st[a][1] for a in arms_st], "s--", color="#b5651d",
                lw=1.3, ms=5, alpha=.8, zorder=2,
                label="same gate, old talker-steering relay (retired 2026-09-02)")
        pts["_steer"] = {a: {"rate": st[a][0], "acc": st[a][1]} for a in arms_st}
    if spec.get("official"):
        ax.axhline(spec["official"], color="#555", ls=":", lw=1.4)
        ax.text(.99, spec["official"] + .006, f"MiniCPM-o 4.5 official {spec['official']*100:.1f} (offline chat mode)",
                ha="right", fontsize=8.5, color="#555")
    if spec.get("chatmode"):
        ax.axhline(spec["chatmode"], color=BLUE, ls="--", lw=1.1, alpha=.6)
        ax.text(.99, spec["chatmode"] - .02, f"same model, offline chat mode (ours) {spec['chatmode']:.3f}",
                ha="right", fontsize=8.5, color=BLUE)
    for name, v in spec.get("others", []):
        ax.axhline(v, color="#777", ls="-.", lw=1)
        ax.text(.99, v + .006, f"{name} {v*100:.1f} (official, offline)", ha="right", fontsize=8.5, color="#777")
    ax.set_xlim(-.03, 1.03)
    ax.set_xlabel("realized escalation rate")
    ax.set_ylabel(ylab)
    ex_col = spec["judge"] + "_expert"
    if ex_col in always.columns and always[ex_col].notna().sum() > .9 * len(always):
        ex = float(always[ex_col].astype(float).mean())
        ax.axhline(ex, color=GREEN, ls=":", lw=1.6)
        ax.text(.01, ex + .006, f"expert text as returned {ex:.3f} (relay-free bound)",
                fontsize=8.5, color=GREEN)
        pts["always"]["expert_bound"] = ex
    ax.set_title(f"{spec['title']}\nnative full duplex · official serving config · deployed 8bq gate\n"
                 f"filled = live arm · hollow = branched on the live onset score", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.12))
    fig.tight_layout()
    fig.savefig(f"figures/native_{pool}_dualview.png", dpi=170)
    plt.close(fig)

    # ---- pareto ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.grid(color=GRID, lw=.7, zorder=0)
    ts = [pts[a]["t50"] for a in ARMS]
    ax.plot(ts, ys, "-", color=BLUE, lw=2, zorder=3)
    for a in ARMS:
        p = pts[a]
        ax.plot(p["t50"], p["acc"], "o", ms=8, color=BLUE, mfc=(BLUE if p["kind"] == "live" else "white"), mew=2, zorder=4)
        ax.annotate(a, (p["t50"], p["acc"]), textcoords="offset points", xytext=(6, -12), fontsize=9, color=BLUE)
    ax.set_xlabel("P50 time from end of user speech to answer complete (s)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{spec['title']} — latency vs accuracy (native full duplex)", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"figures/native_{pool}_pareto.png", dpi=170)
    plt.close(fig)
    print(f"{pool}: " + "  ".join(f"{a}={pts[a]['acc']:.3f}@{pts[a]['rate']:.2f}"
                                  f"({pts[a]['kind'][0]},{pts[a]['t50']:.1f}s)" for a in ARMS))

Path("figures/native_bench_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote figures/native_bench_summary.json")
