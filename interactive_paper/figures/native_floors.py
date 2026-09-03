"""Native never-arm local floors vs the retired harness floors, all pools."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BLUE, GREY = "#2a78d6", "#8a97a5"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
pools = [("frozen", "our pool", "adequate", "frozen_v3_traces.parquet", "heard_ok"),
         ("striviaqa", "TriviaQA", "oab_ok", "striviaqa_v3_traces.parquet", "oab_ok"),
         ("swebq", "WebQ", "oab_ok", "swebq_v3_traces.parquet", "oab_ok"),
         ("sllama", "LlamaQ", "oab_ok", "sllama_v3_traces.parquet", "oab_ok"),
         ("sdqa", "SD-QA", "adequate", "sdqa_v3_traces.parquet", "heard_ok"),
         ("sreason", "Reasoning-zh", "adequate", "sreason_v3_traces.parquet", "heard_ok")]
nat, old, names, cis = [], [], [], []
rng = np.random.default_rng(42)
for p, nice, c, of, oc in pools:
    df = pd.read_parquet(f"data/native_bench/{p}_never_judged.parquet").drop_duplicates("id", keep="last")
    y = df[c].astype(float).to_numpy(); nat.append(y.mean())
    m = y[rng.integers(0, len(y), (5000, len(y)))].mean(1); cis.append((np.percentile(m, 2.5), np.percentile(m, 97.5)))
    t = pd.read_parquet("data/" + of); t = t[t.tier == "never"]; oc = oc if oc in t else "heard_ok"
    old.append(t[oc].astype(float).mean()); names.append(nice)
x = np.arange(len(names)); w = .36
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - w / 2, old, w, color=GREY, label="retired harness loop (turn-based read), never arm")
ax.bar(x + w / 2, nat, w, color=BLUE, label="native full duplex (official config), never arm")
ax.errorbar(x + w / 2, nat, yerr=[[n - lo for n, (lo, hi) in zip(nat, cis)], [hi - n for n, (lo, hi) in zip(nat, cis)]],
            fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
for xi, (o, n) in enumerate(zip(old, nat)):
    ax.text(xi - w / 2, o + .01, f"{o:.3f}", ha="center", fontsize=8.5, color="#444")
    ax.text(xi + w / 2, n + .01, f"{n:.3f}", ha="center", fontsize=8.5, color=BLUE)
ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.set_ylabel("local-only answer accuracy")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Local floor per pool: native full-duplex sessions vs the retired harness loop (same judges)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("figures/native_floors.png", dpi=170)
print("wrote figures/native_floors.png")
