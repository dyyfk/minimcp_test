"""8dc: cross-modal transfer figure (fig:xmodal-transfer).

Left: per-layer text->audio failure AUC vs relative depth for all four
models (6a/6b/6d matched-pair data: same 600 frozen-pool queries as text
and as TTS audio; probes trained on TEXT calib rows, scored on AUDIO calib
rows). Right: frozen-pool probes -> SD-QA (7b: 200 real human recordings,
query-disjoint AND cross-modal -- the strictly harder test).

Inputs (data/): audio_xmodal_{minicpm-o45,minicpm-o26,qwen2.5-omni-7b,
freeze-omni}-audio.json (pulled from gate-data volume),
sdqa_layer_transfer.json (parsed from modal_audio.py::sdqa_report rerun).
Outputs: figures/xmodal_transfer.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")

MODELS = [  # (tag, label, color, lw) -- colors follow 40_lopo_layer_matrix.py
    ("minicpm-o45-audio", "MiniCPM-o 4.5 (duplex)", "#1f5fa8", 2.0),
    ("minicpm-o26-audio", "MiniCPM-o 2.6 (duplex)", "#1baf7a", 1.5),
    ("qwen2.5-omni-7b-audio", "Qwen2.5-Omni (streaming)", "#e4762e", 1.5),
    ("freeze-omni-audio", "Freeze-Omni (frozen backbone)", "#8a6fbf", 1.5),
]
DEPLOY_LAYER = 22  # o4.5 gate layer


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))

    # ---- left: matched-query text->audio transfer, 4 models ----
    for tag, label, color, lw in MODELS:
        with open(os.path.join(DATA, f"audio_xmodal_{tag}.json")) as f:
            curves = json.load(f)["curves"]
        n = len(curves)
        xs = [c["layer"] / (n - 1) for c in curves]
        ys = [c["text2audio"] for c in curves]
        ax1.plot(xs, ys, color=color, lw=lw, label=label)
        if tag == "minicpm-o45-audio":
            xd, yd = DEPLOY_LAYER / (n - 1), ys[DEPLOY_LAYER]
            ax1.plot([xd], [yd], "o", color=color, ms=6, zorder=5)
            ax1.annotate(f"L{DEPLOY_LAYER} (deployed): {yd:.3f}",
                         (xd, yd), xytext=(xd + 0.03, yd + 0.06),
                         fontsize=8, color=color, ha="left")
    ax1.axhline(0.5, color="0.55", lw=0.8, ls=":")
    ax1.set_xlabel("relative depth (layer / depth)")
    ax1.set_ylabel(r"text$\rightarrow$audio failure AUC")
    ax1.set_title("Matched queries, TTS audio\n(train on text states, score audio states)",
                  fontsize=9.5)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax1.set_ylim(0.35, 0.95)

    # ---- right: frozen-pool probes -> SD-QA real speech ----
    with open(os.path.join(DATA, "sdqa_layer_transfer.json")) as f:
        rows = json.load(f)["curves"]
    n = len(rows)
    xs = [r["layer"] / (n - 1) for r in rows]
    arms = [  # (key, label, color, ls, lw)
        ("tx2sdqa_audio", r"text$\rightarrow$SD-QA audio (deploy recipe)",
         "#1f5fa8", "-", 2.0),
        ("au2sdqa_audio", r"audio$\rightarrow$SD-QA audio", "0.45", "--", 1.3),
        ("tx2sdqa_text", r"text$\rightarrow$SD-QA text (content only)",
         "0.45", ":", 1.3),
    ]
    for key, label, color, ls, lw in arms:
        ax2.plot(xs, [r[key] for r in rows], color=color, ls=ls, lw=lw,
                 label=label)
    dep = rows[DEPLOY_LAYER]["tx2sdqa_audio"]
    peak_r = max(rows, key=lambda r: r["tx2sdqa_audio"])
    ax2.plot([DEPLOY_LAYER / (n - 1)], [dep], "o", color="#1f5fa8", ms=6,
             zorder=5)
    ax2.annotate(f"L{DEPLOY_LAYER}: {dep:.3f}",
                 (DEPLOY_LAYER / (n - 1), dep),
                 xytext=(DEPLOY_LAYER / (n - 1) - 0.03, dep + 0.06),
                 fontsize=8, color="#1f5fa8", ha="right")
    ax2.annotate(f"peak L{peak_r['layer']}: {peak_r['tx2sdqa_audio']:.3f}",
                 (peak_r["layer"] / (n - 1), peak_r["tx2sdqa_audio"]),
                 xytext=(peak_r["layer"] / (n - 1) + 0.04,
                         peak_r["tx2sdqa_audio"] + 0.055),
                 fontsize=8, color="#1f5fa8")
    ax2.axhline(0.5, color="0.55", lw=0.8, ls=":")
    ax2.set_xlabel("relative depth (layer / depth)")
    ax2.set_ylabel("failure AUC on SD-QA")
    ax2.set_title("SD-QA real human speech, MiniCPM-o 4.5\n(query-disjoint AND cross-modal)",
                  fontsize=9.5)
    ax2.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax2.set_ylim(0.35, 0.95)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(FIGS, f"xmodal_transfer.{ext}")
        fig.savefig(path, dpi=170)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
