"""Paper-side outputs for the deployed-probe layer ablation.

Reads figures/layer_ablation_{source}.json (from modal_train4_layers.py::
layer_ablation3 or scripts/64) and writes

  figures/layer_ablation_depth.pdf   AUC of the deployed feature set by
                                     decoder layer, one line per external
                                     pool + external mean, frozen-test
                                     dashed, deployed layer marked
  figures/layer_ablation_table.tex   compact table: selected layers x pools
                                     with paired-bootstrap deltas vs L22

Usage: python scripts/65_layer_ablation_paper.py --source eoth3
       [--layers 6,14,18,22,26,30,33,35]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POOL_NAMES = {"striviaqa": "TriviaQA", "swebq": "WebQ", "sllama": "Llama Q.",
              "sdqa": "SD-QA", "sreason": "Reason.\\ zh",
              "frozen-test": "internal"}
ORDER = ["striviaqa", "swebq", "sllama", "sdqa", "sreason", "frozen-test"]


def fmt_delta(d):
    lo, hi = d[1], d[2]
    s = f"{d[0]:+.3f}"
    return s + ("$^{*}$" if (lo > 0 or hi < 0) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="eoth3")
    ap.add_argument("--layers", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    path = a.json or os.path.join(ROOT, "figures", f"layer_ablation_{a.source}.json")
    j = json.load(open(path))
    L = j["layers"]
    ref = j["deployed_layer"]
    pools = [p for p in ORDER if p in j["pools"]]
    ext = [p for p in pools if p != "frozen-test"]

    # ---- table -----------------------------------------------------------
    sel = [int(x) for x in a.layers.split(",") if x.strip()] or L
    sel = [l for l in sel if l in L]
    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(pools) + "cc}")
    lines.append("\\toprule")
    lines.append("layer & " + " & ".join(POOL_NAMES[p] for p in pools)
                 + " & ext.\\ mean & $\\Delta$ vs.\\ L%d \\\\" % ref)
    lines.append("\\midrule")
    ref_row = j["rows"][str(ref)]
    for l in sel:
        r = j["rows"][str(l)]
        cells = []
        for p in pools:
            d = r[p]["delta_vs_ref"]
            star = "" if l == ref or (d[1] <= 0 <= d[2]) else "$^{*}$"
            cells.append(f"{r[p]['auc']:.3f}{star}")
        dm = r["ext_mean_auc"] - ref_row["ext_mean_auc"]
        name = f"L{l}" + (" (deployed)" if l == ref else "")
        if l == ref:
            name = "\\textbf{" + name + "}"
        lines.append(f"{name} & " + " & ".join(cells)
                     + f" & {r['ext_mean_auc']:.3f} & {dm:+.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tex = "\n".join(lines) + "\n"
    out_tex = os.path.join(ROOT, "figures", "layer_ablation_table.tex")
    open(out_tex, "w").write(tex)
    print(tex)

    # ---- figure ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; table only")
        return
    # CVD-safe palette used elsewhere in the paper
    colors = {"striviaqa": "#2b62a8", "swebq": "#e08a1e", "sllama": "#c1272d",
              "sdqa": "#1baf7a", "sreason": "#7b4fb0", "frozen-test": "#6b7280"}
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for p in pools:
        ys = [j["rows"][str(l)][p]["auc"] for l in L]
        ax.plot(L, ys, color=colors.get(p, "k"), lw=1.1, marker="o", ms=2.2,
                ls="--" if p == "frozen-test" else "-",
                label=POOL_NAMES[p].replace("\\ ", " "))
    ax.plot(L, [j["rows"][str(l)]["ext_mean_auc"] for l in L], color="k",
            lw=2.4, label="external mean")
    ax.axvline(ref, color="grey", ls=":", lw=1)
    ax.text(ref + 0.3, ax.get_ylim()[0] + 0.01, f"L{ref} (deployed)",
            fontsize=7, color="grey")
    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("decoder layer (of %d)" % (max(L) + 1))
    ax.set_ylabel("AUC (never-arm local failure)")
    ax.set_title("Deployed three-position probe re-fit at each layer, audio input",
                 fontsize=9)
    ax.grid(alpha=.25)
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="lower center")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "layer_ablation_depth")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    print("wrote", out + ".pdf", "and", out_tex)


if __name__ == "__main__":
    main()
