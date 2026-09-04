#!/usr/bin/env python3
"""Full leave-one-pool-out (LOPO) x layer matrix from the committed Phase-5d sweeps.

The paper reports the late-layer readout failure on a single held-out pool
(hard-math: final-layer AUC 0.372 vs 0.931 at L22) and picks L22 as the argmax
of that same curve.  The per-layer LOPO curves for every pool were already
computed in Phase-5d and are committed as data/layers/layer_sweep_{tag}.json,
so three questions a reader will ask can be answered without any GPU:

  1. Does the final-layer read lose to a mid-network read on the OTHER held-out
     pools, or only on hard-math?
  2. Is 0.931 an artefact of choosing the layer on the fold it is reported on?
     (Nested selection: choose the layer on the other folds only.)
  3. Does any of this hold on audio input, the modality the system deploys on?

Metric provenance (modal_app.py::layer_sweep_report):
  * Rows are the 360-query calibration split, so each held-out pool is ~72 rows.
  * For each layer and pooling, a logistic probe is fit on the four non-held-out
    pools and scored on the held-out pool.
  * If the held-out pool contains both classes the cell is a ROC AUC; if it is
    single-class (the trap pool, where the target fails every query) the cell is
    the MEAN predicted failure score instead.  The two are not comparable, so
    they are kept in separate columns here.
  * No confidence intervals: the sweeps stored point estimates only.  Per-query
    scores live on the gate-data volume; re-running the sweep with a bootstrap
    is the way to add CIs.

Usage:  python interactive_paper/scripts/40_lopo_layer_matrix.py
Writes: interactive_paper/figures/lopo_layer_matrix.json
        interactive_paper/figures/lopo_layer_matrix.png   (if matplotlib present)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAYERS = os.path.join(ROOT, "data", "layers")
FIGS = os.path.join(ROOT, "figures")

# tag -> (label, deployed/mid layer, input modality)
MODELS = [
    ("minicpm-o45", "MiniCPM-o 4.5 (duplex)", 22, "text"),
    ("qwen3-8b", "Qwen3-8B (raw backbone of o4.5)", 22, "text"),
    ("minicpm-o26", "MiniCPM-o 2.6 (duplex)", 21, "text"),
    ("qwen2.5-7b", "Qwen2.5-7B (raw backbone of o2.6)", 21, "text"),
    ("qwen2.5-omni-7b", "Qwen2.5-Omni (streaming omni)", 21, "text"),
    ("minicpm-o45-audio", "MiniCPM-o 4.5 (duplex)", 22, "audio"),
]
# matched fine-tune / raw-control pairs, as in the paper's Table 2 roster
PAIRS = [
    ("minicpm-o45", "qwen3-8b"),
    ("minicpm-o26", "qwen2.5-7b"),
    ("qwen2.5-omni-7b", "qwen2.5-7b"),
]
AUC_FOLDS = ["easy-chat", "easy-fact", "hard-knowledge", "hard-math"]
BAND = (0.50, 0.75)  # relative-depth band, comparable across depths


def load(tag):
    with open(os.path.join(LAYERS, f"layer_sweep_{tag}.json")) as f:
        d = json.load(f)
    assert [c["layer"] for c in d["curves"]] == list(range(d["n_layers"]))
    return d


def series(curves, pos, fold):
    """LOPO curve for one held-out pool, or None if this sweep lacks it."""
    key = f"{pos}_lopo_{fold}"
    if key in curves[0]:
        return [c[key] for c in curves], "auc"
    if key + "_meanscore" in curves[0]:
        return [c[key + "_meanscore"] for c in curves], "meanscore"
    return None, None


def band_idx(n_layers):
    return [i for i in range(n_layers) if BAND[0] <= i / (n_layers - 1) <= BAND[1]]


def analyse():
    out = {"band_relative_depth": list(BAND), "auc_folds": AUC_FOLDS, "models": {}}
    for tag, label, mid, modality in MODELS:
        d = load(tag)
        n, curves = d["n_layers"], d["curves"]
        final = n - 1
        band = band_idx(n)
        rec = {
            "label": label, "modality": modality, "n_layers": n,
            "final_layer": final, "mid_layer": mid,
            "band_layers": [band[0], band[-1]], "positions": {},
        }
        for pos in ("last", "mean"):
            folds = {}
            for fold in AUC_FOLDS + ["trap"]:
                vals, kind = series(curves, pos, fold)
                if vals is None:
                    continue
                # nested layer choice: maximise mean LOPO over the OTHER AUC folds
                others = []
                for g in AUC_FOLDS:
                    if g == fold:
                        continue
                    v, k = series(curves, pos, g)
                    if v is not None and k == "auc":
                        others.append(v)
                nested_layer = nested = None
                if others:
                    avg = [sum(v[i] for v in others) / len(others) for i in range(n)]
                    nested_layer = max(range(n), key=lambda i: avg[i])
                    nested = vals[nested_layer]
                argmax_layer = max(range(n), key=lambda i: vals[i])
                folds[fold] = {
                    "metric": kind,
                    "final": vals[final],
                    "mid": vals[mid],
                    "band_mean": sum(vals[i] for i in band) / len(band),
                    "argmax": vals[argmax_layer], "argmax_layer": argmax_layer,
                    "nested": nested, "nested_layer": nested_layer,
                    "curve": vals,
                }
            aucs = [f for f in AUC_FOLDS if f in folds and folds[f]["metric"] == "auc"]
            summary = {}
            for field in ("final", "mid", "band_mean", "argmax", "nested"):
                vs = [folds[f][field] for f in aucs if folds[f][field] is not None]
                summary[field] = sum(vs) / len(vs) if vs else None
            summary["n_auc_folds"] = len(aucs)
            summary["final_below_mid"] = [f for f in aucs if folds[f]["final"] < folds[f]["mid"]]
            summary["final_below_chance"] = [f for f in aucs if folds[f]["final"] < 0.5]
            rec["positions"][pos] = {"folds": folds, "mean_over_auc_folds": summary}
        rec["oof"] = {p: {"final": curves[final][f"{p}_oof"], "mid": curves[mid][f"{p}_oof"]}
                      for p in ("last", "mean")}
        out["models"][tag] = rec

    out["matched_pairs"] = {}
    for ft, raw in PAIRS:
        a, b = out["models"][ft], out["models"][raw]
        deltas = {}
        for field in ("final", "mid"):
            row = {}
            for fold in AUC_FOLDS:
                fa = a["positions"]["last"]["folds"].get(fold)
                fb = b["positions"]["last"]["folds"].get(fold)
                if fa and fb and fa["metric"] == fb["metric"] == "auc":
                    row[fold] = round(fa[field] - fb[field], 4)
            row["mean"] = round(sum(row.values()) / len(row), 4) if row else None
            deltas[field] = row
        out["matched_pairs"][f"{ft}_vs_{raw}"] = deltas
    return out


def fmt(v, w=6):
    return " " * w if v is None else f"{v:{w}.3f}"


def report(res):
    lines = []
    P = lines.append
    P("LOPO x layer matrix -- last-token and mean-pooled reads")
    P("Rows: the 360-query calibration split; each held-out pool is ~72 queries.")
    P("Cells are AUC on the held-out pool; the trap pool is reported separately")
    P("because it is single-class for some models. No CIs (point estimates only).")
    P("  final  = last decoder layer (the conventional readout)")
    P("  mid    = the layer the paper deploys (L22 of 36 / L21 of 28)")
    P("  nested = layer chosen by the mean LOPO over the OTHER three folds only")
    P("  band   = mean over 0.50-0.75 relative depth")
    P("")
    for pos, posname in (("last", "last-token"), ("mean", "mean-pooled")):
        P(f"=== {posname} read ===")
        P(f"{'model':32s} {'in':5s} {'read':16s} "
          + " ".join(f"{f[:9]:>9s}" for f in AUC_FOLDS) + f" {'mean':>7s}")
        for tag, rec in res["models"].items():
            s = rec["positions"][pos]
            for field, name in (("final", f"final (L{rec['final_layer']})"),
                                ("mid", f"mid (L{rec['mid_layer']})"),
                                ("nested", "nested-selected"),
                                ("band", "band 0.50-0.75")):
                key = "band_mean" if field == "band" else field
                cells = []
                for f in AUC_FOLDS:
                    fd = s["folds"].get(f)
                    cells.append(fmt(fd[key] if fd and fd["metric"] == "auc" else None, 9))
                P(f"{rec['label']:32.32s} {rec['modality']:5s} {name:16s} "
                  + " ".join(cells) + " "
                  + fmt(s["mean_over_auc_folds"][key], 7))
            picks = []
            for f in AUC_FOLDS:
                fd = s["folds"].get(f)
                if fd and fd["metric"] == "auc" and fd["nested_layer"] is not None:
                    picks.append(f"{f[:9]}->L{fd['nested_layer']}")
            P(f"{'':32s} {'':5s} {'nested picks':16s} " + "  ".join(picks))
            below = s["mean_over_auc_folds"]["final_below_mid"]
            chance = s["mean_over_auc_folds"]["final_below_chance"]
            P(f"{'':32s} {'':5s} {'final < mid on':16s} "
              + (", ".join(below) if below else "none")
              + (f"   | below chance: {', '.join(chance)}" if chance else ""))
            P("")
    P("trap pool (SimpleQA) held out, last-token read:")
    for tag, rec in res["models"].items():
        fd = rec["positions"]["last"]["folds"].get("trap")
        if not fd:
            continue
        kind = "AUC" if fd["metric"] == "auc" else "mean failure score (pool is 100% fail)"
        P(f"  {rec['label']:32.32s} {rec['modality']:5s} "
          f"final {fd['final']:.3f}  mid {fd['mid']:.3f}   {kind}")
    P("")
    P("Matched pairs, last-token (fine-tune minus its raw control):")
    for k, v in res["matched_pairs"].items():
        P(f"  {k}")
        for field in ("final", "mid"):
            row = v[field]
            P(f"    at {field:5s}: "
              + ", ".join(f"{f} {row[f]:+.3f}" for f in AUC_FOLDS if f in row)
              + f"  | mean {row['mean']:+.3f}")
    return "\n".join(lines)


def figure(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1), sharey=True)
    show = [("minicpm-o45", "#1f5fa8"), ("qwen3-8b", "#e4762e"),
            ("minicpm-o26", "#1baf7a"), ("qwen2.5-7b", "#8a6fbf")]
    for ax, fold in zip(axes, AUC_FOLDS):
        for tag, color in show:
            rec = res["models"][tag]
            fd = rec["positions"]["last"]["folds"].get(fold)
            if not fd or fd["metric"] != "auc":
                continue
            n = rec["n_layers"]
            xs = [i / (n - 1) for i in range(n)]
            dup = "duplex" in rec["label"]
            ax.plot(xs, fd["curve"], color=color, lw=2.0 if dup else 1.3,
                    ls="-" if dup else "--",
                    label=rec["label"].split(" (")[0])
        ax.axhline(0.5, color="0.55", lw=0.8, ls=":")
        ax.axvspan(BAND[0], BAND[1], color="0.85", alpha=0.45, lw=0)
        ax.set_title(f"held out: {fold}", fontsize=10)
        ax.set_xlabel("relative depth")
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("LOPO AUC (last token)")
    axes[0].legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    fig.suptitle("Leave-one-pool-out transfer by depth, text input "
                 "(shaded: 0.50-0.75 relative depth)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170)
    print(f"wrote {path}")


if __name__ == "__main__":
    res = analyse()
    text = report(res)
    print(text)
    os.makedirs(FIGS, exist_ok=True)
    jpath = os.path.join(FIGS, "lopo_layer_matrix.json")
    with open(jpath, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {jpath}")
    figure(res, os.path.join(FIGS, "lopo_layer_matrix.png"))
