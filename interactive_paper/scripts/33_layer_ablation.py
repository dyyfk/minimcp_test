"""Local (CPU) runner for the deployed-probe layer ablation over shards
pulled from the gate-data volume. Same analysis as
modal_train4_layers.py::layer_ablation3; use this when the volume
files are on disk, e.g.

    modal volume get gate-data 'eoth3_*.npz' data/eoth_pull/
    modal volume get gate-data features_minicpm-o45-audio.parquet data/eoth_pull/
    modal volume get gate-data midlayer_gate_audio_v3.json data/eoth_pull/
    python scripts/33_layer_ablation.py --source eoth3 --pull data/eoth_pull

Writes figures/layer_ablation_{source}.json, .md and (if matplotlib is
installed) figures/layer_ablation_{source}.png: external AUC of the
deployed feature set by depth, one line per pool, deployed layer marked.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import layer_ablation as la  # noqa: E402

EXT_TRACES = (("striviaqa", "striviaqa_traces.parquet"),
              ("swebq", "swebq_traces.parquet"),
              ("sdqa", "sdqa_traces.parquet"),
              ("sllama", "sllama_v2_traces.parquet"),
              ("sreason", "sreason_v2_traces.parquet"))


def _find(pull, data, name):
    for d in (pull, data):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="eoth3", choices=["eoth3", "eoth2"])
    ap.add_argument("--pull", default=os.path.join(ROOT, "data", "eoth_pull"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--layers", default="")
    ap.add_argument("--C", type=float, default=la.DEPLOYED_C)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--with-oof", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    def load(tag):
        return la.load_shards(os.path.join(a.pull, f"{a.source}_{tag}.shard*.npz"))

    feats = pd.read_parquet(_find(a.pull, a.data,
                                  "features_minicpm-o45-audio.parquet"))[
        ["id", "split", "escalate_label"]]
    cal = feats[(feats["split"] == "calib") & feats["escalate_label"].notna()]
    lab = dict(zip(cal["id"], cal["escalate_label"].astype(int)))
    for name in ("expansion_labels.parquet", "expansion2_labels.parquet"):
        df = pd.read_parquet(_find(a.pull, a.data, name))
        df = df[df["escalate_label"].notna()]
        lab.update(dict(zip(df["id"], df["escalate_label"].astype(int))))

    parts = []
    for tag in ("frozen", "expansion", "expansion2"):
        sh = load(tag)
        keep = [j for j, i in enumerate(sh.ids) if i in lab]
        parts.append(sh.subset(keep))
        print(f"train {tag}: {len(keep)}/{sh.n} labeled")
    train = la.Shards(sum((p.ids for p in parts), []),
                      np.concatenate([p.E for p in parts]),
                      np.concatenate([p.M for p in parts]),
                      np.concatenate([p.elen for p in parts]), parts[0].layers)
    y = np.array([lab[i] for i in train.ids])

    pools = []
    for bench, tpath in EXT_TRACES:
        try:
            tr = pd.read_parquet(_find(a.pull, a.data, tpath))
            sh = load(bench)
        except FileNotFoundError as e:
            print(f"{bench}: {e} — skipped")
            continue
        nev = tr[(tr["tier"] == "never") & tr["heard_ok"].notna()]
        lb = dict(zip(nev["id"], 1 - nev["heard_ok"].astype(int)))
        keep = [j for j, i in enumerate(sh.ids) if i in lb]
        if len(keep) < 30:
            continue
        yy = np.array([lb[sh.ids[j]] for j in keep])
        if yy.min() == yy.max():
            continue
        pools.append(la.Pool(bench, sh.subset(keep), yy, external=True))
    tst = feats[(feats["split"] == "test") & feats["escalate_label"].notna()]
    lb = dict(zip(tst["id"], tst["escalate_label"].astype(int)))
    sh = load("frozen")
    keep = [j for j, i in enumerate(sh.ids) if i in lb]
    pools.append(la.Pool("frozen-test", sh.subset(keep),
                         np.array([lb[sh.ids[j]] for j in keep]),
                         external=False))
    for p in pools:
        print(f"pool {p.name}: n={p.sh.n} fail={p.y.mean():.2f}")

    anchor = None
    for name in ("midlayer_gate_audio_v3.json", "gate_v3_frozen_local.json"):
        try:
            anchor = json.load(open(_find(a.pull, a.data, name)))
            break
        except FileNotFoundError:
            continue

    lay = [int(x) for x in a.layers.split(",") if x.strip()] or None
    res = la.run_ablation(train, y, pools, layers=lay, C=a.C, anchor=anchor,
                          B=a.B, with_oof=a.with_oof)
    res.notes.append(f"source={a.source}; local run; pull={a.pull}")
    out = a.out or os.path.join(ROOT, "figures", f"layer_ablation_{a.source}.json")
    la.save_json(res, out)
    md = la.markdown_table(res)
    with open(out.replace(".json", ".md"), "w") as fh:
        fh.write(md + "\n")
    print("\n" + md)
    print(f"wrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    j = res.to_json()
    L = j["layers"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name in j["pools"]:
        ys = [j["rows"][str(l)][name]["auc"] for l in L]
        ax.plot(L, ys, marker="o", ms=3, lw=1.2,
                label=name, ls="--" if name == "frozen-test" else "-")
    ext = [j["rows"][str(l)]["ext_mean_auc"] for l in L]
    ax.plot(L, ext, color="k", lw=2.2, label="external mean")
    ax.axvline(j["deployed_layer"], color="grey", ls=":", lw=1)
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("AUC, deployed feature set")
    ax.set_title("Deployed probe re-fit by layer (audio input)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out.replace(".json", ".png"), dpi=200)
    fig.savefig(out.replace(".json", ".pdf"))
    print(f"wrote {out.replace('.json', '.png')}")


if __name__ == "__main__":
    main()
