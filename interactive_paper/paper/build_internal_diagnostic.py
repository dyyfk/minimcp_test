"""Describe the fixed internal test strata without running or selecting models.

Usage: python build_internal_diagnostic.py --data-dir ../data/native_bench \
    --queries ../data/queries.jsonl
Outputs an appendix table and a JSON ledger with input hashes. Does not edit
the original main Table 1 or teaser. Requires pandas and pyarrow.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CATEGORIES = {
    "easy-chat": "Chat",
    "easy-fact": "Factual QA",
    "hard-knowledge": "Knowledge",
    "hard-math": "Math",
    "trap": "SimpleQA traps",
}


def build(data_dir, queries):
    meta = pd.read_json(queries, lines=True).set_index("id")
    assert meta.index.is_unique
    sources = {queries.name: hashlib.sha256(queries.read_bytes()).hexdigest()}
    frames = {}
    for arm in ["never", "aggressive", "always"]:
        suffix = "" if arm == "never" else "_tts"
        path = data_dir / f"frozen_{arm}{suffix}_judged.parquet"
        df = pd.read_parquet(path).set_index("id")
        assert df.index.is_unique and len(df) == 240
        assert df.adequate.notna().all()
        assert df["query"].eq(meta.loc[df.index, "query"]).all()
        assert meta.loc[df.index, "split"].eq("test").all()
        frames[arm] = df
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = frames["never"].index
    assert all(set(df.index) == set(ids) for df in frames.values())
    d = meta.loc[ids, ["pool", "source"]].copy()
    for arm, df in frames.items():
        d[arm] = df.adequate.astype(float)
        d[arm + "_rate"] = df["mode"].eq("escalated")
    d["audio_gt_30s"] = frames["always"].audio_s > 30
    assert set(d.pool) == set(CATEGORIES)
    rows = []
    for category, label in CATEGORIES.items():
        g = d[d.pool.eq(category)]
        rows.append({"category": category, "label": label, "n": len(g),
                     "local": float(g.never.mean()),
                     "aggressive": float(g.aggressive.mean()),
                     "always": float(g.always.mean()),
                     "aggressive_rate": float(g.aggressive_rate.mean()),
                     "audio_gt_30s_n": int(g.audio_gt_30s.sum()),
                     "always_fired_long_n": int((g.audio_gt_30s & g.always_rate).sum())})
    out = HERE / "revision_data"
    out.mkdir(exist_ok=True)
    doc = {"status": "post-hoc descriptive analysis of all fixed test queries",
           "source_commit": "e74ca0c80f0e20fea8b466424e0307ea621a1403",
           "input_sha256": sources, "runs_per_query_arm": 1,
           "scoring_field": "adequate", "n": len(d), "strata": rows,
           "source_breakdown": d.groupby("source").agg(
               n=("never", "size"), local=("never", "mean"),
               aggressive=("aggressive", "mean"),
               aggressive_rate=("aggressive_rate", "mean")).reset_index().to_dict("records")}
    (out / "internal_diagnostic.json").write_text(json.dumps(doc, indent=2) + "\n")
    lines = [r"\begin{table}[h]", r"\centering\small",
             r"\caption{Descriptive strata of the fixed internal test set. Local, aggressive (A), and always report answer-content accuracy (\%); call rate is the aggressive arm's realized rate. The final column counts question waveforms longer than the expert's 30-second input window. All queries are retained. All 240 queries have unit weight, as in \S\ref{sec:evalproto}.}",
             r"\label{tab:internal-strata}", r"\begin{tabular}{lrrrrrr}",
             r"\toprule",
             r"Stratum & $n$ & Local & A & Always & A call rate & $>30$\,s ($n$) \\",
             r"\midrule"]
    for r in rows:
        cells = [r["label"], str(r["n"])] + [f"{100*r[k]:.1f}" for k in
                 ["local", "aggressive", "always", "aggressive_rate"]] + [str(r["audio_gt_30s_n"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "sections/internal_strata.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data/native_bench")
    parser.add_argument("--queries", type=Path, default=HERE.parent / "data/queries.jsonl")
    args = parser.parse_args()
    build(args.data_dir, args.queries)
