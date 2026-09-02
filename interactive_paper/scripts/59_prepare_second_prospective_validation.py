"""Freeze the untouched three-source P16 validation set before outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


REVISIONS = {
    "boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "qasc": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
}


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def boolq(path: Path):
    rows = []
    for index, row in pd.read_parquet(path).iterrows():
        answer = "Yes" if bool(row.answer) else "No"
        rows.append({"id": f"p16b{index:05d}", "pool": "p16_boolq",
                     "source": "google/boolq:validation",
                     "query": (f"Based on this passage, answer yes or no. "
                               f"Passage: {row.passage} Question: {row.question}?"),
                     "reference_answer": answer, "language": "en",
                     "split": "second_prospective_validation"})
    return pd.DataFrame(rows)


def hellaswag(path: Path):
    rows = []
    for index, row in pd.read_parquet(path).iterrows():
        endings = list(row.endings)
        rendered = ", ".join(f"({'ABCD'[i]}) {value}"
                             for i, value in enumerate(endings))
        answer = int(row.label)
        rows.append({"id": f"p16h{index:05d}", "pool": "p16_hellaswag",
                     "source": "Rowan/hellaswag:validation",
                     "query": (f"Complete the scenario: {row.ctx}. Which ending "
                               f"is most likely? {rendered}"),
                     "reference_answer": f"({'ABCD'[answer]}) {endings[answer]}",
                     "language": "en", "split": "second_prospective_validation"})
    return pd.DataFrame(rows)


def qasc(path: Path):
    rows = []
    for index, row in pd.read_parquet(path).iterrows():
        choices = dict(zip(row.choices["label"], row.choices["text"]))
        key = str(row.answerKey)
        rows.append({"id": f"p16q{index:05d}", "pool": "p16_qasc",
                     "source": "allenai/qasc:validation",
                     "query": str(row.formatted_question),
                     "reference_answer": f"({key}) {choices[key]}",
                     "language": "en", "split": "second_prospective_validation"})
    return pd.DataFrame(rows)


def existing_queries(data_dir: Path, p15: Path):
    values = set(pd.read_parquet(p15)["query"].map(normalize))
    for path in sorted(data_dir.glob("queries*.jsonl")):
        frame = pd.read_json(path, lines=True)
        if "query" in frame:
            values.update(frame["query"].map(normalize))
    for name in ("calib_features.parquet", "expansion_labels.parquet",
                 "expansion2_labels.parquet", "expansion3_labels.parquet",
                 "expansion3zh_labels.parquet", "fresh_labels.parquet"):
        values.update(pd.read_parquet(data_dir / name, columns=["query"])
                      ["query"].map(normalize))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boolq", type=Path, required=True)
    parser.add_argument("--hellaswag", type=Path, required=True)
    parser.add_argument("--qasc", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--p15-selection", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=150)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-characters", type=int, default=300)
    args = parser.parse_args()
    seen = existing_queries(args.data_dir, args.p15_selection)
    frames = [boolq(args.boolq), hellaswag(args.hellaswag), qasc(args.qasc)]
    chosen, diagnostics = [], {}
    for frame in frames:
        normalized = frame["query"].map(normalize)
        overlap = normalized.isin(seen)
        duplicate = normalized.duplicated(keep="first")
        too_long = frame["query"].str.len() > args.max_characters
        eligible = frame[~overlap & ~duplicate & ~too_long].copy()
        eligible["_key"] = eligible.id.map(
            lambda value: stable_key(args.seed, value))
        sample = eligible.sort_values("_key").head(args.per_source)
        if len(sample) != args.per_source:
            raise RuntimeError(f"not enough eligible rows for {frame.pool.iloc[0]}")
        chosen.append(sample.drop(columns="_key"))
        diagnostics[str(frame.pool.iloc[0])] = {
            "source_rows": len(frame), "existing_overlap": int(overlap.sum()),
            "within_source_duplicates": int(duplicate.sum()),
            "over_character_limit": int(too_long.sum()),
            "eligible": len(eligible), "chosen": len(sample),
        }
    selection = pd.concat(chosen, ignore_index=True).sort_values("id")
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(args.selection, index=False)
    canonical = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")) + "\n"
                        for row in selection.to_dict("records"))
    receipt = {
        "status": "frozen_before_tts_native_or_judge_outputs",
        "seed": args.seed, "n": len(selection),
        "counts_by_pool": selection.pool.value_counts().sort_index().to_dict(),
        "max_characters": args.max_characters,
        "characters": int(selection["query"].str.len().sum()),
        "tts_cost_usd_at_15_per_million_characters": float(
            selection["query"].str.len().sum() * 15 / 1_000_000),
        "ordered_id_sha256": hashlib.sha256(
            ("\n".join(selection.id) + "\n").encode()).hexdigest(),
        "canonical_content_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "selection_sha256": sha256(args.selection),
        "raw_sources": {
            "boolq": {"revision": REVISIONS["boolq"], "sha256": sha256(args.boolq)},
            "hellaswag": {"revision": REVISIONS["hellaswag"], "sha256": sha256(args.hellaswag)},
            "qasc": {"revision": REVISIONS["qasc"], "sha256": sha256(args.qasc)},
        },
        "diagnostics": diagnostics,
        "guard": "P16 IDs/content frozen after the P15-selected ensemble and before all P16 outputs.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
