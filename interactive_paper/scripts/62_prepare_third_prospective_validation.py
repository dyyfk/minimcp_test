"""Freeze P17 across SNLI, SST-2, and WiC before any generated output."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


REVISIONS = {
    "snli": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
    "sst2": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "wic": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def source_frames(args):
    snli_rows = []
    names = {0: "(A) entailment", 1: "(B) neutral", 2: "(C) contradiction"}
    for index, row in pd.read_parquet(args.snli).iterrows():
        if int(row.label) not in names:
            continue
        snli_rows.append({
            "id": f"p17n{index:05d}", "pool": "p17_snli",
            "source": "stanfordnlp/snli:validation", "label": int(row.label),
            "query": (f"Premise: {row.premise} Hypothesis: {row.hypothesis} "
                      "Choose (A) entailment, (B) neutral, or (C) contradiction."),
            "reference_answer": names[int(row.label)], "language": "en",
            "split": "third_prospective_validation"})
    sst_rows = []
    for index, row in pd.read_parquet(args.sst2).iterrows():
        label = int(row.label)
        sst_rows.append({
            "id": f"p17s{index:05d}", "pool": "p17_sst2",
            "source": "nyu-mll/glue:sst2:validation", "label": label,
            "query": (f"Is the sentiment of this review positive or negative? "
                      f"Review: {row.sentence}"),
            "reference_answer": "positive" if label else "negative",
            "language": "en", "split": "third_prospective_validation"})
    wic_rows = []
    for index, row in pd.read_parquet(args.wic).iterrows():
        label = int(row.label)
        wic_rows.append({
            "id": f"p17w{index:05d}", "pool": "p17_wic",
            "source": "aps/super_glue:wic:validation", "label": label,
            "query": (f"Does the word '{row.word}' have the same meaning in both "
                      f"sentences? Sentence one: {row.sentence1} Sentence two: "
                      f"{row.sentence2} Answer yes or no."),
            "reference_answer": "yes" if label else "no", "language": "en",
            "split": "third_prospective_validation"})
    return [pd.DataFrame(snli_rows), pd.DataFrame(sst_rows),
            pd.DataFrame(wic_rows)]


def existing_queries(data_dir, selections):
    values = set()
    for path in selections:
        values.update(pd.read_parquet(path)["query"].map(normalize))
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
    parser.add_argument("--snli", type=Path, required=True)
    parser.add_argument("--sst2", type=Path, required=True)
    parser.add_argument("--wic", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, action="append",
                        required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--per-source", type=int, default=150)
    parser.add_argument("--max-characters", type=int, default=300)
    args = parser.parse_args()
    seen = existing_queries(args.data_dir, args.prior_selection)
    chosen, diagnostics = [], {}
    for frame in source_frames(args):
        normalized = frame["query"].map(normalize)
        overlap = normalized.isin(seen)
        duplicate = normalized.duplicated(keep="first")
        too_long = frame["query"].str.len() > args.max_characters
        invalid = overlap | duplicate | too_long
        eligible = frame[~invalid].copy()
        label_values = sorted(eligible.label.unique())
        if args.per_source % len(label_values):
            raise RuntimeError("per-source count must divide evenly by labels")
        per_label = args.per_source // len(label_values)
        parts = []
        for label in label_values:
            part = eligible[eligible.label == label].copy()
            part["_key"] = part.id.map(lambda value: stable_key(args.seed, value))
            parts.append(part.sort_values("_key").head(per_label))
        sample = pd.concat(parts).sort_values("id").drop(columns="_key")
        if len(sample) != args.per_source:
            raise RuntimeError(f"insufficient eligible rows in {frame.pool.iloc[0]}")
        chosen.append(sample.drop(columns="label"))
        diagnostics[str(frame.pool.iloc[0])] = {
            "source_rows": len(frame), "eligible": len(eligible),
            "existing_overlap": int(overlap.sum()),
            "within_source_duplicates": int(duplicate.sum()),
            "over_character_limit": int(too_long.sum()),
            "excluded_union": int(invalid.sum()), "chosen": len(sample),
            "chosen_label_counts": sample.label.value_counts().sort_index().to_dict(),
        }
    selection = pd.concat(chosen, ignore_index=True).sort_values("id")
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(args.selection, index=False)
    canonical = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")) + "\n"
                        for row in selection.to_dict("records"))
    receipt = {
        "status": "frozen_after_alpha2_before_tts_native_or_judge",
        "seed": args.seed, "n": len(selection),
        "counts_by_pool": selection.pool.value_counts().sort_index().to_dict(),
        "characters": int(selection["query"].str.len().sum()),
        "tts_cost_usd_at_15_per_million_characters": float(
            selection["query"].str.len().sum() * 15 / 1e6),
        "ordered_id_sha256": hashlib.sha256(
            ("\n".join(selection.id) + "\n").encode()).hexdigest(),
        "canonical_content_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "selection_sha256": sha256(args.selection),
        "raw_sources": {
            "snli": {"revision": REVISIONS["snli"], "sha256": sha256(args.snli)},
            "sst2": {"revision": REVISIONS["sst2"], "sha256": sha256(args.sst2)},
            "wic": {"revision": REVISIONS["wic"], "sha256": sha256(args.wic)},
        },
        "diagnostics": diagnostics,
        "guard": "P17 was frozen after alpha-2 and cannot be used for its fitting or selection.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
