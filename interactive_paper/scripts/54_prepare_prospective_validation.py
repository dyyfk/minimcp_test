"""Freeze a source-disjoint WinoGrande + SciQ validation set pre-outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def existing_queries(data_dir: Path):
    values = set()
    for path in sorted(data_dir.glob("queries*.jsonl")):
        frame = pd.read_json(path, lines=True)
        if "query" in frame:
            values.update(frame["query"].map(normalize))
    for name in ("calib_features.parquet", "expansion_labels.parquet",
                 "expansion2_labels.parquet", "expansion3_labels.parquet",
                 "expansion3zh_labels.parquet", "fresh_labels.parquet"):
        frame = pd.read_parquet(data_dir / name, columns=["query"])
        values.update(frame["query"].map(normalize))
    return values


def winogrande(path: Path):
    source = pd.read_parquet(path)
    rows = []
    for index, row in source.iterrows():
        row_id = f"p15w{index:04d}"
        answer = int(row.answer) - 1
        options = [str(row.option1), str(row.option2)]
        rows.append({
            "id": row_id, "pool": "p15_winogrande",
            "source": "allenai/winogrande:winogrande_debiased:validation",
            "query": (f"{row.sentence} Which choice best fills the blank? "
                      f"(A) {options[0]}, (B) {options[1]}"),
            "source_question": str(row.sentence),
            "reference_answer": f"({'AB'[answer]}) {options[answer]}",
            "language": "en", "split": "prospective_validation",
        })
    return pd.DataFrame(rows)


def sciq(path: Path, seed: int):
    source = pd.read_parquet(path)
    rows = []
    for index, row in source.iterrows():
        row_id = f"p15s{index:04d}"
        options = [str(row.correct_answer), str(row.distractor1),
                   str(row.distractor2), str(row.distractor3)]
        options = sorted(options, key=lambda value: stable_key(
            seed, f"{row_id}:{value}"))
        answer = options.index(str(row.correct_answer))
        rendered = ", ".join(
            f"({'ABCD'[i]}) {value}" for i, value in enumerate(options))
        rows.append({
            "id": row_id, "pool": "p15_sciq",
            "source": "allenai/sciq:validation",
            "query": f"{row.question} {rendered}",
            "source_question": str(row.question),
            "reference_answer": f"({'ABCD'[answer]}) {options[answer]}",
            "language": "en", "split": "prospective_validation",
        })
    return pd.DataFrame(rows)


def canonical_sha(frame):
    columns = ["id", "pool", "source", "query", "reference_answer",
               "language", "split"]
    payload = "".join(json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in frame[columns].sort_values("id").to_dict("records"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winogrande", type=Path, required=True)
    parser.add_argument("--sciq", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=200)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    seen = existing_queries(args.data_dir)
    chosen, rejected = [], {}
    for frame in (winogrande(args.winogrande), sciq(args.sciq, args.seed)):
        frame["normalized_source_question"] = frame.source_question.map(normalize)
        overlap = frame.normalized_source_question.isin(seen)
        duplicate = frame.normalized_source_question.duplicated(keep="first")
        rejected[str(frame.pool.iloc[0])] = {
            "existing_query_overlap": int(overlap.sum()),
            "within_source_duplicates": int(duplicate.sum()),
        }
        eligible = frame[~overlap & ~duplicate].copy()
        eligible["_key"] = eligible.id.map(
            lambda row_id: stable_key(args.seed, row_id))
        sample = eligible.sort_values("_key").head(args.per_source)
        if len(sample) != args.per_source:
            raise RuntimeError(f"not enough eligible rows in {frame.pool.iloc[0]}")
        chosen.append(sample.drop(columns=["_key", "normalized_source_question",
                                           "source_question"]))
    selection = pd.concat(chosen, ignore_index=True).sort_values("id")
    if selection.id.duplicated().any() or selection["query"].map(
            normalize).duplicated().any():
        raise RuntimeError("prospective selection contains duplicate IDs/queries")
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(args.selection, index=False)
    ids_blob = "\n".join(selection.id) + "\n"
    receipt = {
        "status": "frozen_pre_outcome",
        "seed": args.seed, "n": len(selection),
        "counts_by_pool": selection.pool.value_counts().sort_index().to_dict(),
        "id_list_sha256": hashlib.sha256(ids_blob.encode()).hexdigest(),
        "canonical_content_sha256": canonical_sha(selection),
        "selection_parquet_sha256": sha256(args.selection),
        "raw_sources": {
            "winogrande": {"sha256": sha256(args.winogrande),
                            "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5"},
            "sciq": {"sha256": sha256(args.sciq),
                     "revision": "2c94ad3e1aafab77146f384e23536f97a4849815"},
        },
        "rejected": rejected,
        "characters": int(selection["query"].str.len().sum()),
        "tts_1_usd_at_15_per_million_chars": float(
            selection["query"].str.len().sum() * 15 / 1_000_000),
        "guard": "IDs and content frozen before TTS, native generation, feature extraction, or judging.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
