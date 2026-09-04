"""Summarize fixed-set latency for the semantic+RTJ shadow candidate."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(directory: Path):
    rows = []
    for path in sorted(glob.glob(str(directory / "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def canonical_sha256(rows):
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda row: str(row["id"])))
    return hashlib.sha256(payload.encode()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--rtj-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    semantic = load_jsonl(args.semantic_dir)
    rtj = load_jsonl(args.rtj_dir)
    if len(semantic) != 50 or len(rtj) != 50:
        raise RuntimeError(f"expected 50 semantic/RTJ rows, got {len(semantic)}/{len(rtj)}")
    if any(row.get("error") for row in semantic + rtj):
        raise RuntimeError("latency inputs contain errors")
    if any(len(row["samples"]) != 2 for row in semantic):
        raise RuntimeError("semantic input must contain exactly two samples per row")

    sem = pd.DataFrame({
        "id": [str(row["id"]) for row in semantic],
        "semantic_serial": [row["elapsed_s"] for row in semantic],
        "sample_1": [row["samples"][0]["elapsed_s"] for row in semantic],
        "sample_2": [row["samples"][1]["elapsed_s"] for row in semantic],
    })
    judge = pd.DataFrame({
        "id": [str(row["id"]) for row in rtj],
        "rtj": [row["elapsed_s"] for row in rtj],
    })
    selection = pd.read_parquet(args.selection)[["id", "pool"]]
    selection["id"] = selection["id"].astype(str)
    frame = sem.merge(judge, on="id", validate="one_to_one").merge(
        selection, on="id", validate="one_to_one")
    if len(frame) != 50:
        raise RuntimeError(f"ID join lost rows: {len(frame)}")

    frame["serial_total"] = frame.semantic_serial + frame.rtj
    # These are capacity-planning estimates, not measured concurrent runs.
    frame["parallel_three_replica"] = frame[["sample_1", "sample_2", "rtj"]].max(axis=1)
    frame["parallel_semantic_then_rtj"] = (
        frame[["sample_1", "sample_2"]].max(axis=1) + frame.rtj)

    result = {
        "fixed_rows": len(frame),
        "rows_per_pool": frame.groupby("pool").size().astype(int).to_dict(),
        "errors": 0,
        "latency_seconds": {
            "semantic_two_sample_serial_measured": summary(frame.semantic_serial),
            "rtj_measured": summary(frame.rtj),
            "semantic_plus_rtj_serial_estimate": summary(frame.serial_total),
            "three_replica_parallel_estimate": summary(frame.parallel_three_replica),
            "two_semantic_replicas_then_rtj_estimate": summary(frame.parallel_semantic_then_rtj),
        },
        "semantic_serial_by_pool": {
            pool: summary(group.semantic_serial)
            for pool, group in frame.groupby("pool")
        },
        "notes": [
            "Model loading is excluded.",
            "The fixed set has 10 rows per pool and is not prevalence weighted.",
            "Parallel figures are estimates from matched branch timings, not measured concurrent serving.",
            "Embedding latency is excluded; the dominant cost is native answer generation.",
        ],
        "semantic_stream_sha256": canonical_sha256(semantic),
        "rtj_stream_sha256": canonical_sha256(rtj),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("receipt_sha256", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
