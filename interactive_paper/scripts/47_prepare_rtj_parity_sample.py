"""Freeze an equal-per-pool sample for RTJ transcript/latency parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-pool", type=int, default=10)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    frame["_key"] = [hashlib.sha256(
        f"{args.seed}:{row_id}".encode()).hexdigest() for row_id in frame["id"]]
    sample = (frame.sort_values("_key").groupby("pool", sort=True)
              .head(args.per_pool).sort_values("id").drop(columns="_key"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)
    ids = "\n".join(sample["id"]) + "\n"
    receipt = {"n": len(sample), "per_pool": args.per_pool,
               "seed": args.seed,
               "id_list_sha256": hashlib.sha256(ids.encode()).hexdigest()}
    args.output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
