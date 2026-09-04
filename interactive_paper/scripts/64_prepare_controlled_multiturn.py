"""Freeze deterministic cross-source two-turn sessions from prospective audio.

The target questions already have standalone outcomes.  Replaying each after
an unrelated, completed carrier turn creates a matched context-shift test
without generating new benchmark content or TTS audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(seed: int, value: str):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--id-prefix", default="p19")
    args = parser.parse_args()

    source = pd.read_parquet(args.selection).sort_values("id")
    if source.id.duplicated().any():
        raise RuntimeError("duplicate source IDs")
    missing = [row_id for row_id in source.id
               if not (args.audio_dir / f"{row_id}.wav").exists()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} audio files")
    pools = sorted(source.pool.unique())
    if len(pools) < 2:
        raise RuntimeError("need at least two source pools")

    chosen = []
    for pool_index, pool in enumerate(pools):
        targets = (source[source.pool == pool].assign(
            _key=lambda frame: frame.id.map(
                lambda value: stable(args.seed, f"target:{value}")))
            .sort_values("_key"))
        if not args.all_targets:
            targets = targets.head(args.per_source)
        carrier_pool = pools[(pool_index + 1) % len(pools)]
        carriers = (source[source.pool == carrier_pool].assign(
            _key=lambda frame: frame.id.map(
                lambda value: stable(args.seed, f"carrier:{pool}:{value}")))
            .sort_values("_key"))
        if not args.all_targets:
            carriers = carriers.head(args.per_source)
        if not args.all_targets and (len(targets) != args.per_source or
                                     len(carriers) != args.per_source):
            raise RuntimeError(f"insufficient rows for {pool}")
        carrier_rows = list(carriers.itertuples())
        for target_index, target in enumerate(targets.itertuples()):
            carrier = carrier_rows[target_index % len(carrier_rows)]
            row = {
                "id": f"{args.id_prefix}-{carrier.id}-{target.id}",
                "target_id": str(target.id),
                "target_pool": str(target.pool),
                "target_query": str(target.query),
                "target_reference_answer": str(target.reference_answer),
                "carrier_id": str(carrier.id),
                "carrier_pool": str(carrier.pool),
                "carrier_query": str(carrier.query),
                "language": str(target.language),
                "context_condition": "completed_unrelated_prior_turn",
            }
            for column in ("source", "training_tag"):
                if hasattr(target, column):
                    row[f"target_{column}"] = str(getattr(target, column))
                if hasattr(carrier, column):
                    row[f"carrier_{column}"] = str(getattr(carrier, column))
            chosen.append(row)
    output = pd.DataFrame(chosen).sort_values("id")
    if output.id.duplicated().any() or output.target_id.duplicated().any():
        raise RuntimeError("pair IDs and targets must be unique")
    if (output.target_pool == output.carrier_pool).any():
        raise RuntimeError("carrier source must differ from target source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    canonical = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")) + "\n"
                        for row in output.to_dict("records"))
    receipt = {
        "status": "frozen_before_context_generation",
        "seed": args.seed,
        "rows": len(output),
        "counts_by_target_pool": output.target_pool.value_counts(
            sort=False).sort_index().to_dict(),
        "selection_sha256": sha256(args.output),
        "canonical_content_sha256": hashlib.sha256(
            canonical.encode()).hexdigest(),
        "source_selection_sha256": sha256(args.selection),
        "guard": "Pairs frozen before any two-turn model outputs.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
