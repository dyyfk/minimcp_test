"""Generate resumable paired expert outcomes for expert-gain routing.

This is the local/HF-bundle equivalent of ``modal_benefit.py::train_ceiling``.
It uses the repository's exact expert and judge prompts, writes a checkpoint
after every batch, and never stores ``OPENAI_API_KEY`` in an artifact.

Stage 1 reproduces the original 2310-query recipe.  Additional official
training families can be opted into after Stage 1 is evaluated.

Example (from ``interactive_paper/``)::

    OPENAI_API_KEY=... python scripts/34_generate_expert_gain_labels.py \
      --data-dir /path/to/native_bundle/data \
      --output /safe/path/train_ceiling.parquet \
      --cache-dir /safe/path/expert-cache
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_POOLS = [
    ("queries.jsonl", "calib", "caliboff"),
    ("queries_expansion.jsonl", None, "expoff"),
    ("queries_expansion2.jsonl", None, "exp2off"),
]
EXPANDED_POOLS = [
    ("queries_expansion3.jsonl", None, "exp3off"),
    ("queries_expansion3zh.jsonl", None, "exp3zhoff"),
    ("queries_fresh.jsonl", "train", "freshoff"),
]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_queries(data_dir: Path, include_expanded: bool):
    rows = []
    specs = BASE_POOLS + (EXPANDED_POOLS if include_expanded else [])
    for filename, split, tag in specs:
        block = read_jsonl(data_dir / filename)
        if split:
            block = [row for row in block if row.get("split") == split]
        for row in block:
            row = dict(row)
            row["training_tag"] = tag
            rows.append(row)
    frame = pd.DataFrame(rows)
    if "id" not in frame or "query" not in frame:
        raise RuntimeError("query files must contain id and query")
    frame["id"] = frame["id"].astype(str)
    dup = frame[frame["id"].duplicated(keep=False)]["id"].unique()
    if len(dup):
        raise RuntimeError(f"duplicate training ids across query files: {len(dup)}")
    return frame


def feature_ids(data_dir: Path, tag: str) -> set[str]:
    ids = set()
    for path in sorted(data_dir.glob(
            f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(path, allow_pickle=True)
        ids.update(str(row_id) for row_id in z["ids"])
    if not ids:
        raise FileNotFoundError(f"no official-native features for {tag}")
    return ids


def add_sampling_fields(frame: pd.DataFrame, data_dir: Path):
    """Keep feature-bearing rows and attach the pre-outcome strata."""
    parts = []
    for tag, block in frame.groupby("training_tag", sort=False):
        judged = pd.read_parquet(
            data_dir / f"frozen_native_{tag}_judged.parquet")
        judged = judged.dropna(subset=["adequate"])
        local = (judged.drop_duplicates("id", keep="last")
                 .set_index("id")["adequate"].astype(int).to_dict())
        valid = feature_ids(data_dir, tag)
        block = block[block["id"].isin(valid) & block["id"].isin(local)].copy()
        block["native_failure"] = [1 - local[row_id] for row_id in block["id"]]
        family = block["pool"].fillna(block.get("source"))
        block["source_family"] = family.fillna(tag).astype(str)
        block["language"] = [
            "zh" if re.search(r"[\u3400-\u9fff]", str(query)) else "en"
            for query in block["query"]
        ]
        block["selection_stratum"] = (
            block["source_family"] + "|" + block["language"] + "|y=" +
            block["native_failure"].astype(str))
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    if len(out) != 5228 and set(frame["training_tag"]) == {
            spec[2] for spec in BASE_POOLS + EXPANDED_POOLS}:
        raise RuntimeError(
            f"official full recipe reconstructed {len(out)} rows, expected 5228")
    return out


def stratified_sample(frame: pd.DataFrame, size: int, seed: int):
    if not 0 < size <= len(frame):
        raise ValueError(f"sample size {size} is outside 1..{len(frame)}")
    counts = frame["selection_stratum"].value_counts().sort_index()
    if size < len(counts):
        raise ValueError(
            f"sample size {size} cannot cover {len(counts)} strata")
    ideal = counts * (size / len(frame))
    quota = np.floor(ideal).astype(int).clip(lower=1)
    quota = pd.concat([quota, counts], axis=1).min(axis=1).astype(int)
    while int(quota.sum()) < size:
        eligible = quota.index[quota < counts]
        key = max(eligible, key=lambda k: (ideal[k] - quota[k], counts[k], k))
        quota[key] += 1
    while int(quota.sum()) > size:
        eligible = quota.index[quota > 1]
        key = min(eligible, key=lambda k: (ideal[k] - quota[k], -counts[k], k))
        quota[key] -= 1
    rng = np.random.default_rng(seed)
    selected = []
    for key, block in frame.groupby("selection_stratum", sort=True):
        selected.extend(rng.choice(
            block.index.to_numpy(), size=int(quota[key]), replace=False))
    sampled = frame.loc[selected].sort_values(
        ["training_tag", "source_family", "id"]).reset_index(drop=True)
    return sampled, counts.to_dict(), quota.to_dict()


def write_selection_manifest(path: Path, population: pd.DataFrame,
                             selected: pd.DataFrame, seed: int,
                             population_strata: dict, selected_strata: dict):
    ids_payload = "\n".join(selected["id"]).encode()
    manifest = {
        "recipe": "source_family x language x native_failure proportional; "
                  "minimum one row per stratum",
        "seed": seed,
        "population_n": int(len(population)),
        "selected_n": int(len(selected)),
        "selected_ids_sha256": hashlib.sha256(ids_payload).hexdigest(),
        "population_strata": population_strata,
        "selected_strata": selected_strata,
        "selected_by_tag": selected["training_tag"].value_counts()
        .sort_index().astype(int).to_dict(),
        "selected_by_language": selected["language"].value_counts()
        .sort_index().astype(int).to_dict(),
        "selected_by_native_failure": selected["native_failure"].value_counts()
        .sort_index().astype(int).to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def checkpoint(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


async def process_batch(escalate, batch: pd.DataFrame, args):
    """Keep both clients on one loop and give their async close tasks time."""
    answers = await escalate.ask_expert_many(
        list(batch["query"]), concurrency=args.expert_concurrency,
        effort="low", cache_dir=str(args.cache_dir))
    # OpenAI's async client schedules transport cleanup after the helper
    # returns. Yield before the loop is closed to avoid leaking connections.
    await asyncio.sleep(.1)
    generated = []
    for (_, query), answer in zip(batch.iterrows(), answers):
        generated.append({
            "id": str(query["id"]),
            "pool": query.get("pool"),
            "source": query.get("source"),
            "training_tag": query["training_tag"],
            "query": query["query"],
            "reference_answer": query.get("reference_answer"),
            "answer": answer.get("answer"),
            "latency_s": answer.get("latency_s"),
            "error": answer.get("error"),
            "prompt_tokens": answer.get("prompt_tokens"),
            "completion_tokens": answer.get("completion_tokens"),
            "expert_model": escalate.EXPERT_MODEL,
            "expert_effort": "low",
            "judge_model": escalate.JUDGE_MODEL,
            "judge_effort": escalate.JUDGE_EFFORT,
            "adequate": None,
            "judge_reason": None,
        })
    judge_input = [row for row in generated if row["answer"]]
    judged = await escalate.judge_many(
        judge_input, concurrency=args.judge_concurrency)
    await asyncio.sleep(.1)
    return generated, judged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--include-expanded", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--expert-concurrency", type=int, default=3)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src))
    import escalate

    queries = load_queries(args.data_dir, args.include_expanded)
    queries = add_sampling_fields(queries, args.data_dir)
    population = queries.copy()
    if args.sample_size:
        queries, population_strata, selected_strata = stratified_sample(
            queries, args.sample_size, args.seed)
    else:
        population_strata = (queries["selection_stratum"].value_counts()
                             .sort_index().astype(int).to_dict())
        selected_strata = population_strata
    selection_manifest = (args.selection_manifest or
                          args.output.with_suffix(".selection.json"))
    write_selection_manifest(selection_manifest, population, queries,
                             args.seed, population_strata, selected_strata)
    if args.limit:
        queries = queries.iloc[:args.limit].copy()
    previous = []
    complete = set()
    if args.output.exists():
        old = pd.read_parquet(args.output)
        previous = old.to_dict("records")
        complete = set(old.loc[
            old["adequate"].notna() & old["answer"].notna(), "id"].astype(str))
    pending = queries[~queries["id"].isin(complete)]
    print(f"expert-label recipe: total={len(queries)} complete={len(complete)} "
          f"pending={len(pending)}", flush=True)
    rows_by_id = {str(row["id"]): row for row in previous}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), args.batch_size):
        batch = pending.iloc[start:start + args.batch_size]
        generated, judged = asyncio.run(process_batch(escalate, batch, args))
        judged_by_id = {str(row["id"]): row for row in judged}
        for row in generated:
            verdict = judged_by_id.get(row["id"])
            if verdict:
                row["adequate"] = verdict.get("adequate")
                row["judge_reason"] = verdict.get("judge_reason")
            rows_by_id[row["id"]] = row
        ordered = [rows_by_id[row_id] for row_id in queries["id"]
                   if row_id in rows_by_id]
        checkpoint(args.output, ordered)
        nonnull = sum(row.get("adequate") is not None for row in ordered)
        print(f"checkpoint {len(ordered)}/{len(queries)} rows, "
              f"judged={nonnull}", flush=True)
    result = pd.read_parquet(args.output)
    judged = result["adequate"].dropna()
    errors = int(result["error"].notna().sum())
    print(f"done rows={len(result)} judged={len(judged)} errors={errors} "
          f"expert_acc={float(judged.astype(bool).mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
