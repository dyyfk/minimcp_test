"""Generate resumable expert outcomes for the frozen P25-B expansion."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd


def checkpoint(path: Path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


async def process(escalate, batch: pd.DataFrame, args):
    answers = await escalate.ask_expert_many(
        list(batch["query"]), concurrency=args.expert_concurrency,
        effort="low", cache_dir=str(args.cache_dir))
    await asyncio.sleep(.1)
    generated = []
    for (_, source), answer in zip(batch.iterrows(), answers):
        generated.append({
            "id": str(source["id"]), "pool": str(source["pool"]),
            "source_family": str(source["source_family"]),
            "language": str(source["language"]), "mode": str(source["mode"]),
            "query": str(source["query"]),
            "reference_answer": str(source["reference_answer"]),
            "answer": answer.get("answer"),
            "latency_s": answer.get("latency_s"),
            "error": answer.get("error"),
            "expert_prompt_tokens": answer.get("prompt_tokens"),
            "expert_completion_tokens": answer.get("completion_tokens"),
            "expert_model": escalate.EXPERT_MODEL,
            "expert_effort": "low", "adequate": None,
            "judge_reason": None,
        })
    judged_input = [row for row in generated if row["answer"]]
    judged = await escalate.judge_many(
        judged_input, concurrency=args.judge_concurrency)
    await asyncio.sleep(.1)
    return generated, {str(row["id"]): row for row in judged}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--expert-concurrency", type=int, default=5)
    parser.add_argument("--judge-concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    selection = pd.read_parquet(args.selection).sort_values("id")
    if args.limit:
        selection = selection.head(args.limit)
    old = (pd.read_parquet(args.output) if args.output.exists()
           else pd.DataFrame(columns=["id"]))
    complete = set(old.loc[
        old.get("adequate", pd.Series(index=old.index, dtype=object)).notna(),
        "id"].astype(str))
    pending = selection[~selection.id.astype(str).isin(complete)]
    rows = {str(row["id"]): row for row in old.to_dict("records")}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"expert expansion total={len(selection)} complete={len(complete)} "
          f"pending={len(pending)}", flush=True)
    for offset in range(0, len(pending), args.batch_size):
        batch = pending.iloc[offset:offset + args.batch_size]
        generated, judged = asyncio.run(process(escalate, batch, args))
        for row in generated:
            verdict = judged.get(row["id"])
            if verdict:
                row["adequate"] = verdict.get("adequate")
                row["judge_reason"] = verdict.get("judge_reason")
                row["judge_prompt_tokens"] = verdict.get("judge_prompt_tokens")
                row["judge_completion_tokens"] = verdict.get(
                    "judge_completion_tokens")
                row["judge_cached_prompt_tokens"] = verdict.get(
                    "judge_cached_prompt_tokens")
            rows[row["id"]] = row
        ordered = [rows[str(row_id)] for row_id in selection.id
                   if str(row_id) in rows]
        checkpoint(args.output, ordered)
        judged_count = sum(row.get("adequate") is not None for row in ordered)
        print(f"checkpoint {len(ordered)}/{len(selection)} "
              f"judged={judged_count}", flush=True)
    result = pd.read_parquet(args.output)
    usage_columns = [
        "expert_prompt_tokens", "expert_completion_tokens",
        "judge_prompt_tokens", "judge_completion_tokens",
        "judge_cached_prompt_tokens",
    ]
    usage = result.reindex(columns=usage_columns)
    usage = usage.fillna(0).sum().astype(int).to_dict()
    print(json.dumps({"rows": len(result),
                      "judged": int(result.adequate.notna().sum()),
                      "errors": int(result.error.notna().sum()),
                      "expert_adequate_rate": float(result.adequate.mean()),
                      **usage}, indent=2), flush=True)


if __name__ == "__main__":
    main()
