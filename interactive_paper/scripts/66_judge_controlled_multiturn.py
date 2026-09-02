"""Judge target answers after their frozen completed carrier turns."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd


def write_atomic(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    pairs = pd.read_parquet(args.pairs).drop_duplicates("id")
    metadata = pairs.set_index("id").to_dict("index")
    traces = {}
    for path in sorted(args.trace_dir.glob("controlled_multiturn.rank*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                traces[str(record["id"])] = record
    old = (pd.read_parquet(args.output) if args.output.exists()
           else pd.DataFrame(columns=["id"]))
    old = old.drop_duplicates("id", keep="last")
    have = set(old.loc[old.get("adequate", pd.Series(dtype=object)).notna(),
                       "id"])
    todo = []
    for row_id in sorted(metadata):
        if row_id in have:
            continue
        trace = traces.get(row_id)
        if trace is None:
            raise RuntimeError(f"missing or failed trace for {row_id}")
        error = trace.get("error")
        if error and error != "RuntimeError: target never reached speak onset":
            raise RuntimeError(f"failed trace for {row_id}: {error}")
        row = metadata[row_id]
        todo.append({
            "id": row_id,
            "target_id": row["target_id"],
            "pool": row["target_pool"],
            "query": row["target_query"],
            "reference_answer": row["target_reference_answer"],
            "answer": trace.get("target_answer") or "",
            "no_speak": not bool(trace.get("target_answer")),
            "eot_seen": bool(trace.get("target_eot_seen")),
            "native_no_speak": bool(error),
        })
    print(f"{len(traces)} traces; {len(have)} judged; {len(todo)} pending",
          flush=True)

    async def judge_batch(rows):
        client = escalate._async_client()
        semaphore = asyncio.Semaphore(args.concurrency)

        async def one(row):
            async with semaphore:
                try:
                    response = await client.chat.completions.create(
                        model=escalate.JUDGE_MODEL,
                        reasoning_effort=escalate.JUDGE_EFFORT,
                        max_completion_tokens=escalate.JUDGE_MAX_TOKENS,
                        response_format=escalate._resp_format(
                            "verdict", escalate._JUDGE_SCHEMA),
                        messages=[
                            {"role": "system", "content": escalate.JUDGE_SYSTEM},
                            {"role": "user", "content": escalate._judge_user(
                                row["query"], row["reference_answer"],
                                row["answer"])},
                        ], user=escalate.USER_ID)
                    verdict = json.loads(escalate._content(response))
                    row["adequate"] = bool(verdict["adequate"])
                    row["judge_reason"] = verdict.get("reason", "")
                    usage = response.usage
                    row["prompt_tokens"] = int(usage.prompt_tokens)
                    row["completion_tokens"] = int(usage.completion_tokens)
                    row["cached_prompt_tokens"] = int(getattr(
                        getattr(usage, "prompt_tokens_details", None),
                        "cached_tokens", 0) or 0)
                    row["judge_error"] = None
                except Exception as exc:
                    row.update(adequate=None, judge_reason="",
                               prompt_tokens=None, completion_tokens=None,
                               cached_prompt_tokens=None,
                               judge_error=f"{type(exc).__name__}: {exc}")
                row["escalate_label"] = (None if row["adequate"] is None else
                                         int(not row["adequate"]))
                return row

        try:
            return await asyncio.gather(*(one(dict(row)) for row in rows))
        finally:
            await client.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    accumulated = old.to_dict("records")
    for offset in range(0, len(todo), args.batch_size):
        batch = todo[offset:offset + args.batch_size]
        judged = asyncio.run(judge_batch(batch))
        for _ in range(2):
            failed = [row for row in judged if row["adequate"] is None]
            if not failed:
                break
            replacements = {row["id"]: row for row in
                            asyncio.run(judge_batch(failed))}
            judged = [replacements.get(row["id"], row) for row in judged]
        accumulated.extend(judged)
        result = pd.DataFrame(accumulated).drop_duplicates("id", keep="last")
        write_atomic(result, args.output)
        print(f"checkpoint {min(offset + len(batch), len(todo))}/{len(todo)}; "
              f"errors={int(result.adequate.isna().sum())}", flush=True)
    result = pd.read_parquet(args.output)
    usage = result[["prompt_tokens", "completion_tokens",
                    "cached_prompt_tokens"]].fillna(0).sum().astype(int)
    print(json.dumps({"rows": len(result),
                      "errors": int(result.adequate.isna().sum()),
                      "adequate_rate": float(result.adequate.mean()),
                      **usage.to_dict()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
