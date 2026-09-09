"""Locally judge pass-3 NVDA answers on the 240-query internal test set.

The API key is read from this process's ``OPENAI_API_KEY`` environment
variable. It is not written to disk or included in model prompts.

Run from ``interactive_paper``:

    read -s "OPENAI_API_KEY?OpenAI API key: "; echo
    OPENAI_API_KEY="$OPENAI_API_KEY" \
      uv run --with openai python judge_nvda_internal.py
    unset OPENAI_API_KEY
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE_PULL = HERE / "data" / "gate_pull"
PROBE_DOC = GATE_PULL / "new" / "probe_doc"
OUTPUT = PROBE_DOC / "internal_pass3_judged.json"


def input_rows() -> list[dict]:
    queries = {}
    with (GATE_PULL / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "test":
                queries[str(row["id"])] = row

    answers = {}
    pattern = GATE_PULL / "onset3" / "nvda_answers_frozen.shard*.jsonl"
    for filename in sorted(glob.glob(str(pattern))):
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                query_id = str(row["id"])
                if query_id in queries:
                    if query_id in answers:
                        raise ValueError(f"duplicate pass-3 answer: {query_id}")
                    answers[query_id] = row

    if set(answers) != set(queries):
        missing = sorted(set(queries) - set(answers))
        extra = sorted(set(answers) - set(queries))
        raise ValueError(f"answer coverage mismatch: missing={missing}, extra={extra}")

    return [
        {
            "id": query_id,
            "query": query["query"],
            "reference_answer": query.get("reference_answer"),
            "answer": answers[query_id]["answer"],
            "onset_frame": int(answers[query_id]["onset_frame"]),
        }
        for query_id, query in sorted(queries.items())
    ]


async def judge(rows: list[dict]) -> list[dict]:
    sys.path.insert(0, str(HERE / "src"))
    import escalate

    completed = []
    pending = rows
    for attempt in range(6):
        judged = await escalate.judge_many(pending, concurrency=4)
        completed.extend(row for row in judged if row["adequate"] is not None)
        pending = [row for row in judged if row["adequate"] is None]
        if not pending:
            break
        print(f"judge pass {attempt + 1}: retrying {len(pending)} rows")
        await asyncio.sleep(2**attempt)
    if pending:
        errors = {row["id"]: row["judge_reason"] for row in pending}
        raise RuntimeError(f"judge failed after retries: {errors}")
    return sorted(completed, key=lambda row: row["id"])


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    rows = input_rows()
    print(f"judging {len(rows)} pass-3 internal answers")
    result = asyncio.run(judge(rows))
    payload = {
        "protocol": {
            "model": "gpt-5.4-mini",
            "source": "pass-3 nvda_replay_v2.py answers",
            "split": "queries.jsonl::test",
            "n": len(result),
            "no_answer_onset": sum(row["onset_frame"] < 0 for row in result),
        },
        "rows": result,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
