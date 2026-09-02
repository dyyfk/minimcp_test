"""Judge existing official-native training answers without GPU volumes.

This standalone Modal app intentionally depends only on ``gate-data`` and the
``openai`` secret. It does not import the model dump app or resolve the
``minicpm-o45-weights`` volume.

Usage:

    modal run modal_native_relabel.py::judge_training_official
"""
import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data"
OUT = f"{DATA}/frozen_native"

app = modal.App("native-training-relabel")
gate_data = modal.Volume.from_name("gate-data")
openai_secret = modal.Secret.from_name("openai")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("openai", "pandas", "pyarrow")
         .add_local_dir(os.path.join(HERE, "src"), "/workspace/gate"))

QUERY_FILES = {
    "frozen": f"{DATA}/queries.jsonl",
    "expansion": f"{DATA}/queries_expansion.jsonl",
    "expansion2": f"{DATA}/queries_expansion2.jsonl",
    "expansion3": f"{DATA}/queries_expansion3.jsonl",
    "expansion3zh": f"{DATA}/queries_expansion3zh.jsonl",
    "fresh": f"{DATA}/queries_fresh.jsonl",
}
TARGETS = [
    ("caliboff", "frozen"),
    ("expoff", "expansion"),
    ("exp2off", "expansion2"),
    ("exp3off", "expansion3"),
    ("exp3zhoff", "expansion3zh"),
    ("freshoff", "fresh"),
]


@app.function(image=image, volumes={DATA: gate_data},
              secrets=[openai_secret], timeout=60 * 60)
def judge_native(tag: str, pool: str):
    """Judge one native trace family; existing IDs are skipped."""
    import asyncio
    import glob

    import pandas as pd

    sys.path.insert(0, "/workspace/gate")
    import escalate

    queries = {row["id"]: row for row in
               (json.loads(line) for line in open(
                   QUERY_FILES[pool], encoding="utf-8") if line.strip())}
    traces = {}
    for path in sorted(glob.glob(f"{OUT}_{tag}_traces.jsonl.shard*")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                traces[row["id"]] = row
    if not traces:
        raise FileNotFoundError(f"no official-native traces found for {tag}")

    output = f"{OUT}_{tag}_judged.parquet"
    old = (pd.read_parquet(output) if os.path.exists(output)
           else pd.DataFrame(columns=["id"]))
    have = set(old["id"])
    todo = []
    for trace in traces.values():
        row_id = trace["id"]
        if row_id in have or row_id not in queries:
            continue
        query = queries[row_id]
        todo.append({
            "id": row_id,
            "query": query["query"],
            "reference_answer": query.get("reference_answer"),
            "answer": trace.get("answer_text") or "",
            "no_speak": trace.get("no_speak"),
            "eot_seen": trace.get("eot_seen"),
        })
    print(f">>> {tag}: traces={len(traces)}, existing={len(have)}, "
          f"todo={len(todo)}")
    if not todo:
        return {"tag": tag, "traces": len(traces), "new": 0,
                "parsed": 0}

    judged = asyncio.run(escalate.judge_many(todo, concurrency=8))
    parsed = [row for row in judged if row["adequate"] is not None]
    new = pd.concat([old, pd.DataFrame(judged)], ignore_index=True)
    new = new.drop_duplicates("id", keep="last")
    new.to_parquet(output)
    gate_data.commit()
    print(f">>> {tag}: wrote {len(judged)} rows, "
          f"parsed={len(parsed)}/{len(judged)}")
    return {"tag": tag, "traces": len(traces), "new": len(judged),
            "parsed": len(parsed)}


@app.local_entrypoint()
def judge_training_official():
    calls = [(tag, pool, judge_native.spawn(tag=tag, pool=pool))
             for tag, pool in TARGETS]
    receipts = [call.get() for _tag, _pool, call in calls]
    print(json.dumps(receipts, indent=1))
