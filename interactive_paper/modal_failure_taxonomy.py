"""8bv: classify every native never-arm FAILURE by type, so the probe's lift
can be reported per failure class (perception / knowledge gap / confident
wrong / execution / quality-other).

Inputs (gate-data volume): native_bench/{pool}_never_judged.parquet (local
answer + verdict + onset score) and native_bench/{pool}_always_tts_judged.parquet
(gpt-transcribe transcript of the same audio, expert outcome).
Output: native_bench/failure_types.parquet (id, pool, ftype, rationale).

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_failure_taxonomy.py::classify
"""
import json
import os

import modal

DATA = "/data"
NB = f"{DATA}/native_bench"
from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("failure-taxonomy")
gate_data = modal.Volume.from_name("gate-data")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("pandas", "pyarrow", "openai", "pydantic>=2.11")
       .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
       .add_local_file(os.path.join(_HERE, "modal_app.py"), "/root/modal_app.py"))

POOLS = {"frozen": "adequate", "striviaqa": "oab_ok", "swebq": "oab_ok",
         "sllama": "oab_ok", "sdqa": "adequate", "sreason": "adequate"}

TYPES = ["perception", "knowledge_gap", "confident_wrong", "execution", "quality_other"]
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"ftype": {"type": "string", "enum": TYPES},
                         "rationale": {"type": "string"}},
          "required": ["ftype", "rationale"]}
PROMPT = """You are auditing a small speech assistant that answered a spoken question incorrectly. Classify WHY it failed into exactly one type:

- perception: it misheard or misparsed the spoken question (answers a different question than the one asked, wrong entity from mishearing, garbled key term). Compare the question text with the ASR transcript and with what the answer is actually about.
- knowledge_gap: it heard the question right but does not know the answer: it hedges, says it is unsure, gives a vague or generic non-answer, or declines.
- confident_wrong: it heard the question right and states a specific, confident answer that is simply wrong (misremembered fact, wrong name/date/number stated as fact, no hedging).
- execution: it heard the question right and knows the method, but slips while carrying out multi-step reasoning, arithmetic, counting, ordering, or logic (the error is in the working, not in the recall).
- quality_other: incomplete/cut-off answer, refuses for format reasons, answers in the wrong language, or any failure that fits none of the above.

Question (ground-truth text): {query}
What a transcriber heard from the same audio: {transcript}
Reference answer: {reference}
The assistant's spoken answer: {answer}
Judge's note (if any): {reason}

Respond with the type and a one-sentence rationale."""


@app.function(image=img, volumes={DATA: gate_data}, secrets=[OPENAI], timeout=60 * 60)
def classify(limit: int = 0) -> int:
    import asyncio
    import sys

    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    rows = []
    for pool, col in POOLS.items():
        n = pd.read_parquet(f"{NB}/{pool}_never_judged.parquet").drop_duplicates("id", keep="last")
        a = pd.read_parquet(f"{NB}/{pool}_always_tts_judged.parquet").drop_duplicates("id", keep="last").set_index("id")
        f = n[n[col] == 0]
        for _, r in f.iterrows():
            tr = a["transcript"].get(r["id"], "") if r["id"] in a.index else ""
            rows.append({"id": r["id"], "pool": pool, "query": r["query"],
                         "transcript": tr or "(unavailable)",
                         "reference": str(r["reference_answer"])[:400],
                         "answer": (r["answer"] or "(empty)")[:600],
                         "reason": str(r.get("judge_reason") or "")[:300]})
    out_p = f"{NB}/failure_types.parquet"
    old = pd.read_parquet(out_p) if os.path.exists(out_p) else pd.DataFrame(columns=["id", "pool"])
    have = set(zip(old["id"], old["pool"]))
    todo = [r for r in rows if (r["id"], r["pool"]) not in have]
    if limit:
        todo = todo[:limit]
    print(f">>> {len(rows)} failures, {len(todo)} to classify", flush=True)
    if not todo:
        return 0
    client = escalate._async_client()
    sem = asyncio.Semaphore(8)

    async def one(r):
        p = PROMPT.format(query=r["query"], transcript=r["transcript"], reference=r["reference"],
                          answer=r["answer"], reason=r["reason"] or "none")
        r["ftype"], r["rationale"] = None, None
        for attempt in range(5):
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=escalate.JUDGE_MODEL, reasoning_effort="low",
                        max_completion_tokens=400,
                        messages=[{"role": "user", "content": p}],
                        response_format=escalate._resp_format("ftype", SCHEMA),
                        user=escalate.USER_ID)
                    d = json.loads(resp.choices[0].message.content or "{}")
                    r["ftype"], r["rationale"] = d.get("ftype"), d.get("rationale")
                    return
                except Exception as e:
                    r["rationale"] = f"error: {str(e)[:100]}"
            await asyncio.sleep(min(30, 2 ** attempt))

    async def run():
        await asyncio.gather(*(one(r) for r in todo))
    asyncio.run(run())
    new = pd.concat([old, pd.DataFrame(todo)], ignore_index=True)
    new.to_parquet(out_p)
    gate_data.commit()
    ok = new[new["ftype"].notna()]
    print(">>> classified", len(ok), "/", len(new), flush=True)
    print(ok.groupby(["pool", "ftype"]).size().unstack(fill_value=0).to_string(), flush=True)
    return len(todo)
