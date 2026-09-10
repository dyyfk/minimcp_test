"""Measured GPT-expert dollar cost for the main-table protocol (row
"GPT expert USD/query", tab:transfer).

The frozen runs (modal_native_bench.py) discarded token usage, so this
replays each escalated call's exact recorded ASR uplink transcript through
the identical expert path — responses API, gpt-5.5, reasoning effort=low,
web_search enabled, EXPERT_SYSTEM instructions, max 8192 output tokens;
chat-completions fallback on error, as in the bench — and records billed
usage (input/cached/output incl. hidden reasoning) plus web_search call
counts. Identical transcripts are measured once and shared.

Input : data/_expert_cost_tasks.jsonl (pool, tier, id, transcript, n_pool)
Output: data/expert_cost_replay.jsonl (one row per unique transcript)

Run: PYTHONUTF8=1 modal run modal_expert_cost.py
"""
import hashlib
import json
import pathlib

import modal

app = modal.App("expert-cost-replay")
OPENAI = modal.Secret.from_name("openai")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("openai")
         .add_local_dir("src", remote_path="/root/gate"))

HERE = pathlib.Path(__file__).resolve().parent
N_SHARDS = 12


@app.function(image=image, secrets=[OPENAI], timeout=60 * 60 * 2)
def replay_shard(tasks: list[dict]) -> list[dict]:
    import sys
    import time
    sys.path.insert(0, "/root/gate")
    from escalate import (EXPERT_MODEL, EXPERT_MAX_TOKENS, EXPERT_SYSTEM,
                          USER_ID, _client)
    client = _client()
    out = []
    for t in tasks:
        rec = {"sha": t["sha"], "error": None, "path": "web",
               "input_tokens": None, "cached_tokens": 0,
               "output_tokens": None, "reasoning_tokens": None,
               "web_calls": 0, "answer_chars": 0}
        for attempt in range(4):
            try:
                resp = client.responses.create(
                    model=EXPERT_MODEL, reasoning={"effort": "low"},
                    tools=[{"type": "web_search"}],
                    instructions=EXPERT_SYSTEM, input=t["transcript"],
                    max_output_tokens=EXPERT_MAX_TOKENS, user=USER_ID)
                u = resp.usage
                rec.update(
                    input_tokens=u.input_tokens,
                    cached_tokens=getattr(u.input_tokens_details,
                                          "cached_tokens", 0) or 0,
                    output_tokens=u.output_tokens,
                    reasoning_tokens=getattr(u.output_tokens_details,
                                             "reasoning_tokens", None),
                    web_calls=sum(1 for it in resp.output
                                  if it.type == "web_search_call"),
                    answer_chars=len(resp.output_text or ""), error=None)
                break
            except Exception as e:
                rec["error"] = str(e)[:200]
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(5 * (attempt + 1))
                    continue
                # bench fallback: tool-free chat completions
                try:
                    resp = client.chat.completions.create(
                        model=EXPERT_MODEL, reasoning_effort="low",
                        max_completion_tokens=EXPERT_MAX_TOKENS,
                        messages=[{"role": "system", "content": EXPERT_SYSTEM},
                                  {"role": "user", "content": t["transcript"]}],
                        user=USER_ID)
                    u = resp.usage
                    rec.update(path="chat_fallback",
                               input_tokens=u.prompt_tokens,
                               output_tokens=u.completion_tokens,
                               answer_chars=len(
                                   resp.choices[0].message.content or ""),
                               error=None)
                except Exception as e2:
                    rec["error"] = str(e2)[:200]
                break
        out.append(rec)
    return out


@app.local_entrypoint()
def main(limit: int = 0, retry: bool = False):
    tasks_path = HERE / "data/_expert_cost_tasks.jsonl"
    rows = [json.loads(x) for x in
            tasks_path.read_text(encoding="utf-8").splitlines()]
    seen, uniq = set(), []
    for r in rows:
        tr = r["transcript"]
        if not tr:
            continue
        sha = hashlib.sha256(tr.encode()).hexdigest()[:16]
        if sha in seen:
            continue
        seen.add(sha)
        uniq.append({"sha": sha, "transcript": tr})
    if limit:
        uniq = uniq[:limit]
    out_path = HERE / "data/expert_cost_replay.jsonl"
    prior = []
    if retry:
        prior = [json.loads(x) for x in
                 out_path.read_text(encoding="utf-8").splitlines()]
        failed = {r["sha"] for r in prior if r["error"]}
        prior = [r for r in prior if not r["error"]]
        uniq = [u for u in uniq if u["sha"] in failed]
        print(f">>> retrying {len(uniq)} failed calls")
    print(f">>> {len(rows)} escalations, {len(uniq)} unique transcripts")
    n_shards = 2 if retry else N_SHARDS
    shards = [uniq[i::n_shards] for i in range(n_shards)]
    results = []
    for shard_out in replay_shard.map(shards):
        results.extend(shard_out)
        print(f">>> {len(results)}/{len(uniq)} done")
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in prior + results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    allrows = prior + results
    errs = [r for r in allrows if r["error"]]
    print(f">>> wrote {out_path} ({len(allrows)} rows, {len(errs)} errors)")
