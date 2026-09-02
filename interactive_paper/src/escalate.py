"""OpenAI API roles for the gate: the judge (Phase 2 labels).

  * judge — gpt-5.4-mini grades the small model's answer as adequate / not
            (produces the Phase-2 escalate labels).

(Query GENERATION was removed 2026-07-08: all five pools now come from public
datasets — the GPT-generated trap pool failed its design and generator==judge
is a circularity. See modal_app.build_public_queries.)

The escalation target itself (gpt-5.5) lives in Phase 4 (src/inject etc.).

Provider switched from Anthropic → OpenAI on 2026-07-08 (user has OpenAI
credit, not Anthropic). Model ids from the OpenAI pricing/models pages
(July 2026): judge/gen = gpt-5.4-mini, escalation target = gpt-5.5. These are
GPT-5.x *reasoning* models — hidden reasoning tokens bill as output, so calls
use reasoning_effort="low" (schema adherence without heavy spend) and a
generous max_completion_tokens (reasoning eats the budget before the visible
JSON). All calls use Structured Outputs (response_format json_schema, strict)
for parse-free reliability. The client reads OPENAI_API_KEY from the env
(injected by the Modal `openai` secret). If an id 404s, pin a dated snapshot.

ANTI-FLAG MEASURES (2026-07-24): the account was temporarily suspended for
suspected "model distillation" — the ask_expert_many pattern (batch benchmark
questions → full gpt-5.5 answers) matches OpenAI's harvesting heuristics even
though no expert output ever trains a model here. Hence: (1) every call sends
user=USER_ID so the workload is attributable; (2) EXPERT_SYSTEM declares the
research purpose; (3) ask_expert* take cache_dir — cached answers are returned
verbatim (original latency/usage preserved for the 6c latency stats) so no
query is ever sent to the expert twice; (4) expert concurrency default is 3.
Do NOT bulk re-collect expert answers for pools that already have results.
"""
from __future__ import annotations

JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_EFFORT = "low"        # reasoning_effort for GPT-5.x judge calls
JUDGE_MAX_TOKENS = 4096     # room for reasoning tokens + the small JSON verdict

# Escalation target (Phase 4/5): the big cloud model the gate escalates to.
EXPERT_MODEL = "gpt-5.5"
EXPERT_EFFORT = "medium"    # reasoning depth for the expert answer
EXPERT_MAX_TOKENS = 8192    # generous: hidden reasoning bills as output and can
                           # eat the budget before the visible answer (a 4096 cap
                           # hit finish_reason=length → empty content on hard Qs)
EXPERT_SYSTEM = (
    "You are an expert assistant. Answer the user's question correctly, directly, "
    "and concisely. For a multiple-choice question, state the correct option "
    "letter and its content. (Context: this is a single-turn escalation from an "
    "academic research prototype of a small-to-large model routing gate; your "
    "answer is relayed to the user and used for system evaluation, not for "
    "model training.)"
)

# Stable workload identifier sent as `user` on every call (OpenAI's recommended
# attribution signal) — part of the anti-distillation-flag measures above.
USER_ID = "think-gate-escalation-eval"

# Fixed judge prompt — stored in repo for reproducibility (PLAN Phase 2.2).
JUDGE_SYSTEM = (
    "You are a strict, fair grader of a small language model's answers. "
    "You are given a QUESTION, an optional REFERENCE answer, and a CANDIDATE "
    "answer produced by a small model. Decide whether the CANDIDATE is ADEQUATE, "
    "meaning ALL of: (1) factually correct; (2) any reasoning shown is valid; "
    "(3) it actually and completely answers the question that was asked. "
    "For multiple-choice questions the candidate must land on the correct option "
    "(by letter or by unambiguously stating the option's content). A partially "
    "correct, hedged, refused, off-topic, or truncated answer is NOT adequate. "
    "If a REFERENCE is provided, grade against it. Be strict but do not penalize "
    "harmless differences in format or extra correct detail."
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string",
                   "description": "one sentence justifying the verdict"},
        "adequate": {"type": "boolean"},
    },
    "required": ["reason", "adequate"],
    "additionalProperties": False,
}


def _resp_format(name: str, schema: dict) -> dict:
    """OpenAI Structured Outputs wrapper (strict json_schema)."""
    return {"type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True}}


def _client():
    from openai import OpenAI
    return OpenAI()


def _async_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI()


def _content(resp) -> str:
    """Pull the JSON string from a chat completion, or raise on refusal/truncation."""
    msg = resp.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(f"model refused: {msg.refusal}")
    if not msg.content:
        raise RuntimeError(
            f"empty content (finish_reason={resp.choices[0].finish_reason}); "
            "reasoning likely consumed max_completion_tokens")
    return msg.content


def _judge_user(query: str, reference: str | None, answer: str) -> str:
    ref = f"\n\nREFERENCE answer:\n{reference}" if reference else ""
    return (f"QUESTION:\n{query}{ref}\n\nCANDIDATE answer (small model):\n{answer}\n\n"
            "Grade the candidate.")


def judge(query: str, reference: str | None, answer: str) -> dict:
    """Synchronous single judge call → {'adequate': bool, 'reason': str}."""
    import json
    resp = _client().chat.completions.create(
        model=JUDGE_MODEL, reasoning_effort=JUDGE_EFFORT,
        max_completion_tokens=JUDGE_MAX_TOKENS,
        response_format=_resp_format("verdict", _JUDGE_SCHEMA),
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user",
                   "content": _judge_user(query, reference, answer)}],
        user=USER_ID,
    )
    return json.loads(_content(resp))


async def judge_many(rows: list[dict], concurrency: int = 8) -> list[dict]:
    """Label a list of {query, reference_answer, answer} rows concurrently.

    Returns the same rows with 'adequate', 'judge_reason', 'escalate_label' added.
    """
    import asyncio
    import json

    client = _async_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL, reasoning_effort=JUDGE_EFFORT,
                    max_completion_tokens=JUDGE_MAX_TOKENS,
                    response_format=_resp_format("verdict", _JUDGE_SCHEMA),
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": _judge_user(
                                  row["query"], row.get("reference_answer"),
                                  row["answer"])}],
                    user=USER_ID,
                )
                v = json.loads(_content(resp))
                row["adequate"] = bool(v["adequate"])
                row["judge_reason"] = v.get("reason", "")
            except Exception as e:  # keep the row; mark unjudged
                row["adequate"] = None
                row["judge_reason"] = f"ERROR: {e}"
            row["escalate_label"] = (None if row["adequate"] is None
                                     else int(not row["adequate"]))
            return row

    try:
        return await asyncio.gather(*(one(r) for r in rows))
    finally:
        await client.close()


# ============================ escalation target (gpt-5.5) ==================
def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {"prompt_tokens": None, "completion_tokens": None}
    return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}


def _cache_path(cache_dir: str, query: str, effort: str) -> str:
    """One JSON file per (model, effort, system prompt, query) — a prompt change
    correctly invalidates."""
    import hashlib
    h = hashlib.sha256(
        f"{EXPERT_MODEL}|{effort}|{EXPERT_SYSTEM}|{query}".encode()).hexdigest()
    return f"{cache_dir}/{h}.json"


def _cache_get(cache_dir: str | None, query: str, effort: str) -> dict | None:
    import json
    import os
    if not cache_dir:
        return None
    p = _cache_path(cache_dir, query, effort)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _cache_put(cache_dir: str | None, query: str, effort: str, result: dict):
    """Cache successful results only — the stored dict (original latency_s /
    token usage) is returned verbatim on a hit, so latency stats stay honest."""
    import json
    import os
    if not cache_dir or result.get("error"):
        return
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, query, effort), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)


def ask_expert(query: str, effort: str = EXPERT_EFFORT,
               cache_dir: str | None = None) -> dict:
    """Synchronous single escalation to gpt-5.5 → {answer, latency_s, usage...}.

    Never raises: a failure returns answer=None + error=str (matches
    ask_expert_many) so one bad call can't crash a demo or eval loop.
    With cache_dir, a previously answered query never re-hits the API.
    """
    import time
    hit = _cache_get(cache_dir, query, effort)
    if hit is not None:
        return hit
    t0 = time.time()
    try:
        resp = _client().chat.completions.create(
            model=EXPERT_MODEL, reasoning_effort=effort,
            max_completion_tokens=EXPERT_MAX_TOKENS,
            messages=[{"role": "system", "content": EXPERT_SYSTEM},
                      {"role": "user", "content": query}],
            user=USER_ID,
        )
        out = {"answer": _content(resp), "latency_s": time.time() - t0,
               "error": None, **_usage(resp)}
    except Exception as e:
        out = {"answer": None, "latency_s": time.time() - t0, "error": str(e),
               "prompt_tokens": None, "completion_tokens": None}
    _cache_put(cache_dir, query, effort, out)
    return out


def ask_expert_web(query: str, effort: str = EXPERT_EFFORT) -> dict:
    """DEMO-ONLY expert with OpenAI web_search enabled (user request
    2026-08-24: escalated real-time queries were relaying a refusal —
    the no-tools expert cannot know live values). The measured eval arms
    keep the tool-free ask_expert + its caches; nothing cached here.
    Never raises: failure returns answer=None + error, caller falls back
    to ask_expert."""
    import time
    t0 = time.time()
    try:
        resp = _client().responses.create(
            model=EXPERT_MODEL,
            reasoning={"effort": effort},
            tools=[{"type": "web_search"}],
            instructions=EXPERT_SYSTEM,
            input=query,
            max_output_tokens=EXPERT_MAX_TOKENS,
        )
        ans = (resp.output_text or "").strip()
        out = {"answer": ans or None, "latency_s": time.time() - t0,
               "error": None if ans else "empty output_text", "web": True}
    except Exception as e:
        out = {"answer": None, "latency_s": time.time() - t0,
               "error": str(e), "web": False}
    return out


async def ask_expert_many(queries: list[str], concurrency: int = 3,
                          effort: str = EXPERT_EFFORT,
                          cache_dir: str | None = None) -> list[dict]:
    """Escalate a list of queries to gpt-5.5 concurrently.

    Each result: {answer|None, latency_s, error|None, prompt_tokens, completion_tokens}.
    Failures are captured per-row (answer=None, error=str) so one bad call doesn't
    sink the batch. With cache_dir, cached queries are served locally and only
    misses reach the API (concurrency default 3 — see anti-flag notes up top).
    """
    import asyncio
    import time

    client = _async_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(q):
        hit = _cache_get(cache_dir, q, effort)
        if hit is not None:
            return hit
        async with sem:
            t0 = time.time()
            try:
                resp = await client.chat.completions.create(
                    model=EXPERT_MODEL, reasoning_effort=effort,
                    max_completion_tokens=EXPERT_MAX_TOKENS,
                    messages=[{"role": "system", "content": EXPERT_SYSTEM},
                              {"role": "user", "content": q}],
                    user=USER_ID,
                )
                out = {"answer": _content(resp), "latency_s": time.time() - t0,
                       "error": None, **_usage(resp)}
            except Exception as e:
                out = {"answer": None, "latency_s": time.time() - t0,
                       "error": str(e), "prompt_tokens": None,
                       "completion_tokens": None}
            _cache_put(cache_dir, q, effort, out)
            return out

    try:
        return await asyncio.gather(*(one(q) for q in queries))
    finally:
        await client.close()
