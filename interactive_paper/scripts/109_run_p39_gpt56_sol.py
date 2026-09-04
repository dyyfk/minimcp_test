"""Run the frozen P39 gpt-5.5-low versus gpt-5.6-sol-low comparison."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd


CANDIDATE_MODEL = "gpt-5.6-sol"
OURS_JUDGE_MODEL = "gpt-5.4-mini"
OAB_JUDGE_MODEL = "gpt-4o-2024-08-06"
CANDIDATE_MAX = 4096
OURS_MAX = 2048
OAB_MAX = 512
COST_CAP = 15.0
PRICES = {
    CANDIDATE_MODEL: (4.0 / 1e6, 20.0 / 1e6),
    OURS_JUDGE_MODEL: (.75 / 1e6, 4.5 / 1e6),
    OAB_JUDGE_MODEL: (2.5 / 1e6, 10.0 / 1e6),
}
JUDGE_KIND = {"frozen": "ours", "sdqa": "ours", "sreason": "ours",
              "striviaqa": "oab", "swebq": "oab", "sllama": "oab"}
OAB_PATTERN = """
Your will be given a question, the reference answers to that question, and an answer to be judged. Your tasks is to judge whether the answer to be judged is correct, given the question and reference answers. An answer considered correct expresses or contains the same meaning as at least **one of** the reference answers. The format and the tone of the response does not matter.

You should respond in JSON format. First provide a one-sentence concise analysis for the judgement in field 'analysis', then your judgment in field 'judgment'. For example,
'''json
{{"analysis": "<a one-sentence concise analysis for the judgement>", "judgment": < your final judgment, "correct" or "incorrect">}}
'''

# Question
{instruction}

# Reference Answer
{targets}

# Answer To Be Judged
{answer}

"""


def read_jsonl(pattern):
    rows = []
    for path in sorted(glob.glob(str(pattern))):
        rows.extend(json.loads(line) for line in open(path, encoding="utf-8")
                    if line.strip())
    return rows


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def usage_cost(model, prompt_tokens, completion_tokens):
    input_price, output_price = PRICES[model]
    return prompt_tokens * input_price + completion_tokens * output_price


def total_cost(*groups):
    return sum(float(row.get("cost_usd") or 0) for group in groups for row in group
               if not row.get("error"))


def population(args):
    p38 = pd.read_parquet(args.p38_rows)
    sources = []
    for pool in JUDGE_KIND:
        source = pd.read_parquet(
            args.data_dir / f"{pool}_always_tts_judged.parquet"
        ).drop_duplicates("id", keep="last")
        source = source[["id", "query", "reference_answer"]].copy()
        source["pool"] = pool
        sources.append(source)
    source = pd.concat(sources, ignore_index=True)
    frame = p38.merge(source, on=["pool", "id"], how="left", validate="one_to_one")
    if frame["query"].isna().any():
        raise SystemExit("missing source question")
    frame["reference"] = frame["reference_answer"].where(
        frame["reference_answer"].notna(), None)
    frame["judge_kind"] = frame["pool"].map(JUDGE_KIND)
    frame = frame.sort_values(["pool", "id"]).reset_index(drop=True)
    receipt = "\n".join(f"{r.pool}:{r.id}" for r in frame.itertuples())
    digest = hashlib.sha256(receipt.encode()).hexdigest()
    if len(frame) != 292 or digest != args.population_sha:
        raise SystemExit(f"population mismatch: n={len(frame)} sha={digest}")
    return frame


def candidate_rows(output_dir):
    return read_jsonl(output_dir / "p39_candidate*.jsonl")


def judge_rows(output_dir):
    return read_jsonl(output_dir / "p39_judge*.jsonl")


def candidate_phase(args, frame):
    from openai import OpenAI

    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    output = args.output_dir / f"p39_candidate.rank{args.shard_index}.jsonl"
    old = candidate_rows(args.output_dir)
    done = {str(row["id"]) for row in old if not row.get("error")}
    client = OpenAI()
    for row in frame.iloc[args.shard_index::args.shard_count].itertuples(index=False):
        if row.id in done:
            continue
        record = {"phase": "candidate", "id": row.id, "pool": row.pool,
                  "model": CANDIDATE_MODEL, "reasoning_effort": "low",
                  "error": None}
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=CANDIDATE_MODEL, reasoning_effort="low",
                max_completion_tokens=CANDIDATE_MAX,
                messages=[{"role": "system", "content": escalate.EXPERT_SYSTEM},
                          {"role": "user", "content": row.transcript}],
                user=escalate.USER_ID,
            )
            record["answer"] = escalate._content(response)
            record["prompt_tokens"] = int(response.usage.prompt_tokens)
            record["completion_tokens"] = int(response.usage.completion_tokens)
            record["cost_usd"] = usage_cost(
                CANDIDATE_MODEL, record["prompt_tokens"],
                record["completion_tokens"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = time.perf_counter() - started
        append_jsonl(output, record)
        old.append(record)
        print(f"candidate {row.pool}:{row.id}: error={bool(record['error'])} "
              f"cost=${total_cost(old):.4f}", flush=True)
        if total_cost(old) >= COST_CAP:
            raise SystemExit("P39 cost cap reached")


def judge_phase(args, frame):
    from openai import OpenAI

    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    candidates = {str(row["id"]): row for row in candidate_rows(args.output_dir)
                  if not row.get("error")}
    if len(candidates) != len(frame):
        raise SystemExit(f"candidate coverage {len(candidates)}/{len(frame)}")
    output = args.output_dir / f"p39_judge.rank{args.shard_index}.jsonl"
    old = judge_rows(args.output_dir)
    done = {str(row["id"]) for row in old if not row.get("error")}
    order = sorted(frame.itertuples(index=False), key=lambda row: hashlib.sha256(
        f"{row.pool}:{row.id}:candidate".encode()).hexdigest())
    client = OpenAI()
    for row in order[args.shard_index::args.shard_count]:
        if row.id in done:
            continue
        model = OURS_JUDGE_MODEL if row.judge_kind == "ours" else OAB_JUDGE_MODEL
        record = {"phase": "judge", "id": row.id, "pool": row.pool,
                  "model": model, "judge_kind": row.judge_kind, "error": None}
        started = time.perf_counter()
        try:
            answer = str(candidates[row.id]["answer"])
            if row.judge_kind == "ours":
                response = client.chat.completions.create(
                    model=model, reasoning_effort="low",
                    max_completion_tokens=OURS_MAX,
                    response_format=escalate._resp_format(
                        "verdict", escalate._JUDGE_SCHEMA),
                    messages=[{"role": "system", "content": escalate.JUDGE_SYSTEM},
                              {"role": "user", "content": escalate._judge_user(
                                  str(row.query), row.reference, answer)}],
                    user=escalate.USER_ID,
                )
                verdict = json.loads(escalate._content(response))
                record["adequate"] = bool(verdict["adequate"])
                record["reason"] = verdict["reason"]
            else:
                prompt = (OAB_PATTERN.replace("{instruction}", str(row.query))
                          .replace("{targets}", str(row.reference))
                          .replace("{answer}", answer))
                response = client.chat.completions.create(
                    model=model, max_tokens=OAB_MAX,
                    messages=[{"role": "user", "content": prompt}],
                    user=escalate.USER_ID,
                )
                content = response.choices[0].message.content or ""
                match = re.search(r'"judgment"\s*:\s*"?(correct|incorrect)',
                                  content, re.I)
                if not match:
                    raise ValueError(f"unparseable OAB verdict: {content[:160]}")
                record["adequate"] = match.group(1).lower() == "correct"
            record["prompt_tokens"] = int(response.usage.prompt_tokens)
            record["completion_tokens"] = int(response.usage.completion_tokens)
            record["cost_usd"] = usage_cost(
                model, record["prompt_tokens"], record["completion_tokens"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = time.perf_counter() - started
        append_jsonl(output, record)
        old.append(record)
        running = total_cost(list(candidates.values()), old)
        print(f"judge {row.pool}:{row.id}: error={bool(record['error'])} "
              f"cost=${running:.4f}", flush=True)
        if running >= COST_CAP:
            raise SystemExit("P39 cost cap reached")


def mcnemar(wins, losses):
    n = wins + losses
    return (sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n
            if n else 1.0)


def report_phase(args, frame):
    cs = candidate_rows(args.output_dir)
    js = judge_rows(args.output_dir)
    candidates = {str(row["id"]): row for row in cs if not row.get("error")}
    judges = {str(row["id"]): row for row in js if not row.get("error")}
    missing_c = sorted(set(frame.id) - set(candidates))
    missing_j = sorted(set(frame.id) - set(judges))
    rows = []
    for row in frame.itertuples(index=False):
        if row.id not in candidates or row.id not in judges:
            continue
        rows.append({"id": row.id, "pool": row.pool,
                     "baseline": bool(row.baseline),
                     "candidate": bool(judges[row.id]["adequate"]),
                     "baseline_latency_s": float(row.baseline_latency_s),
                     "candidate_latency_s": float(candidates[row.id]["latency_s"]),
                     "transcript": row.transcript,
                     "baseline_answer": row.baseline_answer,
                     "candidate_answer": candidates[row.id]["answer"]})
    scored = pd.DataFrame(rows)
    wins = int((~scored.baseline & scored.candidate).sum()) if len(scored) else 0
    losses = int((scored.baseline & ~scored.candidate).sum()) if len(scored) else 0

    def metrics(part):
        if not len(part):
            return {"n": 0, "baseline": None, "candidate": None, "delta": None}
        baseline, candidate = float(part.baseline.mean()), float(part.candidate.mean())
        return {"n": len(part), "baseline": baseline, "candidate": candidate,
                "delta": candidate - baseline}

    overall = metrics(scored)
    pools = {pool: metrics(scored[scored.pool.eq(pool)]) for pool in JUDGE_KIND}
    latency_ratio = (float(scored.candidate_latency_s.median()
                           / scored.baseline_latency_s.median())
                     if len(scored) else None)
    cost = total_cost(cs, js)
    p_value = mcnemar(wins, losses)
    gates = {
        "candidate_and_judge_292_of_292": not missing_c and not missing_j,
        "overall_delta_at_least_0.02": overall["delta"] is not None and overall["delta"] >= .02,
        "mcnemar_one_sided_below_0.05": p_value < .05,
        "wins_exceed_losses": wins > losses,
        "every_pool_nonnegative": all(v["delta"] is not None and v["delta"] >= 0
                                      for v in pools.values()),
        "median_latency_ratio_at_most_1.25": latency_ratio is not None and latency_ratio <= 1.25,
        "cost_at_most_15": cost <= COST_CAP,
        "live_unchanged": True,
    }
    result = {
        "experiment": "P39 gpt-5.6-sol low-reasoning expert substitution",
        "status": "pass" if all(gates.values()) else "reject",
        "metrics": {"all": overall, "pools": pools, "wins": wins,
                    "losses": losses, "mcnemar_one_sided_p": p_value,
                    "baseline_median_latency_s": (float(scored.baseline_latency_s.median())
                                                   if len(scored) else None),
                    "candidate_median_latency_s": (float(scored.candidate_latency_s.median())
                                                    if len(scored) else None),
                    "median_latency_ratio": latency_ratio},
        "coverage": {"population": len(frame), "missing_candidates": missing_c,
                     "missing_judges": missing_j},
        "cost_usd": cost, "gates": gates, "activation": "not_authorized",
    }
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    scored.to_parquet(args.rows, index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["candidate", "judge", "report"])
    parser.add_argument("--p38-rows", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population-sha", default=(
        "790b82563f3684189ea7592c1179eb88338d9bb089f65da8595fff3efccd08bf"))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--result", type=Path,
                        default=Path("figures/p39_gpt56_sol_result.json"))
    parser.add_argument("--rows", type=Path,
                        default=Path("figures/p39_gpt56_sol_rows.parquet"))
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard parameters")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = population(args)
    if args.phase == "candidate":
        candidate_phase(args, frame)
    elif args.phase == "judge":
        judge_phase(args, frame)
    else:
        report_phase(args, frame)


if __name__ == "__main__":
    main()
