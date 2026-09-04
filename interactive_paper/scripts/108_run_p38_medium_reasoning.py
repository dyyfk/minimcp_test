"""Run the frozen P38 low-vs-medium expert reasoning comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd


EXPERT_MODEL = "gpt-5.5"
OURS_JUDGE_MODEL = "gpt-5.4-mini"
OAB_JUDGE_MODEL = "gpt-4o-2024-08-06"
EXPERT_MAX = 4096
OURS_MAX = 2048
OAB_MAX = 512
COST_CAP = 25.0
PRICES = {
    EXPERT_MODEL: (5.0 / 1e6, 30.0 / 1e6),
    OURS_JUDGE_MODEL: (.75 / 1e6, 4.5 / 1e6),
    OAB_JUDGE_MODEL: (2.5 / 1e6, 10.0 / 1e6),
}
POOLS = {
    "frozen": ("en", "ours"),
    "striviaqa": ("en", "oab"),
    "swebq": ("en", "oab"),
    "sllama": ("en", "oab"),
    "sdqa": ("en", "ours"),
    "sreason": ("zh", "ours"),
}
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


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")
            if line.strip()]


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def usage_cost(model, prompt_tokens, completion_tokens):
    input_price, output_price = PRICES[model]
    return prompt_tokens * input_price + completion_tokens * output_price


def total_cost(*groups):
    return sum(float(row.get("cost_usd") or 0) for rows in groups for row in rows
               if not row.get("error"))


def population(args):
    gate = json.loads(args.gate.read_text())
    rows = []
    for pool, (lang, judge_kind) in POOLS.items():
        never = pd.read_parquet(args.data_dir / f"{pool}_never_judged.parquet")
        never = never.drop_duplicates("id", keep="last")
        always = pd.read_parquet(
            args.data_dir / f"{pool}_always_tts_judged.parquet"
        ).drop_duplicates("id", keep="last").set_index("id")
        threshold = gate["eot_thresholds_lang"][lang]["balanced"]
        info = never["is_info"].fillna(True)
        selected = never[
            never["score"].notna() & never["score"].ge(threshold) & info
        ]
        for _, source in selected.iterrows():
            base = always.loc[source["id"]]
            if pd.isna(base["expert_answer"]) or not str(base["expert_answer"]).strip():
                continue
            rows.append({
                "id": str(source["id"]),
                "pool": pool,
                "judge_kind": judge_kind,
                "query": str(base["query"]),
                "reference": (None if pd.isna(base["reference_answer"])
                              else str(base["reference_answer"])),
                "transcript": str(base["transcript"]),
                "baseline_answer": str(base["expert_answer"]),
                "baseline_latency_s": float(base["expert_latency_s"]),
            })
    frame = pd.DataFrame(rows).sort_values(["pool", "id"]).reset_index(drop=True)
    receipt = "\n".join(f"{r.pool}:{r.id}" for r in frame.itertuples())
    digest = hashlib.sha256(receipt.encode()).hexdigest()
    if len(frame) != 292 or digest != args.population_sha:
        raise SystemExit(f"population mismatch: n={len(frame)} sha={digest}")
    return frame


def candidate_phase(args, frame):
    from openai import OpenAI

    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    output = args.output_dir / "p38_candidate.jsonl"
    old = read_jsonl(output)
    done = {str(row["id"]) for row in old if not row.get("error")}
    client = OpenAI()
    for row in frame.itertuples(index=False):
        if row.id in done:
            continue
        record = {"phase": "candidate", "id": row.id, "pool": row.pool,
                  "model": EXPERT_MODEL, "reasoning_effort": "medium",
                  "error": None}
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=EXPERT_MODEL,
                reasoning_effort="medium",
                max_completion_tokens=EXPERT_MAX,
                messages=[{"role": "system", "content": escalate.EXPERT_SYSTEM},
                          {"role": "user", "content": row.transcript}],
                user=escalate.USER_ID,
            )
            record["answer"] = escalate._content(response)
            record["prompt_tokens"] = int(response.usage.prompt_tokens)
            record["completion_tokens"] = int(response.usage.completion_tokens)
            record["cost_usd"] = usage_cost(
                EXPERT_MODEL, record["prompt_tokens"], record["completion_tokens"]
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = time.perf_counter() - started
        append_jsonl(output, record)
        old.append(record)
        print(f"candidate {row.pool}:{row.id}: error={bool(record['error'])} "
              f"cost=${total_cost(old):.4f}", flush=True)
        if total_cost(old) >= COST_CAP:
            raise SystemExit("P38 cost cap reached")


def _hash_order(frame):
    items = [(row, arm) for row in frame.itertuples(index=False)
             for arm in ("baseline", "candidate")]
    return sorted(items, key=lambda item: hashlib.sha256(
        f"{item[0].pool}:{item[0].id}:{item[1]}".encode()).hexdigest())


def judge_phase(args, frame):
    from openai import OpenAI

    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    candidates = {str(row["id"]): row for row in read_jsonl(
        args.output_dir / "p38_candidate.jsonl") if not row.get("error")}
    if len(candidates) != len(frame):
        raise SystemExit(f"candidate coverage {len(candidates)}/{len(frame)}")
    output = args.output_dir / "p38_judge.jsonl"
    old = read_jsonl(output)
    done = {(str(row["id"]), row["arm"]) for row in old if not row.get("error")}
    client = OpenAI()
    for row, arm in _hash_order(frame):
        key = (row.id, arm)
        if key in done:
            continue
        answer = (row.baseline_answer if arm == "baseline"
                  else str(candidates[row.id]["answer"]))
        model = OURS_JUDGE_MODEL if row.judge_kind == "ours" else OAB_JUDGE_MODEL
        record = {"phase": "judge", "id": row.id, "pool": row.pool,
                  "arm": arm, "judge_kind": row.judge_kind,
                  "model": model, "error": None}
        started = time.perf_counter()
        try:
            if row.judge_kind == "ours":
                response = client.chat.completions.create(
                    model=model,
                    reasoning_effort="low",
                    max_completion_tokens=OURS_MAX,
                    response_format=escalate._resp_format(
                        "verdict", escalate._JUDGE_SCHEMA),
                    messages=[{"role": "system", "content": escalate.JUDGE_SYSTEM},
                              {"role": "user", "content": escalate._judge_user(
                                  row.query, row.reference, answer)}],
                    user=escalate.USER_ID,
                )
                verdict = json.loads(escalate._content(response))
                record["adequate"] = bool(verdict["adequate"])
                record["reason"] = verdict["reason"]
            else:
                prompt = (OAB_PATTERN.replace("{instruction}", row.query)
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
                model, record["prompt_tokens"], record["completion_tokens"]
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = time.perf_counter() - started
        append_jsonl(output, record)
        old.append(record)
        candidate_rows = list(candidates.values())
        running_cost = total_cost(candidate_rows, old)
        print(f"judge {row.pool}:{row.id} {arm}: "
              f"error={bool(record['error'])} cost=${running_cost:.4f}", flush=True)
        if running_cost >= COST_CAP:
            raise SystemExit("P38 cost cap reached")


def exact_mcnemar_one_sided(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def report_phase(args, frame):
    candidate_rows = read_jsonl(args.output_dir / "p38_candidate.jsonl")
    judge_rows = read_jsonl(args.output_dir / "p38_judge.jsonl")
    candidates = {str(row["id"]): row for row in candidate_rows
                  if not row.get("error")}
    judges = {(str(row["id"]), row["arm"]): row for row in judge_rows
              if not row.get("error")}
    expected_judges = {(row.id, arm) for row in frame.itertuples(index=False)
                       for arm in ("baseline", "candidate")}
    missing_candidates = sorted(set(frame.id) - set(candidates))
    missing_judges = sorted(expected_judges - set(judges))
    rows = []
    for row in frame.itertuples(index=False):
        if row.id not in candidates or any(
                (row.id, arm) not in judges for arm in ("baseline", "candidate")):
            continue
        rows.append({
            "id": row.id, "pool": row.pool,
            "baseline": bool(judges[(row.id, "baseline")]["adequate"]),
            "candidate": bool(judges[(row.id, "candidate")]["adequate"]),
            "baseline_latency_s": row.baseline_latency_s,
            "candidate_latency_s": float(candidates[row.id]["latency_s"]),
            "transcript": row.transcript,
            "baseline_answer": row.baseline_answer,
            "candidate_answer": str(candidates[row.id]["answer"]),
        })
    scored = pd.DataFrame(rows)
    wins = int((~scored.baseline & scored.candidate).sum()) if len(scored) else 0
    losses = int((scored.baseline & ~scored.candidate).sum()) if len(scored) else 0

    def metrics(part):
        if not len(part):
            return {"n": 0, "baseline": None, "candidate": None, "delta": None}
        baseline = float(part.baseline.mean())
        candidate = float(part.candidate.mean())
        return {"n": len(part), "baseline": baseline, "candidate": candidate,
                "delta": candidate - baseline}

    overall = metrics(scored)
    pools = {pool: metrics(scored[scored.pool.eq(pool)]) for pool in POOLS}
    latency_ratio = (float(scored.candidate_latency_s.median()
                           / scored.baseline_latency_s.median())
                     if len(scored) else None)
    api_cost = total_cost(candidate_rows, judge_rows)
    p_value = exact_mcnemar_one_sided(wins, losses)
    gates = {
        "candidate_and_both_judges_292_of_292": (
            not missing_candidates and not missing_judges),
        "overall_delta_at_least_0.02": (
            overall["delta"] is not None and overall["delta"] >= .02),
        "mcnemar_one_sided_below_0.05": p_value < .05,
        "wins_exceed_losses": wins > losses,
        "every_pool_nonnegative": all(
            value["delta"] is not None and value["delta"] >= 0
            for value in pools.values()),
        "median_latency_ratio_at_most_2": (
            latency_ratio is not None and latency_ratio <= 2.0),
        "cost_at_most_25": api_cost <= COST_CAP,
        "live_unchanged": True,
    }
    result = {
        "experiment": "P38 medium-reasoning expert substitution on balanced-tier fires",
        "status": "pass" if all(gates.values()) else "reject",
        "metrics": {"all": overall, "pools": pools, "wins": wins,
                    "losses": losses, "mcnemar_one_sided_p": p_value,
                    "median_latency_ratio": latency_ratio,
                    "baseline_median_latency_s": (
                        float(scored.baseline_latency_s.median()) if len(scored) else None),
                    "candidate_median_latency_s": (
                        float(scored.candidate_latency_s.median()) if len(scored) else None)},
        "coverage": {"population": len(frame),
                     "missing_candidates": missing_candidates,
                     "missing_judges": missing_judges},
        "cost_usd": api_cost,
        "gates": gates,
        "activation": "not_authorized",
    }
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    scored.to_parquet(args.rows, index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["candidate", "judge", "report"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population-sha", default=(
        "790b82563f3684189ea7592c1179eb88338d9bb089f65da8595fff3efccd08bf"))
    parser.add_argument("--result", type=Path,
                        default=Path("figures/p38_medium_reasoning_result.json"))
    parser.add_argument("--rows", type=Path,
                        default=Path("figures/p38_medium_reasoning_rows.parquet"))
    args = parser.parse_args()
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
