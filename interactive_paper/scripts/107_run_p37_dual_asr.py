"""Run and score the frozen P37 paired expert-input experiment.

Phases are explicit and resumable.  Expert output is collected before either
arm is judged; the judge receives only question, reference, and candidate
answer, never the arm name or the competing answer.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd


EXPERT_MODEL = "gpt-5.5"
JUDGE_MODEL = "gpt-5.4-mini"
EXPERT_MAX = 4096
JUDGE_MAX = 2048
EXPERT_IN, EXPERT_OUT = 5.0 / 1e6, 30.0 / 1e6
JUDGE_IN, JUDGE_OUT = .75 / 1e6, 4.5 / 1e6
COST_CAP = 10.0

DUAL_INSTRUCTION = """Two independent speech recognizers transcribed the same user audio below. Either transcript may contain recognition errors. Infer the most likely coherent original question using both transcripts, then answer that question correctly, directly, and concisely. Do not mention the transcripts or this reconciliation instruction.

Transcript {first_label}:
{first}

Transcript {second_label}:
{second}"""


def read_jsonl(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(json.loads(line) for line in open(path, encoding="utf-8")
                    if line.strip())
    return rows


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def source_rows(args):
    taxonomy = pd.read_parquet(args.taxonomy)
    taxonomy = taxonomy[taxonomy["ftype"].eq("perception")].copy()

    public = pd.concat(
        [pd.read_parquet(path) for path in sorted(glob.glob(args.public_asr))],
        ignore_index=True,
    ).drop_duplicates("id", keep="last")
    second = dict(zip(public["id"].astype(str), public["transcript"].astype(str)))
    for row in read_jsonl(args.local_asr):
        if not row.get("error") and str(row.get("transcript", "")).strip():
            second[str(row["id"])] = str(row["transcript"]).strip()
    taxonomy["transcript_b"] = taxonomy["id"].astype(str).map(second)
    missing = taxonomy.loc[taxonomy["transcript_b"].isna(), "id"].tolist()
    if missing:
        raise SystemExit(f"missing second transcript for {missing}")
    return taxonomy.sort_values("id")


def cost(rows):
    total = 0.0
    for row in rows:
        if row.get("error"):
            continue
        if row["phase"] == "expert":
            total += ((row.get("prompt_tokens") or 0) * EXPERT_IN
                      + (row.get("completion_tokens") or 0) * EXPERT_OUT)
        else:
            total += ((row.get("prompt_tokens") or 0) * JUDGE_IN
                      + (row.get("completion_tokens") or 0) * JUDGE_OUT)
    return total


def expert_phase(args, frame):
    from openai import OpenAI

    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    output = args.output_dir / "p37_expert.jsonl"
    old = read_jsonl(str(output)) if output.exists() else []
    done = {(str(row["id"]), row["arm"]) for row in old if not row.get("error")}
    client = OpenAI()
    for _, row in frame.iterrows():
        for arm in ("baseline", "dual"):
            key = (str(row["id"]), arm)
            if key in done:
                continue
            if arm == "baseline":
                prompt = str(row["transcript"])
                order = "A"
            else:
                flip = int(hashlib.sha256(str(row["id"]).encode()).hexdigest(), 16) & 1
                values = ((str(row["transcript"]), str(row["transcript_b"]))
                          if not flip else
                          (str(row["transcript_b"]), str(row["transcript"])))
                prompt = DUAL_INSTRUCTION.format(
                    first_label="1", first=values[0],
                    second_label="2", second=values[1],
                )
                order = "AB" if not flip else "BA"
            record = {"phase": "expert", "id": str(row["id"]),
                      "pool": str(row["pool"]), "arm": arm,
                      "transcript_order": order, "error": None}
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=EXPERT_MODEL, reasoning_effort="low",
                    max_completion_tokens=EXPERT_MAX,
                    messages=[{"role": "system", "content": escalate.EXPERT_SYSTEM},
                              {"role": "user", "content": prompt}],
                    user=escalate.USER_ID,
                )
                record["answer"] = escalate._content(response)
                record["prompt_tokens"] = int(response.usage.prompt_tokens)
                record["completion_tokens"] = int(response.usage.completion_tokens)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_s"] = time.perf_counter() - started
            append_jsonl(output, record)
            old.append(record)
            print(f"expert {row['id']} {arm}: error={bool(record['error'])} "
                  f"cost=${cost(old):.4f}", flush=True)
            if cost(old) >= COST_CAP:
                raise SystemExit("P37 cost cap reached")


def judge_phase(args, frame):
    from openai import OpenAI

    sys.path.insert(0, str(args.source_dir.resolve()))
    import escalate

    expert_rows = read_jsonl(str(args.output_dir / "p37_expert.jsonl"))
    expert = {(str(row["id"]), row["arm"]): row for row in expert_rows
              if not row.get("error")}
    output = args.output_dir / "p37_judge.jsonl"
    old = read_jsonl(str(output)) if output.exists() else []
    done = {(str(row["id"]), row["arm"]) for row in old if not row.get("error")}
    client = OpenAI()
    for _, row in frame.iterrows():
        for arm in ("baseline", "dual"):
            key = (str(row["id"]), arm)
            if key in done:
                continue
            if key not in expert:
                raise SystemExit(f"missing expert output {key}")
            record = {"phase": "judge", "id": str(row["id"]),
                      "pool": str(row["pool"]), "arm": arm, "error": None}
            started = time.perf_counter()
            try:
                reference = None if pd.isna(row["reference"]) else str(row["reference"])
                response = client.chat.completions.create(
                    model=JUDGE_MODEL, reasoning_effort="low",
                    max_completion_tokens=JUDGE_MAX,
                    response_format=escalate._resp_format("verdict", escalate._JUDGE_SCHEMA),
                    messages=[{"role": "system", "content": escalate.JUDGE_SYSTEM},
                              {"role": "user", "content": escalate._judge_user(
                                  str(row["query"]), reference,
                                  str(expert[key]["answer"]))}],
                    user=escalate.USER_ID,
                )
                verdict = json.loads(escalate._content(response))
                record["adequate"] = bool(verdict["adequate"])
                record["reason"] = verdict["reason"]
                record["prompt_tokens"] = int(response.usage.prompt_tokens)
                record["completion_tokens"] = int(response.usage.completion_tokens)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_s"] = time.perf_counter() - started
            append_jsonl(output, record)
            old.append(record)
            total_cost = cost(expert_rows) + cost(old)
            print(f"judge {row['id']} {arm}: error={bool(record['error'])} "
                  f"cost=${total_cost:.4f}", flush=True)
            if total_cost >= COST_CAP:
                raise SystemExit("P37 cost cap reached")


def exact_mcnemar_one_sided(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def report_phase(args, frame):
    experts = read_jsonl(str(args.output_dir / "p37_expert.jsonl"))
    judges = read_jsonl(str(args.output_dir / "p37_judge.jsonl"))
    eok = {(str(row["id"]), row["arm"]): row for row in experts if not row.get("error")}
    jok = {(str(row["id"]), row["arm"]): row for row in judges if not row.get("error")}
    expected = {(str(sample_id), arm) for sample_id in frame["id"]
                for arm in ("baseline", "dual")}
    missing_expert = sorted(expected - set(eok))
    missing_judge = sorted(expected - set(jok))
    rows = []
    for _, source in frame.iterrows():
        sample_id = str(source["id"])
        if any((sample_id, arm) not in jok for arm in ("baseline", "dual")):
            continue
        rows.append({
            "id": sample_id, "pool": str(source["pool"]),
            "baseline": bool(jok[(sample_id, "baseline")]["adequate"]),
            "dual": bool(jok[(sample_id, "dual")]["adequate"]),
            "transcript_a": str(source["transcript"]),
            "transcript_b": str(source["transcript_b"]),
            "baseline_answer": eok[(sample_id, "baseline")]["answer"],
            "dual_answer": eok[(sample_id, "dual")]["answer"],
        })
    scored = pd.DataFrame(rows)
    wins = int((~scored["baseline"] & scored["dual"]).sum()) if len(scored) else 0
    losses = int((scored["baseline"] & ~scored["dual"]).sum()) if len(scored) else 0

    def subset_metrics(mask):
        part = scored[mask]
        if not len(part):
            return {"n": 0, "baseline": None, "dual": None, "delta": None}
        base, dual = float(part["baseline"].mean()), float(part["dual"].mean())
        return {"n": len(part), "baseline": base, "dual": dual, "delta": dual - base}

    all_metrics = subset_metrics(pd.Series(True, index=scored.index))
    frozen_metrics = subset_metrics(scored["pool"].eq("frozen"))
    external_metrics = subset_metrics(~scored["pool"].eq("frozen"))
    total_cost = cost(experts) + cost(judges)
    gates = {
        "second_transcript_36_of_36": len(frame) == 36,
        "paired_expert_and_judge_36_of_36": not missing_expert and not missing_judge,
        "accuracy_delta_at_least_0.10": (
            all_metrics["delta"] is not None and all_metrics["delta"] >= .10),
        "mcnemar_one_sided_below_0.10": exact_mcnemar_one_sided(wins, losses) < .10,
        "regressions_at_most_one": losses <= 1,
        "frozen_and_external_nonnegative": (
            frozen_metrics["delta"] is not None and frozen_metrics["delta"] >= 0
            and external_metrics["delta"] is not None and external_metrics["delta"] >= 0),
        "cost_at_most_10": total_cost <= COST_CAP,
        "live_unchanged": True,
    }
    result = {
        "experiment": "P37 dual-ASR expert repair for perception failures",
        "status": "pass" if all(gates.values()) else "reject",
        "metrics": {"all": all_metrics, "frozen": frozen_metrics,
                    "external": external_metrics, "wins": wins, "losses": losses,
                    "mcnemar_one_sided_p": exact_mcnemar_one_sided(wins, losses)},
        "coverage": {"second_transcripts": len(frame),
                     "missing_expert": missing_expert, "missing_judge": missing_judge},
        "cost_usd": total_cost,
        "gates": gates,
        "activation": "not_authorized",
    }
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    scored.to_parquet(args.rows, index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["expert", "judge", "report"])
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--public-asr", required=True)
    parser.add_argument("--local-asr", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path,
                        default=Path("figures/p37_dual_asr_expert_result.json"))
    parser.add_argument("--rows", type=Path,
                        default=Path("figures/p37_dual_asr_expert_rows.parquet"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = source_rows(args)
    if args.phase == "expert":
        expert_phase(args, frame)
    elif args.phase == "judge":
        judge_phase(args, frame)
    else:
        report_phase(args, frame)


if __name__ == "__main__":
    main()
