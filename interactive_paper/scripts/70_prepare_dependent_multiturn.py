"""Freeze a bilingual, genuinely context-dependent two-turn validation set.

Unlike the P19--P21 replays, the target utterance cannot be answered without
the carrier turn.  The fixtures are deterministic and template-generated so
no model output or judge result influences their construction.  Three task
families exercise linked lookup, constraint selection, and state updates.

The script writes the model pair table, a flat TTS table, and a checksummed
receipt.  TTS/model/judge execution is deliberately separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def linked_lookup(index: int, language: str) -> dict:
    people_en = ["Avery", "Blair", "Casey", "Devon", "Emery"]
    people_zh = ["安然", "白露", "陈曦", "杜衡", "方宁"]
    projects_en = ["Cedar", "Maple", "Orchid", "Quartz", "Willow"]
    projects_zh = ["青松", "白桦", "兰舟", "云石", "柳溪"]
    people = people_zh if language == "zh" else people_en
    projects = projects_zh if language == "zh" else projects_en
    rotation = index % len(projects)
    assigned = projects[rotation:] + projects[:rotation]
    hours = [7 + ((index + 2 * j) % 10) for j in range(5)]
    minutes = [5 * ((index * 3 + j * 7) % 12) for j in range(5)]
    times = [f"{hour:02d}:{minute:02d}" for hour, minute in zip(hours, minutes)]
    ask = (index * 3 + 1) % 5
    if language == "zh":
        assignments = "；".join(
            f"{person}负责{project}" for person, project in zip(people, assigned))
        launches = "；".join(
            f"{project}在{time}启动" for project, time in zip(assigned, times))
        carrier = (f"请记住本轮对话中的两组记录。人员分工：{assignments}。"
                   f"启动时间：{launches}。请只简短确认你记住了。")
        target = f"根据我刚才给出的两组记录，{people[ask]}负责的项目几点启动？"
        reference = f"{times[ask]}（{people[ask]}负责{assigned[ask]}）"
    else:
        assignments = "; ".join(
            f"{person} owns {project}" for person, project in zip(people, assigned))
        launches = "; ".join(
            f"{project} launches at {time}" for project, time in zip(assigned, times))
        carrier = ("Remember these two record sets for this conversation. "
                   f"Assignments: {assignments}. Launches: {launches}. "
                   "Only acknowledge briefly that you have them.")
        target = ("Using the two record sets I just gave you, at what time does "
                  f"the project owned by {people[ask]} launch?")
        reference = f"{times[ask]} ({people[ask]} owns {assigned[ask]})"
    return {"carrier_query": carrier, "target_query": target,
            "target_reference_answer": reference}


def constraint_choice(index: int, language: str) -> dict:
    chosen = index % 4
    letters = list("ABCD")
    max_cost = 70 + 5 * (index % 6)
    min_storage = 100 + 50 * (index % 4)
    rows = []
    for j, letter in enumerate(letters):
        if j == chosen:
            cost, storage, backup = max_cost - 5, min_storage + 50, True
        elif j == (chosen + 1) % 4:
            cost, storage, backup = max_cost + 10, min_storage + 100, True
        elif j == (chosen + 2) % 4:
            cost, storage, backup = max_cost - 15, min_storage - 25, True
        else:
            cost, storage, backup = max_cost - 10, min_storage + 100, False
        rows.append((letter, cost, storage, backup))
    if language == "zh":
        details = "；".join(
            f"方案{letter}每月{cost}美元、{storage}GB、"
            f"{'包含' if backup else '不包含'}云备份"
            for letter, cost, storage, backup in rows)
        carrier = (f"我在比较四个方案：{details}。请先只简短确认你记住了这些信息。")
        target = (f"根据之前的信息，我需要每月不超过{max_cost}美元、至少"
                  f"{min_storage}GB，并且必须包含云备份。唯一符合的是哪个方案？")
        reference = f"方案{letters[chosen]}"
    else:
        details = "; ".join(
            f"plan {letter} costs ${cost} monthly, has {storage} GB, and "
            f"{'includes' if backup else 'does not include'} cloud backup"
            for letter, cost, storage, backup in rows)
        carrier = (f"I am comparing four plans: {details}. For now, only "
                   "acknowledge briefly that you have the information.")
        target = (f"Using the earlier details, I need a monthly cost no more "
                  f"than ${max_cost}, at least {min_storage} GB, and cloud "
                  "backup. Which single plan qualifies?")
        reference = f"Plan {letters[chosen]}"
    return {"carrier_query": carrier, "target_query": target,
            "target_reference_answer": reference}


def state_update(index: int, language: str) -> dict:
    max_cost = 105 + 5 * (index % 7)
    fast = 3 + (index % 2)
    rows = [
        ("A", max_cost - 20, fast + 2, True),
        ("B", max_cost - 5, fast, False),
        ("C", max_cost - 10, fast + 1, True),
        ("D", max_cost + 10, fast - 1, True),
    ]
    if language == "zh":
        details = "；".join(
            f"车次{letter}票价{cost}美元、耗时{hours}小时、"
            f"{'含' if bag else '不含'}托运行李"
            for letter, cost, hours, bag in rows)
        carrier = (f"请记住这些车次：{details}。我的限制是总价不超过"
                   f"{max_cost}美元、耗时不超过{fast + 1}小时，并且需要托运行李。"
                   "请先只简短确认。")
        target = ("情况有变：车次C已售罄，车次B可加5美元购买托运行李。"
                  "结合之前的全部信息，现在哪个剩余车次符合所有限制？")
        reference = (f"车次B；加行李后总价{max_cost}美元，耗时{fast}小时，"
                     "满足全部限制")
    else:
        details = "; ".join(
            f"train {letter} costs ${cost}, takes {hours} hours, and "
            f"{'includes' if bag else 'does not include'} a checked bag"
            for letter, cost, hours, bag in rows)
        carrier = (f"Remember these trains: {details}. My limits are a total "
                   f"cost no more than ${max_cost}, a trip no longer than "
                   f"{fast + 1} hours, and a checked bag. Only acknowledge "
                   "briefly for now.")
        target = ("Update: train C is sold out, and I can add a checked bag to "
                  "train B for $5. Considering all the earlier information, "
                  "which remaining train now meets every limit?")
        reference = (f"Train B; with the bag it costs ${max_cost}, takes "
                     f"{fast} hours, and meets every limit")
    return {"carrier_query": carrier, "target_query": target,
            "target_reference_answer": reference}


BUILDERS = {
    "linked_lookup": linked_lookup,
    "constraint_choice": constraint_choice,
    "state_update": state_update,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--audio-selection", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--per-family-language", type=int, default=20)
    args = parser.parse_args()

    rows = []
    for language in ("en", "zh"):
        for family, builder in BUILDERS.items():
            for index in range(args.per_family_language):
                pair_id = f"p22-{language}-{family}-{index:03d}"
                row = {
                    "id": pair_id,
                    "target_id": f"{pair_id}-target",
                    "target_pool": f"{language}-{family}",
                    "carrier_id": f"{pair_id}-carrier",
                    "carrier_pool": f"{language}-fixture",
                    "language": language,
                    "context_condition": "semantically_dependent_prior_turn",
                    "task_family": family,
                    **builder(index, language),
                }
                rows.append(row)
    rows = sorted(rows, key=lambda row: row["id"])
    frame = pd.DataFrame(rows)
    if frame.id.duplicated().any() or frame.target_id.duplicated().any():
        raise RuntimeError("fixture IDs must be unique")
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.pairs, index=False)

    audio_rows = []
    for row in rows:
        audio_rows.extend([
            {"id": row["carrier_id"], "query": row["carrier_query"]},
            {"id": row["target_id"], "query": row["target_query"]},
        ])
    audio = pd.DataFrame(audio_rows).sort_values("id")
    args.audio_selection.parent.mkdir(parents=True, exist_ok=True)
    audio.to_parquet(args.audio_selection, index=False)

    receipt = {
        "status": "frozen_before_tts_native_or_judge_outputs",
        "rows": len(frame),
        "turn_audio_rows": len(audio),
        "counts_by_pool": frame.target_pool.value_counts().sort_index().to_dict(),
        "counts_by_language": frame.language.value_counts().sort_index().to_dict(),
        "counts_by_family": frame.task_family.value_counts().sort_index().to_dict(),
        "characters": int(audio["query"].str.len().sum()),
        "tts_cost_usd_at_15_per_million_characters": float(
            audio["query"].str.len().sum() * 15 / 1_000_000),
        "pairs_sha256": sha256(args.pairs),
        "audio_selection_sha256": sha256(args.audio_selection),
        "canonical_content_sha256": canonical_sha(rows),
        "guard": ("Fixtures and scoring criteria frozen before all TTS, native "
                  "model, judge, and candidate results."),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
