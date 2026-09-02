"""Freeze an independent bilingual dependent-context validation for P23."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal_order(index: int, language: str) -> dict:
    names_en = ["Atlas", "Birch", "Cobalt", "Delta", "Elm"]
    names_zh = ["朝霞", "白鹭", "苍穹", "丹枫", "银杏"]
    names = names_zh if language == "zh" else names_en
    rotation = index % 5
    ordered = names[rotation:] + names[:rotation]
    base = 8 * 60 + 7 * (index % 6)
    times = [base + offset for offset in (0, 37, 83, 128, 176)]
    rendered = [f"{value // 60:02d}:{value % 60:02d}" for value in times]
    ask = 1 + (index * 2 % 3)
    if language == "zh":
        schedule = "；".join(f"{name}在{time}开始"
                              for name, time in zip(ordered, rendered))
        carrier = f"请记住今天的日程：{schedule}。现在只需简短确认。"
        target = f"根据刚才的日程，紧接在{ordered[ask]}之后开始的是哪个项目，几点开始？"
        reference = f"{ordered[ask + 1]}，{rendered[ask + 1]}"
    else:
        schedule = "; ".join(f"{name} starts at {time}"
                              for name, time in zip(ordered, rendered))
        carrier = f"Remember today's schedule: {schedule}. Only acknowledge briefly now."
        target = (f"From the schedule I gave you, which project starts "
                  f"immediately after {ordered[ask]}, and at what time?")
        reference = f"{ordered[ask + 1]} at {rendered[ask + 1]}"
    return {"carrier_query": carrier, "target_query": target,
            "target_reference_answer": reference}


def arithmetic_ledger(index: int, language: str) -> dict:
    start = 180 + 11 * index
    deposit = 35 + 3 * (index % 7)
    purchase = 22 + 4 * (index % 5)
    fee = 7 + (index % 4)
    refund = 13 + 2 * (index % 6)
    answer = start + deposit - purchase - fee + refund
    if language == "zh":
        carrier = (f"请记住这份账户记录：初始余额{start}美元，随后存入{deposit}美元，"
                   f"购买支出{purchase}美元，又支付手续费{fee}美元。先只简短确认。")
        target = f"在刚才的账户记录基础上，现在又收到{refund}美元退款。最终余额是多少？"
        reference = f"{answer}美元"
    else:
        carrier = (f"Remember this account record: the starting balance is "
                   f"${start}, then a ${deposit} deposit, a ${purchase} purchase, "
                   f"and a ${fee} fee. Only acknowledge briefly for now.")
        target = (f"Continuing the account record I gave you, a ${refund} refund "
                  "now arrives. What is the final balance?")
        reference = f"${answer}"
    return {"carrier_query": carrier, "target_query": target,
            "target_reference_answer": reference}


def seating_swap(index: int, language: str) -> dict:
    names_en = ["Ari", "Bo", "Cy", "Di", "Eli", "Fay"]
    names_zh = ["安琪", "博文", "晨宇", "冬梅", "恩泽", "芳华"]
    names = names_zh if language == "zh" else names_en
    rotation = index % 6
    order = names[rotation:] + names[:rotation]
    first = index % 5
    second = (first + 2 + index % 3) % 6
    if first == second:
        second = (second + 1) % 6
    updated = list(order)
    updated[first], updated[second] = updated[second], updated[first]
    ask = 1 + ((index * 3) % 4)
    if language == "zh":
        carrier = ("请记住六个人从左到右的座位顺序：" + "、".join(order) +
                   "。现在只需简短确认。")
        target = (f"现在把原来第{first + 1}位和第{second + 1}位的人交换。交换后，"
                  f"紧靠{updated[ask]}左边坐的是谁？")
        reference = updated[ask - 1]
    else:
        carrier = ("Remember the left-to-right seating order: " +
                   ", ".join(order) + ". Only acknowledge briefly for now.")
        target = (f"Now swap the people who were originally in positions "
                  f"{first + 1} and {second + 1}. After the swap, who sits "
                  f"immediately to the left of {updated[ask]}?")
        reference = updated[ask - 1]
    return {"carrier_query": carrier, "target_query": target,
            "target_reference_answer": reference}


BUILDERS = {
    "temporal_order": temporal_order,
    "arithmetic_ledger": arithmetic_ledger,
    "seating_swap": seating_swap,
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
                pair_id = f"p23-{language}-{family}-{index:03d}"
                rows.append({
                    "id": pair_id,
                    "target_id": f"{pair_id}-target",
                    "target_pool": f"{language}-{family}",
                    "carrier_id": f"{pair_id}-carrier",
                    "carrier_pool": f"{language}-fixture",
                    "language": language,
                    "context_condition": "independent_semantically_dependent_prior_turn",
                    "task_family": family,
                    **builder(index, language),
                })
    rows.sort(key=lambda row: row["id"])
    frame = pd.DataFrame(rows)
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.pairs, index=False)
    audio_rows = [{"id": row[f"{turn}_id"], "query": row[f"{turn}_query"]}
                  for row in rows for turn in ("carrier", "target")]
    audio = pd.DataFrame(audio_rows).sort_values("id")
    args.audio_selection.parent.mkdir(parents=True, exist_ok=True)
    audio.to_parquet(args.audio_selection, index=False)
    canonical = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")) + "\n"
                        for row in rows)
    receipt = {
        "status": "frozen_after_p22_candidate_before_all_p23_outputs",
        "rows": len(frame), "turn_audio_rows": len(audio),
        "counts_by_pool": frame.target_pool.value_counts().sort_index().to_dict(),
        "counts_by_language": frame.language.value_counts().sort_index().to_dict(),
        "counts_by_family": frame.task_family.value_counts().sort_index().to_dict(),
        "characters": int(audio["query"].str.len().sum()),
        "tts_cost_usd_at_15_per_million_characters": float(
            audio["query"].str.len().sum() * 15 / 1e6),
        "pairs_sha256": sha256(args.pairs),
        "audio_selection_sha256": sha256(args.audio_selection),
        "canonical_content_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "acceptance": ("macro pool AUC delta >= .015, bootstrap lower bound > 0, "
                       "pooled AUC nonnegative, both languages positive, and no "
                       "pool delta below -.01"),
        "guard": "P23 content and gate frozen before TTS/model/judge/scoring.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
