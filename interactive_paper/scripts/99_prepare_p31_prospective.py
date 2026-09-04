"""Freeze P32 source-disjoint prospective standalone questions pre-outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


REVISIONS = {
    "piqa": "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
    "super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
    "openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "social_i_qa": "8835ceb9141d7896d9d968634a9b21ae440e3ec5",
}


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def balanced(frame, count, seed):
    labels = sorted(frame.label.unique())
    if count % len(labels):
        raise RuntimeError(f"count {count} not divisible by {len(labels)} labels")
    parts = []
    for label in labels:
        value = frame[frame.label == label].copy()
        value["_key"] = value.id.map(lambda item: stable_key(seed, item))
        parts.append(value.sort_values("_key").head(count // len(labels)))
    result = pd.concat(parts).drop(columns="_key")
    if len(result) != count:
        raise RuntimeError("insufficient rows for balanced sample")
    return result


def frames():
    piqa = pd.DataFrame(load_dataset(
        "ybisk/piqa", split="validation", revision=REVISIONS["piqa"],
        trust_remote_code=True))
    piqa_rows = []
    for index, row in piqa.iterrows():
        options = [str(row.sol1), str(row.sol2)]
        label = int(row.label)
        piqa_rows.append({
            "id": f"p32p{index:05d}", "pool": "p32_piqa", "label": label,
            "source": "ybisk/piqa:validation", "source_index": index,
            "query": (f"What is the more sensible way to accomplish this goal: "
                      f"{row.goal} (A) {options[0]} (B) {options[1]}"),
            "reference_answer": f"({'AB'[label]}) {options[label]}",
            "language": "en", "split": "p31_prospective_validation",
        })

    copa = pd.DataFrame(load_dataset(
        "aps/super_glue", "copa", split="validation",
        revision=REVISIONS["super_glue"], trust_remote_code=True))
    copa_rows = []
    for _, row in copa.iterrows():
        index = int(row.idx)
        options = [str(row.choice1), str(row.choice2)]
        label = int(row.label)
        relation = "most likely cause" if row.question == "cause" else "most likely effect"
        copa_rows.append({
            "id": f"p32c{index:05d}", "pool": "p32_copa", "label": label,
            "source": "aps/super_glue:copa:validation", "source_index": index,
            "query": (f"Given: {row.premise} Which option is the {relation}? "
                      f"(A) {options[0]} (B) {options[1]}"),
            "reference_answer": f"({'AB'[label]}) {options[label]}",
            "language": "en", "split": "p31_prospective_validation",
        })

    openbook = pd.DataFrame(load_dataset(
        "allenai/openbookqa", "main", split="validation",
        revision=REVISIONS["openbookqa"], trust_remote_code=True))
    openbook_rows = []
    for index, row in openbook.iterrows():
        mapping = dict(zip(row.choices["label"], row.choices["text"]))
        labels = list(row.choices["label"])
        answer_key = str(row.answerKey)
        label = labels.index(answer_key)
        rendered = " ".join(f"({key}) {mapping[key]}" for key in labels)
        openbook_rows.append({
            "id": f"p32o{index:05d}", "pool": "p32_openbookqa", "label": label,
            "source": "allenai/openbookqa:main:validation", "source_index": index,
            "query": f"Answer this science question. {row.question_stem} {rendered}",
            "reference_answer": f"({answer_key}) {mapping[answer_key]}",
            "language": "en", "split": "p31_prospective_validation",
        })

    social = pd.DataFrame(load_dataset(
        "allenai/social_i_qa", split="validation",
        revision=REVISIONS["social_i_qa"], trust_remote_code=True))
    social_rows = []
    for index, row in social.iterrows():
        options = [str(row.answerA), str(row.answerB), str(row.answerC)]
        label = int(row.label) - 1
        rendered = " ".join(
            f"({'ABC'[item]}) {value}" for item, value in enumerate(options))
        social_rows.append({
            "id": f"p32s{index:05d}", "pool": "p32_social_iqa", "label": label,
            "source": "allenai/social_i_qa:validation", "source_index": index,
            "query": f"Context: {row.context} Question: {row.question} {rendered}",
            "reference_answer": f"({'ABC'[label]}) {options[label]}",
            "language": "en", "split": "p31_prospective_validation",
        })
    return [pd.DataFrame(value) for value in (
        piqa_rows, copa_rows, openbook_rows, social_rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-selection", type=Path, action="append",
                        required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--max-characters", type=int, default=360)
    args = parser.parse_args()
    seen = set()
    for path in args.prior_selection:
        seen.update(pd.read_parquet(path)["query"].map(normalize))
    counts = {"p32_piqa": 150, "p32_copa": 90,
              "p32_openbookqa": 148, "p32_social_iqa": 150}
    chosen, diagnostics = [], {}
    for frame in frames():
        pool = str(frame.pool.iloc[0])
        normalized = frame["query"].map(normalize)
        overlap = normalized.isin(seen)
        duplicate = normalized.duplicated(keep="first")
        too_long = frame["query"].str.len() > args.max_characters
        eligible = frame[~(overlap | duplicate | too_long)].copy()
        sample = balanced(eligible, counts[pool], args.seed)
        chosen.append(sample.drop(columns="label"))
        diagnostics[pool] = {
            "source_rows": len(frame), "eligible": len(eligible),
            "prior_overlap": int(overlap.sum()),
            "duplicates": int(duplicate.sum()),
            "over_character_limit": int(too_long.sum()),
            "chosen": len(sample),
            "chosen_label_counts": sample.label.value_counts().sort_index().to_dict(),
        }
    selection = pd.concat(chosen, ignore_index=True).sort_values("id")
    if selection.id.duplicated().any() or selection["query"].map(
            normalize).duplicated().any():
        raise RuntimeError("duplicate prospective rows")
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(args.selection, index=False)
    canonical = "".join(json.dumps(
        row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in selection.to_dict("records"))
    receipt = {
        "status": "frozen_before_tts_native_causal_or_judge_outputs",
        "seed": args.seed, "n": len(selection),
        "counts_by_pool": selection.pool.value_counts().sort_index().to_dict(),
        "characters": int(selection["query"].str.len().sum()),
        "tts_cost_usd_at_15_per_million_characters": float(
            selection["query"].str.len().sum() * 15 / 1_000_000),
        "ordered_id_sha256": hashlib.sha256(
            ("\n".join(selection.id) + "\n").encode()).hexdigest(),
        "canonical_content_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "source_revisions": REVISIONS, "diagnostics": diagnostics,
        "guard": ("P32 IDs/content and gates freeze after P31 historical pass "
                  "and before all P32 outputs."),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
