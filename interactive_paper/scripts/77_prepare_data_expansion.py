"""Freeze the P25-B cross-domain, bilingual, context-heavy expansion.

The output is split into standalone rows and genuine two-turn rows so the
existing native generators can be reused.  Spoken-SQuAD carrier audio is kept
as real speech; all other utterances are rendered only after this selection is
frozen.  No model outcome is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


REVISIONS = {
    "arc": "210d026faf9955653af8916fad021475a3f00453",
    "commonsenseqa": "94630fe30dad47192a8546eb75f094926d47e155",
    "mmlu_pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "xnli": "b8dd5d7af51114dbda02c0e3f6133f332186418e",
    "xquad": "51adfef1c1287aab1d2d91b5bead9bcfb9c68583",
    "ceval": "617524a00b307ff6f9933702f724131fe12ca7ce",
    "spoken_squad": "b55aab98726d0eab95eeef1ee9992a0532b3226e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def normalize(value) -> str:
    return re.sub(r"\W+", " ", str(value).casefold(), flags=re.UNICODE).strip()


def allocate(frame: pd.DataFrame, group: str, count: int, seed: int):
    frame = frame.copy()
    frame["_normalized_query"] = frame["query"].map(normalize)
    frame = frame.drop_duplicates("_normalized_query", keep="first").drop(
        columns="_normalized_query")
    groups = sorted(frame[group].astype(str).unique())
    base, remainder = divmod(count, len(groups))
    available = frame[group].astype(str).value_counts().to_dict()
    quotas = {name: min(available[name], base + int(position < remainder))
              for position, name in enumerate(groups)}
    while sum(quotas.values()) < count:
        eligible = [name for name in groups if quotas[name] < available[name]]
        if not eligible:
            raise RuntimeError(f"only {sum(quotas.values())}/{count} rows "
                               f"available across {group}")
        name = min(eligible, key=lambda value: (quotas[value], value))
        quotas[name] += 1
    pieces = []
    for name in groups:
        quota = quotas[name]
        part = frame[frame[group].astype(str) == name].copy()
        part["_key"] = [stable(seed, str(value)) for value in part["raw_id"]]
        chosen = part.sort_values("_key").head(quota)
        if len(chosen) != quota:
            raise RuntimeError(f"insufficient rows for {group}={name}: "
                               f"{len(chosen)}/{quota}")
        pieces.append(chosen.drop(columns="_key"))
    return pd.concat(pieces, ignore_index=True)


def choices_text(labels, texts):
    return ", ".join(f"({label}) {text}" for label, text in zip(labels, texts))


def standalone(raw: Path, seed: int):
    rows = []
    arc = pd.read_parquet(raw / "ARC-Challenge/train-00000-of-00001.parquet")
    for row in arc.itertuples():
        labels, texts = list(row.choices["label"]), list(row.choices["text"])
        answer = labels.index(str(row.answerKey))
        rows.append({"raw_id": f"arc:{row.id}", "source_family": "ai2_arc_challenge",
                     "pool": "p25b_arc", "language": "en",
                     "query": f"{row.question} Choose one: {choices_text(labels, texts)}",
                     "reference_answer": f"({labels[answer]}) {texts[answer]}"})
    arc_out = pd.DataFrame(rows)
    arc_out = allocate(arc_out, "source_family", 500, seed)

    rows = []
    cs = pd.read_parquet(raw / "data/train-00000-of-00001.parquet")
    for row in cs.itertuples():
        labels, texts = list(row.choices["label"]), list(row.choices["text"])
        answer = labels.index(str(row.answerKey))
        rows.append({"raw_id": f"csqa:{row.id}", "source_family": "commonsense_qa",
                     "pool": "p25b_commonsense", "language": "en",
                     "query": f"{row.question} Choose one: {choices_text(labels, texts)}",
                     "reference_answer": f"({labels[answer]}) {texts[answer]}"})
    cs_out = allocate(pd.DataFrame(rows), "source_family", 600, seed + 1)

    rows = []
    mmlu = pd.read_parquet(raw / "data/test-00000-of-00001.parquet")
    letters = "ABCDEFGHIJ"
    for row in mmlu.itertuples():
        options = list(row.options)
        if not 0 <= int(row.answer_index) < len(options):
            continue
        labels = list(letters[:len(options)])
        query = f"{row.question} Choose one: {choices_text(labels, options)}"
        if len(query) > 650:
            continue
        rows.append({"raw_id": f"mmlupro:{row.question_id}",
                     "source_family": f"mmlu_pro:{row.category}",
                     "pool": "p25b_mmlu_pro", "language": "en",
                     "query": query,
                     "reference_answer": f"({labels[int(row.answer_index)]}) "
                                         f"{options[int(row.answer_index)]}"})
    mmlu_out = allocate(pd.DataFrame(rows), "source_family", 800, seed + 2)

    rows = []
    mbpp = pd.read_parquet(raw / "sanitized/train-00000-of-00001.parquet")
    for row in mbpp.itertuples():
        rows.append({"raw_id": f"mbpp:{row.task_id}", "source_family": "mbpp",
                     "pool": "p25b_mbpp", "language": "en",
                     "query": str(row.prompt), "reference_answer": str(row.code)})
    mbpp_out = allocate(pd.DataFrame(rows), "source_family", 100, seed + 3)

    rows = []
    for path in sorted((raw / "ceval").glob("*/val-*.parquet")):
        subject = path.parent.name
        for row in pd.read_parquet(path).itertuples():
            labels = list("ABCD")
            texts = [str(getattr(row, label)) for label in labels]
            answer = str(row.answer)
            query = f"{row.question} 请选择：{choices_text(labels, texts)}"
            if answer not in labels or len(query) > 650:
                continue
            rows.append({"raw_id": f"ceval:{subject}:{row.id}",
                         "source_family": f"ceval:{subject}",
                         "pool": "p25b_ceval", "language": "zh",
                         "query": query,
                         "reference_answer": f"({answer}) {texts[labels.index(answer)]}"})
    ceval_out = allocate(pd.DataFrame(rows), "source_family", 1000, seed + 4)
    out = pd.concat([arc_out, cs_out, mmlu_out, mbpp_out, ceval_out],
                    ignore_index=True)
    out["id"] = [f"p25bs{index:05d}" for index in range(len(out))]
    out["mode"] = "standalone"
    return out


def centered_context(context: str, answer_start: int, answer: str,
                     limit: int = 360) -> str:
    start = max(0, int(answer_start) - limit // 2)
    stop = min(len(context), start + limit)
    start = max(0, stop - limit)
    excerpt = context[start:stop]
    if answer not in excerpt:
        raise RuntimeError("answer missing from centered XQuAD excerpt")
    return re.sub(r"\s+", " ", excerpt).strip()


def multiturn(raw: Path, output: Path, seed: int):
    blocks = []
    label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for language in ("en", "zh"):
        frame = pd.read_parquet(raw / language / "train-00000-of-00001.parquet")
        rows = []
        for index, row in frame.iterrows():
            label = int(row.label)
            if label not in label_names:
                continue
            if language == "en":
                carrier = f"Premise: {row.premise}"
                target = ("Given that premise, classify this hypothesis as entailment, "
                          f"neutral, or contradiction: {row.hypothesis}")
            else:
                carrier = f"前提：{row.premise}"
                target = ("根据刚才的前提，请判断下面的假设是蕴含、中立还是矛盾："
                          f"{row.hypothesis}")
            rows.append({"raw_id": f"xnli:{language}:{index}", "label": label,
                         "source_family": f"xnli:{language}",
                         "pool": f"p25b_xnli_{language}", "language": language,
                         "carrier_query": carrier, "query": target,
                         "reference_answer": label_names[label],
                         "carrier_audio_kind": "tts"})
        selected = allocate(pd.DataFrame(rows), "label", 500,
                            seed + (10 if language == "en" else 11))
        blocks.append(selected.drop(columns="label"))

    for language in ("en", "zh"):
        frame = pd.read_parquet(
            raw / f"xquad.{language}/validation-00000-of-00001.parquet")
        rows = []
        for index, row in frame.iterrows():
            answers = row.answers
            answer = str(answers["text"][0])
            carrier = centered_context(
                str(row.context), int(answers["answer_start"][0]), answer)
            if language == "en":
                carrier = f"Passage: {carrier}"
                target = f"According to that passage, {row.question}"
            else:
                carrier = f"资料：{carrier}"
                target = f"根据刚才的资料，{row.question}"
            rows.append({"raw_id": f"xquad:{language}:{row.id}",
                         "source_family": f"xquad:{language}",
                         "pool": f"p25b_xquad_{language}", "language": language,
                         "carrier_query": carrier, "query": target,
                         "reference_answer": answer,
                         "carrier_audio_kind": "tts"})
        frame = pd.DataFrame(rows)
        frame["_group"] = "all"
        blocks.append(allocate(frame, "_group", 500,
                               seed + (12 if language == "en" else 13))
                      .drop(columns="_group"))

    spoken_frames = [pd.read_parquet(path) for path in sorted(
        (raw / "data").glob("test-*-of-00021.parquet"))]
    spoken = pd.concat(spoken_frames, ignore_index=True)
    spoken["raw_id"] = [f"spoken_squad:{index}" for index in range(len(spoken))]
    spoken["query"] = spoken["instruction"].astype(str)
    spoken["_normalized_query"] = spoken["query"].map(normalize)
    spoken = spoken.drop_duplicates("_normalized_query", keep="first")
    spoken["_key"] = [stable(seed + 14, value) for value in spoken.raw_id]
    spoken = spoken.sort_values("_key").head(400)
    if len(spoken) != 400:
        raise RuntimeError(f"only {len(spoken)} Spoken-SQuAD rows available")
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    spoken_rows = []
    for row in spoken.itertuples():
        spoken_rows.append({"raw_id": row.raw_id,
                            "source_family": "spoken_squad_real",
                            "pool": "p25b_spoken_squad", "language": "en",
                            "carrier_query": "", "query": str(row.instruction),
                            "reference_answer": str(row.answer),
                            "carrier_audio_kind": "real"})
    blocks.append(pd.DataFrame(spoken_rows))

    out = pd.concat(blocks, ignore_index=True)
    out["id"] = [f"p25bm{index:05d}" for index in range(len(out))]
    out["carrier_id"] = out["id"] + "-carrier"
    out["target_id"] = out["id"] + "-target"
    out["carrier_pool"] = out["source_family"]
    out["target_pool"] = out["pool"]
    out["target_query"] = out["query"]
    out["target_reference_answer"] = out["reference_answer"]
    out["mode"] = "multiturn"

    # Extract the selected real carrier bytes only after stable IDs exist.
    spoken_by_raw = spoken.set_index("raw_id")
    for row in out[out.carrier_audio_kind == "real"].itertuples():
        payload = spoken_by_raw.loc[row.raw_id, "context"]
        (audio_dir / f"{row.carrier_id}.wav").write_bytes(payload["bytes"])
    return out


def canonical_sha(frame: pd.DataFrame, columns) -> str:
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")) + "\n"
                      for row in frame[columns].sort_values("id")
                      .to_dict("records"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, action="append",
                        default=[])
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    single = standalone(args.raw_dir, args.seed)
    multi = multiturn(args.raw_dir, args.output_dir, args.seed)

    seen = set()
    for path in sorted(args.data_dir.glob("queries*.jsonl")):
        frame = pd.read_json(path, lines=True)
        if "query" in frame:
            seen.update(frame["query"].map(normalize))
    for path in args.prior_selection:
        seen.update(pd.read_parquet(path)["query"].map(normalize))
    combined = pd.concat([
        single,
        multi[["raw_id", "source_family", "pool", "language", "query",
               "reference_answer", "id", "mode"]],
    ], ignore_index=True)
    normalized = combined["query"].map(normalize)
    overlap = normalized.isin(seen)
    duplicate = normalized.duplicated(keep=False)
    if overlap.any() or duplicate.any():
        raise RuntimeError(f"query leakage/duplicates: overlap={int(overlap.sum())} "
                           f"duplicate_rows={int(duplicate.sum())}")

    single_out = single[["id", "pool", "source_family", "language", "query",
                         "reference_answer", "raw_id", "mode"]]
    multi_out = multi[["id", "pool", "source_family", "language", "query",
                       "reference_answer", "raw_id", "mode", "carrier_id",
                       "target_id", "carrier_pool", "target_pool",
                       "carrier_query", "carrier_audio_kind", "target_query",
                       "target_reference_answer"]]
    single_path = args.output_dir / "single.parquet"
    pairs_path = args.output_dir / "pairs.parquet"
    selection_path = args.output_dir / "selection.parquet"
    single_out.to_parquet(single_path, index=False)
    multi_out.to_parquet(pairs_path, index=False)
    combined.to_parquet(selection_path, index=False)

    tts_characters = int(single_out["query"].str.len().sum())
    tts_characters += int(multi_out["query"].str.len().sum())
    tts_characters += int(multi_out.loc[
        multi_out.carrier_audio_kind == "tts", "carrier_query"].str.len().sum())
    raw_files = sorted(path for path in args.raw_dir.rglob("*.parquet"))
    receipt = {
        "status": "frozen_before_tts_native_expert_or_judge_outputs",
        "seed": args.seed,
        "rows": int(len(combined)),
        "standalone_rows": int(len(single_out)),
        "multiturn_rows": int(len(multi_out)),
        "real_speech_rows": int((multi_out.carrier_audio_kind == "real").sum()),
        "counts_by_pool": combined.pool.value_counts().sort_index().to_dict(),
        "counts_by_language": combined.language.value_counts().sort_index().to_dict(),
        "source_families": int(combined.source_family.nunique()),
        "tts_characters": tts_characters,
        "projected_tts_usd_at_15_per_million_characters":
            tts_characters * 15 / 1_000_000,
        "selection_sha256": sha256(selection_path),
        "single_sha256": sha256(single_path),
        "pairs_sha256": sha256(pairs_path),
        "canonical_sha256": canonical_sha(
            combined, ["id", "pool", "source_family", "language", "query",
                       "reference_answer", "raw_id", "mode"]),
        "source_revisions": REVISIONS,
        "raw_parquet_sha256": {str(path.relative_to(args.raw_dir)): sha256(path)
                               for path in raw_files},
        "selection_guard": (
            "All rows and source splits are frozen before TTS, MiniCPM answers, "
            "expert answers, judging, feature fitting, or evaluation."),
        "evaluation_guard": (
            "Choose capacity and regularization only with source-family-grouped "
            "OOF. P22/P23 are development-only. Open a new source-disjoint "
            "prospective set only after grouped OOF and both development sets pass."),
    }
    (args.output_dir / "selection_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
