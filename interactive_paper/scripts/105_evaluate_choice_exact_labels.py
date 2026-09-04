"""P35: replace noisy judge labels with high-confidence exact-choice labels.

This is a label-floor diagnostic on the already-opened P25-B standalone set.
It keeps the deployed L22 feature recipe and a single fixed C=3e-4 logistic
readout.  Only rows with a reference option and an unambiguous answer cue are
used; neither answer content nor judge outcomes select a regex or model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


C = 3e-4
RATES = (.15, .30, .50)
REF = re.compile(r"^\s*\(([A-J])\)", re.I)
CUES = tuple(re.compile(pattern, re.I) for pattern in (
    r"\b(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?:is|:|=)\s*(?:option\s+|choice\s+)?[\(\[]?([A-J])\b",
    r"\b(?:the\s+)?(?:correct\s+|final\s+)?(?:option|choice)\s*[\(\[]?([A-J])\b\s*(?:is\s+correct|is\s+the\s+answer)",
    r"\b(?:I\s+(?:would\s+)?(?:choose|select|pick)|choose|select|pick|go\s+with)\s+(?:option\s+|choice\s+)?[\(\[]?([A-J])\b",
    r"(?:答案|正确答案|最终答案)\s*(?:是|为|：|:)\s*(?:选项)?\s*[（(\[]?([A-J])\b",
    r"(?:选择|选)\s*(?:选项)?\s*[（(\[]?([A-J])\b",
))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_choice(text: str) -> str | None:
    hits = [match.group(1).upper() for pattern in CUES
            for match in pattern.finditer(str(text))]
    return hits[0] if hits and len(set(hits)) == 1 else None


def load_npz(directory: Path):
    ids, arrays, paths = [], [], sorted(
        directory.glob("prospective_native_feats.rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"no native feature shards under {directory}")
    for path in paths:
        data = np.load(path)
        ids.extend(map(str, data["ids"]))
        arrays.append(data["X"].astype(np.float32, copy=False))
    matrix = np.concatenate(arrays)
    index = pd.DataFrame({"id": ids, "row": np.arange(len(ids))})
    if index.id.duplicated().any():
        raise RuntimeError("duplicate feature IDs")
    return index, matrix, paths


def artifact_score(path: Path, values):
    artifact = json.loads(path.read_text())
    return values @ np.asarray(artifact["w"], np.float32) + float(artifact["b"])


def top_mask(score, rate):
    count = int(round(rate * len(score)))
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:count]] = True
    return mask


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def macro_auc(y, score, groups):
    values = [auc(y[groups == group], score[groups == group])
              for group in sorted(set(groups))]
    values = [value for value in values if value is not None]
    return float(np.mean(values)), len(values)


def routing(gain, score):
    return float(np.mean([
        gain[top_mask(score, rate)].sum() / len(gain) for rate in RATES]))


def macro_routing(gain, score, groups):
    return float(np.mean([
        routing(gain[groups == group], score[groups == group])
        for group in sorted(set(groups))]))


def summarize(exact_y, judge_y, local_ok, expert_ok, score, groups):
    gain = expert_ok - local_ok
    exact_macro, exact_valid = macro_auc(exact_y, score, groups)
    judge_macro, judge_valid = macro_auc(judge_y, score, groups)
    return {
        "exact_auc_pooled": auc(exact_y, score),
        "exact_auc_macro_source": exact_macro,
        "exact_valid_sources": exact_valid,
        "judge_auc_pooled": auc(judge_y, score),
        "judge_auc_macro_source": judge_macro,
        "judge_valid_sources": judge_valid,
        "routing_pooled": routing(gain, score),
        "routing_macro_source": macro_routing(gain, score, groups),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame.mode == "standalone"]
    local = pd.read_parquet(args.local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    feature_index, matrix, feature_paths = load_npz(args.features)
    if matrix.shape[1:] != (12288,) or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid feature matrix {matrix.shape}")
    frame = frame.merge(local[["id", "answer", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(feature_index, on="id", validate="one_to_one")
    frame["reference_choice"] = frame.reference_answer.astype(str).str.extract(REF)[0].str.upper()
    frame["answer_choice"] = frame.answer.map(answer_choice)
    reference_rows = int(frame.reference_choice.notna().sum())
    frame = frame[frame.reference_choice.notna() & frame.answer_choice.notna()]
    frame = frame.sort_values("id").reset_index(drop=True)
    x = matrix[frame.row.to_numpy()]
    exact_y = (frame.answer_choice != frame.reference_choice).astype(int).to_numpy()
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    judge_y = 1 - local_ok
    groups = frame.source_family.astype(str).to_numpy()
    strat = exact_y * 4 + judge_y * 2 + (expert_ok - local_ok + 1)
    live = artifact_score(args.live_artifact, x)

    oof = np.full(len(frame), np.nan)
    folds = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, strat, groups))
    fold_rows = []
    for fold, (train, test) in enumerate(folds):
        model = LogisticRegression(C=C, max_iter=5000, tol=1e-5)
        model.fit(x[train], exact_y[train])
        oof[test] = model.decision_function(x[test])
        fold_rows.append({"fold": fold,
                          "test_rows": int(len(test)),
                          "test_sources": sorted(set(groups[test])),
                          "exact_failure_rate": float(exact_y[test].mean())})
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions are incomplete")

    baseline = summarize(exact_y, judge_y, local_ok, expert_ok, live, groups)
    candidate = summarize(exact_y, judge_y, local_ok, expert_ok, oof, groups)
    deltas = {key: candidate[key] - baseline[key] for key in (
        "exact_auc_pooled", "exact_auc_macro_source",
        "judge_auc_pooled", "judge_auc_macro_source",
        "routing_pooled", "routing_macro_source")}
    strata = {}
    gain = expert_ok - local_ok
    for column in ("language", "pool"):
        values = frame[column].astype(str).to_numpy()
        strata[column] = {}
        for value in sorted(set(values)):
            mask = values == value
            strata[column][value] = {
                "rows": int(mask.sum()),
                "exact_auc_delta": ((auc(exact_y[mask], oof[mask]) or 0.) -
                                    (auc(exact_y[mask], live[mask]) or 0.)),
                "judge_auc_delta": ((auc(judge_y[mask], oof[mask]) or 0.) -
                                    (auc(judge_y[mask], live[mask]) or 0.)),
                "routing_delta": routing(gain[mask], oof[mask]) - routing(
                    gain[mask], live[mask]),
            }

    coverage = len(frame) / max(1, reference_rows)
    gates = {
        "choice_answer_coverage_ge_0.80": coverage >= .80,
        "exact_macro_auc_delta_ge_0.010": deltas["exact_auc_macro_source"] >= .010,
        "exact_pooled_auc_delta_ge_0.005": deltas["exact_auc_pooled"] >= .005,
        "judge_macro_auc_nonnegative": deltas["judge_auc_macro_source"] >= 0.,
        "judge_pooled_auc_nonnegative": deltas["judge_auc_pooled"] >= 0.,
        "routing_macro_delta_ge_0.005": deltas["routing_macro_source"] >= .005,
        "routing_pooled_nonnegative": deltas["routing_pooled"] >= 0.,
        "language_routing_nonnegative": all(
            row["routing_delta"] >= 0. for row in strata["language"].values()),
        "broad_pool_routing_ge_minus_0.010": all(
            row["routing_delta"] >= -.010 for row in strata["pool"].values()),
    }
    payload = {
        "status": "p35_exact_choice_label_development",
        "protocol": {
            "feature": "unchanged deployed L22 12288-d onset recipe",
            "head": "single raw-feature logistic C=3e-4",
            "training_label": "unambiguous answer-cue choice mismatch",
            "selection": "none; regexes and C frozen before answer parsing",
            "validation_labels": "existing judge adequacy and expert gain",
            "outer_cv": "5-fold source-family-disjoint OOF",
            "pass_action": "unchanged historical P15/P16/P17/P32 exact-choice transfer only",
            "base_trainable_parameters": 0,
            "new_readout_parameters": 12289,
        },
        "rows": int(len(frame)), "reference_choice_rows": reference_rows,
        "coverage": coverage, "source_families": int(frame.source_family.nunique()),
        "exact_failure_rate": float(exact_y.mean()),
        "judge_failure_rate": float(judge_y.mean()),
        "exact_judge_agreement": float(np.mean(exact_y == judge_y)),
        "folds": fold_rows, "baseline": baseline, "candidate_oof": candidate,
        "deltas": deltas, "strata": strata, "gates": gates,
        "all_gates_pass": all(gates.values()),
        "activation_recommended": False, "live_unchanged": True,
        "provenance": {
            "script": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
            "selection": {"path": str(args.selection), "sha256": sha256(args.selection)},
            "local": {"path": str(args.local), "sha256": sha256(args.local)},
            "expert": {"path": str(args.expert), "sha256": sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact), "sha256": sha256(args.live_artifact)},
            "features": [{"path": str(path), "sha256": sha256(path)}
                         for path in feature_paths],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "rows", "reference_choice_rows", "coverage", "source_families",
        "exact_failure_rate", "judge_failure_rate", "exact_judge_agreement",
        "deltas", "strata", "gates", "all_gates_pass")}, indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
