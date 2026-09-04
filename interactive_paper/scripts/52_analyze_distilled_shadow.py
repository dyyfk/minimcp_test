"""Produce an acceptance receipt from new distilled-gate shadow logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
REQUIRED = {
    "id", "language", "live_score", "latency_ms",
    "realized_escalation", "local_outcome", "expert_outcome",
}
CONTEXT_FIELDS = {"turn_index", "has_context", "prior_escalations"}


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mask(frame, score_column, rate):
    mask = pd.Series(False, index=frame.index)
    for _language, group in frame.groupby("language", sort=True):
        k = int(round(len(group) * rate))
        chosen = group.sort_values(
            [score_column, "id"], ascending=[False, True]).index[:k]
        mask.loc[chosen] = True
    return mask.to_numpy()


def auc_or_none(target, score):
    return (None if len(np.unique(target)) < 2 else
            float(roc_auc_score(target, score)))


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, .90)),
            "p95": float(np.quantile(values, .95))}


def outcome_metrics(frame, candidate_column):
    local = frame.local_outcome.astype(int).to_numpy()
    expert = frame.expert_outcome.astype(int).to_numpy()
    native = 1 - local
    benefit = (expert - local) > 0
    live = frame.live_score.to_numpy(dtype=float)
    candidate = frame[candidate_column].to_numpy(dtype=float)
    live_native_auc = auc_or_none(native, live)
    candidate_native_auc = auc_or_none(native, candidate)
    live_benefit_auc = auc_or_none(benefit, live)
    candidate_benefit_auc = auc_or_none(benefit, candidate)
    return {
        "rows": len(frame),
        "failure_rate": float(native.mean()),
        "benefit_rate": float(benefit.mean()),
        "native_auc_live": live_native_auc,
        "native_auc_candidate": candidate_native_auc,
        "native_auc_delta": (None if live_native_auc is None else
                             candidate_native_auc - live_native_auc),
        "benefit_auc_live": live_benefit_auc,
        "benefit_auc_candidate": candidate_benefit_auc,
        "benefit_auc_delta": (None if live_benefit_auc is None else
                              candidate_benefit_auc - live_benefit_auc),
    }


def analyze(log_path: Path, artifact_path: Path, min_rows: int):
    artifact = json.loads(artifact_path.read_text())
    if (artifact.get("status") != "shadow_only" or
            artifact.get("activation_prohibited") is not True):
        raise RuntimeError("candidate artifact is not shadow-safe")
    rows = [json.loads(line) for line in log_path.read_text().splitlines()
            if line.strip()]
    frame = pd.DataFrame(rows)
    candidate_column = ("shadow_score" if "shadow_score" in frame
                        else "distilled_score")
    if candidate_column not in frame:
        raise RuntimeError(
            "missing shadow score field: expected shadow_score")
    missing = REQUIRED - set(frame)
    if missing:
        raise RuntimeError(f"missing shadow fields: {sorted(missing)}")
    if len(frame) < min_rows:
        raise RuntimeError(f"need at least {min_rows} rows, got {len(frame)}")
    if frame.id.astype(str).duplicated().any():
        raise RuntimeError("shadow log contains duplicate IDs")
    value_columns = list(REQUIRED - {"id", "language"}) + [candidate_column]
    if frame[value_columns].isna().any().any():
        raise RuntimeError("shadow log contains null required values")
    for column in ("local_outcome", "expert_outcome",
                   "realized_escalation"):
        if not frame[column].isin([0, 1, False, True]).all():
            raise RuntimeError(f"{column} must be binary")
    context_present = CONTEXT_FIELDS & set(frame)
    if context_present and context_present != CONTEXT_FIELDS:
        raise RuntimeError("context metadata must include turn_index, "
                           "has_context, and prior_escalations together")
    if context_present:
        if not frame.has_context.isin([0, 1, False, True]).all():
            raise RuntimeError("has_context must be binary")
        turn_index = pd.to_numeric(frame.turn_index, errors="coerce")
        prior = pd.to_numeric(frame.prior_escalations, errors="coerce")
        if (turn_index.isna().any() or (turn_index < 1).any()
                or (turn_index % 1 != 0).any()):
            raise RuntimeError("turn_index must be a positive integer")
        if (prior.isna().any() or (prior < 0).any()
                or (prior % 1 != 0).any()):
            raise RuntimeError("prior_escalations must be a nonnegative integer")
        expected_context = turn_index > 1
        if not np.array_equal(frame.has_context.astype(bool).to_numpy(),
                              expected_context.to_numpy()):
            raise RuntimeError("has_context must equal turn_index > 1")

    local = frame.local_outcome.astype(int).to_numpy()
    expert = frame.expert_outcome.astype(int).to_numpy()
    native = 1 - local
    benefit = (expert - local) > 0
    live = frame.live_score.to_numpy(dtype=float)
    candidate = frame[candidate_column].to_numpy(dtype=float)
    budgets = {}
    for tier, rate in RATES.items():
        live_mask = exact_mask(frame, "live_score", rate)
        candidate_mask = exact_mask(frame, candidate_column, rate)
        budgets[tier] = {
            "target_rate": rate,
            "live_realized_rate": float(live_mask.mean()),
            "candidate_realized_rate": float(candidate_mask.mean()),
            "live_accuracy": float(np.where(live_mask, expert, local).mean()),
            "candidate_accuracy": float(np.where(
                candidate_mask, expert, local).mean()),
            "accuracy_delta": float(
                np.where(candidate_mask, expert, local).mean() -
                np.where(live_mask, expert, local).mean()),
            "candidate_harmful_escalation_rate": float(np.mean(
                candidate_mask & (expert < local))),
        }
    realized = frame.realized_escalation.astype(bool).to_numpy()
    context_strata = None
    if context_present:
        context_strata = {}
        for name, mask in {
                "first_turn": ~frame.has_context.astype(bool),
                "follow_up": frame.has_context.astype(bool),
        }.items():
            group = frame.loc[mask]
            context_strata[name] = outcome_metrics(
                group, candidate_column) if len(group) else {"rows": 0}
        context_strata["prior_escalation_rows"] = int(
            (frame.prior_escalations.astype(int) > 0).sum())
    return {
        "status": "shadow_acceptance_receipt",
        "rows": len(frame),
        "languages": frame.language.value_counts().sort_index().to_dict(),
        "native_auc_live": auc_or_none(native, live),
        "native_auc_candidate": auc_or_none(native, candidate),
        "benefit_auc_live": auc_or_none(benefit, live),
        "benefit_auc_candidate": auc_or_none(benefit, candidate),
        "budgets": budgets,
        "observed_policy": {
            "realized_escalation_rate": float(realized.mean()),
            "accuracy": float(np.where(realized, expert, local).mean()),
        },
        "latency_ms": distribution(frame.latency_ms),
        "score_drift": {
            "live_mean": float(live.mean()),
            "live_std": float(live.std()),
            "candidate_mean": float(candidate.mean()),
            "candidate_std": float(candidate.std()),
            "pearson": float(np.corrcoef(live, candidate)[0, 1]),
        },
        "context_strata": context_strata,
        "provenance": {
            "log_sha256": sha256(log_path),
            "artifact_sha256": sha256(artifact_path),
            "artifact_result_sha256": artifact.get(
                "validation", {}).get("result_sha256"),
            "artifact_selection": artifact.get("selection"),
        },
        "note": "This analyzer reports evidence only and cannot activate the candidate.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()
    result = analyze(args.log, args.artifact, args.min_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.output, "sha256", sha256(args.output))


if __name__ == "__main__":
    main()
