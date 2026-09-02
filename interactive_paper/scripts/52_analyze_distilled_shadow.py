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
    "id", "language", "live_score", "distilled_score", "latency_ms",
    "realized_escalation", "local_outcome", "expert_outcome",
}


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


def analyze(log_path: Path, artifact_path: Path, min_rows: int):
    artifact = json.loads(artifact_path.read_text())
    if (artifact.get("status") != "shadow_only" or
            artifact.get("activation_prohibited") is not True):
        raise RuntimeError("candidate artifact is not shadow-safe")
    rows = [json.loads(line) for line in log_path.read_text().splitlines()
            if line.strip()]
    frame = pd.DataFrame(rows)
    missing = REQUIRED - set(frame)
    if missing:
        raise RuntimeError(f"missing shadow fields: {sorted(missing)}")
    if len(frame) < min_rows:
        raise RuntimeError(f"need at least {min_rows} rows, got {len(frame)}")
    if frame.id.astype(str).duplicated().any():
        raise RuntimeError("shadow log contains duplicate IDs")
    if frame[list(REQUIRED - {"id", "language"})].isna().any().any():
        raise RuntimeError("shadow log contains null required values")
    for column in ("local_outcome", "expert_outcome",
                   "realized_escalation"):
        if not frame[column].isin([0, 1, False, True]).all():
            raise RuntimeError(f"{column} must be binary")

    local = frame.local_outcome.astype(int).to_numpy()
    expert = frame.expert_outcome.astype(int).to_numpy()
    native = 1 - local
    benefit = (expert - local) > 0
    live = frame.live_score.to_numpy(dtype=float)
    candidate = frame.distilled_score.to_numpy(dtype=float)
    budgets = {}
    for tier, rate in RATES.items():
        live_mask = exact_mask(frame, "live_score", rate)
        candidate_mask = exact_mask(frame, "distilled_score", rate)
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
        "provenance": {
            "log_sha256": sha256(log_path),
            "artifact_sha256": sha256(artifact_path),
            "artifact_result_sha256": artifact["validation"]["result_sha256"],
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
