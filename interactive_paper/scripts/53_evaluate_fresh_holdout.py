"""One-shot evaluation of the frozen distilled gate on FreshQA heldout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BLOCK = 4096
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
LIVE_SHA256 = "0e6494c2eeac9bcd86c10b5def3cbd32e98bb0765fa2fd8afc8c1b47915ea372"
DISTILLED_SHA256 = "c85e0697788b2f8ce819fc963aa68c5a5ef34e0ae59c4f58b7917ccbf848dbb0"


def load_p3a():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metrics(p3a, target, candidate, live):
    result = {
        "positive_rate": float(np.mean(target)),
        "live_auc": float(roc_auc_score(target, live)),
        "candidate_auc": float(roc_auc_score(target, candidate)),
        "candidate_vs_live_delta_ci": p3a.bootstrap_auc_delta(
            target, candidate, live),
        "budgets": {},
    }
    for tier, rate in RATES.items():
        live_mask = p3a.top_mask(live, rate)
        candidate_mask = p3a.top_mask(candidate, rate)
        result["budgets"][tier] = {
            "target_rate": rate,
            "live_precision": float(target[live_mask].mean()),
            "candidate_precision": float(target[candidate_mask].mean()),
            "live_recall": float(target[live_mask].sum() / target.sum()),
            "candidate_recall": float(
                target[candidate_mask].sum() / target.sum()),
            "selection_overlap": float(np.mean(live_mask == candidate_mask)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p3a = load_p3a()
    live_sha = p3a.sha256(args.live_artifact)
    candidate_sha = p3a.sha256(args.candidate_artifact)
    if live_sha != LIVE_SHA256 or candidate_sha != DISTILLED_SHA256:
        raise RuntimeError(
            f"artifact mismatch: live={live_sha} candidate={candidate_sha}")

    ids, x = p3a.load_feats(args.data_dir, "freshoff")
    positions = {row_id: index for index, row_id in enumerate(ids)}
    metadata_path = args.data_dir / "fresh_labels.parquet"
    judged_path = args.data_dir / "frozen_native_freshoff_judged.parquet"
    metadata = pd.read_parquet(metadata_path)
    judged = (pd.read_parquet(judged_path).drop_duplicates("id", keep="last")
              [["id", "adequate"]])
    heldout = metadata[metadata.split == "heldout"][["id", "pool"]].merge(
        judged, on="id", validate="one_to_one")
    heldout = heldout[heldout.id.isin(positions)].sort_values("id")
    if len(heldout) != 60 or heldout.adequate.isna().any():
        raise RuntimeError(f"expected 60 judged heldout rows, got {len(heldout)}")
    features = x[[positions[row_id] for row_id in heldout.id]]
    live = p3a.artifact_score(args.live_artifact, features)
    candidate_artifact = json.loads(args.candidate_artifact.read_text())
    candidate = (features[:, BLOCK:] @ np.asarray(candidate_artifact["w"]) +
                 float(candidate_artifact["b"]))
    native_failure = 1 - heldout.adequate.astype(int).to_numpy()
    policy_failure = np.where(
        heldout.pool.to_numpy() == "fresh_fast", 1, native_failure)
    result = {
        "status": "one_shot_frozen_holdout",
        "rows": len(heldout),
        "pool_counts": heldout.pool.value_counts().sort_index().to_dict(),
        "official_native_failure": metrics(
            p3a, native_failure, candidate, live),
        "deployed_fresh_policy": metrics(
            p3a, policy_failure, candidate, live),
        "provenance": {
            "live_artifact_sha256": live_sha,
            "candidate_artifact_sha256": candidate_sha,
            "metadata_sha256": p3a.sha256(metadata_path),
            "judged_sha256": p3a.sha256(judged_path),
            "ordered_id_sha256": hashlib.sha256(
                "\n".join(heldout.id).encode()).hexdigest(),
        },
        "note": "Candidate was frozen before this split was opened; no selection or retuning uses these rows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("receipt_sha256", p3a.sha256(args.output))


if __name__ == "__main__":
    main()
