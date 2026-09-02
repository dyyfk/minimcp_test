"""Export the P23 zero-forward context block mix from the frozen live gate.

P22 is development evidence.  Its fixed diagnostic grid selected alpha=.375:
blend 62.5% of the full live logit with 37.5% of the live user-mean block.
Algebraically this scales eot_last/eot_mean8 weights by .625 and leaves the
user_mean weights unchanged.  The artifact is follow-up-only and explicitly
activation-prohibited pending an independent P23 test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BLOCK = 4096


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    live = json.loads(args.live_artifact.read_text())
    if len(live["w"]) != 3 * BLOCK:
        raise RuntimeError("unexpected live feature dimension")
    weights = list(live["w"])
    for index in range(2 * BLOCK):
        weights[index] *= .625
    artifact = {
        "status": "p23_context_block_mix_frozen",
        "activation_prohibited": True,
        "live_gate_unchanged": True,
        "context_only": True,
        "first_turn_behavior": "use unchanged live gate",
        "feature_recipe": live.get("recipe"),
        "w": weights,
        "b": float(live["b"]) * .625,
        "formula": ".625 * live_logit + .375 * live_user_mean_contribution",
        "extra_model_forwards": 0,
        "development": {
            "set": "P22 dependent bilingual conversations",
            "selection_grid": [0, .125, .25, .375, .5, .625, .75, .875, 1],
            "selected_alpha": .375,
            "selection_rule": ("largest macro pool AUC gain among grid points "
                               "with pooled AUC nonnegative versus live"),
            "live_pooled_auc": .6858258928571428,
            "candidate_pooled_auc": .6861049107142857,
            "live_macro_pool_auc": .5577690722489923,
            "candidate_macro_pool_auc": .5797301677564835,
            "positive_pools": 5,
            "total_pools": 6,
        },
        "provenance": {
            "live_artifact_sha256": sha256(args.live_artifact),
            "guard": "Exported before any P23 TTS, model, judge, or score output.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output), "sha256": sha256(args.output),
        "weights": len(weights), "activation_prohibited": True,
    }, indent=2))


if __name__ == "__main__":
    main()
