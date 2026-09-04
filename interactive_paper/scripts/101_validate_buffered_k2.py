"""Open the frozen P36 accuracy gate from already-published native results."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "figures/p36_reasoning_buffered_k2_freeze.json"
SOURCE = ROOT / "figures/two_stage.json"
RESULT = ROOT / "figures/p36_reasoning_buffered_k2_result.json"


freeze = json.loads(FREEZE.read_text())
source = json.loads(SOURCE.read_text())
balanced = source["0.3"]
onset = balanced["d0.0_f1.0"]["pools"]["sreasonk"]
k2 = balanced["d1.0_f1.0"]["pools"]["sreasonk"]
delta = k2 - onset

result = {
    "experiment": freeze["experiment"],
    "status": "mechanics_pass_accuracy_pending_row_level_replay",
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "budget": 0.3,
    "pool": "Reasoning-zh",
    "onset_delivered_accuracy": onset,
    "k2_delivered_accuracy": k2,
    "delta": delta,
    "minimum_delta": 0.03,
    "cached_k2_only_delta_gate_pass": delta >= 0.03,
    "p36_end_to_end_accuracy_gate": "not_evaluable_from_aggregate",
    "latency_cost": "two post-onset answer-chunk intervals; three generated chunks buffered",
    "measurement_limit": (
        "scripts/39_two_stage.py pads rows that ended before k2 with their last "
        "available state and still assigns the exact per-pool quota. P36 instead "
        "releases a pre-k2 short answer locally, so the aggregate k2-only number "
        "is a candidate upper bound, not an exact replay of P36."
    ),
    "required_replay_fields": ["id", "n_post", "local_outcome", "expert_outcome", "k2_score"],
    "activation": "not authorized",
}
RESULT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["cached_k2_only_delta_gate_pass"]:
    raise SystemExit("cached k2-only candidate gate failed")
