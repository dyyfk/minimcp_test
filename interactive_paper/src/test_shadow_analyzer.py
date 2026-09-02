"""Synthetic smoke for the fail-closed shadow receipt analyzer."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


PAPER = Path(__file__).resolve().parent.parent
path = PAPER / "scripts" / "52_analyze_distilled_shadow.py"
spec = importlib.util.spec_from_file_location("shadow_analyzer", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as directory:
    log = Path(directory) / "shadow.jsonl"
    rows = []
    for index in range(20):
        failure = index % 4 == 0
        rows.append({
            "id": f"row-{index:02d}",
            "language": "zh" if index % 5 == 0 else "en",
            "live_score": index / 20,
            "shadow_score": (1 if failure else 0) + index / 100,
            "latency_ms": 2 + index / 10,
            "realized_escalation": index >= 14,
            "local_outcome": int(not failure),
            "expert_outcome": 1,
            "turn_index": 1 if index < 8 else 2 + (index % 3),
            "has_context": index >= 8,
            "prior_escalations": 0 if index < 12 else 1,
        })
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    artifact = PAPER / "data" / "gate_shadow_robust_ensemble.json"
    result = module.analyze(log, artifact, min_rows=20)
    assert result["rows"] == 20
    assert result["native_auc_candidate"] == 1.0
    assert result["budgets"]["balanced"]["candidate_accuracy"] >= 0.9
    assert result["provenance"]["artifact_sha256"]
    assert result["context_strata"]["first_turn"]["rows"] == 8
    assert result["context_strata"]["follow_up"]["rows"] == 12
    assert result["context_strata"]["prior_escalation_rows"] == 8

print("OK (shadow analyzer synthetic smoke)")
