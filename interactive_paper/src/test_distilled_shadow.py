"""Safety contract for the inactive robust-gate shadow integration.

Run: python interactive_paper/src/test_distilled_shadow.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, os.fspath(HERE))
from gate import Probe  # noqa: E402


artifact = json.loads((
    PAPER / "data" / "shadow" / "gate_shadow_robust_ensemble.json").read_text())
source_path = PAPER / "demo_duplex.py"
source = source_path.read_text()
tree = ast.parse(source)
checks = 0


def check(condition, message):
    global checks
    checks += 1
    assert condition, message


check(artifact["status"] == "shadow_only", "artifact stays shadow-only")
check(artifact["activation_prohibited"] is True,
      "artifact explicitly prohibits activation")
check(artifact["live_gate_unchanged"] is True,
      "artifact records that the live gate is unchanged")
check(artifact["selection"]["chosen_alpha"] == 1.0,
      "packaged candidate is the P16 alpha-1 ensemble")
check("thresholds" not in artifact and "eot_thresholds" not in artifact,
      "artifact carries no activation thresholds")
check(artifact["feature_recipe"]["blocks"] == [
    "eot_last", "eot_mean8", "user_mean"],
      "feature recipe is the frozen alpha-1 ensemble recipe")
check(len(artifact["w"]) == artifact["feature_recipe"]["dimension"] == 12288,
      "coefficient dimension matches the feature recipe")
check(0 < Probe(artifact["w"], artifact["b"]).score([0.] * 12288) < 1,
      "artifact loads in the production pure-Python Probe")

fired_values = []
for node in ast.walk(tree):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "fired"
               for target in targets):
            fired_values.append(node.value)
check(len(fired_values) == 1, "demo has one auditable live fired assignment")
shadow_refs = [node for node in ast.walk(fired_values[0])
               if ((isinstance(node, ast.Name) and "shadow" in node.id) or
                   (isinstance(node, ast.Attribute) and
                    "shadow" in node.attr))]
check(not shadow_refs, "shadow state cannot enter the live fired expression")
context_refs = [node.id for node in ast.walk(fired_values[0])
                if isinstance(node, ast.Name) and node.id in {
                    "completed_turns", "prior_escalations"}]
check(not context_refs,
      "observational context state cannot enter the live fired expression")
check('"shadow_v"' in source and '"shadow_score"' in source,
      "shadow values remain observable in websocket events")
check(all(f'"{field}"' in source for field in (
    "turn_index", "has_context", "prior_escalations")),
      "pre-answer context strata remain observable in gate events")
check("_SHADOW_ARTIFACT" in source and ".add_local_file(" in source,
      "the frozen artifact is packaged into the runtime image")
check("gate_shadow_robust_ensemble.json" in source,
      "the runtime packages the robust P16 artifact")

print(f"OK ({checks} distilled-shadow safety checks)")
