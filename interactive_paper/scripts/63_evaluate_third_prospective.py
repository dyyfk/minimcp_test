"""Pinned P17 entrypoint for the shared prospective evaluator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


path = Path(__file__).with_name("60_evaluate_second_prospective.py")
spec = importlib.util.spec_from_file_location("prospective_evaluator", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.CANDIDATE_SHA = (
    "a26829f06561e34cc957e437138ea9b03571c8a487e37ee22e91001ce5541be5")
module.RESULT_STATUS = "one_shot_third_prospective_source_disjoint_validation"


if __name__ == "__main__":
    module.main()
