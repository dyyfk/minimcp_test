import unittest
import json
from pathlib import Path

import numpy as np

from src.nvda_duplex_probe import NvidiaDuplexProbe


def artifact(dim=2):
    width = dim * 3
    return {
        "k_eot": 8,
        "fail": {
            "w": [0.0] * (width - 1) + [1.0],
            "b": 0.0,
            "thresholds": {"balanced": 0.6},
        },
        "act": {"w": [0.0] * width, "b": 10.0, "tau": 0.9},
    }


class NvidiaDuplexProbeTest(unittest.TestCase):
    def test_deployment_artifact_uses_clean_calibration_split(self):
        path = (Path(__file__).parents[1] / "data" / "gate_pull" / "new"
                / "demo" / "gate_demo_nvda.json")
        deployed = json.loads(path.read_text(encoding="utf-8"))
        # Nominal 2,310-row calibration cohort minus 52 no-commit rows.
        # The 240-row frozen internal test must never enter this fit.
        self.assertEqual(deployed["n_calib"], 2258)
        self.assertEqual(deployed["layer"], 30)
        self.assertEqual(len(deployed["fail"]["w"]), 3 * 4480)

    def test_requires_sustained_nonpad_and_full_window(self):
        probe = NvidiaDuplexProbe(artifact())
        tokens = [12, 5, 12, 12, 7, 8, 9, 10, 11, 13, 14, 15]
        decisions = []
        for i, token in enumerate(tokens):
            decision = probe.observe(np.array([i, i + 1], np.float32), token)
            if decision:
                decisions.append(decision)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].onset_frame, 4)
        self.assertEqual(decisions[0].read_frame, 11)

    def test_act_gate_bypasses_failure_head(self):
        art = artifact()
        art["act"] = {"w": [0.0] * 6, "b": -10.0, "tau": 0.9}
        probe = NvidiaDuplexProbe(art)
        decision = None
        for token in [12, 12] + [5] * 8:
            decision = probe.observe(np.ones(2, np.float32), token) or decision
        self.assertIsNotNone(decision)
        self.assertFalse(decision.is_information_request)
        self.assertFalse(decision.fired)


if __name__ == "__main__":
    unittest.main()
