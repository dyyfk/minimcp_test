"""Pass-3 NVDA probe re-mix on VoiceBench AlpacaEval.

The probe is fit on the same 2,481-row calibration cohort as experiments4.py.
The pass-3 AlpacaEval replay supplies only hidden states and IDs. Local and
expert VoiceBench 1--5 outcomes are the frozen outcomes used by the original
re-mix, so only the ranking changes.

Run from this directory:
  uv run --with numpy --with pandas --with pyarrow --with scikit-learn \
    python remix_eval3_valpaca.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments import lr
from experiments3 import LayerAvg
from experiments4 import J, Pass3
from probe_lab import CALIB_TAGS, GP
from remix_eval import remix


LAYERS = (26, 30, 34)
FEATURE_FILE = GP / "onset3" / "nvda_h3_valpaca.L22-38.npz"


def valpaca_features():
    z = np.load(FEATURE_FILE, allow_pickle=True)
    assert [int(x) for x in z["layers"]] == list(J)
    keep = z["onset_frame"] >= 0
    ids = [str(x) for x in z["ids"][keep]]

    def at(layer):
        j = J[layer]
        onset = z["H_onset"][keep, j].astype(np.float32)
        return np.concatenate(
            [
                onset[:, 0],
                onset[:, -1],
                onset.mean(1),
                z["H_run"][keep, j].astype(np.float32),
            ],
            axis=1,
        )

    return ids, np.concatenate([at(layer) for layer in LAYERS], axis=1)


if __name__ == "__main__":
    blocks = ["commit", "onset_last", "onset_mean8", "run_mean"]
    cal = Pass3(
        CALIB_TAGS, lambda tag: GP / "onset_fit" / f"nvda_{tag}.parquet"
    )
    x_cal = np.concatenate(
        [cal.feats(layer, blocks) for layer in LAYERS], axis=1
    )
    model = LayerAvg(lambda: lr(1e-4), len(LAYERS)).fit(x_cal, cal.y)

    ids, x_eval = valpaca_features()
    scores = model.predict_proba(x_eval)[:, 1]
    local = pd.read_parquet(GP / "nvda_scores_valpaca.parquet").set_index("id")
    expert = (
        pd.read_parquet(GP / "nvda_expert_outcomes.parquet")
        .query("pool == 'valpaca'")
        .set_index("id")
    )
    outcomes = pd.DataFrame(
        {
            "id": ids,
            "local": [local["vb_score"].get(i, np.nan) for i in ids],
            "exp": [expert["expert_score"].get(i, np.nan) for i in ids],
        }
    )
    result = remix(outcomes, scores, "local", np.random.default_rng(8))
    report = {
        "probe": "pass-3 three-layer answer onset, L26/L30/L34",
        "metric": "VoiceBench score (1--5)",
        "protocol": "frozen local/expert outcomes; pass-3 probe ranking",
        **result,
    }
    with open("remix_eval3_valpaca.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
