"""Evaluate the pass-3 NVDA probe on the frozen internal test set.

This is the same three-layer answer-onset recipe used by ``remix_eval3.py``,
with two safeguards for the internal split:

* all frozen test IDs are excluded from probe fitting;
* local correctness comes from the judged pass-3 answers, so outcomes and
  hidden states come from the same replay.

Rows without a valid answer-onset read remain in the 240-query denominator and
cannot escalate. Nominal 15/30/50% budgets are applied to the scoreable cohort,
so the report records both nominal and realized all-query call rates.

Run from this directory:

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn \
      python remix_eval3_internal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments import lr
from experiments4 import Pass3
from probe_lab import CALIB_TAGS, GP

HERE = Path(__file__).resolve().parent
JUDGED = HERE / "internal_pass3_judged.json"
OUTPUT = HERE / "internal_pass3_remix.json"
LAYERS = (26, 30, 34)
BLOCKS = ("commit", "onset_last", "onset_mean8", "run_mean")
RATES = (0.15, 0.30, 0.50)
NPERM = 10_000
SEED = 8


def fit_and_score() -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    queries = pd.DataFrame(
        json.loads(line) for line in (GP / "queries.jsonl").read_text().splitlines()
    )
    calib_ids = set(queries.loc[queries["split"] == "calib", "id"].astype(str))
    test_ids = set(queries.loc[queries["split"] == "test", "id"].astype(str))

    captured = Pass3(
        CALIB_TAGS, lambda tag: GP / "onset_fit" / f"nvda_{tag}.parquet"
    )
    features = [captured.feats(layer, BLOCKS) for layer in LAYERS]
    train = np.array(
        [
            tag != "frozen" or query_id in calib_ids
            for tag, query_id in zip(captured.tag, captured.ids)
        ]
    )
    test = np.array([query_id in test_ids for query_id in captured.ids])
    if int(train.sum()) != 2258 or int(test.sum()) != 223:
        raise ValueError(
            f"unexpected probe cohorts: train={train.sum()}, test={test.sum()}"
        )

    # Use training-logit scales. LayerAvg currently recomputes these scales on
    # its evaluation batch; that is transductive. Both forms produce the same
    # fixed-budget decisions here, but training scales are deployable.
    heads = [lr(1e-4).fit(x[train], captured.y[train]) for x in features]
    scales = [
        float(head.decision_function(x[train]).std())
        for head, x in zip(heads, features)
    ]
    logits = np.mean(
        [
            head.decision_function(x[test]) / scale
            for head, x, scale in zip(heads, features, scales)
        ],
        axis=0,
    )
    score_by_id = dict(zip(np.asarray(captured.ids)[test], logits))
    return score_by_id, captured.y[test], np.asarray(captured.ids)[test]


def main() -> None:
    payload = json.loads(JUDGED.read_text())
    judged = pd.DataFrame(payload["rows"])
    queries = pd.DataFrame(
        json.loads(line) for line in (GP / "queries.jsonl").read_text().splitlines()
    )
    test_queries = queries.loc[queries["split"] == "test", ["id", "pool"]].copy()
    expert = pd.read_parquet(GP.parent / "eval_expert.parquet").set_index("id")
    score_by_id, captured_labels, captured_ids = fit_and_score()

    data = test_queries.merge(
        judged[["id", "adequate", "onset_frame"]], on="id", validate="one_to_one"
    )
    data["expert"] = data["id"].map(expert["expert_adequate"])
    data["score"] = data["id"].map(score_by_id)
    if len(data) != 240 or data[["adequate", "expert"]].isna().any().any():
        raise ValueError("internal outcome coverage is incomplete")
    scoreable = data["score"].notna()
    if int(scoreable.sum()) != 223:
        raise ValueError(f"unexpected scoreable count: {scoreable.sum()}")
    if not ((data["onset_frame"] >= 0) == scoreable).all():
        raise ValueError("score coverage does not match pass-3 answer onset")

    local = data["adequate"].astype(float).to_numpy()
    expert_outcome = data["expert"].astype(float).to_numpy()
    scores = data["score"].fillna(-np.inf).to_numpy()
    scoreable_order = np.flatnonzero(scoreable)[
        np.argsort(-scores[scoreable], kind="stable")
    ]
    rng = np.random.default_rng(SEED)

    rows = {
        "always-local": {
            "accuracy": float(local.mean()),
            "call_rate": 0.0,
        }
    }
    for nominal_rate in RATES:
        k = round(nominal_rate * int(scoreable.sum()))
        escalated = np.zeros(len(data), dtype=bool)
        escalated[scoreable_order[:k]] = True
        realized_rate = float(escalated.mean())
        gated = np.where(escalated, expert_outcome, local)
        random_mean = float(
            local.mean()
            + (k / int(scoreable.sum()))
            * (expert_outcome[scoreable] - local[scoreable]).sum()
            / len(data)
        )
        permutations = np.empty(NPERM)
        for index in range(NPERM):
            chosen = rng.choice(np.flatnonzero(scoreable), size=k, replace=False)
            mask = np.zeros(len(data), dtype=bool)
            mask[chosen] = True
            permutations[index] = np.where(mask, expert_outcome, local).mean()
        rows[f"gate@{nominal_rate}"] = {
            "accuracy": float(gated.mean()),
            "call_rate": realized_rate,
            "n_escalated": k,
            "matched_random": random_mean,
            "permutation_p_random_ge_gate": float(
                (permutations >= gated.mean()).mean()
            ),
        }
    rows["matched-random@0.5"] = {
        "accuracy": rows["gate@0.5"]["matched_random"],
        "call_rate": rows["gate@0.5"]["call_rate"],
    }
    rows["always-expert"] = {
        "accuracy": float(expert_outcome.mean()),
        "call_rate": 1.0,
    }

    scoreable_fresh_labels = (
        1
        - data.loc[scoreable]
        .set_index("id")
        .loc[captured_ids, "adequate"]
        .astype(int)
        .to_numpy()
    )
    captured_scores = np.array([score_by_id[query_id] for query_id in captured_ids])
    report = {
        "protocol": {
            "probe": (
                "pass-3 layer-average L26/L30/L34; "
                "commit|onset_last|onset_mean8|run_mean"
            ),
            "fit": "2,258 committed calibration rows; all internal test IDs excluded",
            "local_outcome": "gpt-5.4-mini judgment of the same pass-3 answer",
            "expert_outcome": "frozen eval_expert.parquet::expert_adequate",
            "budget": "top nominal fraction of 223 scoreable rows",
            "no_onset_policy": "17 rows remain local and cannot escalate",
            "n_total": len(data),
            "n_scoreable": int(scoreable.sum()),
            "n_no_onset": int((~scoreable).sum()),
        },
        "scoreable_auc_fresh_labels": float(
            roc_auc_score(scoreable_fresh_labels, captured_scores)
        ),
        "pass2_to_fresh_label_flips_on_scoreable": int(
            (captured_labels != scoreable_fresh_labels).sum()
        ),
        "rows": rows,
        "by_pool": {
            pool: {
                "n": len(group),
                "local": float(group["adequate"].astype(float).mean()),
                "expert": float(group["expert"].astype(float).mean()),
                "scoreable": int(group["score"].notna().sum()),
            }
            for pool, group in data.groupby("pool", sort=True)
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
