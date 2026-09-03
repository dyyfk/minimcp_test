"""Aggregate five source-family folds for a frozen copied-block probe."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RATES = (.15, .30, .50)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def top_mask(score, rate):
    count = int(round(len(score) * rate))
    mask = np.zeros(len(score), dtype=bool)
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    mask[order[:count]] = True
    return mask


def routing(gain, score):
    return float(np.mean([gain[top_mask(score, rate)].sum() / len(gain)
                          for rate in RATES]))


def metrics(rows):
    local = rows.local_ok.to_numpy()
    expert = rows.expert_ok.to_numpy()
    failure, gain = 1 - local, expert - local
    live, candidate = rows.live.to_numpy(), rows.candidate.to_numpy()
    if len(np.unique(failure)) != 2:
        return {"rows": len(rows), "auc_delta": None,
                "routing_delta": routing(gain, candidate) - routing(gain, live)}
    return {
        "rows": len(rows),
        "live_auc": float(roc_auc_score(failure, live)),
        "candidate_auc": float(roc_auc_score(failure, candidate)),
        "auc_delta": float(roc_auc_score(failure, candidate)
                           - roc_auc_score(failure, live)),
        "live_routing": routing(gain, live),
        "candidate_routing": routing(gain, candidate),
        "routing_delta": routing(gain, candidate) - routing(gain, live),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.inputs))]
    if len(paths) != 5:
        raise RuntimeError(f"expected five fold files, found {len(paths)}")
    payloads = [json.loads(path.read_text()) for path in paths]
    folds = [value["split"]["fold_index"] for value in payloads]
    if sorted(folds) != list(range(5)):
        raise RuntimeError(f"invalid folds {folds}")
    keys = ("tap_layer", "copied_layer", "train_mode", "learning_rate",
            "weight_decay", "window", "anchor_alpha")
    config = {key: payloads[0]["configuration"][key] for key in keys}
    if any({key: value["configuration"][key] for key in keys} != config
           for value in payloads[1:]):
        raise RuntimeError("fold configuration mismatch")
    rows = pd.DataFrame([
        row for value in payloads for row in value["predictions"]])
    if rows.id.duplicated().any():
        raise RuntimeError("duplicate OOF prediction IDs")
    pooled = metrics(rows)
    by_source = {name: metrics(group) for name, group in
                 rows.groupby("source_family", sort=True)}
    by_language = {name: metrics(group) for name, group in
                   rows.groupby("language", sort=True)}
    by_pool = {name: metrics(group) for name, group in
               rows.groupby("pool", sort=True)}
    valid_sources = [value for value in by_source.values()
                     if value["auc_delta"] is not None]
    macro_auc_delta = float(np.mean([value["auc_delta"]
                                     for value in valid_sources]))
    macro_routing_delta = float(np.mean([value["routing_delta"]
                                         for value in by_source.values()]))
    gates = {
        "macro_routing_delta_ge_0.005": macro_routing_delta >= .005,
        "pooled_routing_nonnegative": pooled["routing_delta"] >= 0.,
        "macro_native_auc_delta_ge_0.010": macro_auc_delta >= .010,
        "pooled_native_auc_nonnegative": pooled["auc_delta"] >= 0.,
        "language_routing_nonnegative": all(
            value["routing_delta"] >= 0. for value in by_language.values()),
        "broad_pool_routing_ge_minus_0.010": all(
            value["routing_delta"] >= -.010 for value in by_pool.values()),
    }
    output = {
        "status": "opened_development_full_grouped_oof",
        "configuration": config, "rows": len(rows), "pooled": pooled,
        "macro_source": {"auc_delta": macro_auc_delta,
                         "routing_delta": macro_routing_delta,
                         "valid_auc_sources": len(valid_sources)},
        "by_source": by_source, "by_language": by_language,
        "by_pool": by_pool, "gates": gates, "all_gates_pass": all(gates.values()),
        "activation_recommended": False, "live_unchanged": True,
        "claim_boundary": ("Opened-development OOF only; passing requires "
                           "independent evidence before activation."),
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"configuration": config, "pooled": pooled,
                      "macro_source": output["macro_source"],
                      "gates": gates}, indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
