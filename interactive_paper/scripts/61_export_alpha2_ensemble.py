"""Freeze the alpha-2 single-pass ensemble after P16 becomes development."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BLOCK = 4096
ALPHA = 2.0


def support_module():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--distilled-artifact", type=Path, required=True)
    parser.add_argument("--alpha1-receipt", type=Path, required=True)
    parser.add_argument("--p16-result", type=Path, required=True)
    parser.add_argument("--p16-native-dir", type=Path, required=True)
    parser.add_argument("--p16-selection", type=Path, required=True)
    parser.add_argument("--p16-judged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    support = support_module()
    live = json.loads(args.live_artifact.read_text())
    distilled = json.loads(args.distilled_artifact.read_text())
    prior = json.loads(args.alpha1_receipt.read_text())
    p16 = json.loads(args.p16_result.read_text())
    live_mean = float(prior["live_train_score_mean"])
    live_std = float(prior["live_train_score_std"])
    dist_mean = float(prior["distilled_train_score_mean"])
    dist_std = float(prior["distilled_train_score_std"])

    live_w = np.asarray(live["w"], dtype=np.float64)
    distilled_w = np.zeros(3 * BLOCK, dtype=np.float64)
    distilled_w[BLOCK:] = np.asarray(distilled["w"], dtype=np.float64)
    w = live_w / live_std + ALPHA * distilled_w / dist_std
    b = ((float(live["b"]) - live_mean) / live_std
         + ALPHA * (float(distilled["b"]) - dist_mean) / dist_std)
    artifact = {
        "status": "shadow_only", "activation_prohibited": True,
        "live_gate_unchanged": True,
        "layer": 22, "modes": ["eot_last", "eot_mean8", "user_mean"],
        "k_eot": 8, "w": w.tolist(), "b": float(b),
        "feature_recipe": {"blocks": ["eot_last", "eot_mean8", "user_mean"],
                           "dimension": 3 * BLOCK},
        "selection": {
            "alpha": ALPHA,
            "reason": "Alpha 2 was in the pre-P16 grid, retained positive P15 deltas in both sources, and after P16 became development data improved all three P16 sources while clearing +.015 macro AUC.",
            "p17_used_for_selection": False,
        },
        "provenance": {
            "live_artifact_sha256": support.sha256(args.live_artifact),
            "distilled_artifact_sha256": support.sha256(args.distilled_artifact),
            "alpha1_selection_receipt_sha256": support.sha256(
                args.alpha1_receipt),
            "p16_result_sha256": support.sha256(args.p16_result),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    ids, arrays = [], []
    for path in sorted(args.p16_native_dir.glob(
            "prospective_native_feats.rank*.npz")):
        archive = np.load(path, allow_pickle=True)
        ids += [str(value) for value in archive["ids"]]
        arrays.append(archive["X"].astype(np.float32, copy=False))
    x = np.concatenate(arrays)
    order = np.argsort(ids)
    ids = list(np.asarray(ids)[order])
    x = x[order]
    judged = pd.read_parquet(args.p16_judged).drop_duplicates("id").set_index("id")
    labels = np.asarray([1 - int(judged.loc[row_id, "adequate"])
                         for row_id in ids])
    pools = (pd.read_parquet(args.p16_selection).set_index("id")
             .loc[ids, "pool"].to_numpy())
    live_score = x @ live_w + float(live["b"])
    candidate_score = x @ w + b
    p16_deltas = {}
    for pool in sorted(set(pools)):
        mask = pools == pool
        p16_deltas[pool] = float(
            roc_auc_score(labels[mask], candidate_score[mask])
            - roc_auc_score(labels[mask], live_score[mask]))
    receipt = {
        "status": "frozen_before_p17_selection_or_outputs",
        "alpha": ALPHA,
        "p15_grid_row": next(row for row in prior["rows"]
                              if row["alpha"] == ALPHA),
        "p16_alpha1_result": {
            "macro_source_auc_delta": p16["macro_source_auc_delta"],
        },
        "p16_alpha2_auc_delta": p16_deltas,
        "p16_alpha2_macro_auc_delta": float(np.mean(list(p16_deltas.values()))),
        "p16_note": "P16 is development-only for this new artifact; P17 is the independent test.",
        "artifact_sha256": support.sha256(args.output),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
