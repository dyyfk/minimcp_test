"""Freeze a conservative live+distilled ensemble before P16 outputs exist.

P15 is development data after its one-shot P9 evaluation.  The selection rule
chooses the smallest fixed blend that clears +.015 mean AUC on the five old
external pools while improving both P15 sources.  P16 is the independent test.
"""
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
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def support_module():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_p15(native_dir: Path):
    ids, arrays = [], []
    for path in sorted(native_dir.glob("prospective_native_feats.rank*.npz")):
        archive = np.load(path, allow_pickle=True)
        ids += [str(value) for value in archive["ids"]]
        arrays.append(archive["X"].astype(np.float32, copy=False))
    x = np.concatenate(arrays)
    order = np.argsort(ids)
    return list(np.asarray(ids)[order]), x[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--distilled-artifact", type=Path, required=True)
    parser.add_argument("--p15-native-dir", type=Path, required=True)
    parser.add_argument("--p15-selection", type=Path, required=True)
    parser.add_argument("--p15-judged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    support = support_module()
    distilled = json.loads(args.distilled_artifact.read_text())

    train_x, _train_y, _ids, _groups, _blocks = support.collect_training(
        args.data_dir)
    live_train = support.artifact_score(args.live_artifact, train_x)
    distilled_train = (train_x[:, BLOCK:] @ np.asarray(distilled["w"])
                       + float(distilled["b"]))
    live_mean, live_std = float(live_train.mean()), float(live_train.std())
    dist_mean = float(distilled_train.mean())
    dist_std = float(distilled_train.std())

    external = []
    for tag, _label in support.EXTERNAL:
        ids, x = support.load_feats(args.data_dir, tag + "off")
        labels = (pd.read_parquet(
            args.data_dir / f"frozen_native_{tag}off_judged.parquet")
            .drop_duplicates("id", keep="last").set_index("id")["adequate"]
            .to_dict())
        keep = [index for index, row_id in enumerate(ids) if row_id in labels]
        y = np.asarray([1 - int(labels[ids[index]]) for index in keep])
        x = x[keep]
        external.append((tag, y, support.artifact_score(args.live_artifact, x),
                         x[:, BLOCK:] @ np.asarray(distilled["w"])
                         + float(distilled["b"])))

    p15_ids, p15_x = load_p15(args.p15_native_dir)
    p15_labels = args.p15_judged
    judged = pd.read_parquet(p15_labels).drop_duplicates("id").set_index("id")
    p15_y = np.asarray([1 - int(judged.loc[row_id, "adequate"])
                        for row_id in p15_ids])
    pools = (pd.read_parquet(args.p15_selection).set_index("id")
             .loc[p15_ids, "pool"].to_numpy())
    p15_live = support.artifact_score(args.live_artifact, p15_x)
    p15_dist = (p15_x[:, BLOCK:] @ np.asarray(distilled["w"])
                + float(distilled["b"]))

    def score(live, dist, alpha):
        return ((live - live_mean) / live_std
                + alpha * (dist - dist_mean) / dist_std)

    rows, chosen = [], None
    for alpha in ALPHAS:
        external_delta = {
            tag: float(roc_auc_score(y, score(live, dist, alpha))
                       - roc_auc_score(y, live))
            for tag, y, live, dist in external
        }
        p15_delta = {
            pool: float(roc_auc_score(
                p15_y[pools == pool],
                score(p15_live[pools == pool], p15_dist[pools == pool], alpha))
                - roc_auc_score(p15_y[pools == pool], p15_live[pools == pool]))
            for pool in sorted(set(pools))
        }
        row = {"alpha": alpha, "external_auc_delta": external_delta,
               "external_mean_auc_delta": float(np.mean(list(
                   external_delta.values()))), "p15_auc_delta": p15_delta,
               "p15_both_positive": bool(all(v > 0 for v in p15_delta.values()))}
        rows.append(row)
        if (chosen is None and row["external_mean_auc_delta"] >= .015
                and row["p15_both_positive"]):
            chosen = row
    if chosen is None:
        raise RuntimeError("no fixed blend clears the frozen selection rule")
    alpha = float(chosen["alpha"])

    live = json.loads(args.live_artifact.read_text())
    live_w = np.asarray(live["w"], dtype=np.float64)
    dist_w = np.zeros(3 * BLOCK, dtype=np.float64)
    dist_w[BLOCK:] = np.asarray(distilled["w"], dtype=np.float64)
    folded_w = live_w / live_std + alpha * dist_w / dist_std
    folded_b = ((float(live["b"]) - live_mean) / live_std
                + alpha * (float(distilled["b"]) - dist_mean) / dist_std)
    direct = train_x @ folded_w + folded_b
    expected = score(live_train, distilled_train, alpha)
    max_difference = float(np.max(np.abs(direct - expected)))
    if max_difference > 2e-5:
        raise RuntimeError(f"fold mismatch: {max_difference}")

    artifact = {
        "status": "shadow_only", "activation_prohibited": True,
        "live_gate_unchanged": True,
        "layer": 22, "modes": ["eot_last", "eot_mean8", "user_mean"],
        "k_eot": 8, "w": folded_w.tolist(), "b": float(folded_b),
        "feature_recipe": {"blocks": ["eot_last", "eot_mean8", "user_mean"],
                           "dimension": 3 * BLOCK},
        "selection": {"rule": "smallest alpha with old-external mean AUC delta >= .015 and both P15 source deltas > 0",
                      "grid": ALPHAS, "chosen_alpha": alpha,
                      "p16_used_for_selection": False},
        "provenance": {
            "live_artifact_sha256": support.sha256(args.live_artifact),
            "distilled_artifact_sha256": support.sha256(
                args.distilled_artifact),
            "p15_judged_sha256": support.sha256(args.p15_judged),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    receipt = {"status": "frozen_before_p16_outputs", "rows": rows,
               "chosen": chosen, "live_train_score_mean": live_mean,
               "live_train_score_std": live_std,
               "distilled_train_score_mean": dist_mean,
               "distilled_train_score_std": dist_std,
               "fold_max_abs_difference": max_difference,
               "artifact_sha256": support.sha256(args.output)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
