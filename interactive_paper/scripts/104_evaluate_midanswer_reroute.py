"""Evaluate a fixed 15% onset + 15% answer-chunk-4 rerouting policy.

The deployed probe and feature recipe are unchanged.  At an exact 30% total
budget, the policy spends the first half on the onset live ranking, then ranks
the remaining rows by the same probe read after four answer chunks.  Rows that
already ended by chunk four cannot be selected by the second stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(directory, pattern, row_name):
    ids, values, paths = [], [], sorted(directory.glob(pattern))
    for path in paths:
        data = np.load(path)
        ids.extend(map(str, data["ids"]))
        values.append(data["X"])
    matrix = np.concatenate(values) if values else np.empty((0, 0))
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate ids in {pattern}")
    return pd.DataFrame({"id": ids, row_name: np.arange(len(ids))}), matrix, paths


def load_traces(directory, pattern):
    rows, paths = {}, sorted(directory.glob(pattern))
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows, paths


def score(artifact, values):
    payload = json.loads(artifact.read_text())
    return values @ np.asarray(payload["w"], dtype=np.float32) + float(payload["b"])


def top_indices(values, count, eligible=None):
    values = np.asarray(values)
    allowed = np.ones(len(values), dtype=bool) if eligible is None else eligible.copy()
    order = np.lexsort((np.arange(len(values)), -values))
    order = order[allowed[order]]
    return order[:count]


def policy_masks(live, late, available):
    n = len(live)
    target = int(round(.30 * n))
    initial_n = int(round(.15 * n))
    live_mask = np.zeros(n, dtype=bool)
    live_mask[top_indices(live, target)] = True
    candidate = np.zeros(n, dtype=bool)
    candidate[top_indices(live, initial_n)] = True
    remaining = available & ~candidate
    candidate[top_indices(late, target - candidate.sum(), remaining)] = True
    return live_mask, candidate


def metrics(rows):
    local = rows.local_ok.astype(int).to_numpy()
    expert = rows.expert_ok.astype(int).to_numpy()
    failure, gain = 1 - local, expert - local
    live = rows.live.to_numpy()
    late = rows.late.to_numpy()
    available = rows.late_available.astype(bool).to_numpy()
    live_mask, candidate = policy_masks(live, late, available)

    def values(mask):
        selected = max(1, int(mask.sum()))
        return {
            "selected": int(mask.sum()),
            "rate": float(mask.mean()),
            "failure_precision": float(failure[mask].sum() / selected),
            "failure_recall": float(failure[mask].sum() / max(1, failure.sum())),
            "routing_gain": float(gain[mask].sum() / len(rows)),
            "cascade_accuracy": float((local.sum() + gain[mask].sum()) / len(rows)),
            "harmful_rate": float(np.mean(gain[mask] < 0)) if mask.any() else 0.,
        }

    baseline, reroute = values(live_mask), values(candidate)
    output = {
        "rows": len(rows), "failures": int(failure.sum()),
        "late_available": int(available.sum()),
        "late_coverage": float(available.mean()),
        "live_30": baseline, "onset15_late15": reroute,
        "deltas": {
            "failure_precision": reroute["failure_precision"] - baseline["failure_precision"],
            "failure_recall": reroute["failure_recall"] - baseline["failure_recall"],
            "routing_gain": reroute["routing_gain"] - baseline["routing_gain"],
            "cascade_accuracy": reroute["cascade_accuracy"] - baseline["cascade_accuracy"],
            "harmful_rate": reroute["harmful_rate"] - baseline["harmful_rate"],
        },
    }
    if len(np.unique(failure[available])) == 2:
        output["late_available_auc"] = float(
            roc_auc_score(failure[available], late[available]))
        output["onset_available_auc"] = float(
            roc_auc_score(failure[available], live[available]))
        output["late_minus_onset_available_auc"] = (
            output["late_available_auc"] - output["onset_available_auc"])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--midanswer-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--seed-namespace", default="p15-native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mi, mx, mid_paths = load_npz(
        args.midanswer_dir, "midanswer_feats.rank*.npz", "mid_row")
    oi, ox, original_paths = load_npz(
        args.original_dir, "prospective_native_feats.rank*.npz", "original_row")
    if mx.shape[1:] != (12288,) or ox.shape[1:] != (12288,):
        raise RuntimeError(f"invalid feature shapes {mx.shape} {ox.shape}")
    if not np.isfinite(mx).all() or not np.isfinite(ox).all():
        raise RuntimeError("non-finite features")

    frame = pd.read_parquet(args.selection).drop_duplicates("id")
    if "mode" in frame:
        frame = frame[frame["mode"] == "standalone"]
    local = pd.read_parquet(args.local).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    frame = frame.merge(local[["id", "adequate"]].rename(
        columns={"adequate": "local_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(expert[["id", "adequate"]].rename(
        columns={"adequate": "expert_ok"}), on="id", validate="one_to_one")
    frame = frame.merge(oi, on="id", validate="one_to_one")
    frame = frame.merge(mi, on="id", how="left", validate="one_to_one")
    frame = frame.sort_values("id").reset_index(drop=True)
    frame["late_available"] = frame.mid_row.notna()
    live = score(args.live_artifact, ox[frame.original_row.to_numpy()])
    late = np.full(len(frame), -np.inf, dtype=np.float32)
    available = frame.late_available.to_numpy()
    late[available] = score(
        args.live_artifact, mx[frame.loc[available, "mid_row"].astype(int)])
    frame["live"], frame["late"] = live, late

    mid_traces, mid_trace_paths = load_traces(
        args.midanswer_dir, "midanswer_traces.rank*.jsonl")
    original_traces, original_trace_paths = load_traces(
        args.original_dir, "prospective_native_traces.rank*.jsonl")
    fields = ("seed", "onset_chunk", "answer_text", "eot_seen", "error")
    mismatches = []
    for row_id in frame.id.astype(str):
        if row_id not in mid_traces or row_id not in original_traces:
            mismatches.append({"id": row_id, "field": "missing_trace"})
            continue
        for field in fields:
            if mid_traces[row_id].get(field) != original_traces[row_id].get(field):
                mismatches.append({"id": row_id, "field": field})
        expected_seed = int(hashlib.sha256(
            f"{args.seed_namespace}:{row_id}".encode()).hexdigest()[:8], 16)
        if mid_traces[row_id].get("seed") != expected_seed:
            mismatches.append({"id": row_id, "field": "seed_namespace"})
    replay = {"rows": len(frame), "fields": list(fields),
              "mismatch_count": len(mismatches),
              "mismatches": mismatches[:20], "exact": not mismatches}

    pooled = metrics(frame)
    by_source = {name: metrics(rows) for name, rows in
                 frame.groupby("source_family", sort=True)}
    by_pool = {name: metrics(rows) for name, rows in
               frame.groupby("pool", sort=True)}
    by_language = {name: metrics(rows) for name, rows in
                   frame.groupby("language", sort=True)}
    source_deltas = [row["deltas"]["routing_gain"] for row in by_source.values()]
    pool_deltas = [row["deltas"]["routing_gain"] for row in by_pool.values()]
    gates = {
        "replay_exact": replay["exact"],
        "late_coverage_ge_0.80": pooled["late_coverage"] >= .80,
        "exact_0.30_budget": (pooled["onset15_late15"]["selected"]
                              == pooled["live_30"]["selected"]),
        "pooled_routing_delta_ge_0.005": pooled["deltas"]["routing_gain"] >= .005,
        "macro_source_routing_nonnegative": float(np.mean(source_deltas)) >= 0.,
        "failure_precision_nonnegative": pooled["deltas"]["failure_precision"] >= 0.,
        "harmful_rate_nonincreasing": pooled["deltas"]["harmful_rate"] <= 0.,
        "language_routing_nonnegative": all(
            row["deltas"]["routing_gain"] >= 0. for row in by_language.values()),
        "broad_pool_routing_ge_minus_0.010": min(pool_deltas) >= -.010,
    }
    payload = {
        "status": "opened_development_fixed_chunk4_reroute",
        "protocol": {
            "policy": "exact 15% live onset then 15% deployed-probe chunk4 reroute",
            "late_answer_chunk": 4,
            "extra_model_forwards": 0,
            "new_trainable_parameters": 0,
            "base_model_trainable_parameters": 0,
            "selection": "none; deployed probe and fixed chunk4 read",
            "claim_boundary": ("Opened-development diagnostic only. Passing "
                               "authorizes unchanged historical recapture."),
        },
        "pooled": pooled, "macro_source_routing_delta": float(np.mean(source_deltas)),
        "by_source": by_source, "by_pool": by_pool,
        "by_language": by_language, "replay_qc": replay,
        "gates": gates, "all_gates_pass": all(gates.values()),
        "activation_recommended": False, "live_unchanged": True,
        "provenance": {
            "script": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
            "selection": {"path": str(args.selection), "sha256": sha256(args.selection)},
            "midanswer_features": [{"path": str(p), "sha256": sha256(p)}
                                   for p in mid_paths],
            "midanswer_traces": [{"path": str(p), "sha256": sha256(p)}
                                 for p in mid_trace_paths],
            "original_features": [{"path": str(p), "sha256": sha256(p)}
                                  for p in original_paths],
            "original_traces": [{"path": str(p), "sha256": sha256(p)}
                                for p in original_trace_paths],
            "local": {"path": str(args.local), "sha256": sha256(args.local)},
            "expert": {"path": str(args.expert), "sha256": sha256(args.expert)},
            "live_artifact": {"path": str(args.live_artifact),
                              "sha256": sha256(args.live_artifact)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled": pooled,
                      "macro_source_routing_delta": payload["macro_source_routing_delta"],
                      "gates": gates, "all_gates_pass": payload["all_gates_pass"]},
                     indent=2))
    print("receipt_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
