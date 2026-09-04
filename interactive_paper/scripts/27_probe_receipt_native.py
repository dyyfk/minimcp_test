"""Fail-closed training receipt for the deployed native probe.

The gate artifact and feature dumps exist in two serving regimes:
default tags (``calib``, ``test``, ...) and official-config tags
(``caliboff``, ``testoff``, ...). A receipt is only valid when every
feature block comes from the artifact's regime and the reconstructed
training count matches ``gate_native.json:train_n``.

The receipt reports two evaluation targets separately:

* ``native``: answers produced and judged in the matching native regime;
* ``legacy``: older turn/concurrent labels kept for historical comparison.

Usage (from interactive_paper/):

    python scripts/27_probe_receipt_native.py
    python scripts/27_probe_receipt_native.py --config official
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")
GATE = D / "gate_native.json"
EXTERNAL = [("striviaqa", "oab_ok"), ("swebq", "oab_ok"),
            ("sllama", "oab_ok"), ("sdqa", "heard_ok"),
            ("sreason", "heard_ok")]
CORE = [("calib", "calib_features.parquet", False),
        ("exp", "expansion_labels.parquet", False),
        ("exp2", "expansion2_labels.parquet", False),
        ("exp3", "expansion3_labels.parquet", True),
        ("exp3zh", "expansion3zh_labels.parquet", True)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=("auto", "default", "official"),
        default="auto",
        help="feature-dump serving regime (default: derive from gate recipe)")
    parser.add_argument("--output", type=Path,
                        default=D / "probe_receipt_native.json")
    return parser.parse_args()


def load_feats(tag):
    ids, arrays = [], []
    files = sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz"))
    for path in files:
        z = np.load(path, allow_pickle=True)
        ids += [str(i) for i in z["ids"]]
        arrays.append(z["X"])
    if not arrays:
        raise FileNotFoundError(f"no native features for tag {tag}")
    x = np.concatenate(arrays)
    rows = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    rows = rows.drop_duplicates("id", keep="last")
    return list(rows["id"]), x[rows["row"].to_numpy()], files


def artifact_config(artifact, requested):
    if requested != "auto":
        return requested
    recipe = str(artifact.get("recipe", "")).lower()
    if "official" in recipe:
        return "official"
    if recipe:
        return "default"
    raise ValueError("gate artifact has no recipe; pass --config explicitly")


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    return {"n": int(len(y)),
            "fail_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, score)),
            "logloss": float(log_loss(y, score)),
            "acc_at_0.5": float(accuracy_score(y, score >= .5)),
            "majority": float(max(y.mean(), 1 - y.mean()))}


def aligned(ids, x, label):
    label = label[~label.index.duplicated(keep="last")]
    y = label.reindex(ids).to_numpy(dtype=float)
    keep = ~np.isnan(y)
    return x[keep], y[keep].astype(int)


def native_labels(tag):
    path = D / f"frozen_native_{tag}_judged.parquet"
    df = pd.read_parquet(path).dropna(subset=["adequate"])
    adequate = df.drop_duplicates("id", keep="last").set_index("id")[
        "adequate"].astype(int)
    return 1 - adequate


def legacy_internal_labels():
    df = pd.read_parquet(D / "frozen_v3_traces.parquet")
    adequate = df[df["mode"] == "local"].groupby("id")[
        "heard_ok"].max().dropna().astype(int)
    return 1 - adequate


def legacy_external_labels(pool, col):
    df = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
    adequate = df[df["tier"] == "never"].dropna(subset=[col])
    adequate = adequate.drop_duplicates("id", keep="last").set_index(
        "id")[col].astype(int)
    return 1 - adequate


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()
    gate_bytes = GATE.read_bytes()
    artifact = json.loads(gate_bytes)
    config = artifact_config(artifact, args.config)
    suffix = "off" if config == "official" else ""
    w = np.asarray(artifact["w"], dtype=float)
    bias = float(artifact["b"])
    c_value = float(artifact["C"])

    blocks = []
    block_receipts = []
    for base_tag, label_file, optional in CORE:
        tag = base_tag + suffix
        try:
            ids, x, files = load_feats(tag)
            labels = pd.read_parquet(D / label_file).set_index("id")[
                "escalate_label"]
        except FileNotFoundError:
            if optional:
                continue
            raise
        x, y = aligned(ids, x, labels)
        blocks.append((x, y))
        block_receipts.append({
            "feature_tag": tag, "label_file": label_file,
            "n": int(len(y)), "shards": [path.name for path in files]})
    x_core = np.concatenate([part[0] for part in blocks])
    y_core = np.concatenate([part[1] for part in blocks])
    n_core = len(y_core)

    fresh_tag = "fresh" + suffix
    fresh_labels = pd.read_parquet(D / "fresh_labels.parquet")
    fresh_labels = fresh_labels[fresh_labels["escalate_label"].notna()]
    fresh_y = dict(zip(fresh_labels["id"],
                       fresh_labels["escalate_label"].astype(int)))
    fresh_split = dict(zip(fresh_labels["id"], fresh_labels["split"]))
    fresh_ids, fresh_x, fresh_files = load_feats(fresh_tag)
    fresh_rows = [j for j, row_id in enumerate(fresh_ids)
                  if row_id in fresh_y and
                  fresh_split.get(row_id) == "train"]
    x_train = np.concatenate([x_core, fresh_x[fresh_rows]])
    y_train = np.concatenate(
        [y_core, [fresh_y[fresh_ids[j]] for j in fresh_rows]])
    block_receipts.append({
        "feature_tag": fresh_tag,
        "label_file": "fresh_labels.parquet[train]",
        "n": int(len(fresh_rows)),
        "shards": [path.name for path in fresh_files]})

    expected_n = int(artifact["train_n"])
    if len(y_train) != expected_n:
        raise RuntimeError(
            f"receipt refused: reconstructed {len(y_train)} rows from "
            f"{config} features, but gate artifact declares {expected_n}")
    if x_train.shape[1] != len(w):
        raise RuntimeError(
            f"receipt refused: feature dim {x_train.shape[1]} != "
            f"artifact weight dim {len(w)}")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(
        LogisticRegression(C=c_value, max_iter=5000), x_train, y_train,
        cv=cv, method="predict_proba")[:, 1]
    deployed_train_score = sigmoid(x_train @ w + bias)
    thresholds_oof = {
        tier: float(np.quantile(oof[:n_core], 1 - rate))
        for tier, rate in (("conservative", .15), ("balanced", .30),
                           ("aggressive", .50))}

    receipt = {
        "schema": 2,
        "name": "native L22 probe (deployed gate)",
        "repo_commit": git_head(),
        "gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "config": config,
        "feature_suffix": suffix,
        "recipe": artifact.get("recipe"),
        "C": c_value,
        "dim": int(x_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_core": int(n_core),
        "blocks": block_receipts,
        "label_provenance": {
            "training": "legacy turn-based escalate_label parquets",
            "evaluation_primary": "matching native judged answers",
            "evaluation_comparison": "legacy turn/concurrent labels"},
        "train": metrics(y_train, deployed_train_score),
        "oof": metrics(y_train, oof),
        "artifact_thresholds": artifact["eot_thresholds"],
        "recomputed_oof_thresholds": thresholds_oof,
        "evaluations": {}}

    test_tag = "test" + suffix
    test_ids, test_x, _ = load_feats(test_tag)
    x_native, y_native = aligned(
        test_ids, test_x, native_labels(test_tag))
    x_legacy, y_legacy = aligned(
        test_ids, test_x, legacy_internal_labels())
    native_score = sigmoid(x_native @ w + bias)
    internal = {
        "native": metrics(y_native, native_score),
        "legacy": metrics(y_legacy, sigmoid(x_legacy @ w + bias)),
        "budget_ops_native": {}}
    for tier, threshold in artifact["eot_thresholds"].items():
        fire = native_score >= threshold
        internal["budget_ops_native"][tier] = {
            "classification_acc": float(accuracy_score(y_native, fire)),
            "realized_rate": float(fire.mean())}
    receipt["evaluations"]["internal_test"] = internal

    for pool, col in EXTERNAL:
        tag = pool + suffix
        ids, x, _ = load_feats(tag)
        x_native, y_native = aligned(ids, x, native_labels(tag))
        x_legacy, y_legacy = aligned(
            ids, x, legacy_external_labels(pool, col))
        receipt["evaluations"][pool] = {
            "native": metrics(y_native, sigmoid(x_native @ w + bias)),
            "legacy": metrics(y_legacy, sigmoid(x_legacy @ w + bias))}

    args.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"wrote {args.output}: config={config}, n={len(y_train)}, "
          f"OOF AUC={receipt['oof']['auc']:.3f}, native internal AUC="
          f"{internal['native']['auc']:.3f}")


if __name__ == "__main__":
    main()
