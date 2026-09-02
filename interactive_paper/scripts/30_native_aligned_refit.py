"""Refit the official native probe on deployment-aligned labels.

Prerequisite (run in the Modal workspace that owns ``gate-data``):

    modal run modal_native_dump.py::judge_training_official

Pull the resulting ``frozen_native_*off_judged.parquet`` files and the
existing official feature shards, then run this script from
``interactive_paper/``. The deployed gate is never overwritten.

Outputs:

* ``data/gate_native_aligned_candidate.json``
* ``figures/native_aligned_refit.json``
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

D = Path("data")
FIG = Path("figures")
RNG = np.random.default_rng(42)
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
CORE = [
    ("caliboff", "calib_features.parquet"),
    ("expoff", "expansion_labels.parquet"),
    ("exp2off", "expansion2_labels.parquet"),
    ("exp3off", "expansion3_labels.parquet"),
    ("exp3zhoff", "expansion3zh_labels.parquet"),
]
EXTERNAL = [
    ("striviaqa", "oab_ok"),
    ("swebq", "oab_ok"),
    ("sllama", "oab_ok"),
    ("sdqa", "heard_ok"),
    ("sreason", "heard_ok"),
]


def load_feats(tag):
    ids, arrays = [], []
    for path in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(path, allow_pickle=True)
        ids += [str(row_id) for row_id in z["ids"]]
        arrays.append(z["X"])
    if not arrays:
        raise FileNotFoundError(f"no feature shards for {tag}")
    x = np.concatenate(arrays)
    rows = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    rows = rows.drop_duplicates("id", keep="last")
    return list(rows["id"]), x[rows["row"].to_numpy()]


def native_failure(tag):
    path = D / f"frozen_native_{tag}_judged.parquet"
    df = pd.read_parquet(path).dropna(subset=["adequate"])
    adequate = df.drop_duplicates("id", keep="last").set_index("id")[
        "adequate"].astype(int)
    return (1 - adequate).to_dict()


def legacy_internal_failure():
    df = pd.read_parquet(D / "frozen_v3_traces.parquet")
    adequate = df[df["mode"] == "local"].groupby("id")[
        "heard_ok"].max().dropna().astype(int)
    return (1 - adequate).to_dict()


def legacy_external_failure(pool, col):
    df = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
    adequate = df[df["tier"] == "never"].dropna(subset=[col])
    adequate = adequate.drop_duplicates("id", keep="last").set_index(
        "id")[col].astype(int)
    return (1 - adequate).to_dict()


def expert_outcomes(pool, col=None):
    if pool == "internal_test":
        df = pd.read_parquet(D / "frozen_v3_traces.parquet")
        adequate = df[df["mode"] == "escalated"].groupby("id")[
            "heard_ok"].max().dropna().astype(int)
    else:
        df = pd.read_parquet(D / f"{pool}_conclive_traces.parquet")
        adequate = df[df["tier"] == "always"].dropna(subset=[col])
        adequate = adequate.drop_duplicates("id", keep="last").set_index(
            "id")[col].astype(int)
    return adequate.to_dict()


def group_for(meta, tag, row_id):
    if row_id not in meta.index:
        return f"{tag}:unknown"
    row = meta.loc[row_id]
    source = row.get("source") if hasattr(row, "get") else None
    pool = row.get("pool") if hasattr(row, "get") else None
    value = source if pd.notna(source) else pool
    return f"{tag}:{value if pd.notna(value) else 'unknown'}"


def collect_training(artifact):
    x_parts, old_parts, native_parts, group_parts, id_parts = [], [], [], [], []
    block_report = []
    legacy_total = 0
    for tag, label_file in CORE:
        ids, x = load_feats(tag)
        legacy_df = pd.read_parquet(D / label_file).drop_duplicates(
            "id", keep="last").set_index("id")
        old_map = legacy_df["escalate_label"].dropna().astype(int).to_dict()
        new_map = native_failure(tag)
        legacy_rows = [j for j, row_id in enumerate(ids) if row_id in old_map]
        matched = [j for j in legacy_rows if ids[j] in new_map]
        legacy_total += len(legacy_rows)
        x_parts.append(x[matched])
        old_parts.append(np.array([old_map[ids[j]] for j in matched]))
        native_parts.append(np.array([new_map[ids[j]] for j in matched]))
        group_parts.append(np.array([
            group_for(legacy_df, tag, ids[j]) for j in matched]))
        id_parts.append(np.array([ids[j] for j in matched]))
        block_report.append({
            "tag": tag, "legacy_labeled": len(legacy_rows),
            "native_labeled": len(matched),
            "agreement": float(np.mean([
                old_map[ids[j]] == new_map[ids[j]] for j in matched]))})

    tag = "freshoff"
    ids, x = load_feats(tag)
    legacy_df = pd.read_parquet(D / "fresh_labels.parquet").drop_duplicates(
        "id", keep="last").set_index("id")
    old_map = legacy_df["escalate_label"].dropna().astype(int).to_dict()
    judged_map = native_failure(tag)
    native_map = {}
    for row_id, row in legacy_df.iterrows():
        if row["pool"] == "fresh_fast":
            native_map[row_id] = 1
        elif row_id in judged_map:
            native_map[row_id] = judged_map[row_id]
    legacy_rows = [j for j, row_id in enumerate(ids)
                   if row_id in old_map and
                   legacy_df.loc[row_id, "split"] == "train"]
    matched = [j for j in legacy_rows if ids[j] in native_map]
    legacy_total += len(legacy_rows)
    x_parts.append(x[matched])
    old_parts.append(np.array([old_map[ids[j]] for j in matched]))
    native_parts.append(np.array([native_map[ids[j]] for j in matched]))
    group_parts.append(np.array([
        group_for(legacy_df, tag, ids[j]) for j in matched]))
    id_parts.append(np.array([ids[j] for j in matched]))
    block_report.append({
        "tag": tag, "legacy_labeled": len(legacy_rows),
        "native_labeled": len(matched),
        "agreement": float(np.mean([
            old_map[ids[j]] == native_map[ids[j]] for j in matched]))})

    if legacy_total != int(artifact["train_n"]):
        raise RuntimeError(
            f"official feature recipe reconstructed {legacy_total} legacy "
            f"rows, artifact declares {artifact['train_n']}")
    return (np.concatenate(x_parts), np.concatenate(old_parts),
            np.concatenate(native_parts), np.concatenate(group_parts),
            np.concatenate(id_parts), block_report, legacy_total)


def choose_c(x, y, groups):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    best = None
    rows = []
    for c_value in (1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=c_value, max_iter=5000), x, y,
            cv=cv, groups=groups, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, oof))
        rows.append({"C": c_value, "group_oof_auc": auc})
        if best is None or auc > best[1]:
            best = (c_value, auc, oof)
    return best, rows


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def boot_delta(y, new_score, old_score, n_boot=4000):
    idx = np.arange(len(y))
    values = []
    for _ in range(n_boot):
        sample = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[sample])) < 2:
            continue
        values.append(roc_auc_score(y[sample], new_score[sample]) -
                      roc_auc_score(y[sample], old_score[sample]))
    return [float(np.mean(values)),
            *[float(v) for v in np.percentile(values, [2.5, 97.5])]]


def cascade(local_ok, expert_ok, score, rate):
    threshold = float(np.quantile(score, 1 - rate))
    fire = score >= threshold
    return {"acc": float(np.where(fire, expert_ok, local_ok).mean()),
            "rate": float(fire.mean()), "threshold": threshold}


def eval_pool(name, tag, legacy_map, expert_map, artifact, candidate):
    ids, x = load_feats(tag)
    native_map = native_failure(tag)
    rows = [j for j, row_id in enumerate(ids)
            if row_id in native_map and row_id in legacy_map]
    ids = [ids[j] for j in rows]
    x = x[rows]
    y_native = np.array([native_map[row_id] for row_id in ids])
    y_legacy = np.array([legacy_map[row_id] for row_id in ids])
    old_score = sigmoid(x @ np.asarray(artifact["w"]) + artifact["b"])
    new_score = candidate.predict_proba(x)[:, 1]
    result = {
        "n": len(ids),
        "label_agreement": float((y_native == y_legacy).mean()),
        "native_auc_deployed": float(roc_auc_score(y_native, old_score)),
        "native_auc_aligned": float(roc_auc_score(y_native, new_score)),
        "native_auc_delta_ci": boot_delta(y_native, new_score, old_score),
        "legacy_auc_deployed": float(roc_auc_score(y_legacy, old_score)),
        "legacy_auc_aligned": float(roc_auc_score(y_legacy, new_score)),
        "budgets": {}}
    paired = [j for j, row_id in enumerate(ids) if row_id in expert_map]
    local_ok = 1 - y_native[paired]
    expert_ok = np.array([expert_map[ids[j]] for j in paired])
    gain = expert_ok - local_ok
    for tier, rate in RATES.items():
        k = round(len(paired) * rate)
        oracle = float(local_ok.mean() +
                       np.sort(gain)[::-1][:k].sum() / len(paired))
        result["budgets"][tier] = {
            "deployed": cascade(local_ok, expert_ok,
                                 old_score[paired], rate),
            "aligned": cascade(local_ok, expert_ok,
                                new_score[paired], rate),
            "oracle_acc": oracle}
    print(f"{name:<14} n={len(ids):>4} native AUC "
          f"{result['native_auc_deployed']:.3f} -> "
          f"{result['native_auc_aligned']:.3f}")
    return result


def main():
    artifact = json.loads((D / "gate_native.json").read_text())
    if "official" not in str(artifact.get("recipe", "")).lower():
        raise RuntimeError("gate_native.json is not an official-config artifact")
    (x, y_old, y_native, groups, _ids, blocks,
     legacy_total) = collect_training(artifact)
    print(f"matched native labels: {len(y_native)}/{legacy_total}; "
          f"agreement={float((y_old == y_native).mean()):.3f}")
    (c_value, oof_auc, oof), sweep = choose_c(x, y_native, groups)
    candidate = LogisticRegression(C=c_value, max_iter=5000).fit(x, y_native)
    thresholds = {tier: float(np.quantile(oof, 1 - rate))
                  for tier, rate in RATES.items()}

    candidate_artifact = dict(artifact)
    candidate_artifact.update(
        w=candidate.coef_[0].tolist(), b=float(candidate.intercept_[0]),
        C=c_value, train_n=int(len(y_native)), eot_thresholds=thresholds,
        recipe="scripts/30 official-native deployment-aligned candidate; "
               "fresh_fast remains policy-positive")
    (D / "gate_native_aligned_candidate.json").write_text(
        json.dumps(candidate_artifact))

    result = {
        "train": {
            "legacy_recipe_n": legacy_total,
            "native_labeled_n": int(len(y_native)),
            "label_agreement": float((y_old == y_native).mean()),
            "native_fail_rate": float(y_native.mean()),
            "legacy_fail_rate": float(y_old.mean()),
            "group_oof_auc": oof_auc,
            "C": c_value,
            "C_sweep": sweep,
            "blocks": blocks},
        "pools": {}}

    result["pools"]["internal_test"] = eval_pool(
        "internal_test", "testoff", legacy_internal_failure(),
        expert_outcomes("internal_test"), artifact, candidate)
    for pool, col in EXTERNAL:
        result["pools"][pool] = eval_pool(
            pool, pool + "off", legacy_external_failure(pool, col),
            expert_outcomes(pool, col), artifact, candidate)
    (FIG / "native_aligned_refit.json").write_text(
        json.dumps(result, indent=1) + "\n")
    print("wrote data/gate_native_aligned_candidate.json + "
          "figures/native_aligned_refit.json")


if __name__ == "__main__":
    main()
