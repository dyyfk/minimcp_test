"""Select same-forward structured context features, then validate unchanged.

P21 is the only selection set. P22 and P23 are read only after the feature
recipe and regularization are fixed from source-grouped P21 OOF predictions.
They are development validation sets; a passing candidate must still be frozen
before a new prospective test. Nothing here can overwrite the live gate.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


LAYERS = [14, 18, 22, 26, 30]
K_EOT = 8
RNG = np.random.default_rng(20260902)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structured(root):
    shards = sorted(glob.glob(str(Path(root) / "structured" /
                                   "structured_multiturn_feats.rank*.npz")))
    if not shards:
        raise FileNotFoundError(f"no structured shards under {root}")
    keys = ["H_eot", "H_turn_mean", "H_chunk_mean", "H_chunk_delta",
            "eot_len", "chunk_count", "onset_wait"]
    ids = []
    values = {key: [] for key in keys}
    layer_values = None
    for path in shards:
        z = np.load(path, allow_pickle=True)
        ids += [str(value) for value in z["ids"]]
        for key in keys:
            values[key].append(z[key])
        current_layers = [int(value) for value in z["layers"]]
        if layer_values is None:
            layer_values = current_layers
        elif layer_values != current_layers:
            raise RuntimeError("inconsistent structured layer list")
    data = {key: np.concatenate(parts) for key, parts in values.items()}
    data["layers"] = layer_values
    frame = pd.DataFrame({"id": ids}).assign(row=np.arange(len(ids)))
    frame = frame.drop_duplicates("id", keep="last")
    rows = frame["row"].to_numpy()
    return list(frame["id"]), {
        key: (value[rows] if isinstance(value, np.ndarray) else value)
        for key, value in data.items()
    }


def load_dataset(root):
    root = Path(root)
    ids, data = load_structured(root)
    judged = (pd.read_parquet(root / "judged.parquet")
              .dropna(subset=["adequate"])
              .drop_duplicates("id", keep="last").set_index("id"))
    pairs = (pd.read_parquet(root / "pairs.parquet")
             .drop_duplicates("id", keep="last").set_index("id"))
    traces = {}
    for path in sorted((root / "structured").glob(
            "structured_multiturn.rank*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                traces[str(row["id"])] = row
    keep = [index for index, row_id in enumerate(ids)
            if row_id in judged.index and row_id in pairs.index and
            row_id in traces and traces[row_id].get("error") is None]
    selected_ids = [ids[index] for index in keep]
    mismatched = [row_id for row_id in selected_ids
                  if traces[row_id].get("target_answer") !=
                  judged.loc[row_id, "answer"]]
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} replay answers differ from cached judgments; "
            f"first={mismatched[:3]}")
    selected = {key: (value[keep] if isinstance(value, np.ndarray) else value)
                for key, value in data.items()}
    selected["ids"] = selected_ids
    selected["y"] = np.asarray(
        [1 - int(bool(judged.loc[row_id, "adequate"]))
         for row_id in selected_ids], dtype=np.int8)
    selected["pool"] = np.asarray(
        [str(pairs.loc[row_id, "target_pool"]) for row_id in selected_ids])
    selected["language"] = np.asarray([
        str(pairs.loc[row_id].get("language", "unknown"))
        for row_id in selected_ids])
    selected["answer_parity_rows"] = len(selected_ids)
    return selected


def eot_mean(data, layer_index):
    hidden = data["H_eot"][:, layer_index].astype(np.float32)
    lengths = np.clip(data["eot_len"].astype(int), 1, K_EOT)
    mask = (np.arange(K_EOT)[None, :] >=
            K_EOT - lengths[:, None]).astype(np.float32)
    return (hidden * mask[:, :, None]).sum(1) / lengths[:, None]


def part(data, layer, mode):
    index = data["layers"].index(layer)
    if mode == "eot_last":
        return data["H_eot"][:, index, -1].astype(np.float32)
    if mode == "eot_mean":
        return eot_mean(data, index)
    key = {"turn_mean": "H_turn_mean", "chunk_mean": "H_chunk_mean",
           "chunk_delta": "H_chunk_delta"}[mode]
    return data[key][:, index].astype(np.float32)


def live_matrix(data):
    return np.concatenate([part(data, 22, "eot_last"),
                           part(data, 22, "eot_mean"),
                           part(data, 22, "turn_mean")], axis=1)


def artifact_score(path, x):
    artifact = json.loads(Path(path).read_text())
    return x @ np.asarray(artifact["w"]) + float(artifact["b"])


def scalar_geometry(data, live_score):
    columns = [live_score, data["onset_wait"].astype(float),
               np.log1p(data["chunk_count"].astype(float))]
    names = ["live_score", "onset_wait", "log_chunk_count"]
    for layer in LAYERS:
        turn = part(data, layer, "turn_mean").astype(np.float64)
        mean = part(data, layer, "chunk_mean").astype(np.float64)
        delta = part(data, layer, "chunk_delta").astype(np.float64)
        for label, value in (("turn_norm", np.linalg.norm(turn, axis=1)),
                             ("chunk_norm", np.linalg.norm(mean, axis=1)),
                             ("delta_norm", np.linalg.norm(delta, axis=1))):
            columns.append(value)
            names.append(f"L{layer}_{label}")
        denom = np.linalg.norm(turn, axis=1) * np.linalg.norm(mean, axis=1)
        columns.append(np.sum(turn * mean, axis=1) /
                       np.maximum(denom, 1e-12))
        names.append(f"L{layer}_turn_chunk_cos")
    return np.column_stack(columns).astype(np.float32), names


@dataclass(frozen=True)
class Config:
    name: str
    layers: tuple
    modes: tuple
    standardize: bool = False
    scalar: bool = False


def configs():
    output = []
    for layer in LAYERS:
        for mode in ("eot_last", "eot_mean", "turn_mean", "chunk_mean",
                     "chunk_delta"):
            output.append(Config(f"L{layer}_{mode}", (layer,), (mode,)))
        output += [
            Config(f"L{layer}_deployed3", (layer,),
                   ("eot_last", "eot_mean", "turn_mean")),
            Config(f"L{layer}_trajectory3", (layer,),
                   ("turn_mean", "chunk_mean", "chunk_delta")),
        ]
    output += [
        Config("L18_22_26_eot_last", (18, 22, 26), ("eot_last",)),
        Config("L18_22_26_turn_mean", (18, 22, 26), ("turn_mean",)),
        Config("L18_22_26_chunk_delta", (18, 22, 26), ("chunk_delta",)),
        Config("L14_18_22_26_30_eot_last", tuple(LAYERS), ("eot_last",)),
        Config("L22_deployed3_plus_chunk2", (22,),
               ("eot_last", "eot_mean", "turn_mean", "chunk_mean",
                "chunk_delta")),
        Config("scalar_geometry", (), (), standardize=True, scalar=True),
    ]
    return output


def matrix(data, config, live_score):
    if config.scalar:
        return scalar_geometry(data, live_score)[0]
    return np.concatenate([part(data, layer, mode)
                           for layer in config.layers
                           for mode in config.modes], axis=1)


def macro_group_auc(y, score, groups):
    values = []
    for group in sorted(set(groups)):
        mask = groups == group
        if len(np.unique(y[mask])) == 2:
            values.append(roc_auc_score(y[mask], score[mask]))
    return float(np.mean(values)), len(values)


def fit_fold(x, y, train, test, c_value, standardize):
    scaler = StandardScaler().fit(x[train]) if standardize else None
    xt = scaler.transform(x[train]) if scaler is not None else x[train]
    xv = scaler.transform(x[test]) if scaler is not None else x[test]
    model = LogisticRegression(C=c_value, max_iter=5000, tol=1e-5)
    model.fit(xt, y[train])
    return test, model.predict_proba(xv)[:, 1], int(model.n_iter_[0])


def grouped_oof(x, y, groups, c_value, standardize, jobs):
    folds = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    rows = joblib.Parallel(n_jobs=jobs)(
        joblib.delayed(fit_fold)(x, y, train, test, c_value, standardize)
        for train, test in folds)
    score = np.full(len(y), np.nan)
    iterations = []
    for test, values, count in rows:
        score[test] = values
        iterations.append(count)
    return score, iterations


def fit_full(x, y, c_value, standardize):
    scaler = StandardScaler().fit(x) if standardize else None
    xx = scaler.transform(x) if scaler is not None else x
    model = LogisticRegression(C=c_value, max_iter=5000, tol=1e-5).fit(xx, y)
    return model, scaler


def score_full(model, scaler, x):
    xx = scaler.transform(x) if scaler is not None else x
    return model.predict_proba(xx)[:, 1]


def bootstrap_macro_delta(data, candidate, live, n_boot=5000):
    pools = sorted(set(data["pool"]))
    values = []
    for _ in range(n_boot):
        deltas = []
        for pool in pools:
            index = np.flatnonzero(data["pool"] == pool)
            sample = RNG.choice(index, len(index), replace=True)
            y = data["y"][sample]
            if len(np.unique(y)) < 2:
                continue
            deltas.append(roc_auc_score(y, candidate[sample]) -
                          roc_auc_score(y, live[sample]))
        if deltas:
            values.append(np.mean(deltas))
    return [float(np.mean(values)),
            *[float(value) for value in np.percentile(values, [2.5, 97.5])]]


def evaluate(data, candidate, live):
    pooled = {
        "rows": len(data["y"]), "failures": int(data["y"].sum()),
        "live_auc": float(roc_auc_score(data["y"], live)),
        "candidate_auc": float(roc_auc_score(data["y"], candidate)),
    }
    pooled["auc_delta"] = pooled["candidate_auc"] - pooled["live_auc"]
    by_pool = {}
    for pool in sorted(set(data["pool"])):
        mask = data["pool"] == pool
        if len(np.unique(data["y"][mask])) < 2:
            continue
        la = float(roc_auc_score(data["y"][mask], live[mask]))
        ca = float(roc_auc_score(data["y"][mask], candidate[mask]))
        by_pool[pool] = {"rows": int(mask.sum()), "live_auc": la,
                         "candidate_auc": ca, "auc_delta": ca - la}
    by_language = {}
    for language in sorted(set(data["language"])):
        mask = data["language"] == language
        if len(np.unique(data["y"][mask])) < 2:
            continue
        la = float(roc_auc_score(data["y"][mask], live[mask]))
        ca = float(roc_auc_score(data["y"][mask], candidate[mask]))
        by_language[language] = {"rows": int(mask.sum()), "live_auc": la,
                                 "candidate_auc": ca,
                                 "auc_delta": ca - la}
    macro_delta = float(np.mean(
        [row["auc_delta"] for row in by_pool.values()]))
    return {"pooled": pooled, "by_pool": by_pool,
            "by_language": by_language,
            "macro_pool_auc_delta": macro_delta,
            "macro_pool_delta_bootstrap_95ci": bootstrap_macro_delta(
                data, candidate, live)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", action="append", type=Path,
                        required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()

    train = load_dataset(args.train_root)
    live_train = artifact_score(args.live_artifact, live_matrix(train))
    c_raw = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    c_scalar = (.01, .1, 1., 10.)
    sweep = []
    scores = {}
    config_map = {config.name: config for config in configs()}
    for config in configs():
        x = matrix(train, config, live_train)
        for c_value in (c_scalar if config.scalar else c_raw):
            oof, iterations = grouped_oof(
                x, train["y"], train["pool"], c_value,
                config.standardize, args.jobs)
            pooled = float(roc_auc_score(train["y"], oof))
            macro, valid_groups = macro_group_auc(
                train["y"], oof, train["pool"])
            name = f"{config.name}_C{c_value:g}"
            row = {"name": name, "config": config.name, "C": c_value,
                   "dimension": x.shape[1], "pooled_auc": pooled,
                   "macro_pool_auc": macro,
                   "valid_pools": valid_groups, "iterations": iterations}
            sweep.append(row)
            scores[name] = oof
            print(name, "macro", macro, "pooled", pooled, flush=True)

    baseline_rows = [row for row in sweep
                     if row["config"] == "L22_deployed3"]
    baseline = max(baseline_rows, key=lambda row: row["macro_pool_auc"])
    eligible = [row for row in sweep
                if row["pooled_auc"] >= baseline["pooled_auc"] - .005]
    winner = max(eligible, key=lambda row: (row["macro_pool_auc"],
                                            row["pooled_auc"],
                                            -row["dimension"]))
    selected_config = config_map[winner["config"]]
    x_train = matrix(train, selected_config, live_train)
    fitted = fit_full(x_train, train["y"], winner["C"],
                      selected_config.standardize)

    train_eval = evaluate(train, scores[winner["name"]], live_train)
    validations = {}
    for root in args.validation_root:
        data = load_dataset(root)
        live = artifact_score(args.live_artifact, live_matrix(data))
        x = matrix(data, selected_config, live)
        candidate = score_full(*fitted, x)
        validations[root.name] = evaluate(data, candidate, live)

    rules = {
        "train_macro_delta_min": .015,
        "train_pooled_nonnegative": True,
        "each_validation_macro_nonnegative": True,
        "each_validation_pooled_nonnegative": True,
        "minimum_validation_language_delta": -.01,
        "minimum_validation_pool_delta": -.03,
    }
    all_validation_rows = list(validations.values())
    decision = bool(
        train_eval["macro_pool_auc_delta"] >= .015 and
        train_eval["pooled"]["auc_delta"] >= 0 and
        all(row["macro_pool_auc_delta"] >= 0
            for row in all_validation_rows) and
        all(row["pooled"]["auc_delta"] >= 0
            for row in all_validation_rows) and
        all(min((item["auc_delta"] for item in row["by_language"].values()),
                default=0) >= -.01 for row in all_validation_rows) and
        all(min((item["auc_delta"] for item in row["by_pool"].values()),
                default=0) >= -.03 for row in all_validation_rows))

    out = {
        "status": "structured_multiturn_feature_selection",
        "protocol": {
            "selection_set": args.train_root.name,
            "selection_cv": "5-fold source-pool-grouped OOF",
            "selection_rule": "max macro pool AUC among pooled-within-.005-of-best-L22-recipe",
            "validation_sets_opened_only_after_selection": [
                root.name for root in args.validation_root],
            "live_unchanged": True,
        },
        "inputs": {
            "live_artifact_sha256": sha256(args.live_artifact),
            "selection_pairs_sha256": sha256(args.train_root / "pairs.parquet"),
            "selection_judged_sha256": sha256(
                args.train_root / "judged.parquet"),
        },
        "sweep": sweep, "baseline": baseline, "winner": winner,
        "selection_evaluation": train_eval,
        "development_validations": validations,
        "prospective_graduation_rules": rules,
        "decision": {"graduate_to_new_prospective_set": decision},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"baseline": baseline, "winner": winner,
                      "selection": train_eval,
                      "validations": validations,
                      "decision": out["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
