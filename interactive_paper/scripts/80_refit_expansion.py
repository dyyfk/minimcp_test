"""Nested source-family evaluation for the frozen P25-B expansion.

The script joins official-native pre-answer features with paired local and
expert outcomes.  Hyperparameters are selected only in inner source-family
folds; outer folds estimate the complete selection procedure.  P22/P23 are
optional development checks and never participate in selection.  Nothing in
this script overwrites a live or shadow gate artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


RATES = (.15, .30, .50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(directory: Path, pattern: str):
    ids, arrays = [], []
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    for path in paths:
        value = np.load(path, allow_pickle=True)
        ids.extend(str(row_id) for row_id in value["ids"])
        arrays.append(value["X"].astype(np.float32, copy=False))
    matrix = np.concatenate(arrays)
    index = pd.DataFrame({"id": ids, "row": range(len(ids))})
    index = index.drop_duplicates("id", keep="last")
    return index.id.to_list(), matrix[index.row.to_numpy()]


def feature_map(single_dir: Path, multi_dir: Path):
    output = {}
    receipts = []
    for directory, pattern, mode in (
        (single_dir, "prospective_native_feats.rank*.npz", "standalone"),
        (multi_dir, "controlled_multiturn_feats.rank*.npz", "multiturn"),
    ):
        ids, values = load_features(directory, pattern)
        for row_id, value in zip(ids, values):
            if row_id in output:
                raise RuntimeError(f"duplicate feature id {row_id}")
            output[row_id] = value
        receipts.append({"mode": mode, "directory": str(directory),
                         "rows": len(ids), "dimension": values.shape[1],
                         "files": [{"path": str(path), "sha256": sha256(path)}
                                   for path in sorted(directory.glob(pattern))]})
    return output, receipts


def load_training(args):
    selection = pd.read_parquet(args.selection).drop_duplicates("id")
    local = pd.concat([
        pd.read_parquet(args.local_single),
        pd.read_parquet(args.local_multi),
    ], ignore_index=True).drop_duplicates("id", keep="last")
    expert = pd.read_parquet(args.expert).drop_duplicates("id", keep="last")
    local = local[["id", "adequate"]].rename(columns={"adequate": "local_ok"})
    expert = expert[["id", "adequate"]].rename(columns={"adequate": "expert_ok"})
    frame = selection.merge(local, on="id", how="left", validate="one_to_one")
    frame = frame.merge(expert, on="id", how="left", validate="one_to_one")
    features, feature_receipts = feature_map(args.single_dir, args.multi_dir)
    frame["has_feature"] = frame.id.astype(str).isin(features)
    excluded = {
        "missing_local": int(frame.local_ok.isna().sum()),
        "missing_expert": int(frame.expert_ok.isna().sum()),
        "missing_feature": int((~frame.has_feature).sum()),
    }
    frame = frame.dropna(subset=["local_ok", "expert_ok"])
    frame = frame[frame.has_feature].sort_values("id").reset_index(drop=True)
    if len(frame) < 1000:
        raise RuntimeError(f"only {len(frame)} complete training rows")
    x = np.stack([features[str(row_id)] for row_id in frame.id])
    if x.shape[1] != 12288 or not np.isfinite(x).all():
        raise RuntimeError(f"invalid feature matrix {x.shape}")
    local_ok = frame.local_ok.astype(int).to_numpy()
    expert_ok = frame.expert_ok.astype(int).to_numpy()
    if set(np.unique(local_ok)) - {0, 1} or set(np.unique(expert_ok)) - {0, 1}:
        raise RuntimeError("outcomes must be binary")
    return frame, x, local_ok, expert_ok, excluded, feature_receipts


def top_mask(score, rate):
    count = min(len(score), max(0, int(round(len(score) * rate))))
    mask = np.zeros(len(score), dtype=bool)
    if count:
        order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
        mask[order[:count]] = True
    return mask


def routing_objective(gain, score):
    return float(np.mean([
        gain[top_mask(score, rate)].sum() / len(gain) for rate in RATES
    ]))


def macro_routing(gain, score, groups):
    values = [routing_objective(gain[groups == group], score[groups == group])
              for group in sorted(set(groups))]
    return float(np.mean(values)), len(values)


def macro_auc(y, score, groups):
    values = []
    for group in sorted(set(groups)):
        mask = groups == group
        if len(np.unique(y[mask])) == 2:
            values.append(roc_auc_score(y[mask], score[mask]))
    return (float(np.mean(values)) if values else None), len(values)


def score_summary(local_ok, expert_ok, score, groups):
    gain = expert_ok - local_ok
    y_fail = 1 - local_ok
    macro_native, valid_groups = macro_auc(y_fail, score, groups)
    return {
        "routing_objective_pooled": routing_objective(gain, score),
        "routing_objective_macro_source": macro_routing(gain, score, groups)[0],
        "native_failure_auc_pooled": float(roc_auc_score(y_fail, score)),
        "native_failure_auc_macro_source": macro_native,
        "native_failure_valid_sources": valid_groups,
    }


def standardizer(train):
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-4)


def transform(value, mean, scale):
    return np.clip((value - mean) / scale, -10., 10.).astype(np.float32,
                                                              copy=False)


@dataclass(frozen=True)
class Spec:
    family: str
    parameter: float = 0.
    width: int = 0
    dropout: float = 0.
    weight_decay: float = 0.
    learning_rate: float = 0.

    @property
    def name(self):
        if self.family in {"failure_logistic", "benefit_logistic", "gain_ridge"}:
            return f"{self.family}_{self.parameter:g}"
        return (f"{self.family}_w{self.width}_d{self.dropout:g}_"
                f"wd{self.weight_decay:g}_lr{self.learning_rate:g}")

    @property
    def parameter_count(self):
        if self.family.startswith("residual"):
            outputs = 1 if self.family == "residual_failure" else 3
            return 12288 * outputs + outputs + 12288 * self.width + self.width * outputs
        return 12289


SPECS = tuple(
    [Spec("failure_logistic", value) for value in (1e-4, 3e-4, 1e-3)] +
    [Spec("benefit_logistic", value) for value in (1e-4, 3e-4, 1e-3)] +
    [Spec("gain_ridge", value) for value in (100., 1000., 10000.)] +
    [Spec(family, width=width, dropout=dropout, weight_decay=weight_decay,
          learning_rate=learning_rate)
     for family in ("residual_failure", "residual_gain3")
     for width, dropout, weight_decay, learning_rate in (
         (32, .2, 1e-3, 1e-3),
         (128, .2, 1e-3, 1e-3),
         (128, .5, 1e-3, 1e-3),
         (128, .2, 1e-2, 1e-3),
         (128, .2, 1e-3, 3e-4),
         (512, .5, 1e-3, 1e-3),
     )]
)


def fit_linear(spec, x, local_ok, expert_ok):
    gain = expert_ok - local_ok
    weights = np.where(gain < 0, 2., 1.)
    if spec.family == "failure_logistic":
        model = LogisticRegression(C=spec.parameter, max_iter=3000, tol=1e-5)
        model.fit(x, 1 - local_ok, sample_weight=weights)
    elif spec.family == "benefit_logistic":
        model = LogisticRegression(C=spec.parameter, max_iter=3000, tol=1e-5)
        model.fit(x, (gain > 0).astype(int), sample_weight=weights)
    elif spec.family == "gain_ridge":
        model = Ridge(alpha=spec.parameter)
        model.fit(x, gain, sample_weight=weights)
    else:
        raise ValueError(spec.family)
    return model


def predict_linear(spec, model, x):
    if spec.family.endswith("logistic"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def make_mlp(torch, dimension, spec):
    outputs = 1 if spec.family == "residual_failure" else 3

    class ResidualHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(dimension, outputs)
            self.hidden = torch.nn.Linear(dimension, spec.width, bias=False)
            self.dropout = torch.nn.Dropout(spec.dropout)
            self.out = torch.nn.Linear(spec.width, outputs, bias=False)
            torch.nn.init.normal_(self.hidden.weight, std=.005)
            torch.nn.init.normal_(self.out.weight, std=.005)

        def forward(self, value):
            residual = self.out(self.dropout(torch.nn.functional.gelu(
                self.hidden(value))))
            return self.base(value) + residual

    return ResidualHead()


def mlp_score(torch, model, value, family, batch_size=1024):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(value), batch_size):
            logits = model(value[start:start + batch_size])
            if family == "residual_failure":
                score = torch.sigmoid(logits.squeeze(-1))
            else:
                probability = torch.softmax(logits, dim=-1)
                score = probability[:, 2] - probability[:, 0]
            output.append(score.float().cpu().numpy())
    return np.concatenate(output)


def train_mlp(spec, x_train, local_train, expert_train, x_valid,
              local_valid, expert_valid, valid_groups, device, seed,
              max_epochs, patience, batch_size, fixed_epochs=None):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    xt = torch.from_numpy(x_train).to(device)
    xv = torch.from_numpy(x_valid).to(device)
    gain_train = expert_train - local_train
    if spec.family == "residual_failure":
        target = torch.from_numpy((1 - local_train).astype(np.float32)).to(device)
    else:
        target = torch.from_numpy((gain_train + 1).astype(np.int64)).to(device)
    weights = torch.from_numpy(np.where(gain_train < 0, 2., 1.).astype(np.float32)).to(device)
    model = make_mlp(torch, xt.shape[1], spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate,
                                  weight_decay=spec.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = {"metric": -np.inf, "epoch": 0, "state": None}
    stale = 0
    epochs = fixed_epochs or max_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[index])
            if spec.family == "residual_failure":
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), target[index], reduction="none")
            else:
                raw = torch.nn.functional.cross_entropy(
                    logits, target[index], reduction="none")
            loss = (raw * weights[index]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        score = mlp_score(torch, model, xv, spec.family)
        metric = macro_routing(expert_valid - local_valid, score,
                               valid_groups)[0]
        if metric > best["metric"] + 1e-5:
            best = {"metric": metric, "epoch": epoch,
                    "state": {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if fixed_epochs is None:
        model.load_state_dict(best.pop("state"))
    else:
        best = {"metric": None, "epoch": fixed_epochs}
    score = mlp_score(torch, model, xv, spec.family)
    state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    del model, xt, xv, target, weights
    torch.cuda.empty_cache()
    return score, best, state


def choose(rows):
    best_primary = max(row["inner_routing_macro_mean"] for row in rows)
    eligible = [row for row in rows
                if row["inner_routing_macro_mean"] >= best_primary - .001]
    return max(eligible, key=lambda row: (
        row["inner_native_macro_mean"], -row["parameter_count"]))


def fit_predict(spec, x_train, local_train, expert_train, x_test,
                local_test, expert_test, groups_test, args, seed,
                fixed_epochs=None):
    mean, scale = standardizer(x_train)
    xt, xv = transform(x_train, mean, scale), transform(x_test, mean, scale)
    if spec.family.startswith("residual"):
        score, info, state = train_mlp(
            spec, xt, local_train, expert_train, xv, local_test, expert_test,
            groups_test, args.device, seed, args.max_epochs, args.patience,
            args.batch_size, fixed_epochs)
        return score, info, {"mean": mean, "scale": scale, **state}
    model = fit_linear(spec, xt, local_train, expert_train)
    score = predict_linear(spec, model, xv)
    state = {"mean": mean, "scale": scale,
             "coef": np.asarray(getattr(model, "coef_")),
             "intercept": np.atleast_1d(getattr(model, "intercept_"))}
    return score, {"metric": None, "epoch": 0}, state


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"], dtype=np.float32) + float(artifact["b"])


def evaluate_strata(frame, local_ok, expert_ok, candidate, live, column):
    output = {}
    values = frame[column].astype(str).to_numpy()
    for value in sorted(set(values)):
        mask = values == value
        if mask.sum() < 10:
            continue
        gain = expert_ok[mask] - local_ok[mask]
        output[value] = {
            "rows": int(mask.sum()),
            "candidate_routing": routing_objective(gain, candidate[mask]),
            "live_routing": routing_objective(gain, live[mask]),
        }
        output[value]["delta"] = (output[value]["candidate_routing"] -
                                    output[value]["live_routing"])
    return output


def load_dev(spec, state, value, args):
    name, selection_path, feature_dir, judged_path = value.split("::", 3)
    metadata = pd.read_parquet(selection_path).drop_duplicates("id")
    judged = pd.read_parquet(judged_path).dropna(subset=["adequate"])
    judged = judged.drop_duplicates("id", keep="last")
    ids, x = load_features(Path(feature_dir),
                           "controlled_multiturn_feats.rank*.npz")
    index = {row_id: j for j, row_id in enumerate(ids)}
    frame = metadata.merge(judged[["id", "adequate"]], on="id")
    frame = frame[frame.id.astype(str).isin(index)].reset_index(drop=True)
    x = np.stack([x[index[str(row_id)]] for row_id in frame.id])
    local_ok = frame.adequate.astype(int).to_numpy()
    tx = transform(x, state["mean"], state["scale"])
    if spec.family.startswith("residual"):
        import torch
        model = make_mlp(torch, tx.shape[1], spec).to(args.device)
        model.load_state_dict({key: torch.from_numpy(array).to(args.device)
                              for key, array in state.items()
                              if key not in {"mean", "scale"}})
        candidate = mlp_score(torch, model,
                              torch.from_numpy(tx).to(args.device), spec.family)
    elif spec.family.endswith("logistic"):
        logits = tx @ state["coef"].reshape(-1) + state["intercept"][0]
        candidate = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    else:
        candidate = tx @ state["coef"].reshape(-1) + state["intercept"][0]
    live = artifact_score(args.live_artifact, x)
    y = 1 - local_ok

    def metrics(mask):
        if len(np.unique(y[mask])) < 2:
            return {"rows": int(mask.sum()), "delta": None}
        old = float(roc_auc_score(y[mask], live[mask]))
        new = float(roc_auc_score(y[mask], candidate[mask]))
        return {"rows": int(mask.sum()), "live_auc": old,
                "candidate_auc": new, "delta": new - old}

    pool_col = "target_pool" if "target_pool" in frame else "pool"
    pools = {value: metrics(frame[pool_col].astype(str).to_numpy() == value)
             for value in sorted(frame[pool_col].astype(str).unique())}
    languages = {value: metrics(frame.language.astype(str).to_numpy() == value)
                 for value in sorted(frame.language.astype(str).unique())}
    valid_pool = [row["delta"] for row in pools.values()
                  if row["delta"] is not None]
    return name, {
        "pooled": metrics(np.ones(len(frame), dtype=bool)),
        "macro_pool_delta": float(np.mean(valid_pool)),
        "pools": pools, "languages": languages,
        "inputs": {"selection": sha256(Path(selection_path)),
                   "judged": sha256(Path(judged_path))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--single-dir", type=Path, required=True)
    parser.add_argument("--multi-dir", type=Path, required=True)
    parser.add_argument("--local-single", type=Path, required=True)
    parser.add_argument("--local-multi", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--p16-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--dev", action="append", default=[],
                        help="name::selection.parquet::feature_dir::judged.parquet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    frame, x, local_ok, expert_ok, excluded, feature_receipts = load_training(args)
    groups = frame.source_family.astype(str).to_numpy()
    gain = expert_ok - local_ok
    strat = (gain + 1) * 2 + (1 - local_ok)
    outer = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, strat, groups))
    live = artifact_score(args.live_artifact, x)
    p16 = artifact_score(args.p16_artifact, x)
    nested = np.full(len(frame), np.nan)
    outer_rows = []
    history = {spec.name: [] for spec in SPECS}
    epoch_history = {spec.name: [] for spec in SPECS}

    for outer_index, (outer_train, outer_test) in enumerate(outer):
        inner_strat = strat[outer_train]
        inner = list(StratifiedGroupKFold(
            3, shuffle=True, random_state=100 + outer_index).split(
                x[outer_train], inner_strat, groups[outer_train]))
        sweep = []
        for spec_index, spec in enumerate(SPECS):
            routing_values, native_values, epochs = [], [], []
            for inner_index, (train_rel, valid_rel) in enumerate(inner):
                train, valid = outer_train[train_rel], outer_train[valid_rel]
                score, info, _state = fit_predict(
                    spec, x[train], local_ok[train], expert_ok[train], x[valid],
                    local_ok[valid], expert_ok[valid], groups[valid], args,
                    100000 * outer_index + 1000 * spec_index + inner_index)
                routing_values.append(macro_routing(gain[valid], score,
                                                     groups[valid])[0])
                native_values.append(macro_auc(1 - local_ok[valid], score,
                                                groups[valid])[0] or 0.)
                if info["epoch"]:
                    epochs.append(info["epoch"])
            row = {"name": spec.name, "spec": asdict(spec),
                   "parameter_count": spec.parameter_count,
                   "inner_routing_macro_mean": float(np.mean(routing_values)),
                   "inner_native_macro_mean": float(np.mean(native_values)),
                   "inner_routing_by_fold": [float(v) for v in routing_values],
                   "inner_native_by_fold": [float(v) for v in native_values],
                   "selected_epoch": (max(1, int(round(np.median(epochs))))
                                      if epochs else 0)}
            sweep.append(row)
            history[spec.name].append(row["inner_routing_macro_mean"])
            if row["selected_epoch"]:
                epoch_history[spec.name].append(row["selected_epoch"])
            print(f"outer={outer_index} {spec.name} routing="
                  f"{row['inner_routing_macro_mean']:+.6f} native="
                  f"{row['inner_native_macro_mean']:.6f}", flush=True)
        winner = choose(sweep)
        selected = next(spec for spec in SPECS if spec.name == winner["name"])
        score, _info, _state = fit_predict(
            selected, x[outer_train], local_ok[outer_train], expert_ok[outer_train],
            x[outer_test], local_ok[outer_test], expert_ok[outer_test],
            groups[outer_test], args, 900000 + outer_index,
            winner["selected_epoch"] or None)
        nested[outer_test] = score
        outer_rows.append({"outer_fold": outer_index,
                           "train_rows": len(outer_train),
                           "test_rows": len(outer_test),
                           "test_groups": sorted(set(groups[outer_test])),
                           "winner": winner, "sweep": sweep,
                           "test": score_summary(local_ok[outer_test],
                                                 expert_ok[outer_test], score,
                                                 groups[outer_test])})
        print(f"outer={outer_index} winner={winner['name']}", flush=True)

    global_rows = []
    for spec in SPECS:
        rows = [row for outer_row in outer_rows for row in outer_row["sweep"]
                if row["name"] == spec.name]
        global_rows.append({"name": spec.name, "spec": asdict(spec),
                            "parameter_count": spec.parameter_count,
                            "inner_routing_macro_mean": float(np.mean(
                                [row["inner_routing_macro_mean"] for row in rows])),
                            "inner_native_macro_mean": float(np.mean(
                                [row["inner_native_macro_mean"] for row in rows])),
                            "selected_epoch": (max(1, int(round(np.median(
                                epoch_history[spec.name]))))
                                if epoch_history[spec.name] else 0)})
    global_winner = choose(global_rows)
    spec = next(item for item in SPECS if item.name == global_winner["name"])
    score, _info, state = fit_predict(
        spec, x, local_ok, expert_ok, x[:1], local_ok[:1], expert_ok[:1],
        groups[:1], args, 1900000, global_winner["selected_epoch"] or None)
    del score
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.model_output, **state)

    summaries = {
        "live": score_summary(local_ok, expert_ok, live, groups),
        "p16": score_summary(local_ok, expert_ok, p16, groups),
        "nested_selected_procedure": score_summary(
            local_ok, expert_ok, nested, groups),
    }
    selected_summary, live_summary = (summaries["nested_selected_procedure"],
                                      summaries["live"])
    deltas = {key: selected_summary[key] - live_summary[key]
              for key in ("routing_objective_pooled",
                          "routing_objective_macro_source",
                          "native_failure_auc_pooled",
                          "native_failure_auc_macro_source")}
    strata = {column: evaluate_strata(frame, local_ok, expert_ok, nested, live,
                                      column)
              for column in ("language", "mode", "pool")}
    dev = dict(load_dev(spec, state, value, args) for value in args.dev)
    broad_ok = all(row["delta"] >= -.01 for row in strata["pool"].values())
    strata_ok = all(row["delta"] >= 0 for column in ("language", "mode")
                    for row in strata[column].values())
    dev_ok = all(
        row["pooled"]["delta"] is not None and row["pooled"]["delta"] >= 0 and
        row["macro_pool_delta"] >= 0 and
        all(value["delta"] is None or value["delta"] >= -.01
            for value in row["languages"].values()) and
        all(value["delta"] is None or value["delta"] >= -.03
            for value in row["pools"].values())
        for row in dev.values()) if dev else False
    gates = {
        "oof_macro_routing_delta_ge_0.005": deltas["routing_objective_macro_source"] >= .005,
        "oof_pooled_routing_nonnegative": deltas["routing_objective_pooled"] >= 0,
        "oof_macro_native_auc_delta_ge_0.010": deltas["native_failure_auc_macro_source"] >= .01,
        "oof_pooled_native_auc_nonnegative": deltas["native_failure_auc_pooled"] >= 0,
        "language_and_mode_routing_nonnegative": strata_ok,
        "broad_pool_routing_delta_ge_minus_0.010": broad_ok,
        "all_development_gates": dev_ok,
    }
    result = {
        "status": "p25b_expanded_data_nested_group_cv",
        "inputs": {"rows": len(frame), "dimension": x.shape[1],
                   "source_families": int(frame.source_family.nunique()),
                   "excluded": excluded,
                   "selection_sha256": sha256(args.selection),
                   "local_single_sha256": sha256(args.local_single),
                   "local_multi_sha256": sha256(args.local_multi),
                   "expert_sha256": sha256(args.expert),
                   "live_artifact_sha256": sha256(args.live_artifact),
                   "p16_artifact_sha256": sha256(args.p16_artifact),
                   "features": feature_receipts},
        "protocol": {"outer_folds": 5, "inner_folds": 3,
                     "primary": "macro within-source exact-budget net gain averaged over 15/30/50%",
                     "tie_break": "within .001: native macro AUC, then fewer parameters",
                     "harmful_weight": 2., "max_epochs": args.max_epochs,
                     "patience": args.patience, "batch_size": args.batch_size},
        "specs": [{**asdict(spec), "name": spec.name,
                   "parameter_count": spec.parameter_count} for spec in SPECS],
        "outer_folds": outer_rows, "global_sweep": global_rows,
        "global_winner": global_winner,
        "summaries": summaries, "deltas_vs_live": deltas,
        "strata": strata, "development": dev, "gates": gates,
        "graduate_to_new_prospective": bool(all(gates.values())),
        "model_output_sha256": sha256(args.model_output),
        "live_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"winner": global_winner, "summaries": summaries,
                      "deltas_vs_live": deltas, "gates": gates,
                      "graduate": result["graduate_to_new_prospective"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
