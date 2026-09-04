"""Nested source-family CV for higher-capacity native-failure probe heads.

This stage deliberately does not open a new prospective set.  Hyperparameters
are selected in inner source-family folds, and the complete selection procedure
is estimated in held-out outer source-family folds.  Only a material nested-OOF
gain is allowed to graduate to a separately frozen prospective evaluation.
Nothing overwrites the live gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


BLOCK = 4096


def load_p3a_module():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HeadSpec:
    width: int
    dropout: float
    weight_decay: float
    learning_rate: float
    focal_gamma: float

    @property
    def name(self):
        return (f"resmlp_w{self.width}_d{self.dropout:g}_"
                f"wd{self.weight_decay:g}_lr{self.learning_rate:g}_"
                f"focal{self.focal_gamma:g}")


SPECS = (
    HeadSpec(0, 0., 1e-3, 1e-3, 0.),
    HeadSpec(32, .2, 1e-3, 1e-3, 0.),
    HeadSpec(128, .2, 1e-3, 1e-3, 0.),
    HeadSpec(128, .5, 1e-3, 1e-3, 0.),
    HeadSpec(128, .2, 1e-2, 1e-3, 0.),
    HeadSpec(128, .2, 1e-3, 3e-4, 0.),
    HeadSpec(128, .2, 1e-3, 1e-3, 1.),
    HeadSpec(512, .5, 1e-3, 1e-3, 0.),
)


def macro_group_auc(y, score, groups):
    values = []
    for group in sorted(set(groups)):
        mask = groups == group
        if len(np.unique(y[mask])) == 2:
            values.append(roc_auc_score(y[mask], score[mask]))
    return float(np.mean(values)), len(values)


def standardize(train, test):
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    train = np.clip((train - mean) / std, -10., 10.)
    test = np.clip((test - mean) / std, -10., 10.)
    return train.astype(np.float32, copy=False), test.astype(np.float32,
                                                              copy=False)


def make_model(torch, dimension, spec):
    class ResidualMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(dimension, 1)
            if spec.width:
                self.hidden = torch.nn.Linear(
                    dimension, spec.width, bias=False)
                self.dropout = torch.nn.Dropout(spec.dropout)
                self.out = torch.nn.Linear(spec.width, 1, bias=False)
                torch.nn.init.normal_(self.hidden.weight, std=.005)
                torch.nn.init.normal_(self.out.weight, std=.005)

        def forward(self, value):
            if not spec.width:
                return self.base(value).squeeze(-1)
            residual = self.out(self.dropout(torch.nn.functional.gelu(
                self.hidden(value))))
            return (self.base(value) + residual).squeeze(-1)

    return ResidualMLP()


def focal_bce(torch, logits, target, gamma):
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    if gamma <= 0:
        return raw.mean()
    prob = torch.sigmoid(logits)
    pt = torch.where(target > .5, prob, 1 - prob)
    return (((1 - pt) ** gamma) * raw).mean()


def predict(torch, model, x, batch_size=1024):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start:start + batch_size])
            parts.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(parts)


def train_early(x_train, y_train, x_valid, y_valid, valid_groups, spec,
                device, seed, max_epochs, patience, batch_size):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    xt, xv = standardize(x_train, x_valid)
    xt = torch.from_numpy(xt).to(device)
    xv = torch.from_numpy(xv).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).to(device)
    model = make_model(torch, xt.shape[1], spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate,
        weight_decay=spec.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = {"macro_auc": -np.inf, "epoch": 0, "state": None,
            "pooled_auc": None, "valid_groups": None}
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = focal_bce(torch, model(xt[index]), yt[index],
                             spec.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        score = predict(torch, model, xv)
        macro, n_groups = macro_group_auc(y_valid, score, valid_groups)
        pooled = float(roc_auc_score(y_valid, score))
        if macro > best["macro_auc"] + 1e-5:
            best = {
                "macro_auc": macro, "epoch": epoch,
                "state": {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()},
                "pooled_auc": pooled, "valid_groups": n_groups,
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best.pop("state"))
    score = predict(torch, model, xv)
    del model, xt, xv, yt
    torch.cuda.empty_cache()
    return score, best


def train_fixed(x_train, y_train, x_test, spec, epochs, device, seed,
                batch_size):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    xt, xv = standardize(x_train, x_test)
    xt = torch.from_numpy(xt).to(device)
    xv = torch.from_numpy(xv).to(device)
    yt = torch.from_numpy(y_train.astype(np.float32)).to(device)
    model = make_model(torch, xt.shape[1], spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate,
        weight_decay=spec.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = focal_bce(torch, model(xt[index]), yt[index],
                             spec.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    score = predict(torch, model, xv)
    del model, xt, xv, yt
    torch.cuda.empty_cache()
    return score


def logistic_oof(x, y, groups, splits):
    score = np.full(len(y), np.nan)
    cols = np.arange(BLOCK, 3 * BLOCK)
    for train, test in splits:
        model = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
        model.fit(x[train][:, cols], y[train])
        score[test] = model.predict_proba(x[test][:, cols])[:, 1]
    return score


def parameter_count(dimension, width):
    return dimension + 1 + (dimension * width + width if width else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    p3a = load_p3a_module()
    x, y, ids, groups, blocks = p3a.collect_training(args.data_dir)
    outer = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))
    live = p3a.artifact_score(args.live_artifact, x)
    p3a_oof = logistic_oof(x, y, groups, outer)
    nested_oof = np.full(len(y), np.nan)
    fold_rows = []
    inner_history = defaultdict(list)

    for outer_index, (outer_train, outer_test) in enumerate(outer):
        inner_rel = list(StratifiedGroupKFold(
            3, shuffle=True, random_state=100 + outer_index).split(
                x[outer_train], y[outer_train], groups[outer_train]))
        spec_rows = []
        for spec_index, head in enumerate(SPECS):
            fold_metrics = []
            fold_epochs = []
            for inner_index, (inner_train_rel, inner_valid_rel) in enumerate(
                    inner_rel):
                inner_train = outer_train[inner_train_rel]
                inner_valid = outer_train[inner_valid_rel]
                _score, best = train_early(
                    x[inner_train], y[inner_train], x[inner_valid],
                    y[inner_valid], groups[inner_valid], head, args.device,
                    seed=10000 * outer_index + 100 * spec_index + inner_index,
                    max_epochs=args.max_epochs, patience=args.patience,
                    batch_size=args.batch_size)
                fold_metrics.append(best["macro_auc"])
                fold_epochs.append(best["epoch"])
            row = {
                "name": head.name,
                "spec": asdict(head),
                "parameter_count": parameter_count(x.shape[1], head.width),
                "inner_macro_auc_mean": float(np.mean(fold_metrics)),
                "inner_macro_auc_by_fold": [float(v) for v in fold_metrics],
                "best_epoch_by_fold": fold_epochs,
                "selected_epoch": max(1, int(round(np.median(fold_epochs)))),
            }
            spec_rows.append(row)
            inner_history[head.name].append(row["inner_macro_auc_mean"])
            print("outer", outer_index, head.name,
                  row["inner_macro_auc_mean"], row["selected_epoch"],
                  flush=True)
        winner = max(spec_rows, key=lambda row: (
            row["inner_macro_auc_mean"], -row["parameter_count"]))
        head = next(item for item in SPECS if item.name == winner["name"])
        score = train_fixed(
            x[outer_train], y[outer_train], x[outer_test], head,
            winner["selected_epoch"], args.device, 90000 + outer_index,
            args.batch_size)
        nested_oof[outer_test] = score
        outer_macro, n_groups = macro_group_auc(
            y[outer_test], score, groups[outer_test])
        fold_rows.append({
            "outer_fold": outer_index,
            "train_rows": len(outer_train), "test_rows": len(outer_test),
            "test_groups": sorted(set(groups[outer_test])),
            "valid_auc_groups": n_groups,
            "winner": winner,
            "outer_macro_group_auc": outer_macro,
            "outer_pooled_auc": float(roc_auc_score(y[outer_test], score)),
            "sweep": spec_rows,
        })
        print("outer winner", outer_index, winner["name"],
              outer_macro, flush=True)

    global_winner_name = max(
        inner_history, key=lambda name: (np.mean(inner_history[name]),
                                         -parameter_count(x.shape[1], next(
                                             item.width for item in SPECS
                                             if item.name == name))))
    selected_epochs = [
        row["winner"]["selected_epoch"] for row in fold_rows
        if row["winner"]["name"] == global_winner_name]
    if not selected_epochs:
        selected_epochs = [int(round(np.median([
            candidate["selected_epoch"] for row in fold_rows
            for candidate in row["sweep"]
            if candidate["name"] == global_winner_name])))]

    def summarize(name, score):
        macro, n_groups = macro_group_auc(y, score, groups)
        return {
            "name": name,
            "pooled_auc": float(roc_auc_score(y, score)),
            "macro_source_auc": macro,
            "valid_source_groups": n_groups,
        }

    summaries = {
        "live_in_sample_reference": summarize("live", live),
        "p3a_group_oof": summarize("p3a", p3a_oof),
        "nested_capacity_oof": summarize("nested_capacity", nested_oof),
    }
    nonlinear_folds = sum(
        row["winner"]["spec"]["width"] > 0 for row in fold_rows)
    out = {
        "status": "capacity_head_nested_group_cv",
        "inputs": {
            "rows": len(y), "dimension": x.shape[1],
            "source_groups": len(set(groups)), "blocks": blocks,
            "live_artifact_sha256": sha256(args.live_artifact),
        },
        "protocol": {
            "outer_folds": 5, "inner_folds": 3,
            "selection_metric": "mean valid-source AUC in inner folds",
            "early_stopping": "inner validation only; median inner epoch is fixed before outer scoring",
            "preprocessing": "fold-local per-coordinate standardization clipped to [-10,10]",
            "max_epochs": args.max_epochs, "patience": args.patience,
            "batch_size": args.batch_size,
        },
        "specs": [{**asdict(head), "name": head.name,
                   "parameter_count": parameter_count(x.shape[1], head.width)}
                  for head in SPECS],
        "outer_folds": fold_rows,
        "selection_frequency": dict(Counter(
            row["winner"]["name"] for row in fold_rows)),
        "global_winner": {
            "name": global_winner_name,
            "mean_inner_macro_auc": float(np.mean(
                inner_history[global_winner_name])),
            "inner_macro_auc_by_outer": [float(v) for v in
                                          inner_history[global_winner_name]],
            "suggested_full_fit_epochs": max(1, int(round(np.median(
                selected_epochs)))),
        },
        "summary": summaries,
        "graduation_rule": {
            "macro_source_auc_gain_vs_p3a_min": .01,
            "pooled_auc_nonnegative_vs_p3a": True,
            "at_least_four_outer_folds_select_nonlinear": True,
        },
    }
    macro_gain = (summaries["nested_capacity_oof"]["macro_source_auc"] -
                  summaries["p3a_group_oof"]["macro_source_auc"])
    pooled_gain = (summaries["nested_capacity_oof"]["pooled_auc"] -
                   summaries["p3a_group_oof"]["pooled_auc"])
    out["decision"] = {
        "macro_source_auc_gain_vs_p3a": macro_gain,
        "pooled_auc_gain_vs_p3a": pooled_gain,
        "outer_folds_selecting_nonlinear": nonlinear_folds,
        "graduate_to_prospective": bool(
            macro_gain >= .01 and pooled_gain >= 0 and nonlinear_folds >= 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"global_winner": out["global_winner"],
                      "summary": summaries, "decision": out["decision"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
