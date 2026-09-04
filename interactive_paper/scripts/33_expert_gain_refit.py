"""Train a deployment-aligned router on expert gain.

The deployed gate predicts local failure.  This experiment instead joins the
official-native local outcome with a paired expert outcome and trains a
single-dot-product score for ``expert_ok - local_ok``.  It compares direct
gain regression and a positive-benefit classifier with source-grouped CV,
then evaluates exact 15/30/50% escalation budgets on untouched native pools.

Nothing in this script overwrites ``gate_native.json``.

Example (from ``interactive_paper/``)::

    python scripts/33_expert_gain_refit.py \
      --data-dir data \
      --expert-labels data/train_ceiling.parquet \
      --aligned-artifact /path/to/native_bundle/data/gate_native.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
CORE = [
    ("caliboff", "calib_features.parquet"),
    ("expoff", "expansion_labels.parquet"),
    ("exp2off", "expansion2_labels.parquet"),
    ("exp3off", "expansion3_labels.parquet"),
    ("exp3zhoff", "expansion3zh_labels.parquet"),
    ("freshoff", "fresh_labels.parquet"),
]
EXTERNAL = [
    ("striviaqa", "oab_ok"),
    ("swebq", "oab_ok"),
    ("sllama", "oab_ok"),
    ("sdqa", "heard_ok"),
    ("sreason", "heard_ok"),
]
RNG = np.random.default_rng(42)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feats(data_dir: Path, tag: str):
    ids, arrays = [], []
    for path in sorted(data_dir.glob(
            f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(path, allow_pickle=True)
        ids += [str(row_id) for row_id in z["ids"]]
        arrays.append(z["X"].astype(np.float32, copy=False))
    if not arrays:
        raise FileNotFoundError(f"no official-native feature shards for {tag}")
    x = np.concatenate(arrays)
    rows = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    rows = rows.drop_duplicates("id", keep="last")
    return list(rows["id"]), x[rows["row"].to_numpy()]


def native_local(data_dir: Path, tag: str):
    path = data_dir / f"frozen_native_{tag}_judged.parquet"
    df = pd.read_parquet(path).dropna(subset=["adequate"])
    df = df.drop_duplicates("id", keep="last")
    return df.set_index("id")["adequate"].astype(int).to_dict()


def load_expert(paths: list[Path], column: str | None):
    frames = []
    receipts = []
    for path in paths:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in (".jsonl", ".json"):
            df = pd.read_json(path, lines=path.suffix == ".jsonl")
        else:
            raise ValueError(f"unsupported expert-label file: {path}")
        value_col = column
        if value_col is None:
            for candidate in ("expert_ok", "adequate", "heard_ok", "oab_ok"):
                if candidate in df.columns:
                    value_col = candidate
                    break
        if value_col is None or value_col not in df.columns:
            raise ValueError(
                f"{path} has no expert outcome column; pass --expert-column")
        if "id" not in df.columns:
            raise ValueError(f"{path} has no id column")
        part = df[["id", value_col]].rename(columns={value_col: "expert_ok"})
        part = part.dropna(subset=["expert_ok"])
        part["id"] = part["id"].astype(str)
        part["expert_ok"] = part["expert_ok"].astype(int)
        bad = set(part["expert_ok"].unique()) - {0, 1}
        if bad:
            raise ValueError(f"{path} has non-binary expert outcomes: {bad}")
        frames.append(part)
        receipts.append({"path": str(path), "sha256": sha256(path),
                         "rows_nonnull": int(len(part)),
                         "column": value_col})
    expert = pd.concat(frames, ignore_index=True)
    conflicts = expert.groupby("id")["expert_ok"].nunique()
    conflicts = list(conflicts[conflicts > 1].index)
    if conflicts:
        raise RuntimeError(
            f"conflicting expert outcomes for {len(conflicts)} ids")
    expert = expert.drop_duplicates("id", keep="last").set_index("id")
    return expert["expert_ok"].to_dict(), receipts


def group_for(meta: pd.DataFrame, tag: str, row_id: str) -> str:
    if row_id not in meta.index:
        return f"{tag}:unknown"
    row = meta.loc[row_id]
    for col in ("source", "pool"):
        if col in meta.columns and pd.notna(row.get(col)):
            # The same source family can recur in several expansion rounds.
            # Group by the family itself, not tag:family, or CV leaks a family
            # from (say) expoff into the exp2off validation fold.
            return str(row[col])
    return f"{tag}:unknown"


def eval_ids(data_dir: Path) -> set[str]:
    ids = set()
    for tag in ["testoff"] + [pool + "off" for pool, _ in EXTERNAL]:
        try:
            pool_ids, _ = load_feats(data_dir, tag)
        except FileNotFoundError:
            continue
        ids.update(pool_ids)
    return ids


def collect_training(data_dir: Path, expert: dict[str, int]):
    leaked = set(expert) & eval_ids(data_dir)
    if leaked:
        raise RuntimeError(
            f"expert-label input leaks {len(leaked)} official eval ids")
    x_parts, local_parts, expert_parts = [], [], []
    id_parts, group_parts, blocks = [], [], []
    for tag, meta_file in CORE:
        ids, x = load_feats(data_dir, tag)
        local = native_local(data_dir, tag)
        meta = (pd.read_parquet(data_dir / meta_file)
                .drop_duplicates("id", keep="last").set_index("id"))
        rows = []
        for j, row_id in enumerate(ids):
            if row_id not in local or row_id not in expert:
                continue
            if tag == "freshoff" and (
                    row_id not in meta.index or
                    meta.loc[row_id].get("split") != "train"):
                continue
            rows.append(j)
        if not rows:
            blocks.append({"tag": tag, "matched": 0})
            continue
        x_parts.append(x[rows])
        local_parts.append(np.array([local[ids[j]] for j in rows], dtype=int))
        expert_parts.append(np.array([expert[ids[j]] for j in rows], dtype=int))
        id_parts.append(np.array([ids[j] for j in rows]))
        group_parts.append(np.array(
            [group_for(meta, tag, ids[j]) for j in rows]))
        blocks.append({"tag": tag, "matched": len(rows),
                       "local_ok": float(np.mean(
                           [local[ids[j]] for j in rows])),
                       "expert_ok": float(np.mean(
                           [expert[ids[j]] for j in rows]))})
    if not x_parts:
        raise RuntimeError("no expert labels matched official-native features")
    return (np.concatenate(x_parts), np.concatenate(local_parts),
            np.concatenate(expert_parts), np.concatenate(group_parts),
            np.concatenate(id_parts), blocks)


def top_mask(score, rate):
    n = len(score)
    k = min(n, max(0, int(round(n * rate))))
    fire = np.zeros(n, dtype=bool)
    if k:
        # Stable tie-breaking makes receipts repeatable.
        order = np.lexsort((np.arange(n), -np.asarray(score)))
        fire[order[:k]] = True
    return fire


def routing_metrics(local_ok, expert_ok, score):
    gain = expert_ok - local_ok
    out = {}
    for tier, rate in RATES.items():
        fire = top_mask(score, rate)
        out[tier] = {
            "acc": float(np.where(fire, expert_ok, local_ok).mean()),
            "rate": float(fire.mean()),
            "mean_selected_gain": float(gain[fire].mean()) if fire.any() else 0.,
            "beneficial_rate_selected": float((gain[fire] > 0).mean())
            if fire.any() else 0.,
            "harmful_rate_selected": float((gain[fire] < 0).mean())
            if fire.any() else 0.,
        }
    return out


def routing_objective(gain, score):
    return float(np.mean([
        gain[top_mask(score, rate)].sum() / len(gain)
        for rate in RATES.values()
    ]))


@dataclass
class LinearCandidate:
    family: str
    parameter: float
    model: object

    def score(self, x):
        if self.family == "benefit_logistic":
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)

    @property
    def w(self):
        if self.family == "benefit_logistic":
            return self.model.coef_[0]
        return self.model.coef_

    @property
    def b(self):
        value = self.model.intercept_
        return float(value[0] if np.ndim(value) else value)


def fit_family(family, parameter, x, gain, sample_weight=None):
    if family == "benefit_logistic":
        model = LogisticRegression(C=parameter, max_iter=5000)
        model.fit(x, (gain > 0).astype(int), sample_weight=sample_weight)
    elif family == "direct_gain_ridge":
        model = Ridge(alpha=parameter)
        model.fit(x, gain, sample_weight=sample_weight)
    else:
        raise ValueError(family)
    return LinearCandidate(family, parameter, model)


def grouped_oof(x, local_ok, expert_ok, groups):
    gain = expert_ok - local_ok
    # Preserve the rare harmful/beneficial classes across folds where possible.
    strat = gain + 1
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, strat, groups))
    candidates = []
    grids = {
        "benefit_logistic": (1e-4, 3e-4, 1e-3),
        "direct_gain_ridge": (100., 1000., 10000.),
    }
    for family, parameters in grids.items():
        for parameter in parameters:
            oof = np.full(len(gain), np.nan)
            for train, test in cv:
                # Harmful escalations deserve more weight than neutral cases.
                weights = np.where(gain[train] < 0, 2., 1.)
                model = fit_family(
                    family, parameter, x[train], gain[train], weights)
                oof[test] = model.score(x[test])
            objective = routing_objective(gain, oof)
            candidates.append({"family": family, "parameter": parameter,
                               "routing_objective": objective,
                               "oof": oof})
            print(f"{family:<20} {parameter:g}: grouped OOF routing "
                  f"objective {objective:+.4f}")
    best = max(candidates, key=lambda row: row["routing_objective"])
    return best, [{k: v for k, v in row.items() if k != "oof"}
                  for row in candidates]


def artifact_score(path: Path, x):
    artifact = json.loads(path.read_text())
    return x @ np.asarray(artifact["w"]) + float(artifact["b"])


def expert_eval(data_dir: Path, pool: str, col: str | None = None):
    if pool == "internal_test":
        df = pd.read_parquet(data_dir / "frozen_v3_traces.parquet")
        out = (df[df["mode"] == "escalated"].groupby("id")["heard_ok"]
               .max().dropna().astype(int))
    else:
        df = pd.read_parquet(data_dir / f"{pool}_conclive_traces.parquet")
        out = df[df["tier"] == "always"].dropna(subset=[col])
        out = out.drop_duplicates("id", keep="last").set_index("id")[col]
        out = out.astype(int)
    return out.to_dict()


def bootstrap_acc_delta(local_ok, expert_ok, new_score, old_score,
                        rate, n_boot=4000):
    values = []
    idx = np.arange(len(local_ok))
    for _ in range(n_boot):
        sample = RNG.choice(idx, len(idx), replace=True)
        lo, eo = local_ok[sample], expert_ok[sample]
        nf = top_mask(new_score[sample], rate)
        of = top_mask(old_score[sample], rate)
        values.append(np.where(nf, eo, lo).mean() -
                      np.where(of, eo, lo).mean())
    return [float(np.mean(values)),
            *[float(v) for v in np.percentile(values, [2.5, 97.5])]]


def evaluate(data_dir: Path, deployed_path: Path, aligned_path: Path | None,
             candidate: LinearCandidate):
    pools = [("internal_test", "testoff", None)] + [
        (pool, pool + "off", col) for pool, col in EXTERNAL]
    result = {}
    for name, tag, col in pools:
        ids, x = load_feats(data_dir, tag)
        local = native_local(data_dir, tag)
        expert = expert_eval(data_dir, name, col)
        rows = [j for j, row_id in enumerate(ids)
                if row_id in local and row_id in expert]
        ids = [ids[j] for j in rows]
        x = x[rows]
        local_ok = np.array([local[row_id] for row_id in ids])
        expert_ok = np.array([expert[row_id] for row_id in ids])
        gain = expert_ok - local_ok
        deployed_score = artifact_score(deployed_path, x)
        candidate_score = candidate.score(x)
        scores = {"deployed_failure": deployed_score,
                  "gain_candidate": candidate_score}
        if aligned_path:
            scores["aligned_failure"] = artifact_score(aligned_path, x)
        pool_result = {
            "n": len(ids), "local_acc": float(local_ok.mean()),
            "expert_acc": float(expert_ok.mean()),
            "positive_gain_rate": float((gain > 0).mean()),
            "harmful_gain_rate": float((gain < 0).mean()),
            "scores": {}, "bootstrap_candidate_vs_deployed": {}}
        y_benefit = (gain > 0).astype(int)
        for score_name, score in scores.items():
            pool_result["scores"][score_name] = {
                "benefit_auc": float(roc_auc_score(y_benefit, score)),
                "budgets": routing_metrics(local_ok, expert_ok, score)}
        for tier, rate in RATES.items():
            pool_result["bootstrap_candidate_vs_deployed"][tier] = (
                bootstrap_acc_delta(local_ok, expert_ok, candidate_score,
                                    deployed_score, rate))
        result[name] = pool_result
        old30 = pool_result["scores"]["deployed_failure"]["budgets"][
            "balanced"]["acc"]
        new30 = pool_result["scores"]["gain_candidate"]["budgets"][
            "balanced"]["acc"]
        print(f"{name:<14} n={len(ids):>4} 30% cascade "
              f"{old30:.3f} -> {new30:.3f} ({new30-old30:+.3f})")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--expert-labels", type=Path, action="append",
                        required=True)
    parser.add_argument("--expert-column")
    parser.add_argument("--deployed-artifact", type=Path)
    parser.add_argument("--aligned-artifact", type=Path)
    parser.add_argument("--candidate-out", type=Path)
    parser.add_argument("--result-out", type=Path,
                        default=Path("figures/expert_gain_refit.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    deployed = args.deployed_artifact or data_dir / "gate_native.json"
    candidate_out = (args.candidate_out or
                     data_dir / "gate_native_gain_candidate.json")
    expert, expert_receipts = load_expert(
        args.expert_labels, args.expert_column)
    x, local_ok, expert_ok, groups, ids, blocks = collect_training(
        data_dir, expert)
    gain = expert_ok - local_ok
    print(f"paired train n={len(gain)}, local={local_ok.mean():.3f}, "
          f"expert={expert_ok.mean():.3f}, positive={float((gain>0).mean()):.3f}, "
          f"harmful={float((gain<0).mean()):.3f}")
    best, sweep = grouped_oof(x, local_ok, expert_ok, groups)
    print(f"selected {best['family']} {best['parameter']:g}")
    weights = np.where(gain < 0, 2., 1.)
    candidate = fit_family(
        best["family"], best["parameter"], x, gain, weights)
    thresholds = {tier: float(np.quantile(best["oof"], 1 - rate))
                  for tier, rate in RATES.items()}
    artifact = {
        "w": candidate.w.tolist(), "b": candidate.b,
        "score_link": ("sigmoid" if candidate.family == "benefit_logistic"
                       else "identity"),
        "model": candidate.family, "parameter": candidate.parameter,
        "train_n": int(len(gain)), "eot_thresholds": thresholds,
        "recipe": "scripts/33 official-native paired expert-gain candidate",
        "expert_label_receipts": expert_receipts,
    }
    candidate_out.parent.mkdir(parents=True, exist_ok=True)
    candidate_out.write_text(json.dumps(artifact))
    pools = evaluate(data_dir, deployed, args.aligned_artifact, candidate)
    result = {
        "train": {"n": int(len(gain)),
                  "local_acc": float(local_ok.mean()),
                  "expert_acc": float(expert_ok.mean()),
                  "positive_gain_rate": float((gain > 0).mean()),
                  "harmful_gain_rate": float((gain < 0).mean()),
                  "unique_groups": int(len(np.unique(groups))),
                  "blocks": blocks, "expert_receipts": expert_receipts,
                  "selection": {k: v for k, v in best.items()
                                if k != "oof"}, "sweep": sweep},
        "pools": pools,
        "candidate_sha256": sha256(candidate_out),
        "deployed_sha256": sha256(deployed),
        "aligned_sha256": sha256(args.aligned_artifact)
        if args.aligned_artifact else None,
    }
    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    args.result_out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {candidate_out} + {args.result_out}")


if __name__ == "__main__":
    main()
