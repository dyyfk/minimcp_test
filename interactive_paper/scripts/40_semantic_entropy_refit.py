"""Build local semantic-consistency labels and refit an uncertainty probe.

Four answers per row (one cached official + three stochastic native-duplex
samples) are clustered with a local multilingual sentence encoder.  The
source-grouped sweep predicts continuous semantic entropy from the same frozen
hidden features, then tests whether it adds to the P3a native-failure base.
No OpenAI judge is used.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer


BLOCK = 4096
RATES = {"conservative": .15, "balanced": .30, "aggressive": .50}
THRESHOLDS = (.70, .78, .85)
RIDGE_ALPHAS = (100., 1000., 10000., 100000.)


def load_p3a_module():
    path = Path(__file__).with_name("35_feature_conditioning.py")
    spec = importlib.util.spec_from_file_location("feature_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_text(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value if value else "[no answer]"


def read_samples(sample_dir: Path):
    rows = {}
    for path in sorted(sample_dir.glob("semantic_samples.rank*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["id"])] = row
    return rows


def embed_texts(texts, model_dir: Path, batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval().to(device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=256, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            outputs.append(pooled.float().cpu().numpy())
    return np.concatenate(outputs)


def components(similarity, threshold):
    parent = list(range(len(similarity)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(len(similarity)):
        for j in range(i):
            if similarity[i, j] >= threshold:
                union(i, j)
    return [find(i) for i in range(len(similarity))]


def entropy_from_components(labels):
    counts = pd.Series(labels).value_counts().to_numpy(dtype=float)
    probability = counts / counts.sum()
    return float(-(probability * np.log(probability)).sum() /
                 math.log(len(labels)))


def build_labels(selection_path, sample_dir, model_dir, output, batch_size,
                 sample_count=3):
    selection = pd.read_parquet(selection_path).set_index("id")
    samples = read_samples(sample_dir)
    missing = sorted(set(selection.index.astype(str)) - set(samples))
    errors = [row_id for row_id, row in samples.items() if row.get("error")]
    short = [row_id for row_id, row in samples.items()
             if len(row.get("samples", [])) < sample_count]
    if missing or errors or short:
        raise RuntimeError(
            f"sample coverage incomplete: missing={len(missing)} "
            f"errors={len(errors)} short={len(short)}")

    ordered_ids = sorted(selection.index.astype(str))
    texts, spans = [], {}
    for row_id in ordered_ids:
        row = samples[row_id]
        answers = [row.get("official_answer", "")] + [
            sample.get("answer", "")
            for sample in row["samples"][:sample_count]]
        begin = len(texts)
        texts.extend(normalize_text(answer) for answer in answers)
        spans[row_id] = slice(begin, len(texts))
    embedding = embed_texts(texts, model_dir, batch_size)

    labels = []
    for row_id in ordered_ids:
        vectors = embedding[spans[row_id]]
        similarity = np.clip(vectors @ vectors.T, -1, 1)
        upper = similarity[np.triu_indices(len(vectors), 1)]
        sampled = similarity[1:, 1:]
        sampled_upper = sampled[np.triu_indices(len(sampled), 1)]
        row = {
            "id": row_id,
            "mean_pairwise_dissimilarity": float(np.mean(1 - upper)),
            "min_pairwise_similarity": float(np.min(upper)),
            "official_sample_dissimilarity": float(
                np.mean(1 - similarity[0, 1:])),
            "sampled_pairwise_dissimilarity": float(
                np.mean(1 - sampled_upper)) if len(sampled_upper) else np.nan,
        }
        for threshold in THRESHOLDS:
            row[f"entropy_{int(threshold * 100)}"] = entropy_from_components(
                components(similarity, threshold))
        labels.append(row)
    frame = pd.DataFrame(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def fit_base_fold(train, test, x, y):
    cols = np.arange(BLOCK, 3 * BLOCK)
    base = LogisticRegression(C=3e-4, max_iter=3000, tol=1e-5)
    base.fit(x[train][:, cols], y[train])
    return test, base.predict_proba(x[test][:, cols])[:, 1]


def fit_semantic_fold(train, test, x, sem_index, sem_targets, ridge_alpha):
    cols = np.arange(BLOCK, 3 * BLOCK)
    train_set = set(train)
    test_set = set(test)
    sem_train_mask = np.array([index in train_set for index in sem_index])
    sem_test_mask = np.array([index in test_set for index in sem_index])
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(x[sem_index[sem_train_mask]][:, cols],
              sem_targets[sem_train_mask])
    sem_score = ridge.predict(x[sem_index[sem_test_mask]][:, cols])
    return sem_index[sem_test_mask], sem_score


def zfit(values):
    return float(np.mean(values)), float(np.std(values))


def zapply(values, center, scale):
    return (values - center) / max(scale, 1e-8)


def logit(p):
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    return np.log(p) - np.log1p(-p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aligned-artifact", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--reuse-labels", action="store_true")
    args = parser.parse_args()

    labels = (pd.read_parquet(args.labels) if args.reuse_labels else
              build_labels(args.selection, args.sample_dir,
                           args.embedding_model, args.labels, args.batch_size))
    p3a = load_p3a_module()
    x, y, ids, groups, blocks = p3a.collect_training(args.data_dir)
    id_to_index = {row_id: i for i, row_id in enumerate(ids)}
    labels = labels[labels["id"].isin(id_to_index)].copy()
    sem_index = np.array([id_to_index[row_id] for row_id in labels["id"]])

    expert_df = pd.read_parquet(args.expert_labels)
    ecol = "adequate" if "adequate" in expert_df else "expert_ok"
    expert = (expert_df.dropna(subset=[ecol]).drop_duplicates("id", keep="last")
              .set_index("id")[ecol].astype(int).to_dict())
    expert_ok = np.array([expert[ids[i]] for i in sem_index])
    gain = expert_ok - (1 - y[sem_index])
    cv = list(StratifiedGroupKFold(
        5, shuffle=True, random_state=42).split(x, y, groups))

    target_names = [f"entropy_{int(t * 100)}" for t in THRESHOLDS]
    target_names += ["mean_pairwise_dissimilarity",
                     "official_sample_dissimilarity",
                     "sampled_pairwise_dissimilarity"]
    direct_metrics = {}
    for target_name in target_names:
        target = labels[target_name].to_numpy(dtype=float)
        direct_metrics[target_name] = {
            "native_auc": float(roc_auc_score(y[sem_index], target)),
            "benefit_auc": float(roc_auc_score(
                (gain > 0).astype(int), target)),
            "routing_objective": p3a.routing_objective(gain, target),
            "gain_spearman": float(spearmanr(gain, target).statistic),
        }
        print("direct", target_name, direct_metrics[target_name], flush=True)
    base_outputs = joblib.Parallel(n_jobs=min(args.jobs, 5), verbose=10)(
        joblib.delayed(fit_base_fold)(train, test, x, y)
        for train, test in cv)
    base_oof = np.full(len(y), np.nan)
    for test, base_score in base_outputs:
        base_oof[test] = base_score

    sem_targets = labels[target_names].to_numpy(dtype=float)
    tasks = [(ridge_alpha, train, test)
             for ridge_alpha in RIDGE_ALPHAS for train, test in cv]
    outputs = joblib.Parallel(n_jobs=args.jobs, verbose=10)(
        joblib.delayed(fit_semantic_fold)(
            train, test, x, sem_index, sem_targets, ridge_alpha)
        for ridge_alpha, train, test in tasks)

    candidates = []
    offset = 0
    sem_oof_by_alpha = {}
    for ridge_alpha in RIDGE_ALPHAS:
        sem_oof = np.full((len(y), len(target_names)), np.nan)
        for indices, sem_score in outputs[offset:offset + 5]:
            sem_oof[indices] = sem_score
        offset += 5
        sem_oof_by_alpha[ridge_alpha] = sem_oof
    for target_column, target_name in enumerate(target_names):
        target = sem_targets[:, target_column]
        for ridge_alpha in RIDGE_ALPHAS:
            sem_oof = sem_oof_by_alpha[ridge_alpha][:, target_column]
            prediction = sem_oof[sem_index]
            correlation = float(spearmanr(target, prediction).statistic)
            bc, bs = zfit(logit(base_oof[sem_index]))
            sc, ss = zfit(prediction)
            bz = zapply(logit(base_oof[sem_index]), bc, bs)
            sz = zapply(prediction, sc, ss)
            for blend in (0., .1, .25, .5, .75, 1.):
                score = (1 - blend) * bz + blend * sz
                name = f"{target_name}_ridge{ridge_alpha:g}_blend{blend:g}"
                row = {
                    "name": name, "target": target_name,
                    "ridge_alpha": ridge_alpha, "blend": blend,
                    "target_oof_spearman": correlation,
                    "native_oof_auc": float(roc_auc_score(y[sem_index], score)),
                    "benefit_oof_auc": float(roc_auc_score(
                        (gain > 0).astype(int), score)),
                    "routing_oof_objective": p3a.routing_objective(gain, score),
                    "base_center": bc, "base_scale": bs,
                    "semantic_center": sc, "semantic_scale": ss,
                }
                candidates.append(row)
                print(name, correlation, row["native_oof_auc"],
                      row["benefit_oof_auc"], row["routing_oof_objective"],
                      flush=True)

    best_native = max(row["native_oof_auc"] for row in candidates)
    eligible = [row for row in candidates
                if row["native_oof_auc"] >= best_native - .005]
    winner = max(eligible, key=lambda row: row["routing_oof_objective"])
    print("winner", winner, flush=True)

    cols = np.arange(BLOCK, 3 * BLOCK)
    base_model = LogisticRegression(C=3e-4, max_iter=5000, tol=1e-5)
    base_model.fit(x[:, cols], y)
    sem_model = Ridge(alpha=winner["ridge_alpha"])
    sem_model.fit(x[sem_index][:, cols],
                  labels[winner["target"]].to_numpy(dtype=float))

    def candidate_score(xp):
        base = base_model.predict_proba(xp[:, cols])[:, 1]
        semantic = sem_model.predict(xp[:, cols])
        bz = zapply(logit(base), winner["base_center"], winner["base_scale"])
        sz = zapply(semantic, winner["semantic_center"],
                    winner["semantic_scale"])
        return (1 - winner["blend"]) * bz + winner["blend"] * sz

    pools = {}
    eval_specs = [("internal_test", "testoff", None)] + [
        (pool, pool + "off", col) for pool, col in p3a.EXTERNAL]
    for name, tag, col in eval_specs:
        pool_ids, xp = p3a.load_feats(args.data_dir, tag)
        failure = p3a.native_failure(args.data_dir, tag)
        expert_out = p3a.expert_outcomes(args.data_dir, name, col)
        keep = [i for i, row_id in enumerate(pool_ids)
                if row_id in failure and row_id in expert_out]
        kept_ids = [pool_ids[i] for i in keep]
        xp = xp[keep]
        local_ok = np.array([1 - failure[i] for i in kept_ids])
        expert_eval = np.array([expert_out[i] for i in kept_ids])
        gain_p = expert_eval - local_ok
        aligned = p3a.artifact_score(args.aligned_artifact, xp)
        candidate = candidate_score(xp)
        native_y = 1 - local_ok
        benefit_y = (gain_p > 0).astype(int)
        result = {
            "n": len(keep),
            "native_auc_aligned": float(roc_auc_score(native_y, aligned)),
            "native_auc_candidate": float(roc_auc_score(native_y, candidate)),
            "native_auc_delta_ci": p3a.bootstrap_auc_delta(
                native_y, candidate, aligned),
            "benefit_auc_aligned": float(roc_auc_score(benefit_y, aligned)),
            "benefit_auc_candidate": float(roc_auc_score(benefit_y, candidate)),
            "benefit_auc_delta_ci": p3a.bootstrap_auc_delta(
                benefit_y, candidate, aligned),
            "budgets": {},
        }
        for tier, rate in RATES.items():
            af = p3a.top_mask(aligned, rate)
            cf = p3a.top_mask(candidate, rate)
            result["budgets"][tier] = {
                "aligned_accuracy": float(
                    np.where(af, expert_eval, local_ok).mean()),
                "candidate_accuracy": float(
                    np.where(cf, expert_eval, local_ok).mean()),
                "candidate_vs_aligned_delta_ci": p3a.bootstrap_cascade_delta(
                    local_ok, expert_eval, candidate, aligned, rate),
            }
        pools[name] = result

    out = {
        "inputs": {
            "selection_sha256": p3a.sha256(args.selection),
            "labels_sha256": p3a.sha256(args.labels),
            "aligned_sha256": p3a.sha256(args.aligned_artifact),
            "expert_labels_sha256": p3a.sha256(args.expert_labels),
        },
        "train": {"native_n": len(y), "semantic_n": len(labels),
                  "groups": len(set(groups)), "blocks": blocks,
                  "target_means": {name: float(labels[name].mean())
                                   for name in target_names}},
        "direct_semantic_metrics": direct_metrics,
        "selection_rule": "max routing OOF among configs within .005 of best native OOF AUC on semantic pilot rows",
        "sweep": candidates,
        "winner": winner,
        "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
