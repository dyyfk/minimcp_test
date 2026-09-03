"""Layer ablation of the DEPLOYED probe feature set (experiment "R3").

Question (paper review, 2026-09-02): the abstract claims a mid-network
read (L22) is "the practical signal" rather than the final-layer read,
but that claim was established on single-position text-input LOPO
sweeps. On audio input the audio-calibrated final-layer probe is not
worse than L22 on any held-out pool (data/layers/layer_sweep_*.json),
and the deployed probe (eot_last + eot_mean8 + user_mean at L22,
12,288-d, 2,310-row calibration mix, C=1e-4; modal_train2.py::refit3)
was never re-fit at any other depth and scored on the external pools.

This module is the torch-free analysis half: given per-query hidden
windows captured by the streaming replay of modal_train2.py::eoth2_shard
(five layers) or modal_train4_layers.py::eoth3_shard (every decoder
layer), it re-fits the identical three-position feature set at each
captured layer on the identical training rows and scores the identical
evaluation pools with the identical labels, then reports paired
bootstrap differences against the deployed layer.

Everything here runs on CPU from stored .npz shards; it is shared by the
Modal function (volume-resident shards) and scripts/33_layer_ablation.py
(shards pulled with `modal volume get`).

Feature semantics are copied from modal_train2._feat so that numbers
are comparable with the shipped v3 artifact:
  eot_last  — last row of the rolling K_EOT-token tail of the assistant
              prefill (the "end-of-turn read").
  eot_mean  — mean over the valid rows of that tail (eot_len rows,
              right-aligned, zero-padded on the left when shorter).
  user_mean — mean over all user-audio-chunk positions.
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

K_EOT = 8
DEPLOYED_LAYER = 22
DEPLOYED_MODES = ("eot_last", "eot_mean", "user_mean")
DEPLOYED_C = 1e-4
BUDGETS = (("conservative", 0.15), ("balanced", 0.30), ("aggressive", 0.50))


# --------------------------------------------------------------------------
# shards
# --------------------------------------------------------------------------
@dataclass
class Shards:
    """One pool's captured windows.

    E: (n, nL, K_EOT, d) float16   rolling tail of the assistant prefill
    M: (n, nL, d)        float16   mean over user-audio positions
    elen: (n,) int                 number of valid rows in the tail
    layers: (nL,) int              decoder-layer index of each slot
    """
    ids: list
    E: np.ndarray
    M: np.ndarray
    elen: np.ndarray
    layers: np.ndarray

    def subset(self, rows) -> "Shards":
        rows = np.asarray(rows)
        return Shards([self.ids[j] for j in rows], self.E[rows], self.M[rows],
                      self.elen[rows], self.layers)

    @property
    def n(self) -> int:
        return len(self.ids)


def load_shards(pattern: str) -> Shards:
    """Load and concatenate every shard matching a glob pattern.

    Later shards win on duplicate ids (same rule as scripts/26)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no shards match {pattern}")
    ids, E, M, L, layers = [], [], [], [], None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        ids += [str(x) for x in z["ids"]]
        E.append(z["H_eot"])
        M.append(z["H_mean"])
        L.append(z["eot_len"])
        lay = np.asarray(z["layers"]).astype(int)
        if layers is None:
            layers = lay
        elif not np.array_equal(layers, lay):
            raise ValueError(f"layer set differs across shards: {p}")
    E = np.concatenate(E)
    M = np.concatenate(M)
    L = np.concatenate(L)
    # dedup keeping the last occurrence
    last = {}
    for j, i in enumerate(ids):
        last[i] = j
    keep = np.array(sorted(last.values()))
    return Shards([ids[j] for j in keep], E[keep], M[keep], L[keep], layers)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def feat(sh: Shards, layer: int, modes=DEPLOYED_MODES, k_eot: int = K_EOT
         ) -> np.ndarray:
    """Feature matrix for one layer and a tuple of modes (concatenated in
    the given order). Mirrors modal_train2._feat exactly."""
    j = int(np.where(sh.layers == layer)[0][0]) if layer in sh.layers else None
    if j is None:
        raise KeyError(f"layer {layer} not captured (have {list(sh.layers)})")
    parts = []
    for m in modes:
        if m == "eot_last":
            parts.append(sh.E[:, j, -1, :].astype(np.float32))
        elif m == "eot_mean":
            He = sh.E[:, j].astype(np.float32)                 # (n, K, d)
            ln = np.clip(sh.elen.astype(np.int32), 1, k_eot)
            mask = (np.arange(k_eot)[None, :]
                    >= (k_eot - ln[:, None])).astype(np.float32)
            parts.append((He * mask[:, :, None]).sum(1) / ln[:, None])
        elif m == "user_mean":
            parts.append(sh.M[:, j].astype(np.float32))
        else:
            raise ValueError(m)
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# fitting / scoring
# --------------------------------------------------------------------------
def fit_probe(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def oof_auc(X: np.ndarray, y: np.ndarray, C: float, seed: int = 42) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(LogisticRegression(C=C, max_iter=5000), X, y,
                            cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, oof))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def paired_bootstrap_delta(y: np.ndarray, s_ref: np.ndarray,
                           s_new: np.ndarray, B: int = 2000,
                           seed: int = 42):
    """Paired bootstrap over queries of AUC(new) - AUC(ref).

    Returns (delta, lo, hi) with a 2.5/97.5 percentile interval. Draws
    that lose one class are skipped (counted, and never more than a few
    on n >= 100 with a balanced-ish label)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y)
    d0 = roc_auc_score(y, s_new) - roc_auc_score(y, s_ref)
    ds = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        ds.append(roc_auc_score(yy, s_new[idx]) - roc_auc_score(yy, s_ref[idx]))
    ds = np.asarray(ds)
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def failure_recall_at_budgets(y: np.ndarray, s: np.ndarray,
                              budgets=BUDGETS) -> dict:
    """Label-free per-pool quantile thresholds (the paper's transfer
    protocol): escalate the top-r fraction by score; report the share of
    local failures caught (recall) and the precision of the escalated
    set. This is what a fixed call budget buys before the expert."""
    out = {}
    n = len(y)
    order = np.argsort(-s, kind="stable")
    for name, r in budgets:
        k = max(1, int(round(r * n)))
        top = order[:k]
        caught = int(y[top].sum())
        out[name] = {"rate": k / n,
                     "recall": caught / max(1, int(y.sum())),
                     "precision": caught / k}
    return out


# --------------------------------------------------------------------------
# the ablation
# --------------------------------------------------------------------------
@dataclass
class Pool:
    name: str
    sh: Shards
    y: np.ndarray
    external: bool = True


@dataclass
class AblationResult:
    layers: list
    modes: list
    C: float
    n_train: int
    fail_rate: float
    pools: list
    rows: dict = field(default_factory=dict)     # layer -> per-pool metrics
    anchor: dict | None = None
    notes: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"layers": [int(l) for l in self.layers],
                "modes": list(self.modes), "C": self.C,
                "n_train": self.n_train, "fail_rate": self.fail_rate,
                "pools": self.pools, "deployed_layer": DEPLOYED_LAYER,
                "rows": {str(k): v for k, v in self.rows.items()},
                "anchor": self.anchor, "notes": self.notes}


def run_ablation(train: Shards, y_train: np.ndarray, pools: list,
                 layers=None, modes=DEPLOYED_MODES, C: float = DEPLOYED_C,
                 ref_layer: int = DEPLOYED_LAYER, anchor: dict | None = None,
                 B: int = 2000, seed: int = 42, with_oof: bool = False,
                 log=print) -> AblationResult:
    """Fit the `modes` feature set at every layer in `layers` (default:
    every captured layer) on `train`, score every pool, and report
    AUC, paired-bootstrap delta vs `ref_layer`, and budget recall.

    `anchor`: optional shipped artifact dict with keys w, b, layer_set,
    modes — scored on the reference layer of the NEW capture as a
    pipeline sanity check (expected to reproduce the paper's numbers).
    """
    if layers is None:
        layers = [int(l) for l in train.layers]
    layers = [int(l) for l in layers]
    if ref_layer not in layers:
        raise ValueError(f"reference layer {ref_layer} not in {layers}")
    pool_names = [p.name for p in pools]
    res = AblationResult(layers=layers, modes=list(modes), C=C,
                         n_train=int(len(y_train)),
                         fail_rate=float(np.mean(y_train)), pools=pool_names)

    # reference layer first so every other layer can be paired against it
    order = [ref_layer] + [l for l in layers if l != ref_layer]
    ref_scores = {}
    for L in order:
        X = feat(train, L, modes)
        clf = fit_probe(X, y_train, C)
        row = {"train_oof_auc": (oof_auc(X, y_train, C, seed) if with_oof
                                 else None)}
        ext = []
        for p in pools:
            s = clf.decision_function(feat(p.sh, L, modes))
            a = auc(p.y, s)
            entry = {"n": int(len(p.y)), "auc": a,
                     "budget": failure_recall_at_budgets(p.y, s)}
            if L == ref_layer:
                ref_scores[p.name] = s
                entry["delta_vs_ref"] = [0.0, 0.0, 0.0]
            else:
                entry["delta_vs_ref"] = list(
                    paired_bootstrap_delta(p.y, ref_scores[p.name], s, B, seed))
            row[p.name] = entry
            if p.external:
                ext.append(a)
        row["ext_mean_auc"] = float(np.mean(ext)) if ext else None
        res.rows[L] = row
        log(f"  L{L:<3d} " + " ".join(
            f"{p.name[:10]:>10s}={row[p.name]['auc']:.3f}" for p in pools)
            + f"  ext-mean={row['ext_mean_auc']:.3f}")

    if anchor is not None:
        w = np.asarray(anchor["w"], dtype=np.float32)
        b = float(anchor["b"])
        a_layers = [int(l) for l in anchor.get("layer_set", [ref_layer])]
        a_modes = tuple(anchor.get("modes", modes))
        anc = {"layer_set": a_layers, "modes": list(a_modes)}
        for p in pools:
            X = np.concatenate([feat(p.sh, L, a_modes) for L in a_layers],
                               axis=1)
            if X.shape[1] != len(w):
                anc[p.name] = {"error": f"dim {X.shape[1]} != {len(w)}"}
                continue
            anc[p.name] = {"auc": auc(p.y, X @ w + b)}
        res.anchor = anc
        log("  anchor (shipped artifact on new capture): " + " ".join(
            f"{k}={v.get('auc', float('nan')):.3f}" for k, v in anc.items()
            if isinstance(v, dict) and "auc" in v))
    return res


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def markdown_table(res: AblationResult | dict) -> str:
    d = res.to_json() if isinstance(res, AblationResult) else res
    pools = d["pools"]
    ref = d["deployed_layer"]
    head = ("| layer | " + " | ".join(pools) + " | ext mean | Δ ext mean vs "
            f"L{ref} |")
    sep = "|---:|" + "---:|" * (len(pools) + 2)
    lines = [head, sep]
    ref_row = d["rows"][str(ref)]
    for L in d["layers"]:
        r = d["rows"][str(L)]
        cells = []
        for p in pools:
            a = r[p]["auc"]
            lo, hi = r[p]["delta_vs_ref"][1], r[p]["delta_vs_ref"][2]
            star = "" if L == ref or (lo <= 0 <= hi) else "*"
            cells.append(f"{a:.3f}{star}")
        dm = (r["ext_mean_auc"] - ref_row["ext_mean_auc"]
              if r["ext_mean_auc"] is not None else float("nan"))
        tag = " (deployed)" if L == ref else ""
        lines.append(f"| L{L}{tag} | " + " | ".join(cells)
                     + f" | {r['ext_mean_auc']:.3f} | {dm:+.3f} |")
    lines.append("")
    lines.append("`*` = paired-bootstrap 95% interval of AUC(layer) − "
                 f"AUC(L{ref}) excludes zero on that pool.")
    return "\n".join(lines)


def save_json(res: AblationResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res.to_json(), fh, indent=1)
