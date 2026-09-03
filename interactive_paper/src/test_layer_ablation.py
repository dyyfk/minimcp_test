"""CPU unit tests for src/layer_ablation.py on synthetic shards.

Run:  python -m pytest interactive_paper/src/test_layer_ablation.py -q
  or: python interactive_paper/src/test_layer_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_ablation as la  # noqa: E402


def _make_shards(path, n, layers, d=16, k=la.K_EOT, seed=0, signal_layer=None,
                 y=None):
    """Synthetic capture: noise everywhere, plus a label-correlated
    direction injected into ONE layer so that layer should win."""
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, len(layers), k, d)).astype(np.float16)
    M = rng.normal(size=(n, len(layers), d)).astype(np.float16)
    elen = rng.integers(3, k + 1, size=n).astype(np.int16)
    if signal_layer is not None and y is not None:
        j = list(layers).index(signal_layer)
        bump = (2.0 * (2 * y - 1))[:, None].astype(np.float32)
        E[:, j, -1, :4] = (E[:, j, -1, :4].astype(np.float32) + bump
                           ).astype(np.float16)
        M[:, j, :4] = (M[:, j, :4].astype(np.float32) + bump
                       ).astype(np.float16)
    ids = np.array([f"q{seed}_{i:04d}" for i in range(n)])
    np.savez_compressed(path, ids=ids, H_eot=E, H_mean=M, eot_len=elen,
                        layers=np.array(layers))
    return ids


def test_feat_matches_reference_semantics():
    layers = [14, 22, 35]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.shard0.npz")
        _make_shards(p, 5, layers, d=8)
        sh = la.load_shards(os.path.join(td, "x.shard*.npz"))
        X = la.feat(sh, 22, ("eot_last", "eot_mean", "user_mean"))
        assert X.shape == (5, 24)
        j = 1
        # eot_last is the last tail row
        assert np.allclose(X[:, :8], sh.E[:, j, -1, :].astype(np.float32))
        # eot_mean averages only the valid (right-aligned) rows
        for i in range(5):
            ln = int(np.clip(sh.elen[i], 1, la.K_EOT))
            ref = sh.E[i, j, la.K_EOT - ln:, :].astype(np.float32).mean(0)
            assert np.allclose(X[i, 8:16], ref, atol=1e-5)
        assert np.allclose(X[:, 16:], sh.M[:, j].astype(np.float32))
        try:
            la.feat(sh, 9)
            raise AssertionError("missing layer must raise")
        except KeyError:
            pass


def test_load_dedups_and_concats():
    layers = [22, 35]
    with tempfile.TemporaryDirectory() as td:
        _make_shards(os.path.join(td, "t.shard0.npz"), 4, layers, seed=1)
        _make_shards(os.path.join(td, "t.shard1.npz"), 3, layers, seed=1)
        sh = la.load_shards(os.path.join(td, "t.shard*.npz"))
        # seed=1 ids repeat across shards: q1_0000..q1_0003 and q1_0000..q1_0002
        assert sh.n == 4 and len(set(sh.ids)) == 4
        assert list(sh.layers) == layers


def test_bootstrap_identical_scores_is_zero():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200)
    d, lo, hi = la.paired_bootstrap_delta(y, s, s, B=200)
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_budget_recall_monotone():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(scale=0.5, size=300)
    b = la.failure_recall_at_budgets(y, s)
    assert b["conservative"]["recall"] <= b["balanced"]["recall"] \
        <= b["aggressive"]["recall"]
    assert abs(b["balanced"]["rate"] - 0.30) < 0.01


def test_end_to_end_signal_layer_wins():
    layers = [14, 22, 35]
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as td:
        n_tr, n_te = 400, 200
        y_tr = rng.integers(0, 2, n_tr)
        y_te = rng.integers(0, 2, n_te)
        _make_shards(os.path.join(td, "train.shard0.npz"), n_tr, layers,
                     seed=10, signal_layer=22, y=y_tr)
        _make_shards(os.path.join(td, "ext.shard0.npz"), n_te, layers,
                     seed=11, signal_layer=22, y=y_te)
        tr = la.load_shards(os.path.join(td, "train.shard*.npz"))
        te = la.load_shards(os.path.join(td, "ext.shard*.npz"))
        pools = [la.Pool("ext", te, y_te, external=True),
                 la.Pool("internal", te.subset(range(50)), y_te[:50],
                         external=False)]
        # anchor: a probe fit on the same features, shipped-style dict
        X22 = la.feat(tr, 22)
        clf = la.fit_probe(X22, y_tr, 1e-2)
        anchor = {"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
                  "layer_set": [22], "modes": list(la.DEPLOYED_MODES)}
        res = la.run_ablation(tr, y_tr, pools, C=1e-2, anchor=anchor, B=100,
                              log=lambda *_: None)
        j = res.to_json()
        a22 = j["rows"]["22"]["ext"]["auc"]
        a35 = j["rows"]["35"]["ext"]["auc"]
        a14 = j["rows"]["14"]["ext"]["auc"]
        assert a22 > 0.9 and a35 < 0.65 and a14 < 0.65
        lo, hi = j["rows"]["35"]["ext"]["delta_vs_ref"][1:]
        assert hi < 0  # final layer reliably worse in the synthetic world
        assert abs(j["anchor"]["ext"]["auc"] - a22) < 1e-6
        assert j["rows"]["22"]["ext_mean_auc"] == a22  # internal excluded
        md = la.markdown_table(res)
        assert "L22 (deployed)" in md and "L35" in md
        out = os.path.join(td, "out", "r.json")
        la.save_json(res, out)
        assert json.load(open(out))["deployed_layer"] == 22


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
