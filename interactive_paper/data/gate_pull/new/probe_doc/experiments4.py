"""Round-4: the CAUSAL re-capture (pass 3, nvda_replay_v2.py, 2026-09-01 night).

New arrays per layer: H_pre (8 frames before the commit frame) and H_run (mean
over all frames before commit = the online user_mean).  Labels: pass-2
(answer text drifts between passes, label flips ~1.3 %; drift is measured
and reported here).

Variants (all L2-logistic + StandardScaler, C=1e-4, same as deployed):
  A0'  deployed read on pass-3 arrays      onset_last | onset_mean8 | user_mean    (post-commit, non-causal user_mean)
  F1   deployed read, causal user_mean     onset_last | onset_mean8 | run_mean     (fixes the train/serve skew)
  S1   strictly pre-commit                 pre_last | pre_mean8 | run_mean          (nothing after the commit frame)
  S2   at-commit                           commit_frame | pre_mean8 | run_mean      (commit frame = first answer frame)
  S3   at-commit + pre window              commit_frame | pre_last | pre_mean8 | run_mean
  X1   everything                          onset 3-block + pre_last + pre_mean8 + run_mean
  + layer sweeps for S1/S2 over L22..L38, and the v2 layer-avg architecture on F1/S2 blocks.

Selection metric (pre-declared, as before): LOPO within-held-pool mean, then
pooled LOPO; externals reported, never used to select.

    uv run --with numpy --with pandas --with pyarrow --with scikit-learn python experiments4.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import lr  # noqa: E402
from experiments2 import eligible_pools, macro_auc  # noqa: E402
from experiments3 import LayerAvg, paired_boot  # noqa: E402
from probe_lab import CALIB_TAGS, EXT_POOLS, GP  # noqa: E402

HERE = Path(__file__).resolve().parent
S3 = GP / "onset3"
LAYERS = [22, 26, 30, 34, 38]
J = {L: i for i, L in enumerate(LAYERS)}


class Pass3:
    """Pass-3 arrays for a set of tags, committed rows only, pass-2 labels."""

    def __init__(self, tags, labelfile):
        ids, A, ONS = [], {k: [] for k in ("H_eot", "H_mean", "H_onset", "H_pre", "H_run")}, []
        self.tag = []
        for tag in tags:
            z = np.load(S3 / f"nvda_h3_{tag}.L22-38.npz", allow_pickle=True)
            assert [int(x) for x in z["layers"]] == LAYERS
            ids += [str(x) for x in z["ids"]]; self.tag += [tag] * len(z["ids"])
            for k in A:
                A[k].append(z[k])
            ONS.append(z["onset_frame"])
        A = {k: np.concatenate(v) for k, v in A.items()}; ONS = np.concatenate(ONS); self.tag = np.array(self.tag)
        lab = {}
        for tag in tags:
            df = pd.read_parquet(labelfile(tag))
            for _, r in df.iterrows():
                if pd.notna(r["escalate_label"]):
                    lab[str(r["id"])] = int(r["escalate_label"])
        keep = np.array([i in lab and ONS[n] >= 0 for n, i in enumerate(ids)])
        self.ids = [i for i, k in zip(ids, keep) if k]; self.tag = self.tag[keep]
        self.A = {k: v[keep] for k, v in A.items()}; self.ons = ONS[keep]
        self.y = np.array([lab[i] for i in self.ids])
        q = {}
        for f in ("queries.jsonl", "queries_expansion.jsonl", "queries_expansion2.jsonl"):
            for line in (GP / f).open(encoding="utf-8"):
                r = json.loads(line); q[str(r["id"])] = r["pool"]
        self.pool = np.array([q.get(i, t) for i, t in zip(self.ids, self.tag)])

    def feats(self, L, blocks):
        j = J[L]; A = self.A; parts = []
        for b in blocks:
            if b == "onset_last": parts.append(A["H_onset"][:, j, -1])
            elif b == "onset_mean8": parts.append(A["H_onset"][:, j].mean(1))
            elif b == "commit": parts.append(A["H_onset"][:, j, 0])
            elif b == "user_mean": parts.append(A["H_mean"][:, j])
            elif b == "run_mean": parts.append(A["H_run"][:, j])
            elif b == "pre_last": parts.append(A["H_pre"][:, j, -1])
            elif b == "pre_mean8": parts.append(A["H_pre"][:, j].mean(1))
            elif b == "eot_last": parts.append(A["H_eot"][:, j, -1])
            elif b == "eot_mean8": parts.append(A["H_eot"][:, j].mean(1))
            else: raise KeyError(b)
        return np.concatenate(parts, 1).astype(np.float32)


def run(name, make_clf, fc, cal, ext, seeds=(42,)):
    t = time.time(); X = fc(cal); y = cal.y; pools = cal.pool; elig = eligible_pools(y, pools)
    oofs = []; oof = None
    for s in seeds:
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            p[te] = make_clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oofs.append(float(roc_auc_score(y, p))); oof = p if oof is None else oof
    lopo = np.zeros(len(y))
    for g in np.unique(pools):
        te = pools == g; lopo[te] = make_clf().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    lm, lw, lper = macro_auc(lopo, y, pools, elig); om, _, _ = macro_auc(oof, y, pools, elig)
    full = make_clf().fit(X, y)
    scores = {tag: full.predict_proba(fc(e))[:, 1] for tag, e in ext.items()}
    er = {tag: float(roc_auc_score(ext[tag].y, scores[tag])) for tag in EXT_POOLS}; er["mean"] = float(np.mean(list(er.values())))
    rep = dict(name=name, dims=int(X.shape[1]), oof=float(np.mean(oofs)), oof_sd=float(np.std(oofs)), oof_macro=om, lopo=float(roc_auc_score(y, lopo)),
               lopo_macro=lm, lopo_worst=lw, lopo_per=lper, ext=er, secs=round(time.time() - t, 1))
    print(f"{name:52s} OOF {rep['oof']:.4f} oofM {om:.4f} | LOPO {rep['lopo']:.4f} lopoM {lm:.4f} worst {lw:.3f} | "
          f"ext {er['striviaqa']:.3f}/{er['swebq']:.3f}/{er['sllama']:.3f}/{er['sdqa']:.3f}={er['mean']:.4f} ({rep['secs']}s)", flush=True)
    return rep, scores


class ExtWrap:
    """paired_boot expects ext.pools[tag]['y']."""

    def __init__(self, ext):
        self.pools = {t: {"y": e.y} for t, e in ext.items()}


if __name__ == "__main__":
    t0 = time.time()
    cal = Pass3(CALIB_TAGS, lambda tag: GP / "onset_fit" / f"nvda_{tag}.parquet")
    ext = {tag: Pass3([tag], lambda t: GP / "onset" / f"nvda_{t}_ext2.parquet") for tag in EXT_POOLS}
    print(f"pass-3 calib n={len(cal.y)} fail={cal.y.mean():.3f}; ext " + ", ".join(f"{t} n={len(e.y)}" for t, e in ext.items()) + f" ({time.time()-t0:.0f}s)")

    # --- answer-text drift pass2 -> pass3 (labels are pass-2) ---------------------------------
    p2 = {}; p3 = {}
    for f in glob.glob(str(GP / "onset" / "nvda_answers_*.shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line); p2[str(r["id"])] = (r["answer"], r["onset_frame"])
    for f in glob.glob(str(S3 / "nvda_answers_*.shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line); p3[str(r["id"])] = (r["answer"], r["onset_frame"])
    common = [i for i in p3 if i in p2]
    txt = np.mean([p2[i][0] != p3[i][0] for i in common]); ons = np.mean([p2[i][1] != p3[i][1] for i in common])
    ons_d = np.array([p3[i][1] - p2[i][1] for i in common if p2[i][1] >= 0 and p3[i][1] >= 0])
    drift = dict(n=len(common), answer_text_changed=float(txt), commit_frame_changed=float(ons),
                 commit_shift_frames_median=float(np.median(np.abs(ons_d))), commit_shift_gt2_frames=float((np.abs(ons_d) > 2).mean()))
    print("pass2->pass3 drift:", drift)

    # --- how different is the causal user_mean from the offline one? ----------------------------
    j = J[30]; a = cal.A["H_mean"][:, j].astype(np.float32); b = cal.A["H_run"][:, j].astype(np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)
    print(f"cos(user_mean, run_mean) @L30: median {np.median(cos):.4f}, p10 {np.percentile(cos, 10):.4f}, min {cos.min():.4f}")

    V = {
        "A0' deployed read (pass-3): onset_last|onset_mean8|user_mean @L30": (30, ["onset_last", "onset_mean8", "user_mean"]),
        "F1 deployed read, causal user_mean: onset_last|onset_mean8|run_mean @L30": (30, ["onset_last", "onset_mean8", "run_mean"]),
        "F1 @L26": (26, ["onset_last", "onset_mean8", "run_mean"]),
        "S1 strictly pre-commit: pre_last|pre_mean8|run_mean @L30": (30, ["pre_last", "pre_mean8", "run_mean"]),
        "S2 at-commit: commit|pre_mean8|run_mean @L30": (30, ["commit", "pre_mean8", "run_mean"]),
        "S3 at-commit + pre window: commit|pre_last|pre_mean8|run_mean @L30": (30, ["commit", "pre_last", "pre_mean8", "run_mean"]),
        "X1 everything @L30": (30, ["onset_last", "onset_mean8", "commit", "pre_last", "pre_mean8", "run_mean"]),
        "E0 old eot read (pass-3): eot_last|eot_mean8|user_mean @L34": (34, ["eot_last", "eot_mean8", "user_mean"]),
        "S1 vs eot: pre_last|pre_mean8|user_mean @L34": (34, ["pre_last", "pre_mean8", "user_mean"]),
    }
    for L in (22, 26, 34, 38):
        V[f"S1 @L{L}"] = (L, ["pre_last", "pre_mean8", "run_mean"])
        V[f"S2 @L{L}"] = (L, ["commit", "pre_mean8", "run_mean"])
    out = []; S = {}
    for n, (L, blocks) in V.items():
        rep, sc = run(n, lambda: lr(1e-4), lambda d, L=L, b=blocks: d.feats(L, b), cal, ext, seeds=(42,) if "@L" in n and n[:2] in ("S1", "S2") and "run_mean" in blocks and L != 30 else (0, 1, 42))
        out.append(rep); S[n] = sc
        json.dump(dict(drift=drift, results=out), open(HERE / "results_round4.json", "w"), indent=1)

    # layer-avg (v2 architecture) on the causal blocks
    for n, blocks in (("V2c layer-avg ×3 (26,30,34): commit|onset_last|onset_mean8|run_mean", ["commit", "onset_last", "onset_mean8", "run_mean"]),
                      ("V2s layer-avg ×3 (26,30,34): commit|pre_last|pre_mean8|run_mean [causal]", ["commit", "pre_last", "pre_mean8", "run_mean"])):
        fc = lambda d, b=blocks: np.concatenate([d.feats(L, b) for L in (26, 30, 34)], 1)
        rep, sc = run(n, lambda: LayerAvg(lambda: lr(1e-4), 3), fc, cal, ext, seeds=(0, 1, 42)); out.append(rep); S[n] = sc
        json.dump(dict(drift=drift, results=out), open(HERE / "results_round4.json", "w"), indent=1)

    print("\n== paired bootstrap of external-mean Δ vs A0' (pass-3 deployed read) ==")
    ew = ExtWrap(ext); rng = np.random.default_rng(0); base = S[list(V)[0]]
    for r in out:
        if r["name"] not in S or r["name"].startswith("A0'"):
            continue
        d = paired_boot(base, S[r["name"]], ew, rng)
        r["delta_ext"] = dict(mean=float(d.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))
        print(f"  {r['name']:70s} Δext {d.mean():+.4f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
    json.dump(dict(drift=drift, results=out), open(HERE / "results_round4.json", "w"), indent=1)
    print("\n== ranked by lopoM, then LOPO ==")
    for r in sorted(out, key=lambda r: (-r["lopo_macro"], -r["lopo"])):
        print(f"  lopoM {r['lopo_macro']:.4f} LOPO {r['lopo']:.4f} OOF {r['oof']:.4f} ext {r['ext']['mean']:.4f}  {r['name']}")
