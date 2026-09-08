"""Benchmark-overlap (data-leak) audit (8bz, Jisen 09-04): does any
external-pool question appear in the calibration merge the deployed
probe was trained on?

Stage 1 scans every calibration query file against the five external
pools: normalized exact match + token-Jaccard >= 0.8 near-duplicate
screen. Stage 2 refits the deployed 5,228-row recipe (script 31
parts, C=3e-4) with the colliding rows removed and compares external
AUC (official-config dumps, native labels) and never/always remix
accuracy at 15/25/40% per-pool quantile tiers.

Result 2026-09-04: 8 collisions (Speech TriviaQA 4/250 exact, Llama
Questions 2/250 exact + 2/250 near, others 0); 7 of the matched
calibration rows are in the deployed merge. Removing them moves
striviaqa AUC .8113 -> .8109, sllama .7427 -> .7411, and every remix
accuracy by < 0.5 point. No transfer result rests on the overlap.

Usage (from interactive_paper/):
  set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\41_leak_audit.py
Outputs: data/leak_check.json + printed comparison.
"""
import json
import re
import types
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = Path("data")
m = types.ModuleType("sw")
exec(open("scripts/40_threshold_sweep.py").read().split(
    "if __name__")[0], m.__dict__)

CALIB_FILES = ["queries_public.jsonl", "queries_gen.jsonl",
               "queries_expansion.jsonl", "queries_expansion2.jsonl",
               "queries_expansion3.jsonl", "queries_expansion3zh.jsonl",
               "queries_expansion4zh.jsonl", "queries_expansion5rs.jsonl",
               "queries_fresh.jsonl"]
EXT_FILES = {"striviaqa": "queries_striviaqa.jsonl",
             "swebq": "queries_swebq.jsonl",
             "sllama": "queries_sllama.jsonl",
             "sdqa": "queries_sdqa.jsonl",
             "sreason": "queries_sreason.jsonl"}
DEPLOYED_PARTS = ["caliboff", "expoff", "exp2off", "exp3off",
                  "exp3zhoff"]


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def toks(s):
    return set(norm(s).split())


def load_jsonl(p):
    out = []
    for l in open(p, encoding="utf-8"):
        if l.strip():
            q = json.loads(l)
            out.append((q.get("id"),
                        q.get("text") or q.get("query")
                        or q.get("question", "")))
    return out


def scan():
    calib = []
    for f in CALIB_FILES:
        if (D / f).exists():
            calib += [(f, i, t) for i, t in load_jsonl(D / f)]
    calib_norm = {}
    for f, i, t in calib:
        n = norm(t)
        if n:
            calib_norm.setdefault(n, []).append((f, i))
    calib_tok = [(f, i, t, toks(t)) for f, i, t in calib if t]

    report, leak_ids = {}, set()
    for name, qf in EXT_FILES.items():
        exact, near = [], []
        for i, t in load_jsonl(D / qf):
            n = norm(t)
            if n in calib_norm:
                exact.append([i, t, calib_norm[n]])
                leak_ids.update(ci for _, ci in calib_norm[n] if ci)
                continue
            ts = toks(t)
            if not ts:
                continue
            best, bj = None, 0.0
            for cf, ci, ct, cts in calib_tok:
                inter = len(ts & cts)
                if inter:
                    j = inter / len(ts | cts)
                    if j > bj:
                        bj, best = j, (cf, ci, ct)
            if bj >= 0.8:
                near.append([i, t, list(best), round(bj, 2)])
                if best[1]:
                    leak_ids.add(best[1])
        report[name] = {"exact": exact, "near": near}
        print(f"{name}: exact={len(exact)}  near(J>=0.8)={len(near)}")
    with open(D / "leak_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return leak_ids


def deployed_merge():
    ids, Xs, ys = [], [], []
    for tag in DEPLOYED_PARTS:
        i, X = m.load_feats(tag)
        y = pd.read_parquet(
            D / f"frozen_native_{tag}_judged.parquet").set_index(
            "id")["escalate_label"].reindex(i).to_numpy(float)
        k = ~np.isnan(y)
        ids += [q for q, kk in zip(i, k) if kk]
        Xs.append(X[k])
        ys.append(y[k].astype(int))
    fl = pd.read_parquet(D / "fresh_labels.parquet")
    fl = fl[fl["escalate_label"].notna()]
    lab_f = dict(zip(fl["id"], fl["escalate_label"].astype(int)))
    nf = pd.read_parquet(
        D / "frozen_native_freshoff_judged.parquet").set_index(
        "id")["escalate_label"]
    for i, p in zip(fl["id"], fl["pool"]):
        if p != "fresh_fast" and i in nf.index and pd.notna(nf[i]):
            lab_f[i] = int(nf[i])
    split_f = dict(zip(fl["id"], fl["split"]))
    ids_fr, X_fr = m.load_feats("freshoff")
    tr_j = [j for j, i in enumerate(ids_fr)
            if i in lab_f and split_f.get(i) == "train"]
    ids += [ids_fr[j] for j in tr_j]
    X = np.concatenate(Xs + [X_fr[tr_j]])
    y = np.concatenate(ys + [[lab_f[ids_fr[j]] for j in tr_j]])
    return ids, X, y


def eval_probe(w, b, name):
    for pool, tag in (("striviaqa", "striviaqaoff"),
                      ("sllama", "sllamaoff")):
        pids, PX = m.load_feats(tag)
        s = 1 / (1 + np.exp(-(PX @ w + b)))
        smap = dict(zip(pids, s))
        lab = pd.read_parquet(
            D / f"frozen_native_{tag}_judged.parquet").dropna(
            subset=["escalate_label"]).drop_duplicates("id", keep="last")
        yl = lab.set_index("id")["escalate_label"]
        kk = [i for i in yl.index if i in smap]
        auc = roc_auc_score(yl.loc[kk].astype(int),
                            [smap[i] for i in kk])
        never = m.load_arm(pool, "never")
        always = m.load_arm(pool, "always")
        common = never.index.intersection(always.index)
        never, always = never.loc[common], always.loc[common]
        sc = pd.Series({i: smap.get(i, np.nan) for i in common})
        accs = []
        for r in (.15, .25, .40):
            t = float(np.nanquantile(list(smap.values()), 1 - r))
            fire = ((sc >= t)
                    & never["is_info"].fillna(True).astype(bool))
            accs.append(float(
                np.where(fire, always["y"], never["y"]).mean()))
        print(f"  [{name}] {pool}: AUC={auc:.4f}  remix acc"
              " @15/25/40% = "
              + " ".join(f"{a:.3f}" for a in accs))


def main():
    leak_ids = scan()
    ids, X, y = deployed_merge()
    mask = np.array([i in leak_ids for i in ids])
    print(f"deployed merge {len(y)} rows, colliding rows: {mask.sum()}")
    dep = json.loads((D / "gate_native.json").read_text())
    eval_probe(np.array(dep["w"]), dep["b"], "deployed   ")
    clf = LogisticRegression(C=3e-4, max_iter=5000).fit(
        X[~mask], y[~mask])
    eval_probe(clf.coef_[0], clf.intercept_[0], "no-leak fit")


if __name__ == "__main__":
    main()
