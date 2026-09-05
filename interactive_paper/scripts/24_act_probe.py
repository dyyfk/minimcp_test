"""Floor-act probe (8bh): info-seeking vs floor-management, same L22
read point as the failure probe.

Positives: the 2310 native question features (calib+exp+exp2).
Negatives: flooract stim features (stop commands, backchannels, acks,
fillers; TTS'd, dumped standalone).

Also quantifies the bug being fixed: failure-probe (gate_native)
scores on the floor stims -> false-fire rate per tier BEFORE the act
gate, and the residual rate AFTER (act misclassified AND above thr).

Act threshold: the 0.5-percentile of question OOF scores — costs at
most ~0.5% of real escalations, kills the rest of the false fires.

Usage: .venv_boot\\Scripts\\python.exe scripts\\24_act_probe.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

D = Path("data")


def load_feats(tag):
    ids, X = [], []
    for p in sorted(D.glob(f"frozen_native_{tag}_feats.shard*.npz")):
        z = np.load(p, allow_pickle=True)
        ids += list(z["ids"])
        X.append(z["X"])
    if not X:
        raise FileNotFoundError(f"no native feats for tag {tag}")
    X = np.concatenate(X)
    df = pd.DataFrame({"id": ids}).assign(row=range(len(ids)))
    df = df.drop_duplicates("id", keep="last")
    return list(df["id"]), X[df["row"].to_numpy()]


def main():
    parts = [load_feats(t)[1] for t in ("calib", "exp", "exp2")]
    # 8bj: reqq/reqqx request-phrased positives (standalone + in-
    # context 2nd turn). 8ch: stopq/stopqpc stop-prefixed question
    # compounds + reqqpc plain questions in the POST-BARGE-CUT context
    # — the live gate ruled "Stop, stop. What is the stock price of
    # Apple today?" a floor turn (P(info)=.336) and never escalated.
    for t in ("reqq", "reqqx", "stopq", "stopqpc", "reqqpc"):
        try:
            Xr = load_feats(t)[1]
            parts.append(Xr)
            print(f"{t} positives: {len(Xr)}")
        except FileNotFoundError:
            print(f"{t} positives: none yet")
    Xq = np.concatenate(parts)
    neg_parts, neg_cats = [], []
    qs = {json.loads(l)["id"]: json.loads(l)
          for l in open(D / "queries_flooract.jsonl", encoding="utf-8")
          if l.strip()}
    # 8ch: floorpc (stop commands dumped post-cut) is NOT in the fit:
    # 71/72 never commit after a cut (no onset, no gate read), and the
    # single one that does scores inseparably with the postcut question
    # positives — an n=1 unfixable singleton that dragged the joint-gap
    # threshold from ~.51 to .67, taxing live questions to guard a
    # 1-in-72-commits exposure. Reported below instead.
    for t, sfx in (("flooract", ""), ("flooractx", "+ctx")):
        try:
            ids_t, Xt = load_feats(t)
            neg_parts.append(Xt)
            neg_cats += [qs[i]["pool"].split("-")[1] + sfx
                         for i in ids_t]
            print(f"{t} negatives: {len(Xt)}")
        except FileNotFoundError:
            print(f"{t} negatives: none yet")
    Xf = np.concatenate(neg_parts)
    fa_ids = None
    cats = np.array(neg_cats)
    print(f"positives (questions): {len(Xq)}  negatives (floor): "
          f"{len(Xf)}  cats: "
          + ", ".join(f"{c}:{(cats == c).sum()}"
                      for c in sorted(set(cats))))

    # ---- the bug, quantified: failure-probe fire on floor stims -----
    nat = json.loads((D / "gate_native.json").read_text())
    wn, bn = np.array(nat["w"]), nat["b"]
    s_fail = 1 / (1 + np.exp(-(Xf @ wn + bn)))
    print("\nfailure-probe scores on floor stims (the bug):")
    for tier, thr in nat["eot_thresholds"].items():
        fire = (s_fail >= thr)
        per = {c: round(float(fire[cats == c].mean()), 2)
               for c in sorted(set(cats))}
        print(f"  {tier:<13} thr={thr:.3f}  false-fire "
              f"{fire.mean():.2f}  by cat {per}")

    # ---- act probe ---------------------------------------------------
    X = np.concatenate([Xq, Xf])
    y = np.concatenate([np.ones(len(Xq)), np.zeros(len(Xf))])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (1e-4, 3e-4, 1e-3):
        oof = cross_val_predict(
            LogisticRegression(C=C, max_iter=5000,
                               class_weight="balanced"),
            X, y, cv=cv, method="predict_proba")[:, 1]
        a = roc_auc_score(y, oof)
        print(f"  C={C}: OOF AUC={a:.4f}")
        if best is None or a > best[1]:
            best = (C, a, oof)
    C, auc, oof = best
    clf = LogisticRegression(C=C, max_iter=5000,
                             class_weight="balanced").fit(X, y)

    q_oof = oof[:len(Xq)]
    f_oof_all = oof[len(Xq):]
    # 8bj: NOT the q0.5-percentile (that broke on live mic speech —
    # both distributions shift into the calibration gap once real
    # context enters the tail). With in-context negatives in the mix,
    # take the center of the JOINT gap, clipped to [0.3, 0.7].
    lo = float(np.percentile(f_oof_all, 99.5))
    hi = float(np.percentile(q_oof, 0.5))
    act_thr = float(np.clip((lo + hi) / 2, 0.3, 0.7))
    print(f"joint gap: neg p99.5={lo:.4f}  pos p0.5={hi:.4f}  "
          f"-> act_thr={act_thr:.4f}")
    lost_q = float((q_oof < act_thr).mean())
    f_oof = oof[len(Xq):]
    passed_floor = f_oof >= act_thr
    print(f"\nact thr={act_thr:.4f} (q0.5% of question OOF): loses "
          f"{lost_q:.3%} questions; floor stims passing act gate: "
          f"{passed_floor.mean():.2%}")
    residual = passed_floor & (s_fail >= nat["eot_thresholds"]["balanced"])
    print(f"residual false-fire at balanced AFTER act gate: "
          f"{residual.mean():.2%} (was "
          f"{(s_fail >= nat['eot_thresholds']['balanced']).mean():.2%})")
    for c in sorted(set(cats)):
        m = cats == c
        print(f"  {c:<8} act-pass {passed_floor[m].mean():.2%}  "
              f"residual-fire {residual[m].mean():.2%}")

    # 8ch report-only: the excluded post-cut stop singleton
    try:
        _, Xpc = load_feats("floorpc")
        s_pc = clf.predict_proba(Xpc)[:, 1]
        print(f"floorpc (excluded, n={len(Xpc)}): act scores "
              f"{np.round(s_pc, 3).tolist()} vs thr {act_thr:.3f} "
              f"(71/72 post-cut stops never commit at all)")
    except FileNotFoundError:
        pass

    art = {"w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
           "layer": 22, "act_threshold": act_thr, "C": C,
           "oof_auc": round(float(auc), 4),
           "n_pos": int(len(Xq)), "n_neg": int(len(Xf)),
           "recipe": "scripts/24 floor-act probe (8bh): escalate only "
                     "if act>=thr (info-seeking) AND P(fail)>=tier thr"}
    (D / "gate_act.json").write_text(json.dumps(art))
    out = {"false_fire_before": {t: round(float((s_fail >= thr).mean()), 3)
                                 for t, thr in
                                 nat["eot_thresholds"].items()},
           "act_oof_auc": round(float(auc), 4),
           "act_thr": act_thr, "questions_lost": round(lost_q, 5),
           "floor_pass_rate": round(float(passed_floor.mean()), 4),
           "residual_fire_balanced": round(float(residual.mean()), 4)}
    Path("figures/act_probe.json").write_text(json.dumps(out, indent=1))
    print("\nwrote data/gate_act.json + figures/act_probe.json")


if __name__ == "__main__":
    main()
