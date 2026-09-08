"""Internal-gain analysis ($0): where can the internal number actually
move, using the existing r0/r1/r2 arms only.

A. Threshold capture curve: sweep the aggressive threshold downward,
   per run; realized rate vs delivered acc (branched never/always
   remix), vs the matched-rate random mixture and the oracle-at-rate
   upper bound. Answers "can a threshold move buy the fixable misses
   without paying more than random for them?"
B. Fixable-miss score placement: where do the (local wrong, unfired,
   always right) rows sit in the score distribution?
C. >30s hard-knowledge truncation audit: on always-arm fires, how much
   audio was consumed at onset (onset_chunk+1 vs n_chunks)? If
   consumed ~= full, the v2 causal-prefix fix hands the expert the
   whole question these rows lost under v1's tail-30s slice.
D. Chat relay-loss mechanism: expert_answer vs relay_text length on
   the always-arm relay-loss rows (clean_expert max_chars=400 cap).

Usage: set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\54_internal_gain_analysis.py
Output: data/paper_revision_pr0907/internal_gain_analysis.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
OUT = D / "paper_revision_pr0907"
RUNS = ["r0", "r1", "r2"]

qs = {q["id"]: q for q in (json.loads(l) for l in open(
    D / "queries.jsonl", encoding="utf-8") if l.strip())
    if q.get("split") == "test"}


def load(run, arm):
    fn = f"frozen_{arm}_judged.parquet" if arm == "never" \
        else f"frozen_{arm}_tts_judged.parquet"
    base = D / "native_bench" if run == "r0" \
        else D / "native_bench_repeats" / run
    df = pd.read_parquet(base / fn).drop_duplicates("id", keep="last")
    df = df.set_index("id")
    df["cat"] = pd.Series({i: q["pool"] for i, q in qs.items()})
    return df


out = {"capture_curve": {}, "fixable_scores": {}, "truncation": {},
       "relay_loss": {}}

# A + B: per run, branch on the never-arm onset score
for run in RUNS:
    nv, al = load(run, "never"), load(run, "always")
    ids = nv.index.intersection(al.index)
    nv, al = nv.loc[ids], al.loc[ids]
    sc = nv["score"].astype(float).fillna(-1.0)   # non_commit rows never fire
    info = nv["is_info"].fillna(True).astype(bool)
    a = nv["adequate"].astype(float)
    b = al["adequate"].astype(float)
    fixable = (a == 0) & (b == 1)
    harmful = (a == 1) & (b == 0)
    rows = []
    for rate in np.arange(.30, .96, .05):
        thr = float(np.quantile(sc[info & (sc >= 0)],
                                1 - min(rate / info.mean(), 1)))
        fire = (sc >= thr) & info
        acc = float(np.where(fire, b, a).mean())
        rnd = float(fire.mean() * b.mean() + (1 - fire.mean()) * a.mean())
        # oracle at the same number of fires: fire exactly on the rows
        # where always beats never, best-first
        k = int(fire.sum())
        gain = (b - a)
        orc_idx = gain.sort_values(ascending=False).index[:k]
        orc = float(np.where(pd.Series(ids).isin(orc_idx).to_numpy(),
                             b.to_numpy(), a.to_numpy()).mean())
        rows.append({"target_rate": round(float(rate), 2),
                     "realized": float(fire.mean()),
                     "acc": acc, "acc_random_matched": rnd,
                     "acc_oracle_at_rate": orc,
                     "fixable_captured": int((fire & fixable).sum()),
                     "fixable_total": int(fixable.sum()),
                     "harmful_fired": int((fire & harmful).sum())})
    out["capture_curve"][run] = rows
    pr = (sc.rank(pct=True))
    out["fixable_scores"][run] = {
        "fixable_n": int(fixable.sum()),
        "fixable_score_pctile_mean": float(pr[fixable].mean()),
        "fixable_score_pctile_list": [round(float(x), 3)
                                      for x in sorted(pr[fixable])],
        "blocked_by_act_gate": int((fixable & ~info).sum()),
    }

# C: consumed-at-fire for >30s hard-knowledge (always arm = every row
# fires, so onset_chunk is the commit point for all of them)
for run in RUNS:
    al = load(run, "always")
    m = (al["audio_s"] > 30) & (al["cat"] == "hard-knowledge")
    sub = al[m]
    consumed = (sub["onset_chunk"] + 1).clip(upper=sub["n_chunks"])
    frac = consumed / sub["n_chunks"]
    out["truncation"][run] = {
        "n_gt30s_knowledge": int(m.sum()),
        "consumed_frac_mean": float(frac.mean()),
        "consumed_frac_p10": float(frac.quantile(.10)),
        "full_question_heard_v2": int((frac >= 0.999).sum()),
        "v1_tail30_lost_start": int((sub["audio_s"] > 30).sum()),
        "always_acc_on_these": float(sub["adequate"].astype(float).mean()),
    }

# D: relay loss mechanism (always arm, expert right / delivered wrong)
al = load("r0", "always")
if "adequate_expert" in al.columns:
    f = al["fired"].astype(bool)
    e = al["adequate_expert"].astype(float)
    d = al["adequate"].astype(float)
    loss = f & (e == 1) & (d == 0)
    sub = al[loss]
    out["relay_loss"]["r0_always"] = {
        "n": int(loss.sum()),
        "by_cat": sub["cat"].value_counts().to_dict(),
        "expert_len_mean": float(sub["expert_answer"].str.len().mean()),
        "relay_len_mean": float(sub["relay"].str.len().mean()),
        "relay_capped_at_400": int((sub["relay"].str.len()
                                    .between(350, 460)).sum()),
        "note": "clean_expert max_chars=400 sentence-capped relay",
    }

OUT.mkdir(exist_ok=True)
(OUT / "internal_gain_analysis.json").write_text(json.dumps(out, indent=1))

print("=== capture curve (r0/r1/r2 mean) ===")
print("rate  acc    random  oracle  fix_cap/tot  harmful")
for i in range(len(out["capture_curve"]["r0"])):
    rs = [out["capture_curve"][r][i] for r in RUNS]
    print(f"{rs[0]['target_rate']:.2f} "
          f"{np.mean([x['acc'] for x in rs]):.3f}  "
          f"{np.mean([x['acc_random_matched'] for x in rs]):.3f}   "
          f"{np.mean([x['acc_oracle_at_rate'] for x in rs]):.3f}   "
          f"{np.mean([x['fixable_captured'] for x in rs]):4.1f}/"
          f"{np.mean([x['fixable_total'] for x in rs]):.0f}    "
          f"{np.mean([x['harmful_fired'] for x in rs]):4.1f}")
print("\n=== fixable score percentiles ===")
for r in RUNS:
    fs = out["fixable_scores"][r]
    print(f"{r}: n={fs['fixable_n']} mean pctile "
          f"{fs['fixable_score_pctile_mean']:.2f} "
          f"act-gate-blocked={fs['blocked_by_act_gate']}")
print("\n=== >30s hard-knowledge consumed-at-fire ===")
for r in RUNS:
    t = out["truncation"][r]
    print(f"{r}: n={t['n_gt30s_knowledge']} consumed_frac "
          f"{t['consumed_frac_mean']:.3f} (p10 {t['consumed_frac_p10']:.3f}) "
          f"full={t['full_question_heard_v2']} "
          f"always_acc={t['always_acc_on_these']:.3f}")
print("\n=== relay loss ===")
print(json.dumps(out["relay_loss"], indent=1))
