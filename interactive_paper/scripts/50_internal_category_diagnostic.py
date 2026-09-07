"""Internal (frozen 240) per-category diagnostic from the EXISTING
8bu arms — no new runs. (revision follow-up, 2026-09-07)

A. Reproduce the per-category Local/Aggressive/escalation table.
B. Explain "no escalation but score moved": unfired rows are
   independent samples of the same stochastic decoder (top_k=20,
   modal_native_bench.py:204) judged independently -> paired flip
   rates between never and each tier's UNFIRED rows, per category,
   plus text-identity check.
C. Failure taxonomy per category using never/aggressive/always:
   fixable_miss (local wrong, unfired, always right), no_benefit_esc
   (fired, delivered wrong, always wrong too), relay_or_expert_loss
   (fired, delivered wrong, always right), plus expert ceiling
   (always acc).

Usage: set PYTHONUTF8=1 && .venv_boot\Scripts\python.exe scripts\50_internal_category_diagnostic.py
Output: data/paper_revision_pr0907/internal_category_diagnostic.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
NB = D / "native_bench"
OUT = D / "paper_revision_pr0907"

qs = {q["id"]: q for q in (json.loads(l) for l in open(
    D / "queries.jsonl", encoding="utf-8") if l.strip())
      if q.get("split") == "test"}
cat = {i: q["pool"] for i, q in qs.items()}
src = {i: q.get("source") for i, q in qs.items()}
CATS = ["easy-chat", "easy-fact", "hard-knowledge", "hard-math", "trap"]

def arm(name):
    f = f"frozen_{name}_judged.parquet" if name == "never" \
        else f"frozen_{name}_tts_judged.parquet"
    df = pd.read_parquet(NB / f).drop_duplicates("id", keep="last")
    df = df.set_index("id")
    df["cat"] = pd.Series(cat)
    return df

arms = {a: arm(a) for a in
        ["never", "conservative", "balanced", "aggressive", "always"]}
ids = arms["never"].index
for a, df in arms.items():
    assert set(df.index) == set(ids), a

out = {"decoder": "duplex.streaming_generate(top_k=20) - stochastic; "
                  "each arm is an independent sample per query",
       "table": {}, "unfired_flips": {}, "taxonomy": {},
       "gsm8k_focus": {}}

# A. per-category table
nv, ag, al = arms["never"], arms["aggressive"], arms["always"]
for c in CATS:
    m = nv["cat"] == c
    out["table"][c] = {
        "n": int(m.sum()),
        "local_acc": float(nv.loc[m, "adequate"].mean()),
        "aggressive_acc": float(ag.loc[m, "adequate"].mean()),
        "always_acc": float(al.loc[m, "adequate"].mean()),
        "aggressive_esc_rate": float(ag.loc[m, "fired"].mean()),
    }

# B. unfired-row paired flips vs never, per tier and category
for tier in ["conservative", "balanced", "aggressive"]:
    t = arms[tier]
    unf = ~t["fired"].astype(bool)
    rec = {}
    for c in CATS + ["ALL"]:
        m = unf & ((t["cat"] == c) if c != "ALL" else True)
        a = nv.loc[m[m].index, "adequate"].astype(float)
        b = t.loc[m[m].index, "adequate"].astype(float)
        same_txt = (nv.loc[m[m].index, "answer"].fillna("")
                    == t.loc[m[m].index, "answer"].fillna(""))
        rec[c] = {"n_unfired": int(m.sum()),
                  "never_acc": round(float(a.mean()), 4) if m.sum() else None,
                  "tier_acc": round(float(b.mean()), 4) if m.sum() else None,
                  "flip_wrong_to_right": int(((a == 0) & (b == 1)).sum()),
                  "flip_right_to_wrong": int(((a == 1) & (b == 0)).sum()),
                  "identical_answer_text": int(same_txt.sum())}
    out["unfired_flips"][tier] = rec

# GSM8K focus: hard-math unfired under aggressive
m = (ag["cat"] == "hard-math") & (~ag["fired"].astype(bool))
gs = m[m].index
a = nv.loc[gs, "adequate"].astype(float)
b = ag.loc[gs, "adequate"].astype(float)
out["gsm8k_focus"] = {
    "sources": {s: int((pd.Series(src).reindex(gs) == s).sum())
                for s in set(pd.Series(src).reindex(gs))},
    "n_unfired": int(len(gs)),
    "never_acc_on_these": round(float(a.mean()), 4),
    "aggressive_acc_on_these": round(float(b.mean()), 4),
    "flips_w2r": int(((a == 0) & (b == 1)).sum()),
    "flips_r2w": int(((a == 1) & (b == 0)).sum()),
    "identical_text": int((nv.loc[gs, "answer"].fillna("")
                           == ag.loc[gs, "answer"].fillna("")).sum()),
    "flip_ids_r2w": list(gs[(a == 1) & (b == 0)]),
}

# C. taxonomy per category (aggressive vs never/always)
fired = ag["fired"].astype(bool)
dv = ag["adequate"].astype(float)
nvv = nv["adequate"].astype(float)
alv = al["adequate"].astype(float)
for c in CATS + ["ALL"]:
    m = (ag["cat"] == c) if c != "ALL" else pd.Series(True, index=ag.index)
    out["taxonomy"][c] = {
        "n": int(m.sum()),
        "fired": int((m & fired).sum()),
        "unfired_local_wrong_always_right_fixable_miss":
            int((m & ~fired & (nvv == 0) & (alv == 1)).sum()),
        "unfired_local_wrong_always_wrong_no_headroom":
            int((m & ~fired & (nvv == 0) & (alv == 0)).sum()),
        "fired_delivered_right": int((m & fired & (dv == 1)).sum()),
        "fired_delivered_wrong_always_wrong_expert_limit":
            int((m & fired & (dv == 0) & (alv == 0)).sum()),
        "fired_delivered_wrong_always_right_relay_or_var":
            int((m & fired & (dv == 0) & (alv == 1)).sum()),
        "fired_but_local_was_right_no_benefit_risk":
            int((m & fired & (nvv == 1)).sum()),
    }

OUT.mkdir(exist_ok=True)
(OUT / "internal_category_diagnostic.json").write_text(
    json.dumps(out, indent=1))
print(json.dumps(out["table"], indent=1))
print("\n-- unfired flips (aggressive) --")
print(json.dumps(out["unfired_flips"]["aggressive"], indent=1))
print("\n-- gsm8k focus --")
print(json.dumps({k: v for k, v in out["gsm8k_focus"].items()
                  if k != "flip_ids_r2w"}, indent=1))
print("\n-- taxonomy --")
print(json.dumps(out["taxonomy"], indent=1))
