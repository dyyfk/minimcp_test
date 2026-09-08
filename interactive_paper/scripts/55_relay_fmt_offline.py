"""Offline relay-formatter completeness diagnostic ($0, Exp-2 step 1):
run the shipped v2 formatter and the conservative v3 candidate over the
ALREADY-RECORDED expert texts (no expert/GPU/judge calls) and measure
what survives. This is a relay ablation diagnostic only — it selects
nothing and is never a live arm.

Checks:
  A. The three known regression cases (q0091 "64", q0046 "36", q0486
     valid <button> HTML) — recorded for regression tracking, NOT used
     to pick the final config.
  B. Reference-token retention on always-arm fired rows whose expert
     text was judged right: does the formatted text still contain the
     reference answer (normalized substring, or all of its digit
     tokens)? Reported overall and on the expert-right/delivered-wrong
     relay-loss rows.
  C. Length distributions (chars) per formatter.
  D. Residual display-syntax audit: rows whose formatted text still
     carries markdown/LaTeX tokens the TTS would read aloud.

Usage: set PYTHONUTF8=1 && .venv_boot/Scripts/python.exe scripts/55_relay_fmt_offline.py
Output: data/paper_revision_pr0907/relay_fmt_offline.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import relay_fmt  # noqa: E402

D = Path("data")
OUT = D / "paper_revision_pr0907" / "relay_fmt_offline.json"
POOLS = ["frozen", "sreason", "sllama", "striviaqa", "sdqa"]
FMTS = {"v2": relay_fmt.clean_expert_v2, "v3": relay_fmt.format_spoken_v3}
JCOL = {"frozen": "adequate", "sreason": "adequate", "sdqa": "adequate",
        "sllama": "oab_ok", "striviaqa": "oab_ok"}


def norm(t):
    t = re.sub(r"[^0-9a-zA-Z一-鿿]+", " ", str(t).lower())
    return re.sub(r"s+", " ", t).strip() if False else " ".join(t.split())


def retains(formatted, ref):
    """Reference survives: normalized substring, or every digit token
    of a short reference is present."""
    f, r = norm(formatted), norm(ref)
    if not r:
        return None
    if r in f:
        return True
    digs = re.findall(r"[0-9]+(?:[.][0-9]+)?", str(ref))
    if digs:
        fd = set(re.findall(r"[0-9]+(?:[.][0-9]+)?", str(formatted)))
        return all(d in fd for d in digs)
    toks = r.split()
    hit = sum(1 for w in toks if w in f)
    return hit >= max(1, int(0.8 * len(toks)))


RESID = re.compile(r"(?:frac|[|#`]|[*]{2}|(?<![0-9a-zA-Z])text[{])")

out = {"regression_cases": {}, "retention": {}, "lengths": {},
       "residual_syntax": {}}

frames = {}
for pool in POOLS:
    p = D / "native_bench" / f"{pool}_always_tts_judged.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p).drop_duplicates("id", keep="last").set_index("id")
    df = df[df["fired"].astype(bool) & df["expert_answer"].notna()]
    df = df[df["expert_answer"].astype(str).str.strip().astype(bool)]
    frames[pool] = df

# A. regression cases (frozen pool)
fz = frames["frozen"]
for qid, needle in [("q0091", "64"), ("q0046", "36"), ("q0486", "<button")]:
    e = str(fz.loc[qid, "expert_answer"])
    row = {}
    for name, fn in FMTS.items():
        got = fn(e)
        row[name] = {"text": got, "keeps_answer": needle.lower() in got.lower()}
    row["recorded_v1_relay"] = str(fz.loc[qid, "relay_text"])
    out["regression_cases"][qid] = row

# B/C/D per pool
for pool, df in frames.items():
    jc = JCOL[pool]
    ec = jc + "_expert"
    has_expert_verdict = ec in df.columns and df[ec].notna().any()
    raw_ret = pd.Series([retains(f, r) for f, r in
                         zip(df["expert_answer"], df["reference_answer"])],
                        index=df.index, dtype="object")
    stats = {"raw_expert": {
        "n": int(len(df)),
        "ref_retained": float(raw_ret[raw_ret.notna()].astype(bool).mean()),
        "len_mean": float(df["expert_answer"].astype(str).str.len().mean())}}
    if has_expert_verdict:
        eok0 = df[ec].astype(float) == 1
        loss0 = eok0 & (df[jc].astype(float) == 0)
        mm = raw_ret.notna()
        stats["raw_expert"]["ref_retained_on_relay_loss"] = (
            float(raw_ret[mm & loss0].astype(bool).mean())
            if (mm & loss0).any() else None)
    for name, fn in FMTS.items():
        formatted = df["expert_answer"].astype(str).map(fn)
        lens = formatted.str.len()
        ret = [retains(f, r) for f, r in
               zip(formatted, df["reference_answer"])]
        ret = pd.Series(ret, index=df.index, dtype="object")
        m = ret.notna()
        row = {"n": int(len(df)),
               "ref_retained": float(ret[m].astype(bool).mean()),
               "n_with_ref": int(m.sum()),
               "len_mean": float(lens.mean()),
               "len_p90": float(lens.quantile(.9)),
               "n_empty": int((lens == 0).sum()),
               "residual_syntax": int(formatted.str.contains(RESID).sum())}
        if has_expert_verdict:
            eok = df[ec].astype(float) == 1
            row["ref_retained_expert_right"] = float(
                ret[m & eok].astype(bool).mean())
            loss = eok & (df[jc].astype(float) == 0)
            row["relay_loss_rows"] = int(loss.sum())
            row["ref_retained_on_relay_loss"] = (
                float(ret[m & loss].astype(bool).mean())
                if (m & loss).any() else None)
        stats[name] = row
    out["retention"][pool] = stats

OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False),
               encoding="utf-8")

print("=== regression cases ===")
for qid, row in out["regression_cases"].items():
    print(f"{qid}: v2_keeps={row['v2']['keeps_answer']} "
          f"v3_keeps={row['v3']['keeps_answer']}")
print()
print("=== retention (always-arm fired rows, recorded expert texts) ===")
print(f"{'pool':10} {'fmt':10} {'n':>4} {'ref_kept':>8} {'exp_right':>9} "
      f"{'loss_kept':>9} {'len_mean':>8} {'empty':>5} {'resid':>5}")
for pool, stats in out["retention"].items():
    for name, r in stats.items():
        er = r.get("ref_retained_expert_right")
        lk = r.get("ref_retained_on_relay_loss")
        print(f"{pool:10} {name:10} {r['n']:4d} {r['ref_retained']:8.3f} "
              f"{(f'{er:9.3f}' if er is not None else '        -')} "
              f"{(f'{lk:9.3f}' if lk is not None else '        -')} "
              f"{r['len_mean']:8.1f} "
              f"{r.get('n_empty', 0):5d} {r.get('residual_syntax', 0):5d}")
