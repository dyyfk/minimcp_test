"""Paper-revision P0 manifest (pr0907, rev2 after coauthor review):
inventory every native_bench run on the gate-data volume, join with
the locally-mirrored judged parquets, and emit manifest.json /
artifact_index.jsonl / summary.json.

No GPU, no API: volume listings were dumped with `modal volume ls
--json` into data/paper_revision_pr0907/vol_listings/.

rev2 changes (review items 3/4/5):
- experiment_family / protocol_version / in_paper / provenance_basis
  per entry; gate SHA only where a basis can be stated, else null.
- timing recomputed from each entry's OWN judged parquet
  (timing_source + sha256 recorded); steer and tts no longer share a
  timing source; per-variant endpoints listed; all outputs in seconds.
- smoke shards listed separately from formal shards.
- n_scored / n_total / missing_scores made explicit.
- expert timeout proxy renamed and rule stated; no silent excepts.

Usage (from interactive_paper/):
  set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\49_paper_revision_manifest.py
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
OUT = D / "paper_revision_pr0907"
NB = D / "native_bench"

CODE_V1 = {"commit": "51fdb4e", "file": "interactive_paper/modal_native_bench.py",
           "committed": "2026-09-02", "n_commits_at_run_time": 1,
           "note": "single commit, unmodified through e74ca0c - the "
                   "reviewed snapshot is the code that produced every "
                   "native_v1 run"}
GATE_SHA = hashlib.sha256((D / "gate_native.json").read_bytes()).hexdigest()
V1_BASIS = ("inferred, not a per-run record: bench script has a single "
            "commit (51fdb4e, 2026-09-02); volume gate_native.json mtime "
            "2026-09-01 22:59 PDT and unchanged since; runs dated "
            "2026-09-02/03 by shard mtime")
SERVING = {"artifact": "/data/gate_native.json (8bq, volume mtime 2026-09-01 22:59 PDT)",
           "gate_sha256": GATE_SHA, "layer": 22, "k_eot": 8,
           "sys_prompt": "You are a friendly assistant.", "top_k": 20,
           "force_listen_count": 3, "rates_nominal": {"conservative": .15,
           "balanced": .30, "aggressive": .50},
           "act_gate": "/data/gate_act.json (8bh dialogue-act floor filter)",
           "expert": "gpt-5.5 web low effort (fallback: non-web)",
           "uplink_asr": "gpt-transcribe", "max_wait_s": 150}
JUDGE = {"striviaqa": "OpenAudioBench gpt-4o", "swebq": "OpenAudioBench gpt-4o",
         "sllama": "OpenAudioBench gpt-4o", "valpaca": "VoiceBench gpt-4o-mini 1-5",
         "frozen": "ours ref-anchored gpt-5.4-mini",
         "sdqa": "ours ref-anchored gpt-5.4-mini",
         "sreason": "ours ref-anchored gpt-5.4-mini"}
ACC_COL = {"frozen": "adequate", "striviaqa": "oab_ok", "swebq": "oab_ok",
           "sllama": "oab_ok", "sdqa": "adequate", "sreason": "adequate",
           "valpaca": "vb_score"}
POOLS = ["frozen", "striviaqa", "swebq", "sllama", "sdqa", "sreason", "valpaca"]
TIERS = ("conservative", "balanced", "aggressive", "always")

TIMING_ENDPOINTS = {
    "local_only": "onset_overrun_s (answer-onset chunks past end of query "
                  "audio, 1 s/chunk) + answer_ms/1000 (TEXT generation "
                  "wall time; generate_audio=False, no audio synthesized)",
    "tts_relay": "local rows as local_only; escalated rows: "
                 "onset_overrun_s + stall_ms/1000 + wait_chunks*1s + "
                 "relay_synth_ms/1000 (teacher-forced re-synthesis wall "
                 "time) + relay_audio_s (synthesized relay duration)",
    "steer_relay": "local rows as local_only; escalated rows: "
                   "onset_overrun_s + stall_ms/1000 + wait_chunks*1s + "
                   "relay_ms/1000 (relay TEXT generation wall time; no "
                   "relay audio was synthesized in steer runs)",
}


def classify(arm):
    """-> (experiment_family, protocol_version, in_paper, code, basis)"""
    if arm.startswith("old_cleaner"):
        return ("legacy_old_cleaner", "tts_relay(pre-cleaner-fix)", False,
                None, "filename prefix only; producing code version not "
                      "recorded")
    if arm in ("agg0", "aggmlp"):
        return ("8cu_official_rejudge", "trace_dump(no serving loop)",
                False, "modal_native_dump.py (8cu line, 2026-09-07)",
                "concurrent 8cu experiment (RESULTS 8cu-1..4); NOT part "
                "of this revision task; gate/serving fields not "
                "applicable")
    if arm == "never":
        return ("native_v1", "local_only", True, CODE_V1["commit"], V1_BASIS)
    if arm.endswith("_tts"):
        return ("native_v1", "tts_relay", True, CODE_V1["commit"], V1_BASIS)
    if arm in TIERS:
        return ("native_v1_steer", "steer_relay", False, CODE_V1["commit"],
                V1_BASIS)
    return ("unclassified", None, False, None, "unrecognized arm name")


def timing_from(df, protocol):
    need = {"local_only": ["onset_chunk", "n_chunks", "answer_ms"],
            "tts_relay": ["onset_chunk", "n_chunks", "answer_ms", "mode",
                          "stall_ms", "wait_chunks", "relay_synth_ms",
                          "relay_audio_s"],
            "steer_relay": ["onset_chunk", "n_chunks", "answer_ms", "mode",
                            "stall_ms", "wait_chunks", "relay_ms"]}
    cols = need.get(protocol)
    if cols is None:
        return None, f"no timing definition for protocol {protocol}"
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, f"columns missing from parquet: {missing}"
    on = (df["onset_chunk"].fillna(df["n_chunks"]) + 1
          - df["n_chunks"]).clip(lower=0)
    local_t = on + df["answer_ms"].fillna(0) / 1000
    if protocol == "local_only":
        t = local_t
    else:
        esc = df["mode"] == "escalated"
        relay_s = (df["relay_synth_ms"].fillna(0) / 1000
                   + df["relay_audio_s"].fillna(0)
                   if protocol == "tts_relay"
                   else df["relay_ms"].fillna(0) / 1000)
        esc_t = (on + df["stall_ms"].fillna(0) / 1000
                 + df["wait_chunks"].fillna(0) + relay_s)
        t = pd.Series(np.where(esc, esc_t, local_t), index=df.index)
    return {"reconstructed_t_mean_s": float(t.mean()),
            "reconstructed_t_p50_s": float(np.percentile(t, 50)),
            "reconstructed_t_p95_s": float(np.percentile(t, 95))}, None


root = json.load(open(OUT / "vol_listings/native_bench_root.json"))
pool_ls = {p: json.load(open(OUT / f"vol_listings/pool_{p}.json"))
           for p in POOLS}

index = []
for e in root:
    if e["type"] == "file":
        index.append({"volume": "gate-data", "path": e["filename"],
                      "size": e["size"], "modified": e["created_modified"]})
for p, ls in pool_ls.items():
    for e in ls:
        index.append({"volume": "gate-data", "path": e["filename"],
                      "size": e["size"], "modified": e["created_modified"]})
for rec in index:
    lp = NB / Path(rec["path"]).name
    if lp.exists() and lp.suffix == ".parquet":
        rec["local_mirror"] = str(lp).replace("\\", "/")
        rec["sha256_local"] = hashlib.sha256(lp.read_bytes()).hexdigest()
    rec["export"] = "modal volume get gate-data " + rec["path"]

runs, summary = [], {}
for p in POOLS:
    shards, smokes = {}, {}
    for e in pool_ls[p]:
        n = Path(e["filename"]).name
        mm = re.match(r"(.+?)\.jsonl\.shard\d+$", n)
        ms = re.match(r"(.+?)\.jsonl\.smoke$", n)
        if mm:
            shards.setdefault(mm.group(1), []).append(
                {"file": e["filename"], "size": e["size"],
                 "modified": e["created_modified"]})
        elif ms:
            smokes.setdefault(ms.group(1), []).append(e["filename"])
    for arm in sorted(set(shards) | set(smokes)):
        files = shards.get(arm, [])
        fam, proto, in_paper, code, basis = classify(arm)
        jp = NB / f"{p}_{arm}_judged.parquet"
        entry = {"pool": p, "arm": arm,
                 "experiment_family": fam, "protocol_version": proto,
                 "in_paper": in_paper, "provenance_basis": basis,
                 "n_shards": len(files),
                 "shard_dates": sorted({f["modified"] for f in files}),
                 "raw_shards": [f["file"] for f in files],
                 "smoke_files": smokes.get(arm, []),
                 "smoke_note": ("smoke files are implementation checks; "
                                "never part of formal samples"
                                if smokes.get(arm) else None),
                 "judged_parquet": (f"native_bench/{p}_{arm}_judged.parquet"
                                    if jp.exists() else None),
                 "judge": JUDGE[p], "code": code,
                 "gate_sha256": (GATE_SHA if fam in
                                 ("native_v1", "native_v1_steer") else None),
                 "cached_remix": False,
                 "executed": bool(files or smokes.get(arm))}
        if jp.exists():
            df = pd.read_parquet(jp).drop_duplicates("id", keep="last")
            col = ACC_COL[p]
            fired = (df["fired"].fillna(False).astype(bool) if "fired" in df
                     else pd.Series(False, index=df.index))
            scored = df[col].notna() if col in df else pd.Series(
                False, index=df.index)
            entry["stats"] = {
                "n_total": int(len(df)),
                "n_scored": int(scored.sum()),
                "missing_scores": int((~scored).sum()),
                "missing_scores_reason": (
                    "judge returned no parseable verdict after retries; "
                    "rows kept, NOT scored as 0"
                    if int((~scored).sum()) else None),
                "metric": col,
                "acc_or_score_over_scored": (
                    float(df.loc[scored, col].astype(float).mean())
                    if scored.any() else None),
                "escalation_rate": float(fired.mean()),
                "expert_error_rows": (
                    int((fired & df["expert_answer"].fillna("")
                         .str.startswith(("[thinker failed", "[error"))).sum())
                    if "expert_answer" in df else None),
                "timeout_proxy_rows_wait_ge_145": (
                    int((fired & (df["wait_chunks"].fillna(0) >= 145)).sum())
                    if "wait_chunks" in df else None),
                "timeout_proxy_rule": "PROXY: wait_chunks >= 145 of "
                    "MAX_WAIT_S=150; no explicit timeout status or "
                    "timestamp was logged in v1",
                "onset_before_last_chunk_fired": (
                    int(((df["onset_chunk"] < df["n_chunks"] - 1)
                         & fired).sum())
                    if {"onset_chunk", "n_chunks"} <= set(df.columns)
                    else None),
                "audio_gt_30s": (int((df["audio_s"] > 30).sum())
                                 if "audio_s" in df else None),
            }
            timing, why = timing_from(df, proto)
            entry["timing"] = timing
            entry["timing_missing_reason"] = why
            entry["timing_source"] = str(jp).replace("\\", "/")
            entry["timing_source_sha256"] = hashlib.sha256(
                jp.read_bytes()).hexdigest()
            entry["timing_endpoints"] = TIMING_ENDPOINTS.get(proto)
        runs.append(entry)
    summary[p] = {r["arm"]: {**r.get("stats", {}),
                             **(r.get("timing") or {})}
                  for r in runs if r["pool"] == p and "stats" in r}

man = {"experiment_id": "pr0907", "created": "2026-09-07",
       "revision": 2,
       "purpose": "P0 inventory of existing native-bench runs for the "
                  "RTCA paper revision; no new model runs",
       "code_native_v1": CODE_V1, "serving_native_v1": SERVING,
       "judges": JUDGE,
       "timing_definition": {
           "name": "reconstructed timing diagnostic",
           "units": "all reported values in SECONDS; *_ms trace fields "
                    "divided by 1000",
           "per_protocol_endpoints": TIMING_ENDPOINTS,
           "caveat": "server-side reconstruction from per-turn trace "
                     "fields. The local path never synthesizes audio "
                     "(generate_audio=False) and no client-side clock, "
                     "playback timestamp, or produced-audio recording "
                     "exists in any v1 run. These numbers are NOT "
                     "audible latency and the arms' endpoints are NOT "
                     "symmetric - do not present them as a single "
                     "'audio-ready time'."},
       "judge_field_definition": "field=delivered scores relay TEXT on "
                                 "fired turns / local answer TEXT otherwise "
                                 "- never an ASR transcript of produced audio",
       "runs": runs}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
with open(OUT / "artifact_index.jsonl", "w", encoding="utf-8") as fh:
    for rec in index:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"{len(runs)} run entries, {len(index)} artifacts")
for r in runs:
    s = r.get("stats", {})
    t = r.get("timing") or {}
    print(f"{r['pool']:10s} {r['arm']:22s} {r['experiment_family']:20s} "
          f"n={s.get('n_scored', '-'):>4}/{s.get('n_total', '-'):>4} "
          f"t_mean={t.get('reconstructed_t_mean_s', float('nan')):.2f}")
