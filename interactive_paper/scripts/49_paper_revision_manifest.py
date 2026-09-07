"""Paper-revision P0 manifest (pr0907): inventory every native_bench
run on the gate-data volume, join with the locally-mirrored judged
parquets, and emit manifest.json / artifact_index.jsonl / summary.json.

No GPU, no API: volume listings were dumped with `modal volume ls
--json` into data/paper_revision_pr0907/vol_listings/.

Usage (from interactive_paper/):
  set PYTHONUTF8=1 && .venv_boot\\Scripts\\python.exe scripts\\49_paper_revision_manifest.py
"""
import hashlib
import json
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
OUT = D / "paper_revision_pr0907"
NB = D / "native_bench"
m = types.ModuleType("sw")
exec(open("scripts/40_threshold_sweep.py").read().split("if __name__")[0], m.__dict__)

CODE = {"commit": "51fdb4e", "file": "interactive_paper/modal_native_bench.py",
        "committed": "2026-09-02", "n_commits": 1,
        "note": "single commit, no local modifications - the reviewed "
                "snapshot e74ca0c IS the code that produced every run"}
GATE_SHA = hashlib.sha256((D / "gate_native.json").read_bytes()).hexdigest()
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
    shards = {}
    for e in pool_ls[p]:
        n = Path(e["filename"]).name
        mm = re.match(r"(.+?)\.jsonl\.(shard\d+|smoke)$", n)
        if mm:
            shards.setdefault(mm.group(1), []).append(
                {"file": e["filename"], "size": e["size"],
                 "modified": e["created_modified"]})
    for arm, files in sorted(shards.items()):
        smoke = all("smoke" in f["file"] for f in files)
        jp = NB / f"{p}_{arm}_judged.parquet"
        entry = {"pool": p, "arm": arm, "smoke_only": smoke,
                 "n_shards": sum("shard" in f["file"] for f in files),
                 "shard_dates": sorted({f["modified"] for f in files}),
                 "raw_shards": [f["file"] for f in files],
                 "judged_parquet": (f"native_bench/{p}_{arm}_judged.parquet"
                                    if jp.exists() else None),
                 "judge": JUDGE[p], "code": CODE["commit"],
                 "gate_sha256": GATE_SHA,
                 "relay_variant": ("tts" if arm.endswith("_tts") else
                                   "steer" if arm.split("_")[0] in
                                   ("conservative", "balanced", "aggressive",
                                    "always") else None),
                 "cached_remix": False, "executed": True}
        if jp.exists():
            df = pd.read_parquet(jp).drop_duplicates("id", keep="last")
            col = ACC_COL[p]
            fired = (df["fired"].fillna(False).astype(bool) if "fired" in df
                     else pd.Series(False, index=df.index))
            err_exp = (df["expert_answer"].fillna("").str.startswith(
                ("[thinker failed", "[error")) if "expert_answer" in df
                else pd.Series(False, index=df.index))
            def c(name, default=0):
                return (df[name] if name in df
                        else pd.Series(default, index=df.index))
            entry["stats"] = {
                "n": int(len(df)),
                "escalation_rate": float(fired.mean()),
                "acc_or_score": (float(df[col].dropna().astype(float).mean())
                                 if col in df else None),
                "acc_n_judged": int(df[col].notna().sum()) if col in df else 0,
                "expert_error_rows": int((fired & err_exp).sum()),
                "expert_timeout_rows": int(
                    (fired & (c("wait_chunks").fillna(0) >= 145)).sum()),
                "onset_before_last_chunk_fired": int(
                    ((c("onset_chunk", np.nan) < c("n_chunks", np.nan) - 1)
                     & fired).sum()),
                "audio_gt_30s": int((c("audio_s", np.nan) > 30).sum()),
            }
            try:
                arm0 = arm.replace("_tts", "")
                la_df = (m.load_arm(p, arm0) if arm0 in
                         ("never", "always", "conservative", "balanced",
                          "aggressive") else None)
                if la_df is not None:
                    entry["stats"]["lat_mean_s"] = float(la_df["t_s"].mean())
                    entry["stats"]["lat_p50_s"] = float(
                        np.percentile(la_df["t_s"], 50))
                    entry["stats"]["lat_p95_s"] = float(
                        np.percentile(la_df["t_s"], 95))
            except Exception:
                pass
        runs.append(entry)
    summary[p] = {r["arm"]: r["stats"] for r in runs
                  if r["pool"] == p and "stats" in r}

man = {"experiment_id": "pr0907", "created": "2026-09-07",
       "purpose": "P0 inventory of existing native-bench runs (8bu line) "
                  "for the RTCA paper revision; no new model runs",
       "code": CODE, "serving": SERVING, "judges": JUDGE,
       "latency_definition": {
           "formula": "escalated: onset_overrun_chunks + stall_ms + "
                      "wait_chunks(s) + relay_synth_ms + relay_audio_s; "
                      "local: onset_overrun_chunks + answer_ms",
           "caveat": "server-side composite ESTIMATE. local arm runs "
                     "as_duplex(generate_audio=False): answer_ms is TEXT "
                     "generation wall time, no local audio is synthesized. "
                     "relay audio duration comes from a separate "
                     "teacher-forced TTS re-synthesis of the relay text. "
                     "No client-side playback clock exists in any run."},
       "judge_field_definition": "field=delivered scores relay TEXT on "
                                 "fired turns / local answer TEXT otherwise "
                                 "- never an ASR transcript of produced audio",
       "runs": runs}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
with open(OUT / "artifact_index.jsonl", "w", encoding="utf-8") as fh:
    for rec in index:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"{len(runs)} runs, {len(index)} artifacts")
for r in runs:
    s = r.get("stats", {})
    print(f"{r['pool']:10s} {r['arm']:22s} shards={r['n_shards']} "
          f"n={s.get('n', '-'):>4} esc={s.get('escalation_rate', '-')}")
