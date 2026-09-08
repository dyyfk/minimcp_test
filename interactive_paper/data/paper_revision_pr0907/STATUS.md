# STATUS — paper-revision P0 inventory (experiment_id: pr0907)

**Revision 2 (2026-09-07)** after coauthor review of `dd33f7c`.
Scope: P0 inventory + CPU-only corrections. No model was started for
this deliverable; the only paid calls in the pr0907 line are (a) the
field=expert rejudge of the two frozen arms (~$3) and (b) the
internal r1/r2 repeat arms separately authorized by the user before
this review (in flight; reported separately, never merged into v1
artifacts).

## Review-item dispositions (rev2)

1. **Missing dependencies** — scripts 40/41/43, threshold_sweep /
   leak_check / leak_exclude / gate_native_noleak jsons, and the six
   part-label judged parquets are now committed. `scripts/48` fails
   hard (SystemExit) on any threshold mismatch and resolves every
   source file (`queries.jsonl` added to the scan; 0 unmatched ids).
   **Clean-checkout reproduction**: `git worktree` at the rev2 commit,
   84 non-git inputs injected only after sha256 verification against
   `input_manifest.jsonl` (38 feature shards, 364 MB, kept on volume
   `gate-data` root with per-file download commands) and
   `artifact_index.jsonl` (46 judged parquets). 48+49 rerun there:
   all five text outputs byte-identical, `calibration_oof.parquet`
   value-identical. Commands + env in `repro_record.json`.
2. **Sweep-grid versioning** — `thresholds.json` now contains
   `noleak_nominal_grid_lang` (10/15/20/25/30/40/50% per language,
   fresh OOF of the no-leak fit, with fit gate SHA, train-id hash,
   quantile-base id hash, language rule, score kind, script) and the
   old grid renamed `historical_remix_grid` with **null fit metadata**
   and an explicit limitation: different merge/zh construction, not
   re-derived, exploration-only (its internal remix consumed
   already-inspected test outcomes — it can never be promoted to
   independent validation).
3. **Manifest provenance** — 46 entries (STATUS previously said 45 —
   wrong; the old_cleaner raw-shard entry was the missed one). Every
   entry now carries `experiment_family` (native_v1 /
   native_v1_steer / 8cu_official_rejudge / legacy_old_cleaner),
   `protocol_version`, `in_paper`, `provenance_basis`. The v1 gate
   SHA + commit are explicitly labeled *inferred* (single-commit
   script + volume mtime ordering — no per-run record exists); 8cu
   and old_cleaner entries get their own code attribution or null.
   Smoke shards are listed in `smoke_files`, never in `raw_shards`.
4. **Steer timing** — timing is now computed from each entry's own
   judged parquet (`timing_source` + `timing_source_sha256` recorded);
   steer arms use `relay_ms` (their own endpoint — no relay audio was
   synthesized), TTS arms use synth+audio-duration. Corrected values
   match the reviewer's independent recomputation (striviaqa steer
   aggressive 6.97s mean, always 12.76s, balanced 4.72s; swebq always
   14.50s). `n_scored`/`n_total`/`missing_scores` are explicit
   (striviaqa steer aggressive = 231/250 scored; missing = judge
   returned no parseable verdict after retries; never scored as 0).
   Steer vs TTS remain different protocols — not causally comparable.
5. **Unknown-state metadata** — `calibration_oof.parquet` now has
   `source_split` ("unknown" where the source file never recorded
   one) and `experiment_role` (fit_core+quantile_base /
   fit_fresh_train; **no row has a validation or held-out role**).
   `timeout_proxy_rows_wait_ge_145` replaces the old name, with its
   rule stated (proxy only — v1 logged no explicit timeout status or
   timestamp). All manifest timing is in seconds with per-arm
   endpoints, renamed **"reconstructed timing diagnostic"** — the
   local path synthesizes no audio and no client clock exists, so
   none of it is audible latency or a uniform "audio-ready time".
   Valpaca (AlpacaEval) timing is now included (its trace fields
   exist; only my earlier summary had dropped it). Silent
   `except: pass` removed.

## 1. 已有且可复现 (exists, reproducible end-to-end)

**Threshold sweep / leak audit / no-leak gate (8bz, 8cb)** — scripts
40/41/43 + their four artifacts committed; per-sample export
(`scripts/48`) rebuilds the deployed merge (5,228 = 4,986 core +
242 fresh train), seed-42 5-fold OOF for both the deployed fit and
the 5,221-row no-leak refit, per-sample id / source_file /
source_split / experiment_role / lang / native label / fold / OOF /
leak flags. All 12 per-language thresholds reproduce the shipped
artifacts (fail-hard check); deployed OOF AUC .8477, no-leak .8493.
Clean-checkout reproduction passed (see item 1 above).

**Gate identity** — volume `/data/gate_native.json` byte-identical to
the repo copy, SHA-256 `0e6494c2eeac9bcd86c10b5def3cbd32e98bb0765fa2fd8afc8c1b47915ea372`
(8bq, 15/30/50). `gate_native_noleak.json` (SHA-256 `2b9b9d29…`) is
**not on the volume**; RATES in the bench code are still 15/30/50:
the no-leak gate and the 15/25/40 tiers were never deployed and no
run used them.

**Native benchmark runs** — full inventory in `manifest.json`
(46 entries) + `artifact_index.jsonl` (~680 volume files; SHA-256
where a local mirror exists). Every pool has never +
conservative/balanced/aggressive/always `_tts` at full pool size;
duplicates all listed (striviaqa steer trio, swebq steer always,
legacy old_cleaner shards, 8cu agg0/aggmlp rejudges, smoke files
separated). Steer vs TTS deliver very different judged scores
(striviaqa always .728 vs .960) — separate protocols, do not mix.

## 2. 已有但缺原始记录 (exists, but raw-record gaps)

- Per-turn `_ms`/`wait` trace fields exist per row, but **no client
  clock, no playback timestamps, no stored local/relay audio, no ASR
  of produced audio** anywhere. Existing numbers support only the
  reconstructed-timing-diagnostic reading, per-arm endpoints as in
  `manifest.json`.
- The audio actually sent to the expert was not persisted (only the
  300-char uplink ASR transcript); the snapshot rule is knowable from
  code, not artifacts.
- Timeout is a proxy (`wait_chunks >= 145`), not an explicit status.
- Per-run gate/commit provenance was never logged at run time — v1
  attribution rests on the inference documented in
  `provenance_basis`; 8cu-family metadata not verified here (nulls).
- Modal stdout of the 09-02 sweep not archived.

## 3. 尚未运行 (not yet run)

- No-leak gate deployment + 15/25/40 live arms.
- Client-side timing / audio capture / ASR-of-audio scoring (P1 §4);
  causal expert-input snapshot fix.
- Independent validation split (no current artifact row has one; the
  8bz internal remix used inspected test outcomes — exploration only).
- P2 new-benchmark holdout audit.
- Formal 10,350-session batch: **not started and not authorized**;
  the in-flight internal r1/r2 repeats are the smaller,
  separately-authorized variance measurement of the v1 protocol
  (240 × 5 arms × 2 new repeats), reported outside the v1 artifacts.

## Three evaluation boundaries — unchanged from rev1, quantified

All present in every v1 run (single-commit code): (1) expert gets the
last 30 s of the full wav — fired turns with onset before the last
chunk (frozen always 26/238, valpaca 38/199, sreason 18/202) leaked
unconsumed audio to the expert, and the 46 internal queries >30 s
(43/60 hard-knowledge + 3 hard-math) reach the expert with their
beginning cut off — the hard-knowledge "expert capability limit" in
the category diagnostic is confounded with this truncation and cannot
be split with existing data; (2) local path `generate_audio=False`;
(3) delivered-judge scores text, never produced-audio ASR.

## Minimal P1 order (per review — agreed)

1. CPU-only script fixes: causal expert snapshot (consumed-prefix,
   no new truncation; explicit strategy if waiting for full
   question), both-path audio synthesis, monotonic-clock endpoints,
   explicit timeout/non-commit status, run_id/repeat_id keying
   (run_id landed in the rev2 commit).
2. Smoke set covering >30s knowledge, zh, early-onset, long answers,
   timeout/resume — implementation check only.
3. Independent validation: either re-fit after carving a validation
   split out of the fit data, or collect + audit new non-fit data.
   The current 5,221-row artifact cannot double as validation.
4. Freeze gate/threshold/serving/judge/schema after validation, then
   formal repeats. The 10,350-session figure is the formal-stage
   scale, not a smoke budget.
5. Internal stays 240 at original ratios; report per-category gains,
   escalation, input truncation, expert failure, relay loss, repeat
   spread; GSM8K zero-escalation cross-arm delta is checked in the
   in-flight repeat report.

## Deliverable mapping

| spec name | here |
|---|---|
| STATUS.md | this file (rev2) |
| manifest.json | rev2 (46 runs; families; per-entry timing sources) |
| calibration_manifest.jsonl / calibration_oof.parquet | rev2 (source_split, experiment_role) |
| gate_native_noleak.json | copy; canonical `data/gate_native_noleak.json` (committed) |
| leak_audit.json / thresholds.json | rev2 (grids labeled + provenance) |
| validation_sweep.parquet | not yet (P1); `validation_sweep_source.json` = historical remix, exploration only |
| runs/<run_id>/… | in-flight repeats will land as `native_bench_repeats/{r1,r2}/` (volume) |
| summary.json | rev2 |
| artifact_index.jsonl / input_manifest.jsonl / repro_record.json | rev2 |
| scripts | 40/41/43 (committed), 48/49 (rev2), 50 (category diagnostic), 51 (repeat variance, pending data) |
