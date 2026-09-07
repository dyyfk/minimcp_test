# STATUS — paper-revision P0 inventory (experiment_id: pr0907, 2026-09-07)

Scope: P0 of the revision task list — inventory + export of existing
results. **No model was started; no GPU/API cost incurred** (only
`modal volume ls/get` on cached artifacts). Reviewed-code baseline:
commit `e74ca0c`; every native-bench run was produced by
`interactive_paper/modal_native_bench.py` at its **only** commit
`51fdb4e` (2026-09-02), which is byte-identical in `e74ca0c` — so the
snapshot Codex reviewed IS the code that produced all paper data.

## 1. 已有且可复现 (exists, reproducible end-to-end)

**Threshold sweep / leak audit / no-leak gate (8bz, 8cb).** All three
artifacts and their scripts are in the repo and re-run locally (CPU
only, feature shards mirrored under `interactive_paper/data/`):

- `data/threshold_sweep.json` ← `scripts/40_threshold_sweep.py`
- `data/leak_check.json`, `data/leak_exclude.json` ← `scripts/41_leak_audit.py`
- `data/gate_native_noleak.json` ← `scripts/43_noleak_artifact.py`

New per-sample export (`scripts/48_paper_revision_export.py`) rebuilds
the deployed merge — **5,228 rows = 4,986 core (parts caliboff/expoff/
exp2off/exp3off/exp3zhoff) + 242 fresh train rows** — with seed-42
StratifiedKFold(5), C=3e-4, and dumps per-sample id / source file /
split / lang / native label / fold id / OOF score for BOTH the
deployed fit and the no-leak refit (5,221 rows after dropping the 7
`leak_exclude.json` ids):

- `calibration_oof.parquet` — 5,228 rows, all columns above, zero
  missing source files.
- `calibration_manifest.jsonl` — id/source/split/lang/label/excluded.
- `thresholds.json` — threshold RULE (per-language quantile of
  core-calibration OOF at 1−nominal, fresh rows excluded from the
  quantile base), deployed + no-leak per-language thresholds for both
  tier sets (15/30/50 current, 15/25/40 proposed), the full 5–50%
  nominal sweep grid, and a validation block.
- **Validation: all 12 recomputed thresholds (en/zh × 3 tiers × both
  gates) match the shipped artifacts to 4 decimals**; deployed OOF AUC
  .8477, no-leak .8493, testoff guard as logged in RESULTS 8cb.

**Gate identity.** Volume `/data/gate_native.json` (mtime 2026-09-01
22:59 PDT) is byte-identical to the repo copy, SHA-256
`0e6494c2eeac9bcd…` (full hash in `thresholds.json`). It is the 8bq
artifact with 15/30/50 per-language thresholds. **`gate_native_noleak.json`
does NOT exist on the volume; RATES in `modal_native_bench.py` are
still {15/30/50}. Therefore: the no-leak gate and the proposed
15/25/40 tiers have NEVER been deployed and no run ever used them.**
15/25/40 exists only as a remix recommendation (8bz) — exactly as
RESULTS 8cb says ("swap happens at the tier re-run").

**Native benchmark runs (8bu line, all 2026-09-02, judged 09-02/09-03).**
Complete inventory in `manifest.json` (45 run entries: pool × arm ×
relay-variant, raw shard paths + dates, judged parquet, judge,
gate SHA, per-arm stats) and `artifact_index.jsonl` (~530 volume files
with size/date; SHA-256 recorded for the 46 judged parquets that are
mirrored locally in `data/native_bench/`). Headline counts (n /
escalation / acc / mean-lat s): every pool has never + conservative_tts
+ balanced_tts + aggressive_tts + always_tts at full pool size
(frozen 240, striviaqa 250, swebq 250, sllama 250, sdqa 200,
sreason 202, valpaca 199). Duplicate/legacy runs are all listed, not
just the best: striviaqa also has steer-relay aggressive/balanced/always,
swebq has steer always and `old_cleaner_always_tts` (raw shards only,
never judged), plus `.smoke` shards (excluded from judged parquets).

⚠️ The two relay variants are **different protocols with very
different judged scores** (striviaqa always: steer .728 vs tts .960;
aggressive: .740 vs .884) — steer lets the talker paraphrase the
expert answer, tts relays it verbatim. The paper's current tables use
the `_tts` runs. Do not mix them in one table.

## 2. 已有但缺原始记录 (exists, but raw-record gaps)

- **Per-escalation raw timing**: stall_ms / wait_chunks / relay_synth_ms
  / relay_audio_s / expert_latency_s / answer_ms are logged per row in
  the jsonl shards and judged parquets — but there is **no client-side
  clock anywhere**: no playback timestamps, no stored local/relay
  audio, no ASR of produced audio (see §boundaries). The existing
  latency numbers support "server-side estimated audio-ready time"
  claims only.
- **Expert request input boundary**: the audio actually sent to the
  expert was not persisted; only the ASR transcript (`transcript`,
  truncated to 300 chars) survives. The snapshot rule is knowable from
  code, not from artifacts.
- **8bu run logs**: Modal stdout of the 09-02 sweep was not archived;
  shard mtimes + row payloads are the surviving record.
- **agg0/aggmlp judged parquets** (volume mtime 2026-09-07): these are
  the concurrent 8cu line (patched-model official rejudge, see
  RESULTS 8cu-1..4) — *not* part of this revision task; listed in the
  index for completeness. Same for `backdiff_smoke_*.json` (09-07).

## 3. 尚未运行 (not yet run)

- No-leak gate deployment + 15/25/40 tier live arms (P1 §5).
- Any independent repeats (every existing arm is a single pass;
  `_todo()` skips by query id, so re-invoking the old command would
  NOT produce a second repeat — needs run_id-keyed resume as specified).
- Client-side timing / real audio capture / ASR-of-played-audio
  scoring (P1 §4). No existing run can be reinterpreted to provide these.
- Independent validation split for tier selection (P1 §5.3): does not
  exist yet; calibration OOF was the only threshold basis, and the
  external pools have all been looked at.
- P2 new-benchmark holdout audit.

## Three evaluation boundaries (P0 §3) — verified against the actual code, with data quantification

All three behaviors are **still present**; no newer/fixed version of
the bench code exists anywhere (single commit, no local diff, no
alternate script on the volume).

1. **Expert input** (`modal_native_bench.py:400` `snap = au[-30*16000:]`):
   the expert gets the last 30 s of the **full query wav file**,
   regardless of how much the local model had consumed at fire time.
   Quantified from the judged parquets
   (`onset_before_last_chunk_fired` per run in manifest.json): fired
   turns where onset came before the last audio chunk = frozen always
   26/238, aggressive 18/124; valpaca always 38/199; sreason always
   18/202; sdqa always 13/200; striviaqa/swebq ≈ 0. On those rows the
   expert saw audio the local model had not yet consumed. Converse
   defect: queries >30 s (46/240 frozen internal) have their
   *beginning* cut from the expert's input.
2. **Local audio** (`modal_native_bench.py:155`
   `as_duplex(generate_audio=False)`): the local path **never
   synthesizes audio**. `answer_ms` is text-generation wall time.
   Relay audio duration comes from a separate teacher-forced TTS
   re-synthesis of the relay text (`relay_audio_s` + `relay_synth_ms`).
   The published latency = server-side composite estimate
   (onset-overrun + stall + wait + synth + relay-audio-duration for
   escalated; onset-overrun + answer_ms for local) — it is **not**
   client-audible latency and should be reported as
   "server-side estimated time-to-audio", asymmetric between arms
   (local arm has no synthesis term at all).
3. **Judge field** (`modal_native_bench.py:532-537`):
   `field="delivered"` scores the **relay TEXT on fired turns / local
   answer TEXT otherwise**. It is never an ASR transcript of produced
   or played audio (`transcript` holds the *uplink* ASR of the user's
   question). No audio-transcript accuracy exists in any current run.

## P1 readiness + cost estimate (NOT started)

Prerequisites met: no-leak gate artifact validated; per-language
thresholds for 10–50% nominal already computed (`thresholds.json`);
what is missing before launch is (i) the §4 script fixes (causal
expert snapshot, both-path audio synthesis + explicit timing names,
run_id-keyed resume) and (ii) a frozen validation split.

Estimated first batch (690 queries × {local, 3 tiers, always} × 3
repeats = 10,350 sessions; escalations ≈ 690×3×(.09+.22+.40+1.0) ≈
3,540 expert calls, assuming realized rates track the 8bz remix):

- GPU: ~30 s H100/session local-arm, ~90 s escalated (audio feed +
  expert wait sleep + relay + both-path TTS adds vs the 8bu numbers)
  → ≈145 H100-h ≈ **$580–700** on Modal.
- API: 3,540 × (gpt-transcribe + gpt-5.5-web-low) ≈ **$100–200**;
  judges ≈ 10,350 delivered-text + 10,350 audio-ASR-text ≈ **$40–80**;
  ASR of produced audio ≈ **$30–60**.
- Wall clock: ~1.5–2 days at 8 workers.
- Reusable: nothing directly (protocol changes make old arms a
  separate-version comparison column, as required); the old arms
  remain the paper's "v1 protocol" table.

These are order-of-magnitude estimates from the 8bu per-row timing
fields; the P1 smoke set (§4, ~10 fixed queries) will calibrate them
before full launch.

## Deliverable mapping (spec → this bundle)

| spec name | here |
|---|---|
| STATUS.md | this file |
| manifest.json | `manifest.json` (runs + serving/judge/latency defs) |
| calibration_manifest.jsonl | same name |
| calibration_oof.parquet | same name |
| gate_native_noleak.json | same name (copy; canonical: `data/gate_native_noleak.json`, SHA-256 in thresholds.json) |
| leak_audit.json | same name (leak_check + leak_exclude bundle) |
| thresholds.json | same name |
| validation_sweep.parquet | **not yet** (P1 §5 output); 8bz remix grid = `validation_sweep_source.json` (copy of threshold_sweep.json — calibration-remix, NOT independent validation) |
| runs/<run_id>/… | **not yet** (P1); existing single-pass runs indexed in manifest.json + artifact_index.jsonl |
| summary.json | same name (per pool×arm stats of existing runs) |
| artifact_index.jsonl | same name |
| scripts | `scripts/40,41,43` (existing) + `scripts/48_paper_revision_export.py`, `scripts/49_paper_revision_manifest.py` (new) |

Large artifacts stay on Modal volume `gate-data` (paths + sizes +
export commands in `artifact_index.jsonl`; SHA-256 present where a
local mirror exists). `vol_listings/` holds the raw `modal volume ls
--json` dumps; `vol_copies/` (not committed) holds the volume gate
artifacts verified byte-identical to the repo copies.
