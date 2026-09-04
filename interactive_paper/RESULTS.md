# RESULTS — Zero-Training Escalation Gate for MiniCPM-o 4.5

Execution log. One section per Phase: numbers, figures, decisions, gotchas,
and abandoned routes. This is the raw material for the eventual paper.

Seed fixed at **42** everywhere. Models: small = MiniCPM-o 4.5 (9B); judge =
`gpt-5.4-mini`; escalation target = `gpt-5.5`.

> **2026-07-08 — provider switch: Anthropic → OpenAI.** The user has OpenAI
> credit, not Anthropic. Judge + query-gen moved to `gpt-5.4-mini`, escalation
> target to `gpt-5.5` (GPT-5.x reasoning models; `reasoning_effort="low"`,
> generous `max_completion_tokens`). `src/escalate.py` ported to the OpenAI SDK
> (Structured Outputs `response_format` json_schema, strict). Modal secret is now
> `openai`; the separate app is `think-gate-gen`; `build_claude_queries` →
> `build_gen_queries`; `queries_claude.jsonl` → `queries_gen.jsonl`. Sections
> below written before this date still describe the original Claude design.

---

## Setup / environment (2026-07-07)

Lives as a **subdir** inside the existing `dyyfk/minimcp_test` repo
(`interactive_paper/`), reusing that project's Modal infra:

- **Modal** workspace `rhe9527`, client 1.5.1. Invoke via the `modal` CLI
  (the anaconda base python has no modal module; Python310's does). Prefix
  `PYTHONUTF8=1` on Windows or the CLI's ✓ output crashes on cp936.
- **Weights reused**: MiniCPM-o 4.5 already on the `minicpm-o45-weights` Volume
  (downloaded 2026-06-30 by the sibling `modal_app.py::download_weights`), so
  Phase 0 needs no HF pull — mounts read-only at `/workspace/models/MiniCPM-o-4_5`.
- **Validated stack** (from the sibling project, unchanged): torch 2.8.0+cu128,
  transformers **4.51.0** (4.52+ breaks MiniCPM's Resampler init),
  `minicpmo-utils[all]` (librosa 0.9.0), `setuptools<81`, SDPA (no flash-attn).
  `image` in `modal_app.py` adds `openai` + `scikit-learn` + `pandas` +
  `pyarrow` on top; `src/` mounted at `/workspace/gate` (add_local_dir last).
- **`gate-data` Volume** created for query pool + feature store + preds.
- **OpenAI model ids** (pricing/models pages, July 2026): judge/gen
  `gpt-5.4-mini`, escalation `gpt-5.5`. GPT-5.x reasoning models — hidden
  reasoning tokens bill as output; calls use `reasoning_effort="low"` +
  `max_completion_tokens`. If an id 404s, pin a dated snapshot.

### BLOCKER (Phase 2): openai Modal secret

Phase 2's judge labeling and easy-chat/trap query generation need OpenAI inside
the container. User action required (once):

```
modal secret create openai OPENAI_API_KEY=sk-...
```

Phases 0–1 (small model only) proceed without it.

---

## Phase 0 — Modal env + model smoke test ✅ GO (2026-07-07)

`modal run interactive_paper/modal_app.py::smoke` — image built in 69s, ran on H100.

- **Load**: 16.1 s | **VRAM 16.4 GB** with `init_vision=False, init_audio=False,
  init_tts=False` — text-only load is ~4 GB lighter than the full duplex model,
  leaving the H100 nearly empty. Text `model.chat(content=[prompt])` works fine
  with the audio/vision encoders uninitialized.
- **Model internals** (for the Phase-1 hook):
  - top module `MiniCPMO`, single child `llm` = **`Qwen3ForCausalLM`**
  - `num_hidden_layers=36`, `hidden_size=4096`, `vocab_size=151748`
  - `llm.named_children = ['model', 'lm_head']` → backbone `model.llm.model`,
    last decoder layer `model.llm.model.layers[35]`, head `model.llm.lm_head`.
- **Decode speed** (greedy, bf16, SDPA):

  | probe | out tok | s | tok/s |
  |-------|--------:|--:|------:|
  | chat (octopus) | 50 | 3.7 | 13.6 |
  | math (GSM8K)   | 94 | 2.8 | 33.6 |
  | gpqa (QM)      | 81 | 2.5 | 32.8 |

  The `chat` 13.6 tok/s is a short-output artifact (fixed prefill/decode overhead
  amortized over only 50 tokens); **sustained rate ≈ 33 tok/s**, well above the
  ≥ 20 tok/s gate.
- **Coherence**: all three answers fluent, correct English (3-hearts octopus fact;
  GSM8K worked step-by-step; commutator → Heisenberg uncertainty). No Chinese
  probe run yet, but the duplex project already confirmed bilingual output.

**Verdict: GO.** No downgrade needed — MiniCPM-o 4.5 hooks cleanly (standard
Qwen3 backbone under `model.llm`), so the plan's fallback to 2.6 / Qwen3-8B is
not triggered.

---

## Phase 1 — signal extraction (hook-based capture) ✅ GO (2026-07-07)

Design decision: **don't re-implement MiniCPM's chat template / decode loop**
(it lives in remote code and is fragile). Instead let `model.chat()` generate
greedily as normal and *passively observe* via two forward hooks —
`model.llm.model.layers[35]` (last-layer hidden) and `model.llm.lm_head`
(logits). `src/decode.py::chat_with_signals`. Signals therefore come from the
model's real generation; the returned text IS the model's answer.

Per forward pass we grab the last-position hidden + last-position logits. Forward
0 = prefill → `h_prompt` (whole-prompt representation) + the 1st-token
distribution; forwards 1..K = per-step. Store first-K scalar signals
(entropy/margin) + `h_prompt` [4096] + `h_mean8` (mean of first ≤8 step hiddens).

`modal run interactive_paper/modal_app.py::signal_check`:

- **Faithfulness**: hooked text **identical** to plain `chat()` output → hooks
  don't perturb generation. ✓
- **Overhead**: reported −28% (hooked 49.9 vs plain 35.9 tok/s) — this is
  warmup-order noise (the plain call ran first and paid CUDA kernel warmup). The
  real conclusion is the hooks add **no meaningful slowdown** (a `.float().cpu()`
  copy + a float32 softmax over the vocab, only for the first 16 steps). Well
  inside the < 30 % gate. ✓
- **Sanity** (entropy in nats, first-K steps):

  | query | n_forward | mean entropy@16 | entropy[0] |
  |-------|----------:|----------------:|-----------:|
  | easy ("capital of France?") | 9 | 0.185 | 0.54 |
  | hard (1-D well ⟨x²⟩ derivation) | 17 | **0.885** | **3.86** |

  Hard entropy ≈ 4.8× easy; the first generated token is already strongly
  discriminative (0.54 vs 3.86). Direction matches intuition — wiring is correct. ✓

**Verdict: GO.** Signal capture is faithful, cheap, and directionally sane.

---

## Phase 2 — calibration dataset + discriminative analysis (⭐ GO/NO-GO GATE)

**Status: ✅ COMPLETE (2026-07-08) — verdict GO, see below.** (Was paused at
the secret blocker; resolved by the OpenAI switch + `openai` secret.)

Pipeline (all in `modal_app.py`, pools/formatting in `src/queries.py`, Claude in
`src/escalate.py`):

1. `build_public_queries` (CPU, **no secret**) — GSM8K tail + MATH-500 (hard-math
   150), MMLU-Pro (+GPQA if reachable) (hard-knowledge 150), TriviaQA (easy-fact
   100) → `queries_public.jsonl`.
2. `build_claude_queries` (CPU, **needs secret**) — easy-chat 150 (中英各半) +
   trap 50 via `claude-opus-4-8` → `queries_claude.jsonl`.
3. `finalize_queries` — merge, assign `id`, stratified 60/40 calib/test split
   (seed 42) → `queries.jsonl`. **Run once over all 600** (ids are sequential, so
   re-finalizing would renumber and invalidate any already-collected signals).
4. `run_signals` (H100 ×4 shards) — `chat_with_signals` per query → per-shard
   parquet with answer + entropy/margin/h_prompt/h_mean8.
5. `label` (CPU, **needs secret**) — `claude-opus-4-8` judge → `adequate`;
   `escalate_label = 1 - adequate` → `calib_features.parquet`.
6. `calibrate` (CPU, sklearn) — ROC-AUC per signal (scalar entropy/margin @K∈{4,8,16},
   logistic probes on `h_prompt` / `h_mean8`, combined LR), per-pool AUC
   (the trap-pool "probe beats entropy" story), `roc.png`, and the go/no-go verdict.

### GOTCHA — Modal resolves `Secret.from_name` for *every* function in an app

First `build_public_queries` run failed with **"Secret 'anthropic' not found"**
even though it makes no LLM call: Modal hydrates the secrets of *all*
`@app.function`s when you run *any* function in that app. Fix: put the two
LLM-dependent functions on a **separate app object** (`gen_app =
modal.App("think-gate-gen")`) in the same file — `modal run …::build_public_queries`
then hydrates only `app` and skips the `openai` secret. (Creating even a
placeholder secret is intentionally gated behind the user in auto mode.)

### BLOCKER (unchanged) — the user must create the real secret

```
modal secret create openai OPENAI_API_KEY=sk-...
```

After that, the gate is reached with:
`build_gen_queries` → `finalize_queries` → `run_signals` → `label` → `calibrate`.

### ⭐ FINAL VERDICT (2026-07-08, all-public rerun): **GO — best AUC 0.828**

**Rerun with 100% public datasets** (user decision: no LLM-generated eval
queries — the GPT-generated pools are gone). easy-chat = dolly-15k (75 en,
short no-context instructions) + shibing624/alpaca-zh (75 zh); trap =
**SimpleQA** (basicv8vc/SimpleQA, 50). 600 total, 360 calib / 240 test
(seed 42, ids re-frozen — this supersedes the 597-query run below).
Supervised end-to-end by `supervisor.sh` (log: `pipeline_watch.log`).

- signals: 600/600 on 4×H100, 0 failures. label: 600/600, 0 judge errors.
- Escalate rate by pool: **trap 1.000 (!)**, hard-knowledge .480, easy-fact
  .340, easy-chat .207, hard-math .187; overall **.358**.
- **SimpleQA fixed the trap pool** — MiniCPM failed ALL 50 (GPT-generated
  traps: only .102). Confirms the "use public benchmarks" call. Side effect:
  single-class pool → within-trap AUC undefined; the "probe rescues entropy
  on traps" story is still untestable within-pool (would need a trap set with
  some successes — e.g. PopQA stratified by popularity, for later).

ROC-AUC (600, 5-fold CV):

| signal | AUC (vs 597-run) |
|---|---|
| **probe_h_prompt** | **0.828** (0.812) |
| combined | 0.823 (0.790) |
| probe_h_mean8 | 0.776 (0.805) |
| max_entropy@4 (best scalar) | 0.696 (0.624) |

Conclusions carried / updated:
1. Probe-on-h_prompt remains the signal; conclusion ROBUST to the dataset
   swap (0.812 → 0.828). Entropy stays far behind (best scalar 0.696).
2. h_prompt now clearly beats h_mean8 (0.828 vs 0.776) — pre-decode gating
   (fire before the first token) is the design for Phase 3.
3. Per-pool: probe wins hard-math (.891) and hard-knowledge (.779); entropy
   wins easy-fact (.845 vs .762) and easy-chat (.644 vs .535 — chat failures
   are near-invisible to the probe; caveat for the paper).
4. combined (0.823) no longer hurts vs probe (0.828) but adds nothing —
   Phase-3 gate stays probe-only.

Cost of the rerun: ~$6 GPU + ~$2.5 API. Stopped at the gate per plan.

### Overfit audit (2026-07-08, user challenge — calib-only, test untouched)

User asked whether the probe overfits to our pool. `modal_app.py::audit`
(CPU, calib 360 only — also fixes a process slip: `calibrate` had been doing
CV over all 600 incl. test; the plan-compliant headline is calib-only):

| check | result |
|---|---|
| in-sample AUC | 1.000 (p≫n memorizes, as expected) |
| **calib-only 5-fold CV** | **0.821** (headline barely moves vs 0.828 full) |
| **pool-identity-only oracle** | **0.715** — pool membership alone buys most of the aggregate AUC |
| LOPO easy-chat / easy-fact | 0.704 / 0.717 (transfers OK) |
| LOPO hard-knowledge | 0.606 (weak) |
| **LOPO hard-math** | **0.372 — WORSE than chance; inverts** |
| **LOPO trap** | mean escalate-score **0.232** on 100%-fail questions — would NOT escalate them |

**Verdict: the user's suspicion is substantially correct.** The probe is
honest *within* the calibration distribution (0.821 out-of-fold), but a large
share of the aggregate number is composition shortcut (oracle 0.715), and it
does **not** transfer to unseen query types: trained without math it actively
misranks math failures (0.372), and trained without traps it scores
looks-simple-but-fatal SimpleQA questions as LOW-risk (0.232) — exactly the
"reads surface familiarity, not self-knowledge" failure mode.

Implications recorded for the paper & Phase 3:
- Claim must be scoped: "zero-training gate calibrated on a deployment-like
  query mix", NOT "universal difficulty detector". The LOPO table itself is
  an honest & interesting result (probe generalization is distribution-bound).
- Phase-3 gate remains viable for in-distribution use; deployment requires
  the calibration mix to resemble traffic (or periodic recalibration).
- Candidate stronger baselines to try later: verbalized self-eval / p(True)
  (also zero-training, may transfer better), per-pool calibration, more
  diverse calibration pools.

---

### [SUPERSEDED] first run (597, with GPT-generated pools): GO — AUC 0.812

Full Phase-2 execution (post provider switch, all five steps in one session):

- `build_gen_queries` ✅ — **197** generated (easy-chat 148, trap 49;
  gpt-5.4-mini under-delivered 2+1 vs the requested 150+50 — accepted).
  Validated key + model id + Structured Outputs in one shot (~$0.50).
- `finalize_queries` ✅ — **597 total** (hard-math 150, hard-knowledge 150,
  easy-fact 100, easy-chat 148, trap 49), split **358 calib / 239 test**
  (seed 42). Ids `q0000…` now FROZEN — do not re-run finalize.
- `run_signals` ✅ — 4× H100, 597/597 rows, 0 failures →
  `signals.shard{0..3}.parquet`.
- `label` ✅ — gpt-5.4-mini judged 597/597, **0 judge errors**.
  Escalate rate by pool: hard-knowledge .500, easy-fact .350, hard-math .187,
  easy-chat .128, **trap .102**; overall **.271**.
- `calibrate` ✅ → `roc.png` (pulled to `figures/roc.png`).

ROC-AUC (predicting small-model failure), full 597:

| signal | AUC |
|---|---|
| **probe_h_prompt** | **0.812** |
| probe_h_mean8 | 0.805 |
| combined (entropy+margin+probe LR) | 0.790 |
| max_entropy@16 (best scalar) | 0.626 |
| mean_margin@16 (worst) | 0.490 |

Findings vs expectations:

1. **The linear probe on the prompt representation is the whole story** —
   0.812 vs 0.626 for the best scalar. Entropy/margin are far weaker than
   hoped; the *combined* LR (0.790) is actually WORSE than the probe alone,
   i.e. the scalars add noise, not signal. Phase 3's gate score should be
   probe-only (one dot product — still zero-cost online).
2. **h_prompt ≈ h_mean8** (0.812 vs 0.805): the prompt's last hidden state
   already carries the difficulty signal — decode-time hiddens add nothing.
   Big practical win: the gate can fire **before the first token is decoded**.
3. **The trap pool failed its design** (escalate rate 0.102, expected high):
   gpt-5.4-mini's "confident-but-wrong" traps were mostly answerable by
   MiniCPM. With ~5 positives in n=49, its per-pool AUCs (entropy .855 /
   probe .555) are noise — the planned "probe rescues entropy on traps"
   narrative is UNSUPPORTED on this data. Either regenerate traps harder
   (stronger gen model / verify small-model-fails before accepting) or drop
   that storyline.
4. Per-pool (where n gives signal): probe > entropy on easy-chat
   (.769/.593) and hard-knowledge (.749/.673); entropy > probe on easy-fact
   (.859/.692); tie on hard-math (~.86 both). Entropy is good exactly where
   failures are knowledge-retrieval flavored; the probe is more uniform.

Per PLAN discipline #5: **stopped here and reported to the user.** Phase 3
(threshold gate) is unblocked on a GO.

## Phase 3 — online threshold gate ✅ (2026-07-09)

**Goal**: turn the Phase-2 probe into a real-time trigger + pick deployable
thresholds. Done entirely on **CPU** — every `h_prompt` is already in
`calib_features.parquet`, so the online score is a deterministic replay; no GPU
decode was spent to validate the trigger logic.

### Design decisions (both forced by the data, deviating from PLAN §Phase-3)

1. **Pre-decode single-shot, not streaming EMA.** PLAN designed the gate as
   EMA + k-consecutive hysteresis over per-step scores — sensible for a scalar
   that evolves during decode (entropy/margin). But Phase 2 found the winning
   signal is the probe on `h_prompt`, a **single score available at prefill**
   (h_prompt 0.828 > decode-time h_mean8 0.776). So the headline gate fires
   **before the first token** from one score. `src/gate.py::EscalationGate` still
   implements the full EMA/hysteresis/cooldown machinery (pure-Python, unit-tested,
   needed for the duplex Phase 6); the headline runs it in single-shot mode
   (`k_consecutive=1, ema_alpha=1.0`), where it degenerates to `score >= threshold`.
2. **Tiers by escalation BUDGET, not precision target.** At base rate 0.322 and
   AUC 0.83, PLAN's "precision >= 0.80 default" is only reachable at ~0 recall
   (degenerate — first two attempts pinned all thresholds to 1.0). Escalation rate
   is the real cost knob and the exact axis Phase 5 sweeps, so tiers are set at
   target escalate rates {conservative .15 / balanced .30 / aggressive .50}.

### Overfit-aware threshold calibration (the non-obvious part)

The shipped probe is fit on all 360 calib rows, but with 4096 dims / n=360 the
data is **linearly separable → in-sample AUC 1.000 at every C** (regularization
shrinks score magnitudes, not in-sample ranking). Picking thresholds on in-sample
scores is therefore meaningless. Fixes in `modal_app.py::fit_gate`:

- **thresholds live on 5-fold OOF scores** (seed 42), which reflect deployment
  generalization (~0.83), not the memorized 1.0.
- **C-regularization sweep** picks C by OOF AUC (tie → smaller C):
  C=0.001 (OOF **0.828**) > 0.01 (.822) > 1.0 (.821) > 0.1 (.818). Heavy L2 both
  maximizes OOF AUC *and* compresses the shipped probe's score scale so an
  OOF-quantile threshold transfers to it.

`gate_config.json` (pulled to repo, 4096-float probe + 3 thresholds) is the
artifact Phases 4–5 load. `src/gate.py` = `Probe` (sigmoid(w·h+b), one dot
product) + `EscalationGate`; `src/test_gate.py` = 28 pure-Python checks (pass).

### Realized operating points on calib (OOF scores, `gate_eval`)

| tier | thr | escalate | precision | recall |
|------|----:|---------:|----------:|-------:|
| conservative | 0.933 | 0.150 | 0.722 | 0.336 |
| balanced     | 0.475 | 0.300 | 0.657 | 0.612 |
| aggressive   | 0.070 | 0.500 | 0.539 | 0.836 |

Per-pool trigger rate (conservative / balanced / aggressive):

| pool | esc-rate | cons | bal | aggr |
|------|---------:|-----:|----:|-----:|
| **trap** (SimpleQA, 100% fail) | 1.00 | **0.80** | 1.00 | 1.00 |
| hard-knowledge | 0.40 | 0.18 | 0.42 | 0.73 |
| easy-fact | 0.23 | 0.08 | 0.22 | 0.43 |
| hard-math | 0.22 | 0.09 | 0.20 | 0.32 |
| easy-chat | 0.18 | 0.01 | 0.10 | 0.32 |

Reads exactly as hoped: even the **conservative** tier catches **80% of the
100%-fail trap questions** while false-triggering easy-chat only 1%. Trigger rate
tracks pool failure rate everywhere **except hard-math** (under-caught: 0.09/0.20/
0.32, below its difficulty) — the same "probe reads knowledge-difficulty better
than math-difficulty" weakness the Phase-2 LOPO audit exposed, now visible
in-distribution too. Caveat carries to the paper.

**Validation**: `EscalationGate.from_config` (single-shot) reproduced the
`score >= threshold` decision on all 360 calib rows → deployment trigger logic
is correct. Phase-3 cost ≈ $0 (3 short CPU runs).

**Next (Phase 4)**: escalation chain E2E — trigger → distilled query → GPT-5.5 →
inject/paraphrase. `chat_gated` (live pre-decode stop on the H100) is deferred to
Phase 4, where the escalation chain actually consumes the trigger; Phase 3's CPU
replay already validates the gate numerically.

---

## Phase 4 — escalation chain E2E ✅ (2026-07-09)

**Goal**: trigger → distilled query → gpt-5.5 → inject/paraphrase, end-to-end.
Built `src/distill.py` (`distill_query`), `src/inject.py` (`paraphrase`),
`src/escalate.py::ask_expert`/`ask_expert_many` (gpt-5.5, error-safe, token-usage
capture), and `decode.generate`. `modal_app.py::e2e_demo` runs the full chain on
hard test queries and prints a readable trace.

Verified on the trace (PLAN Phase-4 go/no-go): **distilled queries are faithful**
(single-turn queries are already standalone, so distillation is near-identity —
its real payoff is multi-turn/duplex, Phase 6; the Phase-5 eval therefore
escalates the *original* query) and **the paraphrase relays the expert answer
accurately**. Example — a Sn(gray→white) equilibrium-temperature question: small
model went down the wrong equation; gate fired (score 0.988); gpt-5.5 returned the
reference `C. −3.5 °C`; small model paraphrased it faithfully.

**Gotcha**: gpt-5.x hidden reasoning bills as output and can consume the whole
`max_completion_tokens` before the visible answer (empty content,
`finish_reason=length`). Raised the expert cap 4096→**8192** and made `ask_expert`
error-safe (returns `error` instead of raising). At 8192 the full 240-query eval
hit **0 truncation errors**.

---

## Phase 5 — system evaluation ✅ (2026-07-09) ⭐ RQ2 answered

**Goal**: the accuracy-vs-escalation-rate tradeoff on the frozen **test split**
(240). Four conditions; judge = `gpt-5.4-mini`, blind to source.

Pipeline: `eval_expert` (gpt-5.5 answers all 240, judged → big-only) →
`eval_paraphrase` (small model relays each expert answer, judged → hybrid outcome)
→ `eval_assemble` (probe-scores the stored test `h_prompt`, sweeps the gate
threshold to draw the curve, compares to random escalation, computes latency/cost,
writes `figures/tradeoff.png`). Small-only answers/labels were already in
`calib_features.parquet` from Phase 2 (no recompute).

### Headline numbers (test, n=240)

| condition | accuracy | escalation |
|---|---:|---:|
| small-only (MiniCPM-o 4.5) | **0.588** | 0% |
| big-only (gpt-5.5, raw) | **0.917** | 100% |
| big-only relayed via small (paraphrase) | 0.879 | 100% |
| hybrid-gate **conservative** | 0.679 / 0.671¹ | 14.2% |
| hybrid-gate **balanced** | 0.779 / 0.767¹ | 32.9% |
| hybrid-gate **aggressive** | 0.833 / 0.821¹ | 52.9% |

¹ accuracy as *expert-inject* / *small-model-paraphrase*.

### Key results

1. **The gate beats random escalation at every operating point** (the central
   RQ2 claim). Area between the gate curve and the random-escalation line
   (∫(acc_gate − acc_rand) d rate) = **+0.054**. Concretely at 33% escalation the
   gate reaches **0.779** vs random's ~0.696 (+8.3 pts); +4.4 pts at 14%, +7.1 at
   53%. `figures/tradeoff.png` — both gate curves bow well above the diagonal and
   rise steeply early (the gate escalates the highest-risk queries first).
2. **Escalating 33% of traffic recovers ~58% of the small→big accuracy gap**
   (0.588→0.779 of the 0.588→0.917 span). The gate buys most of the big model's
   accuracy at a third of its cost/latency.
3. **Paraphrase (relay) tax ≈ 1–4 pts**: routing the expert answer back through
   the small model costs 0.917→0.879 at full escalation (and ~1 pt per tier). It's
   the price of natural spoken relay; a deployment that can surface the expert
   answer directly avoids it.
4. **big-only is not a ceiling of 1.0**: gpt-5.5 scores 0.917 overall — perfect on
   hard-math (1.00) but only **0.65 on trap** (SimpleQA long-tail facts stump even
   the big model). So the trap pool caps how much *any* escalation can help there.

### Latency & cost

- Latency (s): expert gpt-5.5 **P50 3.0 / P95 24.4**; small-model paraphrase
  **P50 1.0 / P95 5.8**. Small-only decode ≈ 33 tok/s (Phase 0). The escalation
  chain's latency is dominated by the expert call.
- Cost: gpt-5.5 big-only = **$1.12 / 100 queries** (30.5k in + 84.4k out tokens
  over 240, at $5/$30 per M). Hybrid scales ~linearly with escalation rate, so
  balanced ≈ $0.37 / 100q for the expert calls.

**Phase-5 spend** ≈ $3 API (240 expert + ~480 judge) + ~$3 GPU (paraphrase +
demo). Total project spend to date ≈ **$32**.

### Caveats carried to the paper

- Accuracy figures are single-run greedy (seed 42); no judge-variance bars.
- The gate curve is drawn by thresholding stored test `h_prompt` scores — a true
  online run would recompute the identical score at prefill (Phase-3 established
  the deployment scorer reproduces it). `chat_gated`'s live decode-stop was not
  needed for the accuracy eval and remains unimplemented (a latency optimization).
- hard-math is under-escalated by the gate (Phase 2/3 weakness), yet gpt-5.5 would
  answer those perfectly — the biggest missed opportunity the gate leaves on the
  table. A math-aware signal is the clearest follow-up.

### Pool-oracle baseline added to the figure (2026-07-29, grilling session)

The figure's only opponent was random escalation — the weakest possible —
while the Phase-2 audit already showed pool identity alone buys AUC 0.715
of 0.821. `eval_assemble` now draws the system-level version: a
**pool-oracle router** (score = the query's pool CALIB fail rate; true type
labels, no instance information, no test leakage; ties within a pool =
straight segments between pool-boundary points). Result:

- oracle-vs-random area **+0.042** vs the gate's **+0.054** → the internal
  signal's in-distribution residual is **+0.012 area (~22%)** — the
  accuracy-curve counterpart of the AUC decomposition, and slightly
  harsher (AUC said ~1/3 was instance-level).
- calib pool fail rates driving the oracle: trap 1.00 > hard-knowledge .40
  > easy-fact .23 > hard-math .22 > easy-chat .18.
- `figures/tradeoff.png` regenerated (purple line hugs the gate curve);
  paper `system.tex` text + caption updated same session. The paper claim
  now rests explicitly on transfer (LOPO), not the in-distribution margin.
- Open: is +0.012 even significant at n=240? Folded into the
  statistics-hardening todo (paired bootstrap gate-vs-oracle).

---

## Phase 5b — audit deep-dive + p(True) baseline ⭐ (2026-07-14)

User unblocked budget ("$2000 Modal credit, use at your own discretion").
Three experiments closing the report's biggest holes. All numbers below.

### audit2 — math-inversion root cause + evidence-chain closure + CIs (CPU, ~$0)

`modal_app.py::audit2`, calib-only (360), test untouched except pre-stored
Phase-5 outcomes for CIs.

**[A] Math LOPO inversion root-caused.** Within math calib (n=90): gsm8k
fail=0.04, math500 fail=0.55 — failure ≈ "is it MATH-500" (corr +0.592). The
LOPO probe (trained without math) scores MATH-500 *lower* risk than GSM8K
(corr(score, is_math500) = −0.246): trained on knowledge/fact/chat/trap, it
reads terse symbolic competition problems as "safe" and verbose GSM8K stories
as riskier. Not length-driven (corr ≈ 0). Within-source LOPO AUC: math500
0.704 (OK!), gsm8k 0.455 — the inversion is mostly a *between-source* ranking
error. Confirms: the probe reads surface style, not solve-ability.

**[B] Evidence chain closed: h_prompt ⊇ pool identity.** 5-way pool classifier
on h_prompt: **95.8%** 5-fold accuracy. Dummies-only OOF AUC 0.678; h_prompt
0.822; **dummies+h_prompt 0.821 — adding explicit pool identity to the probe
adds NOTHING**, i.e. h_prompt already contains the full type shortcut.
Within-pool AUC of the probe's OOF scores (type shortcut controlled):
easy-chat .693 / easy-fact .793 / hard-knowledge .634 / hard-math .847,
macro-mean **0.742** vs aggregate 0.822.

**[C] Bootstrap 95% CIs (2000 resamples).** calib OOF AUC 0.822 [.777, .863].
Test headline: small 0.588 [.525, .650], big 0.917 [.879, .950], paraphrase
0.879 [.833, .917]; hybrid cons/bal/aggr 0.679 [.617, .733] / 0.779 [.725,
.829] / 0.833 [.783, .879]. **Gate-vs-random area +0.0541 [+0.0399, +0.0677]**
— significantly > 0. (Judge variance still unbounded — single judge.)

### ⭐ p(True) verbalized self-eval — the probe was reading the wrong signal

`collect_ptrue` (4×H100, ~$1): two zero-training, zero-calibration signals per
query, both = P(Yes) read off the first-token logits (Yes/No token mass;
median mass 1.0, so the read is clean):
- **ptrue_pre**: "Would you answer this correctly?" *before* answering.
- **ptrue_post**: "Is this proposed answer correct?" given its stored answer.

`ptrue_analyze` (calib rows for probe comparability):

| signal | AUC | needs training? |
|---|---:|---|
| **ptrue_post** | **0.899** | no |
| pre+post mean | 0.877 | no |
| probe OOF | 0.821 | probe fit on calib |
| **ptrue_pre** | 0.807 | no |

Per-pool (the transfer test — p(True) has no calibration pool, so this IS its
LOPO): easy-chat .708/.812 (pre/post), easy-fact .764/.855, hard-knowledge
.688/.797, **hard-math .809/.944 (probe LOPO: 0.377)**. No inversion anywhere.

**Trap (100%-fail SimpleQA): ptrue_pre mean score 0.945** vs 0.309 non-trap —
100% of traps rank above the non-trap P70. The probe-LOPO scored them 0.328
(would not escalate). **The model KNOWS it doesn't know these long-tail facts
— you just have to ask it before it answers.** ptrue_post degrades on traps
(0.360): after generating a confident-wrong answer the model starts believing
it. Introspect BEFORE answering.

### p(True) tradeoff on frozen test (n=240) — better AUC → better curve

`ptrue_gate_eval` (CPU; thresholds = calib score quantiles, no fitting):

| signal | area vs random | acc @ ~30% esc | thr transfer calib→test |
|---|---:|---:|---|
| probe | +0.0541 | 0.667 (esc 0.12!) | poor (rates drift) |
| ptrue_pre | +0.0594 | 0.779 (esc 0.36) | good |
| **ptrue_post** | **+0.0682** | **0.821 (esc 0.38)** | good |

`figures/tradeoff_ptrue.png`. Deployment note: ptrue_pre costs one extra short
prefill pre-decode (fits the streaming design); ptrue_post needs the full
small answer first (draft-then-check — natural for the System-1/2 loop, and
what step-2 injection consumes anyway).

**Step-1 conclusion REVISED**: genuine zero-training self-knowledge EXISTS and
is verbalizable; the h_prompt linear probe was reading query-type familiarity
instead. The gate should be built on verbalized self-eval (pre-answer for
pre-decode firing, post-answer check where latency allows), with the probe as
a free auxiliary at best.

### Cross-backbone replication (in flight)

`run_signals_hf` / `label_hf` / `run_ptrue_hf` / `xmodel_report` added
(`src/hf_decode.py` = vanilla-HF hook mirror of decode.py). Backbones:
qwen3-8b (MiniCPM's family, raw; thinking disabled) + mistral-7b-instruct-v0.3
(different family). Judge = same gpt-5.4-mini rubric; same 600 queries.

**qwen3-8b (600/600 signals, 0 judge errors).** Fail rates: chat .387,
fact .470, knowledge .507, math .180, trap .980 (49/50 — one success, so trap
AUC computable). Raw no-think Qwen3-8B fails much MORE on the easy pools than
MiniCPM (.387 vs .207 chat) — omni fine-tune + system prompt differences.

| check | MiniCPM-o 4.5 | qwen3-8b | replicates? |
|---|---:|---:|---|
| probe OOF AUC (calib) | 0.822 | **0.838** | ✅ |
| pool-oracle AUC | 0.715 | 0.704 | ✅ type shortcut |
| pool-classifier acc | 0.958 | 0.958 | ✅ |
| max_entropy@4 AUC | 0.696 | **0.468 (useless)** | ❌ entropy is fragile |
| LOPO hard-math | **0.372 (inverts)** | **0.961 (fine!)** | ❌ **does NOT replicate** |
| LOPO trap | score 0.23 (miss) | AUC 0.966 | ❌ |
| LOPO chat/fact/knowledge | .70/.72/.61 | .78/.60/.73 | ~ |
| ptrue_post AUC | 0.899 | **0.897** | ✅ almost exactly |
| ptrue_pre AUC | 0.807 | 0.736 (knowledge .563 weak) | ~ |
| ptrue_pre on trap | 0.945 mean score | **AUC 0.939** | ✅ knows-it-doesn't-know |

**The headline surprise: the LOPO transfer failure is MiniCPM-SPECIFIC.** On
raw Qwen3-8B the h_prompt probe transfers fine across pools (math 0.961, trap
0.966) — no inversion anywhere. So "linear probes can't transfer across query
types" is NOT a universal law; MiniCPM's omni fine-tuning (or its different
failure profile — qwen fails 37–51% on every non-math pool, giving LOPO
training much broader positive coverage) restructures what the probe can read.
Caveat: the two models' label distributions differ a lot, so representation
vs. label-coverage explanations are confounded — resolved by `lopo_matched`
below.

**`lopo_matched` — confound resolved: it's the REPRESENTATION.** Subsampling
qwen's LOPO training pools to MiniCPM's exact per-pool fail rates (chat .21,
fact .34, knowledge .48, trap 1.0; matched train n=246) leaves qwen's
LOPO-math at **0.962–0.968 over 5 subsample seeds** (unmatched 0.961). Label
coverage is ruled out; raw Qwen3-8B's h_prompt linearly encodes transferable
difficulty that MiniCPM-o's (same architecture, omni fine-tuned) does not.

**What replicates cleanly: p(True).** ptrue_post ≈ 0.90 on BOTH backbones, and
pre-answer trap introspection holds (0.939 AUC) — the "model knows it doesn't
know long-tail facts if you ask before it answers" finding is now 2-for-2.
Entropy, meanwhile, collapsed to chance on qwen (0.468) — scalar uncertainty
is the least portable signal of all.

**mistral-7b (600/600, 0 judge errors).** Much weaker model (fail .519
overall; knowledge .711, math .556). Download fought back (unauthenticated
Xet stall → partial snapshot → 403 on `consolidated.safetensors` → missing
sentencepiece; fixes: HF_HUB_DISABLE_XET=1, no config.json short-circuit,
ignore `consolidated*`, sentencepiece in the GPU image).

| check | MiniCPM-o | qwen3-8b | mistral-7b |
|---|---:|---:|---:|
| probe OOF AUC | 0.822 | 0.838 | 0.758 |
| pool-oracle AUC | 0.715 | 0.704 | **0.730 (probe adds only +.03!)** |
| pool-classifier acc | 0.958 | 0.958 | 0.953 |
| max_entropy@4 | 0.696 | 0.468 | 0.501 |
| LOPO math / fact | .372 / .717 | .961 / .596 | .817 / **.445** |
| ptrue_post AUC | **0.899** | **0.897** | **0.814** |
| ptrue_pre AUC | 0.807 | 0.736 | 0.723 |
| ptrue on trap | pre .945 score | pre .939 AUC | post .901 AUC |

### Cross-backbone synthesis (3 models)

1. **ptrue_post is the most portable signal**: 0.899/0.897/0.814 — beats the
   trained probe on ALL THREE backbones, with zero training/calibration.
2. **The type shortcut is universal**: h_prompt encodes pool identity at ~95%
   on all three; oracle AUC 0.70–0.73. On mistral the probe's aggregate edge
   over the oracle is a mere +0.028 — the probe ≈ a type classifier there.
3. **Probe transferability is a property of the BACKBONE, not the method**:
   LOPO transfer is fine on qwen (all ≥ .60, math .96), partial on mistral
   (fact .445 inverts), catastrophic on MiniCPM (math .372, trap missed).
   `lopo_matched` rules out label coverage — it's the representation.
4. **Entropy is the least portable signal**: ≈ chance on 2 of 3 backbones.
5. **Pre-answer trap introspection holds on all three** — "the model knows it
   doesn't know long-tail facts if asked before answering" is now 3-for-3.

Phase-5b total spend ≈ $12 GPU + $6 API. Project total ≈ **$50**.

---

## Phase 5c — duplex-generalization matched pairs ⭐ (2026-07-17)

**Framing correction (user):** this project is FOR full-duplex models —
qwen3-8b/mistral-7b (and every model below) are **controls for the RQ1 signal
findings, never system baselines**. Question: does "the omni fine-tune
destroys probe transferability" generalize across duplex models, or is it
MiniCPM-specific? Design: matched pairs — one raw backbone vs its own
audio/omni/duplex fine-tunes, so each pair is its own control.

**Models** (all ungated, downloaded to the weights volume): Qwen2.5-7B family
= raw `qwen2.5-7b` vs `qwen2.5-omni-7b` (streaming talker–thinker omni FT) vs
`minicpm-o26` (true duplex FT, same Qwen2.5-7B base). Second pair:
`qwen2-7b` raw vs `qwen2-audio-7b` (audio-understanding FT, no duplex).
Second family raw: `glm4-9b-chat-hf`. Rejected: Moshi (no raw counterpart,
can't answer text queries); deferred: Kimi-Audio (dual-stream forward breaks
vanilla generate), GLM-4-Voice (ChatGLM custom layout + likely capability
confound, see below). New plumbing: `omni_image` (transformers 4.57.6) +
`run_signals_omni`/`run_ptrue_omni` (Qwen2.5-Omni Thinker, Qwen2-Audio
`.language_model`), `run_signals_mo`/`run_ptrue_mo` (o2.6 via decode.py;
needs pre-seeding the transformers_modules cache — 4.51's copier misses
image_processing_minicpmv.py), `lopo_matched2` (generalized fail-rate
matching, both directions).

### Headline: LOPO transfer degrades with duplex-ness of the fine-tune

| LOPO (h_prompt probe) | qwen2.5-7b raw | qwen2.5-omni | minicpm-o26 | [o4.5, Qwen3 base] |
|---|---:|---:|---:|---:|
| hard-math | **0.809** | 0.745 | **0.538 (≈chance)** | 0.372 (inverts) |
| hard-knowledge | 0.680 | 0.613 | **0.526 (≈chance)** | 0.61 |
| probe OOF AUC | 0.798 | 0.724 | 0.752 | 0.822 |
| ptrue pre/post | .744/.857 | .749/.703 | .604/.777 | .807/.899 |

Gradient on ONE backbone: raw > omni-streaming > duplex. It is not the audio
modality per se — the closer the fine-tune is to full-duplex training, the
more transferable difficulty info is washed out of h_prompt.

**`lopo_matched2` deconfound (CPU, ~$0):** subsampling raw qwen2.5-7b's LOPO
training pools to o2.6's exact fail rates leaves math at **0.825 [.801,.840]**
(unmatched 0.809; o2.6 actual 0.538); knowledge 0.667 vs o2.6's 0.526. Label
coverage is ruled out on a SECOND backbone — representation damage is now
deconfounded 2-for-2 (Qwen3 pair in 5b, Qwen2.5 pair here).

### Honest complications

1. **Raw baselines vary a lot** (LOPO math: qwen2 .552, glm4 .670, qwen2.5
   .809, mistral .817, qwen3 .961) — the defensible statistic is the
   within-pair Δ, not absolute AUC. "Inversion" (<0.5) remains duplex-only;
   raw models are 5-for-5 non-inverted (mistral fact .445 the one exception).
2. **Qwen2-Audio pair is confounded and uninformative**: the audio FT crushed
   capability itself (math fail .256→.744, knowledge .60→.79), so the failure
   distribution changed under the probe (math LOPO .552→.672, chat .777→.533;
   matched rerun unstable [.34,.61]). Capability collapse ≈ floor effect —
   documented as a negative, not evidence either way. GLM-4-Voice was skipped
   for the same expected confound + adaptation cost.
3. **GLM-4 breaks the p(True) streak**: glm4-9b-chat-hf ptrue_post 0.685 <
   probe 0.798 (was 4-for-4 the other way). "p(True) beats the probe" softens
   to "on most backbones; it depends on self-eval calibration quality."
   Also qwen2.5-omni is the first model with ptrue pre (.749) > post (.703),
   and o2.6 inverts o4.5's trap pattern (pre .604 weak, post .875 strong).

**Step-1 narrative upgrade:** for the paper's target audience (duplex-model
builders) this is a directly actionable caution — hidden-state probes that
work on a raw backbone degrade to chance after duplex fine-tuning (matched-
pair, label-matched evidence on 2 backbones), while verbalized self-eval,
though also dented, stays usable. Gate design conclusion unchanged: behavior-
level signals (p(True)) over representation-level probes for duplex targets.

Phase-5c spend ≈ $25 GPU + $8 API. Project total ≈ **$85**.

---

## Phase 5d — layer × position sweep: destroyed vs relocated ⭐ (2026-07-20)

5c established that the (last layer, last prompt token) probe's transfer
degrades with duplex-ness, but read only that ONE point of the network. Rival
explanations: (a) **destroyed** — duplex training washes difficulty info out
of the model; (b) **relocated** — duplex training repurposes the late-layer /
last-token readout (streaming turn control lives there) and the info survives
elsewhere. User hypothesis going in: (b).

**Method:** prefill-only forward per query (no generation, labels reused from
the 5b/5c judge runs → $0 API), hooks on EVERY decoder layer capturing both
the last-prompt-token hidden and the mean over all prompt positions
(`src/layers.py`, `collect_layers_{hf,omni,mo}`, `layer_sweep_report`,
float16 npz on the volume). Per layer × pooling: OOF AUC + LOPO, calib rows,
same estimators as `xmodel_report`. **Faithfulness check passed:** final-layer
numbers reproduce 5c's generation-time hooks exactly (qwen2.5-7b math .809=.809,
o2.6 .540≈.538, o4.5 .366≈.372, omni .746≈.745, qwen3 .958≈.961).

### Headline: (b) relocated — more precisely, OVERWRITTEN AT THE READOUT

LOPO hard-math, last-token pooling:

| model | best mid-layer | final layer | shape |
|---|---:|---:|---|
| qwen3-8b (raw) | 0.964 (L33/36) | 0.958 | plateau L10→end, no cliff |
| **minicpm-o45 (duplex)** | **0.931 (L22/36)** | **0.366** | cliff in last 4 layers, INVERTS |
| qwen2.5-7b (raw) | 0.893 (L21/28) | 0.809 | mild late dip |
| qwen2.5-omni (streaming) | 0.794 (L16/28) | 0.746 | whole curve depressed, no cliff |
| **minicpm-o26 (duplex)** | **0.822 (L21/28)** | **0.540** | cliff in last ~5 layers |

- **o4.5's famous math inversion (.372) is a readout artifact.** Mid-network,
  the duplex model carries near-raw transferable difficulty info (within-pair
  Δ at best layer −0.03; at final layer −0.59). The collapse is sharply
  localized: L31 .757 → L32 .654 → L33 .492 → L34 .357.
- **o2.6 same signature** (L21 .822 → L27 .540); knowledge likewise (L19 .748
  → L27 .526 ≈ chance).
- **Mean-pooling survives to the end** on the duplex models (o4.5 math ~.80
  at L35; o2.6 .674 at L27) → the damage is position-specific (last token)
  as well as depth-specific (late layers). Both raw models keep last-token
  transfer through the final layer, so the cliff is a fine-tune effect, not
  an architecture generic.
- **Fine-tune types differ in damage profile:** true duplex = severe but
  LOCAL (mid-layers intact, readout cliff/inversion); omni-streaming =
  mild but DIFFUSE (all layers depressed ~.1, no cliff). Also of note: even
  raw backbones peak mid-network, not at the readout (qwen2.5 .893 vs .809).
- in-mix OOF stays high at the duplex readout (o4.5 L35 .835) — consistent
  with 5b's "the readout still supports type recognition + in-distribution
  probing; it's the *transferable self-knowledge* component that's gone".

### Revisions to earlier conclusions

1. 5b's "probe ≈ query-type recognition, math inversion suggests no
   self-knowledge signal" → **the self-knowledge signal exists and is strong
   (.93 LOPO math), the standard readout just can't see it on duplex models.**
2. 5c's "duplex FT washes difficulty info out of h_prompt" → **"duplex FT
   overwrites the late-layer last-token readout; mid-network info is intact."**
   (Consistent with the mechanism: that readout is exactly where a streaming
   head must encode turn-control state.)
3. Gate design for duplex targets: representation probes are BACK on the
   table — read a mid-layer (~60% depth) instead of the final layer. p(True)
   remains the zero-plumbing option; the mid-layer probe is the zero-latency
   option (no extra forward). Two-stage design unchanged otherwise.

Figure: `figures/layer_sweep.png` (5 models × {last,mean} × {LOPO math,
LOPO knowledge, OOF}). Curves: `layer_sweep_{tag}.json` on the volume.

Phase-5d spend ≈ $8 GPU + $0 API. Project total ≈ **$93**.

---

## Phase 5e — mid-layer probe in the RQ2 tradeoff (2026-07-20)

Does 5d convert into a better gate? `midlayer_gate_eval` (CPU, $0): train a
probe on CALIB rows at a mid layer of o4.5 (layer chosen from 5d's calib-only
curves — L22 = calib LOPO-math peak; L18–L30 swept for sensitivity), score
the frozen test split, same curve/area protocol + the same rows as
`ptrue_gate_eval`. Deployed final-layer probe and p(True) re-evaluated
in-run as references.

| signal (test n=240) | area | stage |
|---|---:|---|
| probe_final (deployed cfg) | +0.054 | pre-decode |
| ptrue_pre | +0.059 | pre-decode |
| **midlayer_L22 (last-token)** | **+0.064** | **pre-decode** |
| ptrue_post | +0.068 | post-draft |

- **The mid-layer probe is the best PRE-DECODE signal** — beats both the
  deployed final-layer probe (+0.054→+0.064, closes ~68% of the gap to
  ptrue_post) and ptrue_pre. ptrue_post keeps the overall crown but needs
  the full draft answer first (latency = a whole generation) + an extra
  forward; the mid-layer probe costs literally nothing at prefill.
- Sensitivity: L20–L30 all ≥ +0.056 (L22 best; L18 +0.053) — not knife-edge.
  Mean-pooling uniformly worse in-mix (L22 +0.058) — mid-depth **last-token**
  is the sweet spot, matching 5d.
- Caveat: quantile-threshold transfer is still probe-weak (esc 0.12 realized
  at nominal .15; p(True) transfers rates much better) — a deployed mid-layer
  gate needs Phase-3-style score-scale calibration (C-compression). Area
  (threshold-free) is the headline metric here.
- Note the in-mix area ranking compresses the 5d story: test has the same
  pool mix as calib, where even the damaged readout scores +0.054 via type
  recognition. The mid-layer probe's LOPO robustness (math .93 vs .37) is
  the bigger deployment argument and doesn't show in this table.

**Two-stage gate design, final form:** stage 1 (prefill, free) = mid-layer
probe — now the best zero-latency signal on the duplex target; stage 2
(post-draft, optional) = ptrue_post draft-check. Figure:
`figures/tradeoff_midlayer.png`.

Phase-5e spend ≈ $0. Project total ≈ **$93**.

---

## Phase 6a — audio-input replication ⭐ (2026-07-20)

Does the signal stack survive when the SAME frozen 600-query pool enters
through o4.5's audio channel? Arm A = TTS matched pairs (user-approved:
OpenAI `tts-1`, voice `alloy`, en+zh, 0 truncated; query CONTENT unchanged —
public-benchmark pool, only the modality is synthetic, matching
Spoken-SQuAD/VoiceBench practice). Arm B (SD-QA real-speech validation) still
open. New code: `modal_audio.py` (tag `minicpm-o45-audio`, same file formats
as the text pipeline so `label_hf`/`layer_sweep_report` ran unchanged);
`audio_report` adds paired fail rates + per-layer cross-modal transfer.
Smoke: chat accepts raw 16 kHz numpy; **pure-audio content (no text
instruction) answers the question** (no transcription behavior) → collection
used `content=[audio]` only. Judge: 0 errors on 600.

### Headline 1: the 5d readout cliff is TEXT-INPUT-SPECIFIC

Audio→audio LOPO hard-math (last-token): mid-layers L12–L16 hit **.93–.96**
(peak L16 .960) — the self-knowledge signal is fully present on audio input.
But the final layer **does not invert on audio: L35 = .936** (text input:
L35 = .366). The duplex fine-tune's late-layer/last-token damage — the
central 5d finding — appears only when the prompt is text. On the model's
NATIVE modality the standard readout survives end-to-end. Mechanism note:
the audio chat template may append wrapper text tokens after the audio
(last-prompt-token position ≠ last audio token); un-audited — flagged as the
first follow-up before leaning on this in the paper.

### Headline 2: cross-modal probe transfer works (calibrate on text, deploy on speech)

Per-layer probes trained on TEXT calib rows, scored on AUDIO calib rows
(audio labels), and vice versa:

| region | text→audio | audio→text |
|---|---:|---:|
| early (L00–L08) | .54–.60 | .57–.71 |
| mid (L19–L25) | **.82–.87** (L23 .867) | .73–.86 (L23 .857) |
| late (L28–L35) | .84–.86 | .83–.86 |
| deployed layer L22 | **.855** | .797 |

Early layers are modality-specific; from ~55% depth the representation is
modality-shared and a text-calibrated probe reads audio hiddens at ~.86 AUC.
The cheap deployment recipe (calibrate on text data, run on speech) is
viable at the 5e gate layer.

### Headline 3: the modality tax, and p(True)'s trap introspection collapses

Paired fail rates (same queries, n=600): audio input costs +5 to +15 pts —
easy-chat .207→.307, easy-fact .340→.390, hard-knowledge .480→.627,
hard-math .187→.247, trap 1.00→.96. Audio p(True): pre AUC .786 (text .807),
post .860 (text .899) — mild degradation overall, BUT the 5b trap story does
not survive the modality: **trap mean p_yes_pre = .556 on audio** (text
≈ .055, i.e. escalation score .945 → .444). Asked before answering, the
model no longer knows it doesn't know when the trap question arrives as
speech — near coin-flip. Role reversal vs the text story: on audio the
probe is the robust signal and pre-answer p(True) is the fragile one.

Artifacts: `signals_/layers_/features_/ptrue_minicpm-o45-audio.*`,
`layer_sweep_minicpm-o45-audio.json`, `audio_xmodal_minicpm-o45-audio.json`
on gate-data; wavs in `/data/audio_pool/`. Pipeline (cwd=interactive_paper,
PYTHONUTF8=1): tts_pool → run_audio_signals → label_hf --tag
minicpm-o45-audio → run_audio_ptrue → layer_sweep_report → audio_report.
Gotcha: Modal auto-mounts only the entry module — modal_audio.py's images
add modal_app.py via `add_local_file` or containers die on import.

### ASR audit: perception vs introspection (user challenge, same day)

Could the trap collapse just be the model MIS-HEARING the question (rare
entities + TTS)? Three-arm test (`collect_asr`/`asr_report`): the model
transcribes each wav, then TEXT ptrue_pre runs on its OWN transcript.

| pool | WER mean/med | p_yes text | transcript | audio |
|---|---|---:|---:|---:|
| easy-chat | .077/.000 | .728 | .715 | .800 |
| easy-fact | .040/.000 | .663 | .632 | .860 |
| hard-knowledge | .224/.114 | .498 | .410 | .606 |
| hard-math | .131/.083 | .867 | .771 | .953 |
| **trap** | **.074/.058** | **.055** | **.074** | **.556** |

**Perception hypothesis REFUTED for trap:** (1) trap WER .074 — heard
almost perfectly; (2) on its own transcript p_yes snaps back to .074 ≈ text
.055 — the self-knowledge is THERE and accessible the moment the same heard
content is re-presented as text; (3) the well-heard subset (WER≤.15, n=43)
still collapses (audio p_yes .582), the misheard 7 are actually LOWER
(.394); (4) corr(WER, p_yes_audio) = −.115 ≈ none. → The collapse is in the
audio-input introspection READOUT, not perception. Elegant symmetry with 5d:
probe's late readout is text-fragile, p(True)'s verbalized readout is
audio-fragile — in both cases the knowledge survives, a readout breaks.

Secondary: p_yes_audio > p_yes_text on EVERY pool (+.07 to +.50).
hard-knowledge is the one pool where perception DOES contribute (WER .224,
long MCQ options — consistent with its +14.7pt fail-rate jump). Possible
cheap fix for audio p(True): "repeat-then-judge" (transcribe, then text
ptrue_pre on the transcript) — recovers trap introspection at
~question-length extra decode.

**Log-odds decomposition (`ptrue_shift_report`, $0): the audio shift is NOT
a uniform prior.** Per-query paired Δlogit(p_yes) medians: easy-chat +0.19,
hard-knowledge +0.63, easy-fact +1.19, hard-math +2.50, **trap +4.37**
(global median +1.12; trap excess over global **+3.25**; 30/50 traps flip
from the No side to the Yes side). A single "audio makes it overconfident"
logit bias is refuted — the shift is graded. Revised mechanism hypothesis:
in audio context the VERBALIZED self-assessment regresses to TYPE-level
priors ("chat → easy", "math → I can do math", "factual question → sure"),
and INSTANCE-level evidence (which specific entity) fails to reach the
verbal judgment — while instance-level info demonstrably stays in the
representation (audio LOPO within held-out pools .93+; the probe ranks
instances fine). Explains the full gradient: shift magnitude tracks how
much the correct judgment depends on instance vs type (chat: type suffices;
trap: instance is everything). Discriminating experiments RUN (same day,
`collect_ptrue_arms`/`arms_report`, n=600):

| trap p_yes | text | filler-audio+text (ctx) | audio+text-dup (dup) | audio |
|---|---:|---:|---:|---:|
| | .055 | **.001** | **.034** | .556 |

Δlog-odds vs text: ctx −3.28, dup **−0.44 (full recovery)**, audio +4.37.
**Both arms land on the binding hypothesis:** (ctx) irrelevant audio in
context does NOT inflate p_yes — the context-prior/persona story is refuted
(if anything filler audio depresses p_yes everywhere: fact .663→.268, chat
.728→.528 — audio context biases toward caution, the opposite of
overconfidence); (dup) giving the SAME question as text tokens alongside
the audio fully restores trap introspection (.034 ≈ .055) even though the
audio is still present. Mechanism, final form: **the verbalized
self-assessment performs its instance check (do I know THIS entity?) over
text-token pathways; audio-embedding tokens don't feed it** — while the
instance evidence demonstrably sits in the shared representation (probe
reads it at .93+). Practical fix confirmed twice over: any text
re-presentation of the question (ground-truth dup here, own-transcript in
the ASR audit at .074) restores the signal.

### TTS-template control: the cliff tracks INPUT MODALITY, not speak mode

Mechanism probe (`collect_layers_ttstpl`, tag `minicpm-o45-ttstpl`): same
600 TEXT queries, but prefilled under the speak-mode template
(`use_tts_template=True`, no TTS weights needed). If the 5d cliff came from
"prepare-to-speak" processing, it should move with the template flag.

LOPO hard-math, last-token, late layers:

| input | L31 | L32 | L33 | L34 | L35 | mid peak |
|---|---:|---:|---:|---:|---:|---:|
| text, plain template (5d) | .757 | .654 | .492 | .357 | .366 | .931 (L22) |
| text, TTS template | .689 | .586 | .519 | .468 | **.362** | .949 (L19) |
| audio (6a) | .903 | .901 | .918 | .920 | **.936** | .960 (L16) |

**The cliff is unchanged under the speak-mode template (L35 .362 ≈ .366)**
— it does not track the output-mode flag. Combined with audio's clean L35,
the operative variable is the modality of the CONTEXT (text tokens vs audio
embeddings), not the template or the speaking intent. Mean-pooling again
survives in all three conditions (ttstpl mean L35 math .801). This also
weighs against the trivial "audio's last position is just a template text
token" artifact story: if late layers damaged all text-token processing at
the readout position, the audio template's text wrapper tail would show the
cliff too — it doesn't. Revised mechanism claim: **the duplex fine-tune
re-purposed late-layer last-position processing of text-token contexts
specifically; audio-token contexts (protected by heavy ASR/understanding
training pressure) retain the faithful readout.** Prompt-tail audit stays
open but demoted (this control covers its main scenario).

### Judge validation: gpt-5.5 re-judge (user request, 2026-07-21)

Both o4.5 answer sets re-judged with gpt-5.5 (`rejudge`, JUDGE_MODEL
monkeypatched, max_tokens 8192; 0 errors). **Agreement with gpt-5.4-mini:
text 0.962, audio 0.945** — disagreement concentrated exactly in easy-chat
(no-reference subjective pool: .907 text / .853 audio, flips balanced in
both directions); reference-backed pools ≥ .95, easy-fact and trap at/near
1.0 (audio trap: 5.5 says 1.000 fail vs mini .960). Escalate rates move
≤4pts in any pool. `rescore55` under 5.5 labels: audio ptrue pre .786→.794,
post .860→.864, L22 probe OOF .815→.805 — **every headline number moves
≤.010; no conclusion changes.** Verdict: mini judge validated, keep
gpt-5.4-mini as default (5.5 labels stored in features_gpt55_{tag}.parquet).
Closes the "judge variance" open gap from 5b. Cost ≈ $25.

### Phase-6 streaming feasibility smoke ✅ (2026-07-21)

Headless duplex loop works on our pinned image — NO demo framework needed
(`streaming_smoke` in modal_audio.py, all 5 stages green):

1. **API surface**: remote code ships `streaming_prefill(session_id, msgs,
   omni_mode=True, is_last_chunk, ...)`, `streaming_generate(...,
   teacher_forcing_text='')`, `get_sys_prompt(mode='omni')`,
   `reset_session()`. **`teacher_forcing_text` = the official control point
   for the stall-phrase injection** — the biggest Phase-6 unknown, solved.
2. 14×1s chunks of a TTS wav prefill cleanly (gotcha: tail chunk must be
   zero-padded to 1s — a <0.1s residual under-fills the apm conv (kernel 3)
   and crashes; also pass `is_last_chunk=True` on the final chunk).
3. End-of-turn `streaming_generate` answers the HEARD math question
   correctly, yielding (text, is_final) increments.
4. **Gate insertion point verified**: L22 hook fires once per chunk prefill,
   shape (1, 18, 4096) — ~18 tokens/s of audio; per-chunk mid-layer probe +
   Phase-3 EMA gate is implementable as designed.
5. Same-session follow-up TEXT turn works (the `<result>` relay analog) —
   with a caveat that IS the step-2 problem: injected "expert result: 42"
   conflicting with the model's own $100 calculation → the model pushed back
   and asked to reconcile rather than relaying. Naive injection is not a
   straight relay; the inject prompt (or teacher forcing) must carry
   authority/formatting. First empirical contact with step 2.

Remaining Phase-6 work is now pure design/engineering (no unknowns):
per-chunk probe scores → EMA/hysteresis gate → teacher-forced stall phrase →
expert call → result injection; latency timers per segment.

Open: (a) SD-QA arm B; (b) prompt-tail audit (demoted, see above);
(c) audio latency numbers (audio prefill is longer — the mid-layer
early-exit argument gets stronger).

Phase-6a spend ≈ $45 incl. audits (TTS $1 + 4×H100 collection/ptrue/asr/
ttstpl + judge). Project total ≈ **$138**.

---

## Phase 6b — do the audio findings generalize? (2026-07-21, in progress)

User challenge: o2.6 is same-family — weak generalization evidence. Plan:
o2.6 = within-family robustness; **qwen2.5-omni = the cross-family test**
(finding 2 + the finding-3 duplex-vs-generic discriminator); qwen2-audio =
optional non-duplex control. Moshi/GLM-4-Voice/Kimi documented as blocked
(no text path / architecture). Pre-registered: finding 1's audio side may
stay MiniCPM-scoped (no other duplex family is runnable); finding 3's o2.6
replication has limited power (its TEXT trap introspection was already weak,
p_yes .196 vs o4.5's .055). `modal_audio.py` parametrized (`mtag`);
`audio_report` generalized; omni audio path = new `omni_image_au` +
`Qwen2_5OmniProcessor` (gotchas: needs pillow AND torchvision — the
processor loads image/video processors too; smoke: audio math answered
correctly, hooks 28×3584 OK).

### o2.6 replication (same 600 wavs; collection+label+ptrue+sweep, ~$35)

- **Finding 1 ✅ direction replicates:** audio last-token LOPO math — mid
  peak L22 .761, **final L27 .664** vs text final **.540** (text cliff
  L21 .822→.540). The text-side cliff is absent on audio (mild −.10 dip,
  no approach to chance). Signal overall weaker than o4.5 (.76 vs .96
  peak — weaker model, higher fail rates).
- **Finding 2 ✅ replicates:** cross-modal transfer onset ~L13/28 (~46%
  depth), plateau .74–.80 (peak text→audio L18 .799; o4.5 plateau ~.86).
  Early layers .58–.67. Same shape, lower ceiling.
- **Finding 3 ✅ broad direction, different signature:** audio ptrue_pre
  AUC **.491 ≈ chance** (text .604 was already weak); ptrue_post .805.
  Per-pool p_yes: non-trap pools DROP (chat .712→.585, fact .652→.471,
  math .699→.511) while trap RISES (.196→.345) — everything compresses
  toward ~.5: on the weaker duplex model the audio verbal self-assessment
  loses discrimination entirely, rather than o4.5's trap-specific collapse
  with preserved type ranking. Unified claim: **audio input degrades
  pre-answer verbalized self-assessment on both duplex generations** (o4.5:
  instance component lost; o2.6: all discrimination lost).
- Modality tax o2.6: chat +4.7, fact +4.0, knowledge +14.0, **math +16.0**,
  trap .96→1.00 — larger than o4.5's, consistent with a weaker audio
  front-end.

### qwen2.5-omni cross-family results ⭐ (same 600 wavs, ~$30)

**The finding-3 discriminator came back clean: the omni-streaming control
does NOT collapse — the audio introspection failure is DUPLEX-SPECIFIC.**

| model | FT type | trap p_yes pre text→audio | audio ptrue_pre AUC (text) |
|---|---|---|---|
| minicpm-o45 | duplex | .055 → **.556** collapse | .786 (.807) trap dead |
| minicpm-o26 | duplex | .196 → .345 | **.491 ≈ chance** (.604) |
| **qwen2.5-omni** | omni-streaming | .279 → **.213 INTACT** | .727 (.749) −.02 only |

Omni's audio p_yes actually moves DOWN on every pool except math (chat
.547→.460, fact .570→.457, trap .279→.213) — no overconfidence shift, no
discrimination loss. Same duplex-vs-omni gradient as 5c/5d. **Unified paper
claim now fully supported: duplex fine-tuning damages self-knowledge
READOUTS — the probe's late-layer readout in its text blind spot (5d) and
the verbalized readout in its audio blind spot (6a) — while the omni
control keeps both and the mid-layer signal survives everywhere.**
(Omni quirk persists: ptrue_post < pre on audio too, .599 < .727 — it was
already the only pre>post model on text.)

- **Finding 2 replicates cross-family**: transfer onset ~L06-08/28 (~25%
  depth — EARLIER than MiniCPM's ~50%, consistent with omni's tighter
  audio-text alignment and no duplex damage), plateau L18–L27 ≈ .80–.83
  both directions (L23 text→audio .826).
- **Finding-1 consistency**: omni audio sweep has no cliff (last-token math
  final .757 ≈ text .746; peaks .82 mid) — matches "no duplex FT → no
  cliff in either modality"; its text side was diffuse-depressed, audio
  similar.
- Modality tax omni: chat +14.0, knowledge +10.0, math +6.0, fact +5.0,
  trap 0 (.98=.98).

**Phase 6b verdict:** finding 2 = MiniCPM×2 + omni (cross-family) ✅;
finding 3 = duplex-specific (two duplex generations collapse, omni control
intact) ✅ — now the same shape as finding 1's raw>omni>duplex gradient;
finding 1 audio-side = MiniCPM-scoped as pre-registered (no other runnable
duplex family).

---

## Phase 6c — ablation vs o4.5's own thinking + latency ⭐ (2026-07-22)

User requirement: prove the solution ≥ o4.5's built-in capability
(`enable_thinking`), and benchmark latency end-to-end per request.

### Component latency bench (`latency_bench`, 50 q × text/audio, CUDA-synced,
3-warmup excluded, per-query interleaved; P50/P95 ms)

| config | text | audio |
|---|---:|---:|
| **L22 truncated decision (the gate)** | **20 / 25** | **45 / 104** |
| TTFT (full prefill + 1st token) | 36 / 47 | 68 / 144 |
| ptrue_pre (short prefill + 1 tok) | 39 / 63 | 67 / 169 |
| full answer (= ptrue_post entry fee) | 1855 / 7013 | 3507 / 7391 |

**The gate decides BEFORE the first token** (20 < 36 ms text; 45 < 68 ms
audio) — pre-TTFT escalation is real, and all pre-decode signals sit far
inside the 200–300 ms voice turn-taking budget. Audio latency tax ≈ 2×.
Decode 36.3 tok/s (matches Phase 0). Closes 6a open item (c).

### Thinking ablation (`collect_think` 600 q, enable_thinking=True,
max 2048 tok, gpt-5.4-mini judge 0 errors)

Per-pool: thinking helps math (fail .187→.127) and knowledge (.480→.427),
does nothing on trap (.98), and HURTS easy-chat (.207→.233). Overhead
P50 +8 to +24 s per query (P95 up to +63 s). think_used .57–.92 (the
hybrid mode skips thinking on some chat).

### End-to-end policy table (`e2e_latency_report`; test n=240; fast
latencies = per-pool bench medians [n_forward store is k-capped, unusable];
think = per-query measured; cloud = per-query expert_latency from Phase 5)

| policy | acc | lat mean | P50 | P95 |
|---|---:|---:|---:|---:|
| fast-only | .588 | 3.0 | 3.5 | 4.2 |
| **all-THINK (o4.5's own)** | .637 | **22.4** | 17.2 | 60.1 |
| gated-think @.33 | .613 | 12.1 | 3.6 | 47.6 |
| **gated-cloud @.15** | **.688** | **5.3** | 3.6 | 10.1 |
| **gated-cloud @.33** | **.787** | **6.5** | 3.6 | 20.8 |
| gated-cloud @.50 | .858 | 7.0 | 3.6 | 24.4 |

**Verdict: proven, with domination.** gated-cloud beats all-THINK on BOTH
axes at every escalation rate — already at 15% escalation: +5.1 acc pts at
4.2× lower mean latency; at 33%: +15.0 pts at 3.4× lower. gated-think (the
self-escalation tier) is weak (.613): the gate predominantly flags
knowledge/trap failures, which thinking cannot fix (thinking's gains are
execution/math) — a clean mechanistic reason why external escalation is
necessary, not just better. Thinking-tier idea documented and closed.

Phase-6c spend ≈ $45 (think 600 incl. long generations + judge + bench).

---

## Phase 6d — Freeze-Omni: the frozen-backbone control (2026-07-22)

Freeze-Omni (arXiv 2411.00774; speech encoder + adapter → FROZEN
Qwen2-7B-Instruct + state-head duplex) separates "duplex operation" from
"backbone weight updates". Integration: `modal_freeze.py` (their pins torch
2.2/transformers 4.45.2 — audioLLM manipulates legacy tuple KV; ptrue =
text tokens + chat_template['suffix'] appended to the audio KV via
DynamicCache.from_legacy_cache — without the suffix close the Yes/No mass
is 0.00; audioEncoderProcessor vendored to avoid the flask import chain;
text side = the `qwen2-7b` tag verbatim, same weights).

### Primary readout test: CONFOUNDED by capability collapse (pre-registered)

Audio fail rates explode vs the same weights on text: math .240→**.713**,
chat .333→.720, knowledge .600→.927 (fact +.04, trap .96→.98 only ones
stable). Audio probe: OOF peaks ~.70 (type recognition), **LOPO math
.44–.55 ≈ chance at every layer** — but with labels this collapsed the
readout question is unanswerable here (same verdict class as qwen2-audio
in 5c: documented negative).

### Two informative residues

1. **Finding-3 control still holds**: trap p_yes text .165 → audio **.112**
   — the verbal knows-it-doesn't-know SURVIVES audio on the frozen
   backbone (moves toward honesty, like omni; opposite of both duplex
   models). Third non-duplex model without the collapse. (Caveat: with
   audio capability collapsed, "No" is also the calibrated easy answer —
   pre AUC only .612, post .842.)
2. **Cross-modal transfer is ≈ DEAD on identical weights**: text→audio
   .52–.60, audio→text .34–.54 (inverts at L20) — versus .80–.86 on every
   end-to-end-trained model. **The modality-shared mid-layer core (finding
   2) is not free: it is CREATED by training the backbone on the modality.**
   An adapter alone aligns well enough to converse, but audio-context
   hiddens live off the text manifold — probes don't transfer, and task
   capability craters.

Combined three-way story: end-to-end multimodal training BUILDS the shared
semantic core (Freeze-Omni lacks it); duplex-style training additionally
DAMAGES the readouts (MiniCPM×2); omni-streaming gets both right (core
present, readouts intact). The paper's mechanism section now has all four
quadrants populated.

Phase-6d spend ≈ $45 (download + smoke ×4 + 600 chunked-streaming
collection + judge + qwen2-7b text layers). Project total ≈ **$290**.

---

## Phase 7a — collaborator follow-ups: fork profiling + escalation overlap (2026-07-24)

Meeting feedback (Jisen, Changyi) triaged into: (1) fork-at-layer-k
profiling — Jisen's "branch at ~L10, probe in parallel, both finish
together"; (2) result-feedback overlap — Jisen's "traditional routing
doesn't need to feed results back to a talker; can the cloud result be
ready by the next turn?"; (3) qwen3-omni probe test (Changyi) — assessed,
deferred (below); (4) routing-literature positioning (Changyi) — written
into TECHNICAL_REPORT §7. Items 1–2 run same-day; step-2 injection design
is the next real phase.

### Fork profiling: the decision is ready at 57% of prefill — at the quality peak

`prefill_timing` (H100, 25 q × text/audio × all 36 layers, truncated-forward
= same methodology as 6c's timed_l22, CUDA-synced, 2-query warmup) +
`fork_report` (joins 5d layer-sweep quality; figures/fork_pareto.png;
prefill_timing.parquet on volume). Architecture note first: the "fork" is
free by construction — the probe is one 4096-d dot product (~µs) on a
hidden state the forward computes anyway; nothing waits. The real question
is WHERE the signal is sufficient:

| layer | text ms (% of prefill+1tok) | text OOF | audio ms (%) | audio OOF |
|---|---:|---:|---:|---:|
| L11 (Jisen's ~L10) | 8.1 (35%) | .794 | 24.1 (48%) | .760 |
| L16 | 10.4 (45%) | .797 | 26.4 (53%) | .780 |
| **L22 (deployed)** | **13.2 (57%)** | **.866 = peak** | 30.6 (61%) | .815 |
| L23 | 13.7 (59%) | .850 | 29.3 (59%) | **.822 = peak** |
| L35 (final) | 19.3 (84%) | .835 | 36.6 (73%) | .813 |
| prefill+1tok | 23.2 | — | 50.1 | — |

- **L22 is simultaneously the earliest-useful AND the best layer** (in-mix
  OOF .866 beats the final layer's .835; 5d showed the same for LOPO .93 vs
  .37) — deciding at 57% of prefill costs nothing in quality. Escalation
  can therefore LAUNCH while the last 43% of prefill + decode still runs.
- L10-ish is too early on this model: .794 (−.07 from peak) and early
  layers are modality-specific (6a: cross-modal transfer dead before ~50%
  depth) — the fork belongs at ~55–60% depth, not 30%.
- Audio pays the encoder front-end (~17 ms: audio L1 21.4 ms vs text L1
  3.8 ms) at EVERY fork depth; L1→L22 then adds only ~9 ms. (Both arms'
  L0 rows are inflated by per-query first-call overhead — the L0 point in
  fork_pareto.png is an artifact, ignore it.)
- Absolute times here (23 ms prefill) are lower than 6c's chat-path bench
  (TTFT 36 ms) — different call overhead; the robust statistic is the
  ratio. Both agree the decision predates the first output token.

### Escalation overlap: Jisen's "result by next turn" — yes for the mix, not for escalated traps

`overlap_report` (CPU $0): per-query Phase-5 gpt-5.5 latencies (P50 3.0 s /
P95 24.4 s) vs pool-matched measured local answer durations (6c bench);
timeline = gate fires at l22 → cloud call in parallel with local decode.
P(expert result ready before the talker finishes + slack), figures/overlap.png:

| pool | text+0s | text+2s | text+5s | audio+0s | audio+2s | audio+5s |
|---|---:|---:|---:|---:|---:|---:|
| easy-chat | .43 | .66 | .86 | .77 | .86 | .92 |
| easy-fact | .02 | .62 | .94 | .56 | .87 | .95 |
| hard-knowledge | .48 | .61 | .74 | .44 | .60 | .73 |
| hard-math | .66 | .91 | .93 | .71 | .91 | .93 |
| trap | **.00** | .00 | .26 | .02 | .09 | .38 |
| ALL (test mix) | .40 | .65 | .81 | .58 | .75 | .84 |
| **escalated @.33** | **.20** | **.39** | **.60** | **.31** | **.47** | **.63** |

Stall needed after the local answer ends (escalated @.33): text P50 3.1 s /
P90 28.9 s; audio P50 1.9 s / P90 26.8 s (@.50: 1.8/0.3 s P50).

- **The overlap story works for the traffic mix (40–58%) but the gate
  selects against it**: escalated queries skew trap/knowledge, whose local
  answers are SHORT (trap overlap ≈ 0 — the talker finishes "…is X" in ~1 s
  while gpt-5.5 thinks for 3+). Math is the good case (.66–.91): long local
  answers buy the cloud time.
- Design consequence: same-turn delivery needs only **P50 one stall
  sentence (~2–3 s)**; the P90 tail (~27 s) is gpt-5.5's own reasoning
  latency, not our plumbing — argues for a fast-expert tier and/or streamed
  partial results in step 2. Audio deployment is structurally friendlier
  (utterances are ~2× longer).
- Caveats: expert latency measured Modal-us-east→OpenAI (includes RTT);
  local durations at max_new_tokens=256 (mild underestimate for the
  longest answers); bench n=10/pool → cross-product estimate.
- Paper figure: `figures/timeline_scenarios.{png,pdf}`
  (`figures/timeline_scenarios.py`, runs locally on the pulled parquets'
  medians) — panel (a) ms-scale fork (decision at 20 ms < TTFT 36 ms),
  panel (b) audio-channel occupancy: pre-answer routing (2.7 s dead air)
  vs gated hard-math (full overlap) / easy-fact (1.8 s stall) / trap
  (7.4 s gap, unbridgeable — gpt-5.5 trap P50 8.2 s is the slowest pool
  while trap drafts are the shortest, the structural worst case).

### Changyi's qwen3-omni proposal — disposition

The logic ("if a non-duplex omni's last layers are probeable, the damage is
duplex not modality") is exactly the already-run qwen2.5-omni control:
no cliff in either modality (5d text final .746 ≈ audio .757, 6b), plus 6a's
within-model converse (o4.5 audio L35 .936 vs text .366 — same weights, no
cliff on the native modality) and the ttstpl control. Qwen3-Omni-30B-A3B
(turn-based streaming per model card, premise correct) would add a
same-generation-as-o4.5 omni control (n=2), ~$25 + MoE/thinker integration
(transformers from source) — worthwhile as reinforcement, deferred by
priority call 2026-07-24. Stronger version of the same test: a NEW
open-weight full-duplex model as a pre-registered prediction ("it should
show the text-input late-layer cliff") — watch Qwen3.5-Omni (2026-03,
weights unconfirmed) and DuplexOmni (arXiv 2606.09186).

Phase-7a spend ≈ $2. Project total ≈ **$292**.

---

## Phase 7b — SD-QA real-speech validation ⭐ arm B closes 6a (2026-07-25)

The promised arm B: 200 REAL human recordings (VoiceBench sd-qa, USA split;
10 more dialect splits available for future robustness work), NQ-style
factoid questions with reference answers. Same questions run BOTH as typed
text and as the human recording (matched pairs); judged by gpt-5.4-mini
(0 errors both arms). New code in modal_audio.py: build_sdqa /
collect_sdqa / collect_ptrue_sdqa / sdqa_report. Note: the new OpenAI key
was verified first (openai_ping, pinned us-east per the post-suspension
policy) — the old key died with the suspension.

**Verdict: the audio-side findings are NOT TTS artifacts — all three
replicate on real speech.**

1. **Modality tax replicates:** fail rate .400 text → .450 audio (+.05,
   consistent with fact/knowledge-type content).
2. **Audio overconfidence replicates:** paired p_yes_pre shift +.089
   (audio > text on 62% of queries); on the failure subset, mean p_yes_pre
   .415 text vs .581 audio — asked before answering, real speech makes the
   model claim ability it doesn't have, exactly as with TTS.
3. **Layer structure replicates** (frozen-pool probes → sdqa = transfer
   across BOTH content and modality, a strictly harder test than 6a's
   matched pairs):
   - early layers dead (≤L11 all arms ~.42–.55, au→au even sub-.5),
     rising sharply after ~50% depth — finding 2's boundary;
   - audio late layers stay usable through L35 (au→sdqa-audio .74–.77 at
     L28–L35) — no audio-side collapse on real speech;
   - cross-modal deploy recipe (text-calib → real-speech audio): band
     .76–.80 at L22–L26 (peak .800 @ L25), then declining toward L35
     (.714) — mid-layer is where to read, same design rule as 6a
     (magnitudes below 6a's .86 as expected: content is also OOD here).
4. p(True) on real speech: pre AUC .769 text / .771 audio (discrimination
   preserved); post .813 text / .743 audio (post degrades on audio, same
   direction as 6a).

Honest scope note: the TEXT-side math cliff was not re-testable here
(sd-qa has no math; tx→sdqa-text late layers hold .79–.80, consistent with
the cliff being sharpest on math per 5d) — but the cliff claim never
involved synthetic audio in the first place, so it needed no arm-B
protection. My pre-registered criterion "text side collapses" was
miscalibrated for a factoid-only pool; the criteria that matter (audio
late-layer usability, mid-layer cross-modal band, overconfidence) all pass.

Phase-7b spend ≈ $18 (GPU collection ~$15 + judge ~$2). **Part 1 (the gate)
is now closed:** design final (mid-layer probe stage 1 + p_post stage 2),
mechanism characterized to the observational limit, latency measured (6c),
thinking ablated (6c), cross-family checked (6b/6d), real-speech validated
(7b).

---

## Phase 8 — chat_gated: the gate goes live ⭐ Part-2 milestone 1 (2026-07-25)

First end-to-end LIVE duplex loop (`modal_stream.py`): stream the wav in 1s
chunks → per-chunk L22 probe → EMA gate → mid-stream fire → expert races in
a background thread while the model keeps listening → canned stall phrase →
authority-framed `<result>` injection in the same session → relay. n=25
test queries (5/pool), balanced tier. Artifact: `midlayer_gate_audio.json`
(L22 audio-calib probe, OOF AUC .843, quantile thresholds — new, distinct
from the final-layer `gate_config.json`).

### What worked (the demo exists)

- **Mid-stream firing is real**: hard-knowledge 5/5 escalated at 12–66% of
  the audio (mean 40%) — the gate interrupts WHILE the user is talking,
  exactly as designed. easy/math 0/15 escalated (good restraint).
- **Stall via assistant-role prefill: P50 24 ms.** (Engineering find #1:
  `teacher_forcing_text` KEEPS GENERATING after the forced text — the model
  answered instead of stalling. Fix: prefill the stall as an
  assistant-role message = canned-filler semantics, ~24 ms, then
  `tts_filler` plays it in deployment.)
- **Relay works**: inject → spoken-form answer in P50 642 ms, expert
  answers relayed faithfully (spot-checks correct).
- **Real-time overlap delivers 5/6**: replay feeds audio faster than real
  time, so raw `expert_wait` is a replay artifact (engineering find #2);
  simulating the true timeline (remaining listening + 3 s stall):
  q0150 4.1s expert vs 21.5s budget ✓, q0152 15.1 vs 28.9 ✓, q0160 4.2 vs
  44.5 ✓, q0163 5.2 vs 16.0 ✓, q0164 12.9 vs 55.6 ✓ — **the expert answer
  is ready before the model needs to speak in 5/6 cases**.
- The 6th is the predicted failure shape (7a live-confirmed): **trap =
  short question (4.1s) × late fire (80%) × slow expert (16.4s reasoning
  on an obscure entity) → 12.6s dead air after the stall.** The step-2
  design problem in one row.

### Honest gaps (next milestone's worklist)

1. **Chunk-score calibration** (engineering find #3): offline thresholds
   are quantiles of FULL-prefill scores; live per-chunk scores are noisier
   and the zero-padded tail chunk dilutes. Shipped k=1/raw-score as the
   milestone fix; trap under-fires (1/5 vs offline expectation ~5/5) —
   needs proper chunk-level threshold calibration (collect chunk-score
   distributions on calib, re-quantile).
2. Heard-accuracy small-n: overall .360 (esc .500 n=6, local .316 n=19;
   includes 4 missed traps ≈ all wrong + judged easy-chat). Not a
   headline number at n=25 — the RQ2-style curve needs the calibrated
   gate + larger n.
3. Milestone shortcuts to lift later: expert query = pool text form (live
   ASR distill next), text-only output (no TTS), no barge-in, stall
   assumed 3 s spoken.

Fixes for the trap gap to evaluate next: progressive filler ("let me look
that up… one moment"), expert effort=low for entity lookups (Phase-5
showed trap needs retrieval not reasoning), or partial-answer streaming.

### Chunk-threshold calibration cycle (same day)

Step-1 fix for find #3: ran the EXACT live streaming procedure on all 360
calib wavs (`collect_chunk_scores`), calibrated the firing statistic on the
resulting per-chunk sequences (`calibrate_chunk_gate`), re-ran the same 25
live sessions.

- **Live chunk signal is weaker than offline: best statistic (mean_top2)
  AUC .764 vs offline full-prefill .843** — streaming 1s-boundary reads
  lose ~.08 AUC. (max .759, max_unpadded .757, last_unpadded .637.)
- Re-run @balanced (thr .52): escalation 24%→40% (target 30%; overshoot
  partly a semantics mismatch — thresholds quantiled on mean_top2 but the
  gate fires on single-chunk max). Knowledge 5/5 fire@27% (earlier),
  math 1/5, fact 3/5 (small-n overshoot vs predicted .12).
- **Heard accuracy 0.360 → 0.560** (escalated .700, local .467) — the
  calibrated gate converts directly into user-heard accuracy.
- **Trap still 1/5.** Calibration was necessary but not sufficient:
  predicted trap fire even on calib is only .40 — the live per-chunk trap
  signal is intrinsically weak (short audio, entity lands at the end next
  to the padded tail; offline's strong trap catch came from the full-
  prefill read after the WHOLE question). Trap under-fire is now a
  signal-placement problem, not a threshold problem.

Next design decision (not auto-iterated): an end-of-turn second read
(pre-TTFT verdict check approximating the offline h_prompt position)
vs unpadded-tail scoring vs accepting trap → p(True)-style stage-2.
Cycle spend ≈ $12.

Phase-8 spend ≈ $27 incl. calibration cycle. Project total ≈ **$353**.

---

## Phase 8b — conflict injection: the pushback problem quantified ⭐ (2026-07-30)

Grilling-session decisions (2026-07-30, all recorded in Notion "Full duplex
issues" + todo.tex): Part-2 acceptance = paper-grade live tradeoff curve;
end-of-turn read becomes the deciding gate with mid-stream chunk gate demoted
to speculative prefetch (+ veto); dead-air fix = deadline-aware expert effort
selection; expert query = talker self-transcript; prefetch = query-level
speculative execution (completeness curve = acceptance rate); offline
query-feature router baseline added to the worklist. This experiment is
worklist item ① — it had to run BEFORE the full live sweep so relay
non-compliance can't contaminate the live curve.

Design (`plan_conflict` / `conflict_sessions` / `conflict_report` in
modal_stream.py): 172 live audio sessions on 270 eligible calib queries
(ref + wav + audio label; easy-chat excluded — no references). Conflicting
"expert answers" fabricated by within-pool derangement of reference answers
(deterministic, zero LLM generation). Four framings: F0 neutral, F1 = the
deployed RELAY_TMPL, F2 strong-override, F3 assistant-seeding
(teacher_forcing_text speaks "the answer is: X" and continues). Judge
gpt-5.4-mini, structured {comply | pushback | lip_service | other}.

**Results:**

1. **The deployed channel is clean: model-wrong × correct-inject (component
   C, the actual live workload) comply = 1.00 (n=24).** The live curve is
   not contaminated by relay refusal.
2. **Layer-0 hypothesis CONFIRMED** (component A, conflict × F1 across the
   gate-score range): comply rises .42 → .67 → .75 → .75 across score
   quartiles; resistance (pushback+lip) is .25 in the most-confident
   quartile vs .08 elsewhere. The same signal that triggers escalation
   predicts relay compliance — "a model that knows it doesn't know is
   willing to listen." In the high-score half, hard-knowledge (the dominant
   escalation type) complies 1.00 (n=13).
3. **Framing ladder on the worst case** (confident-correct × conflict,
   component B): F0 comply .00 (other .54) / F1 .53 / F2 .79 (zero
   pushback+lip) / **F3 seeding BACKFIRES: comply .25, lip_service .62** —
   the model speaks the seeded answer then walks it back in its own
   continuation. Ladder layer 2 (assistant-prefill seeding) is empirically
   REFUTED for conflicting content: the model treats its own forced speech
   as revisable.
4. **Silent-override on math**: "other" concentrates in hard-math — the
   model neither relays nor disputes; it re-derives its own solution and
   states its own number (F1 conflict: math comply .57 / silent .26 vs
   non-math .66 / .20). Bare numeric answers carry no authority against
   the model's own computation chain. Harmless with correct experts
   (agreement is indistinguishable from compliance) but rules out
   "verbatim number relay" claims on math.
5. **Swallow rate** (model-wrong × conflict, D, n=11): comply .64,
   pushback .00 — an unconfident model has no internal resistance to wrong
   expert answers. Expert quality is the accuracy ceiling; now quantified.

**Decision for the full sweep: keep F1_authority.** The deployed regime
(C-like) is already perfect with F1; F2 only helps the rare
wrong-expert × confident-model corner and would plausibly raise the D
swallow rate (untested at F2 — deliberate); F3 rejected. F2 stays on the
shelf as the knob if live relay non-compliance appears.

Phase-8b spend ≈ $6 (GPU ~$5 + judge ~$1). Project total ≈ **$359**.

---

## Phase 8c — end-of-turn read + completeness curve: the trap fix works, prefetch dies ⭐ (2026-07-30)

Worklist ②/③a (collect_eot_scores / eot_gate_report / completeness_curve in
modal_stream.py; 360 calib replays ~$3 + 947 mini-judge calls <$1).

**1. The end-of-turn read RECOVERS AND EXCEEDS the offline discrimination:
AUC .887** (offline full-prefill reference .843; live chunk stats
max .759 / mean_top2 .764). Implementation: after the last audio chunk,
prefill an assistant turn with a single space — the L22 hook then reads the
assistant-start position, the streaming analogue of the offline h_prompt
read. (Empty-string assistant content raises; single space works.)

**2. Trap is saved: fire@balanced .90** (fail rate .97; was .20 live /
.40 predicted with chunk thresholds). Full per-pool fire@balanced:
trap .90, knowledge .41 (fail .56), math .22 (fail .24), easy-chat .16,
easy-fact .17. eot quantile thresholds written to midlayer_gate_audio.json
(conservative/balanced/aggressive = .846/.713/.555).

**3. Completeness curve kills partial-question prefetch** (the pre-agreed
wait-k decision rule): acceptance ("would an expert's answer to the partial
transcript answer the full question?") is .07 / .19 / .51 at 25/50/75% of
the words. At the chunk gate's typical knowledge fire point (27–40% of
audio) acceptance is ~.2–.4 → 6–8 of every 10 speculative expert calls
would be wasted. Per pool @25/50/75: knowledge .22/.41/.72, math
.02/.10/.61, trap .00/.10/.27, easy-fact .00/.17/.35, easy-chat
.00/.02/.24. **Decision (auto-executed per the agreed rule): drop
partial-question speculative escalation; the expert starts at end of
turn.** The prefetch threshold scan agrees from the signal side: 80%
early-fire recall costs 50% prefetch rate with 25% waste.

**4. Coherence finding (paper-worthy): trap's acceptance is the worst
(.00/.10/.27) — these questions back-load their semantic core, which is
the SAME underlying property that made the mid-stream probe under-fire on
trap** (Phase 8 signal-placement diagnosis). Two independent measurements
— hidden-state discrimination and semantic content — agree that the
information arrives at the end. One property, two manifestations.

System consequences: live loop v2 simplifies (no speculative expert calls
→ probation pressure drops; end-of-turn read decides pre-TTFT; on fire:
self-transcribe (~1s, overlapped with the stall TTS) → expert). Dead-air
now rests entirely on deadline-aware effort selection (worklist ④ = the
critical path). Milestone-1's "5/6 overlap" is retired as oracle-inflated.

Phase-8c spend ≈ $4. Project total ≈ **$363**.

---

## Phase 8d — effort × query-form characterization: effort is free, ASR is the leak ⭐ (2026-07-30)

Worklist ④+③b (`effort_characterize` / `effort_report`). Subset = the 72
test queries above the balanced offline-score quantile (the gate's
escalation population). Arms: transcript×{low, medium} + gold×low (~216
gpt-5.5 calls, concurrency 3, cached, spaced batches); gold×medium = the
frozen Phase-5 answers (free). Judge gpt-5.4-mini. Transcripts = the
talker's own 6a ASR outputs (deployed query form, decision 4).

**Four-column decomposition (acc, n=72):**

| pool | gold-med | gold-low | tr-med | tr-low | ASR tax | effort tax |
|---|---|---|---|---|---|---|
| easy-chat (5) | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| easy-fact (8) | .88 | .88 | .88 | .88 | 0 | 0 |
| hard-knowledge (32) | .84 | .84 | .53 | .56 | **+.31** | ≈0 |
| hard-math (10) | 1.00 | **1.00** | .30 | .30 | **+.70** | **0** |
| trap (17) | .71 | .59 | .53 | .53 | +.18 | 0 (gold −.12) |
| ALL | .85 | .82 | .57 | .58 | **+.28** | **≈0** |

1. **The effort tax is ≈ zero everywhere — including gold math (low 1.00 =
   med 1.00).** GSM8K-class problems are solved by gpt-5.5 at any effort;
   the only hint of effort sensitivity is trap on gold (.71→.59, ≈2
   queries, n=17 — noise-level). The grilling's effort-sensitivity
   prediction ("math = execution-bound, effort helps") is REFUTED at
   expert scale on this pool mix: the expert is so far above the task that
   effort never binds. Combined with 6c (talker's own thinking also can't
   fix escalated failures), the unified statement: **on gate-escalated
   traffic, extra compute buys nothing at either scale — the failures are
   knowledge-bound.**
2. **Latency, however, doubles at medium: transcript P50 5.8→9.1s, P95
   23.9→78.8s.** Medium effort = 2–3× the wait for zero accuracy.
3. **Policy decision (auto, per the measurement): fixed-LOW effort for all
   escalations.** Dead-air replay (transcribe 1s + stall 3s cover):
   fixed-low P50 3.8s / P90 16.8s / silent>8s 29%, vs fixed-med 7.1 /
   41.2 / 49%. Score-conditioned effort is REJECTED as unnecessary
   complexity (same acc as fixed-low, worse latency). The deadline-aware
   design collapses to the simplest possible policy — which is what
   measuring the dimensions was for.
4. **The ASR-distill tax is the new dominant leak: −.28 overall, −.70 on
   math, −.31 on knowledge.** The talker's own transcription (WER .074
   trap / .224 knowledge per 6a) halves expert accuracy on the escalated
   subset (.85→.57): small slips are fatal for math (a lost number kills
   the problem; +.70 tax despite decent-looking WER) and costly for
   knowledge entities. **WER is a poor predictor of downstream damage.**
   This caps the live curve's escalated-arm accuracy well below the
   Phase-5 oracle-text numbers — the honest deployment ceiling.

**Post-hoc scope correction (failure inspection):** the failing math rows
are MATH-500 formula problems — the TTS reads LaTeX aloud and the
round-trip destroys the symbolic structure (`31/11111` → `31/111`;
`x^4+5x^3+9x^2` → "x cubed x four plus five x squared";
`x^10+(13x-1)^10` → `(x-10)^10+13(x-10)^5`). GSM8K-style word problems
transcribe cleanly (see measure_asr_timing samples). So the math +.70 is
largely a **TTS-of-LaTeX artifact — nobody speaks LaTeX at a voice
assistant — and overstates deployment harm; the deployment-real ASR tax is
the knowledge-entity −.31** (and trap −.18). Paper scope note: the speech
channel is evaluated on speakable content; formula-symbolic math is out of
scope for the audio arm (it already carries the text-side cliff story).
Live ASR timing measured: P50 2.2s / P95 3.4s (math-length utterances,
conservative) — replaces the 1s placeholder in dead-air accounting.

**ASR-tax attribution COMPLETE (user then scoped ASR out of the paper —
"channel property, not our contribution"; user decision 2026-07-30 late).**
Five-arm table (acc, n=72, expert low): gold .82 / self-1best .58 /
self+robust-prompt .58 / self-kbest-GER .56 / **Whisper .62**. Verdicts:
(1) both self-rescues NEGATIVE — the robust prompt changes nothing, and
k-best GER trades a small trap gain (.53→.59) for a knowledge LOSS
(.56→.44: five divergent long transcripts confuse complex MCQs more than
they repair); (2) external ASR recovers a modest fraction (knowledge
.56→.62, trap →.59 = gold-low level); (3) math is .30 in EVERY
transcript arm — the TTS-of-LaTeX artifact, four-way consistent; (4) the
knowledge gap Whisper can't close (.62 vs .84) points upstream: the TTS
pronunciation of obscure entities is itself lossy — channel-inherent.
Paper treatment: dual-view curve (end-to-end honest heard-acc + a
channel-controlled expert-inject view from frozen gold answers, $0) with
the difference labeled as channel cost; ASR robustness cited as
orthogonal work. Consequence: the always-live arm is CANCELLED (its
~168 calls unneeded — the channel-controlled ceiling and random line
synthesize from frozen data). Also measured today and now moot as a
direction: expert reasoning effort ≈ zero accuracy effect at either
query form — escalated failures are knowledge-bound at both scales.

**Mitigation (a) tested same day (user-approved, 72 calls): NEGATIVE.**
A recognition-errors warning prepended to the transcript
(`ROBUST_PREFIX`, form=robust) changes nothing: ALL .58 = plain .58
(knowledge .59 vs .56, trap .47 vs .53, math .30 unchanged — all within
±1 query). Interpretation: the damaging errors are entity substitutions
that leave an internally-coherent transcript — no textual residue to
correct from. **The information is destroyed at transcription, not
noised; the ASR tax is not prompt-fixable.** Remaining options: (b)
better transcription decoding, (c) an external-ASR decomposition arm
(Whisper transcripts × low, ~72 calls — would quantify how much of the
tax is MiniCPM's ASR quality vs inherent audio ambiguity), or (d) accept
and document as the honest cascaded-escalation ceiling. User to pick;
the sweep currently ships the honest self-transcript pipeline.

Phase-8d spend ≈ $8 (expert ~$6 + judge ~$2). Project total ≈ **$371**.

### 8e — live sweep day 1: floor + conservative (2026-07-30)

v2 smoke n=25 passed (overall .440; the 512-token cap fix lifted local
.294→.412 — the default streaming cap was truncating long math answers,
our artifact not the model's stopping). Latencies: eot read 22ms, stall
26ms, relay 678ms. Local answers after the probe turn are clean (no
history contamination). Then the first two full 240-query live arms:

| arm | esc | heard-acc | escalated | local |
|---|---|---|---|---|
| **never** (live floor) | 0% | **.375** | — | .375 |
| **conservative** | 14% | **.450** | .471 | .447 |

- The honest live floor is .375 — well below the offline chat-mode
  small-only .588 (text) because it stacks the audio modality tax AND the
  streaming-answer mode (speech-style short answers that plan aloud then
  stop; real deployment behavior, now measured at n=240). This is exactly
  why the never endpoint had to be run live rather than replay-synthesized.
- conservative (+14% escalation) buys +7.5 pts over the floor.
- Two engineering incidents burned ~$25 GPU, both documented in code:
  cross-app hydration (gen_app entrypoint required) and concurrent
  same-file volume appends (last-committer-wins; per-shard trace files
  now, matching the collect_* convention).

**Same-day continuation (user said keep going): all four arms landed.**

| tier | esc | heard-acc | escalated arm | local arm |
|---|---|---|---|---|
| never (floor) | 0% | **.400** | — | .400 |
| conservative | 14% | **.446** | .441 | .447 |
| balanced | 35% | **.529** | .536 | .526 |
| aggressive | 55% | **.633** | .621 | .648 |

- The live curve rises monotonically, +.23 accuracy over 55% escalation.
  Against the offline text curve (.588 floor → .679/.779/.833 at
  14/33/53%) the live curve sits ≈.19–.20 lower at every operating point
  — the measured price of real streaming deployment, decomposable with
  the day's experiments: audio modality tax + streaming-answer tax
  (floor .588→.400) and ASR-distill tax on the escalated arm
  (expert .85→.57 on the escalation population).
- Judge-variance note: never was judged .375 in the first (partial)
  report run and .400 in the final run — gated_report re-judges each
  invocation (±.02–.03 single-judge noise). The persisted
  gated_traces_v2.parquet freezes the final labels; bootstrap/DeLong must
  run on THAT, not on fresh judgings.
- Still open: the always-escalate ceiling arm (needs ~168 non-subset
  transcript expert calls — user decision pending) — also needed for a
  proper live random-escalation reference line; bootstrap CIs; the
  offline-vs-live figure.

Day-1 total spend ≈ $75 (GPU ~$60 + expert ~$8 + judge ~$7). Project
total ≈ **$446**.

### 8f — query-feature router baseline: the type-shortcut, isolated (2026-07-30)

`router_baseline` ($0, frozen data): TF-IDF (word 1-2gram + char 3-5gram)
→ LR on calib TEXT labels — a RouteLLM-style external router that sees
only the query surface. Three readouts, all as the grilling predicted:

- **in-mix**: OOF AUC .669 — below even the pool-oracle (.715), far below
  the probe (.828). Test tradeoff area **+.040** vs probe_final +.054 /
  midlayer_L22 +.064 / ptrue_post +.068 (same test set, same frozen
  expert answers). Internal signals beat the external router at every
  operating point (@30%: .717 vs probe-era .779).
- **LOPO: total collapse** — easy-chat .466, easy-fact .384,
  hard-knowledge .505, hard-math .574 (vs mid-layer probe .931); trap
  (single-class) mean score .230 = it MISSES the 100%-fail pool.
  Query-feature routing IS the type shortcut and nothing else; held-out
  types kill it, exactly the structural argument (7a positioning) now
  with numbers.
- Scope: a same-data router, not RouteLLM's 100k-preference-trained one —
  the full LLMRouterBench protocol comparison stays future work; this
  baseline isolates the information source (query surface vs internal
  state) under identical data.

Project total ≈ **$426** (router run $0).

### 8g — false-premise pool audition: the gate's third blind spot (2026-07-30)

User challenge: the trap pool conflates knowledge difficulty with
transcription difficulty (obscure entities are hard to hear by design) —
design a transcription-fair trap. WER-stratifying the existing pool was
REJECTED as cherry-picking (endogenous selection: the stratum is defined
by the system's own ASR; the dual-view counterfactual curve answers the
same question with zero item selection). The principled candidate:
**false-premise questions** (failure mode = premise checking; every word
common by design — criterion fixed before data selection). Pool:
thunlp/FalseQA (ACL 2023, public, 687 FPQ test rows), n=60 audition,
standard protocol (build_falseqa_pool / tts_falseqa / falseqa_eval /
falseqa_report, ~$8).

**Audition verdict: FAILS as a trap pool — and the failure is a
finding.** Small-model fail rate .63 text / .47 audio (substantial, not
SimpleQA's 1.00); expert adequacy .80 (escalation would pay); but the
**end-of-turn gate is blind to these failures: fire@balanced .18 (vs
trap's .90), and eot scores barely separate failed from passed audio
answers (.568 vs .507)**. The model answering along a false premise does
not "feel" unable — the mid-layer signal reads missing-knowledge, not
broken-question. Third documented blind spot (easy-chat failures, decode-
time math, now premise checks): the claim sharpens to "the pre-answer
signal detects retrieval-type failures specifically." FalseQA therefore
goes into the signal-boundary analysis, NOT into the tradeoff pools.
Curiosity noted (n=60, not pushed): audio fails LESS than text (.47 vs
.63) — the voice mode's shorter, more hedged answers may accidentally
challenge premises more often.

Phase-8g spend ≈ $8. Project total ≈ **$462**.

### 8h — bootstrap + dual-view figure: Part-2 closeout ($0, 2026-07-30)

`live_dualview` in modal_stream.py (CPU, frozen data only — reads
`gated_traces_v2.parquet` + `eval_expert.parquet`, NO re-judging, no
expert calls; 240 ids common to all four arms; paired bootstrap 10k
resamples, seed 42):

| tier | esc | heard-acc [95% CI] | Δ vs floor [CI] | gold-inject [CI] | channel cost [CI] |
|---|---:|---|---|---|---|
| never | 0% | .400 [.338,.463] | — | .400 | — |
| conservative | 14% | .446 [.383,.508] | +.046 [−.008,+.100] n.s. | .500 [.438,.562] | +.054 [+.029,+.083] * |
| balanced | 35% | .529 [.467,.592] | +.129 [+.067,+.188] * | .637 [.575,.700] | +.108 [+.071,+.150] * |
| aggressive | 55% | .633 [.571,.692] | +.233 [+.171,+.296] * | .767 [.713,.821] | +.133 [+.088,+.179] * |
| always (synth) | 100% | — | — | .917 [.879,.950] | — |

- **Balanced and aggressive beat the live floor significantly; the
  conservative delta (+.046) is n.s. at n=240** — stated in the paper for
  honesty. Channel cost is significant at every escalating arm.
- Dual-view figure `figures/live_dualview.png` (+ numbers in
  `live_dualview.json`, both fetched into the repo + paper/figures/):
  honest heard-acc curve (blue, CIs) vs channel-controlled gold-inject
  counterfactual (green, escalated rows re-scored with frozen gold expert
  answers) + synthesized always endpoint + random reference (pairs with
  the green view — its outcomes are gold), offline text curve for
  context. Two annotated gaps = audio+streaming tax (floor .588→.400)
  and speech-channel cost (ASR-distill + relay).
- In the channel-controlled view the gate clears random at every arm
  (+.03/+.06/+.08).
- **Paper sync (same session): new §"The Gate Goes Live" (sections/
  live.tex: loop design + 8b conflict injection + live curve
  Table~tab:live + dual-view figure + FalseQA boundary), router-baseline
  paragraph in system.tex, discussion rewritten (pushback resolved by the
  gate's own signal + three-failure-species taxonomy), abstract/intro
  updated to "both steps executed", limitations updated (live-loop
  text-mode output, conservative n.s.), todo.tex items marked DONE.
  TECHNICAL_REPORT.md bumped to v4 (§8b).**

### 8i — latency profile of the live sweep + real-session timelines ($0, 2026-08-05)

User ask: "what did escalation actually cost us — mean / P95 / P99 — and
show real timestamped sessions." All from the frozen
`gated_traces_v2.parquet` per-session timers (no re-runs). Scripts:
`figures/latency_profile.py` (numbers → `latency_profile.{txt,json}`),
`figures/timeline_live.py` (figure). Reconstruction: local total =
eot_read + answer decode; escalated total = eot_read + max(stall prefill,
expert round-trip) + relay decode (the expert thread launches at the gate
decision, concurrent with the stall prefill). `expert_latency_s` is the
TRUE API latency (cache-corrected; min 0.98 s, no timeouts, no ~0 s cache
artifacts in the 250 escalated rows). Text-mode pipeline; speech
synthesis not included.

**Component timers (rows where the stage ran; seconds):**

| stage | n | mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| eot gate read | 960 | .03 | .03 | .03 | .05 |
| stall prefill | 250 | .03 | .03 | .04 | .19 |
| local answer decode | 710 | 4.57 | 2.34 | 15.90 | 17.86 |
| relay decode | 250 | 1.14 | 0.69 | 3.64 | 8.61 |
| expert round-trip (gpt-5.5 low) | 250 | 7.28 | 4.78 | 20.77 | 32.80 |

**Per-arm total response latency (query end → answer text done; s), and
the loss vs the never floor at the same percentile:**

| arm | esc | mean | P50 | P95 | P99 | Δmean | ΔP50 | ΔP95 | ΔP99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| never | 0% | 4.61 | 2.02 | 15.76 | 17.79 | — | — | — | — |
| conservative | 14% | 4.93 | 2.76 | 16.31 | 20.32 | +0.32 | +0.74 | +0.55 | +2.53 |
| balanced | 35% | 6.27 | 4.00 | 18.06 | 30.37 | +1.66 | +1.98 | +2.30 | +12.58 |
| aggressive | 55% | 6.58 | 4.69 | 18.01 | 32.94 | +1.97 | +2.67 | +2.25 | +15.15 |

- **The average price is small; the tail price is real.** Balanced buys
  +.13 heard-acc for +1.7 s mean / +2.0 s P50 — but the P99 doubles
  (17.8 → 30.4 s), and the P99 loss is entirely the expert tail: within
  escalated rows the expert round-trip is 85–88% of total wall time
  (relay decode P50 ≈ 0.7 s, gate + stall prefill ≈ 50 ms combined are
  noise). P95 moves little (+2.3 s) because at 35% escalation the 95th
  percentile is still mostly local long-decode rows; P99 (n=240, ≈2
  worst rows — noisy) is where escalation shows.
- Escalated-row totals: conservative P50 9.2 / P95 21.8; balanced P50
  6.1 / P95 22.4; aggressive P50 5.4 / P95 21.9 (conservative escalates
  the hardest queries → slowest experts, mean 8.9 s vs aggressive 6.7 s).
- **Figure `figures/timeline_live.png`** — three REAL balanced-arm
  sessions, every event a recorded timer: (a) q0254 escalated, expert
  ready at 1.9 s while the talker is still voicing the stall → seamless
  handoff, zero dead air; (b) q0593 (trap) escalated, expert 10.3 s
  outlives the stall → 5.9 s dead air (the 8e/6c prediction,
  live-confirmed on a real session); (c) q0388 not escalated → local
  answer, first token 0.4 s. Speech bars are the one estimated quantity
  (150 wpm; the live loop outputs text — RESULTS 8e).
- **Paper sync (same session): latency paragraph + Table~tab:latency +
  Figure~fig:timeline-live added to sections/live.tex
  (§"The live curve"); figure copied to paper/figures/.**

### 8j — router training receipt + RouterBench grounding (~$1, 2026-08-05)

User ask: "what is the router's training accuracy — give a receipt, and a
score on a routing bench." Two runs, both CPU-only: `router_baseline`
re-run with a full receipt, and new `router_bench` on **RouterBench**
(withmartian/routerbench 0-shot, public — no self-made data).

**Training receipt (8f router, `router_baseline.json` → `receipt`):**
TF-IDF (word 1-2gram + char_wb 3-5gram, min_df=2) → LR (C=1.0, L2,
lbfgs, max_iter=3000), 5-fold stratified OOF, seed 42. n_train=360 calib
(escalate rate .322), 15,103 features.

| split | logloss | acc | majority |
|---|---:|---:|---:|
| train (in-sample) | .377 | .917 | .678 |
| calib OOF (= eval) | .588 | **.678** | **.678** |
| test | .629 | .613 | .588 |

- **The headline answer: at the 0.5 threshold the router's eval accuracy
  exactly equals the majority-class rate** (and test is +.025 over it).
  360 queries of surface text buy a weak ranking signal (OOF AUC .669)
  and no usable classifier — train↔OOF loss gap .38→.59 is textbook
  small-n overfit. AUCs reproduce 8f exactly.

**RouterBench (`router_bench.json`): n=36,497, pair mixtral-8x7b-chat
(correct .568) → gpt-4-1106-preview (.843), label = weak incorrect
(escalate rate .432), same TF-IDF+LR recipe.**

- **In-domain (trained on RouterBench, 5-fold OOF): AUC .710, acc .660
  (majority .568), logloss .615; deferral area over random +.033
  (pair acc @30% escalation .691).** 100× the training data buys
  .669→.710 — the recipe lands at our pool-oracle's level (.715) and
  stays far below the probe (.828). **The 8f baseline is not
  data-starved, it is information-starved — not a strawman.**
- **Leave-one-benchmark-out (the LOPO mirror on public data): the
  format-disjoint benchmarks sit at chance** — hellaswag .502
  (n=10,042), grade-school-math .509 (n=7,450), winogrande .498,
  arc-challenge .581; Chinese_character_riddles .112 (fail rate .98 —
  misses the near-100%-fail pool, the trap-pool signature again). MMLU
  subjects read .55–.69 only because the other 56 subjects stay in
  training (within-format type prior transfers; cross-format it dies).
- **Zero-shot transfer (our calib-trained router → RouterBench): AUC
  .440, area −.022** — below chance; the 8f router learned our pools'
  surface regularities and nothing portable.
- Scope: pair routing readout (AUC/acc/deferral curve), not the full
  RouterBench cost-quality AIQ protocol; that and preference-trained
  routers (RouteLLM 100k) remain the LLMRouterBench future-work item.
- **Paper sync (same session): receipt + RouterBench numbers written
  into the trained-router paragraph in sections/system.tex
  (\citep{hu2024routerbench} already in refs.bib); todo.tex item
  updated; TECHNICAL_REPORT.md bumped to v5 (§8b.5).**

Project total ≈ **$447** (RouterBench run ~$1: 16 CPU + 32 GB, ~25 min).

### 8k — the same receipt for our probes ($0, 2026-08-05)

User follow-up on 8j: "then how does OUR trained probe do on this
accounting?" `probe_receipt` — the identical receipt (train/OOF/test
logloss, acc@0.5 vs majority, AUC, budget-threshold classification acc)
for the two shipped internal gates, exact shipped recipes
(`probe_receipt.json`):

| signal | OOF AUC | OOF acc (maj) | test AUC | test acc (maj) |
|---|---:|---:|---:|---:|
| router 8f (query surface) | .669 | .678 (**= .678**) | .721 | .613 (.588) |
| text h_prompt probe, LR C=.001 | .828 | **.772** (.678) | .819 | **.779** (.588) |
| audio L22 probe (live gate) | .843 | **.764** (.592) | .879 | **.800** (.512) |

- **The probes pass the accounting the router failed**: at the same 0.5
  threshold they clear majority by +.09/+.19 (text, OOF/test) and
  +.17/+.29 (audio) where the router cleared it by +.000/+.025. Same
  n=360 labels, same LR machinery — the difference is purely the input
  representation. (Prediction miss, recorded: I expected C=0.001 score
  compression to pin probe acc@0.5 at majority too — wrong; the internal
  separation survives even heavy regularization.)
- Audio rows use the audio-modality label set (calib esc rate .408, test
  .488 → majority .512) — not the same labels as the text rows; compare
  within-row, not across.
- Budget-threshold classification acc (test): text .704/.775/.700 at
  15/30/50%; audio .692/.750/.800 (realized rates track targets:
  .142/.329/.529 text, .204/.329/.496 audio).
- Honesty notes: the text probe memorizes in-sample even at C=0.001
  (train logloss .009, acc 1.000; 4096 dims ≫ n=360) — all headline
  numbers are OOF/test. Audio probe test logloss .446 — the
  best-calibrated signal we have. AUC headlines reproduce (.828 text
  OOF, .843 audio OOF, .879 audio test).
- **Figure `figures/receipt_compare.png`** (`receipt_figure`, reads both
  receipt jsons — no hardcoded numbers): (a) acc@0.5 vs majority per
  split, (b) train→OOF→test logloss per signal. Loss verdicts: audio
  probe healthy (.278/.483/.446 — eval below base-rate entropy, test
  below OOF, best-calibrated); text probe memorizes in-sample
  (.009 train vs .688 OOF — OOF logloss WORSE than predicting the base
  rate, i.e., uncalibrated probabilities, but held-out acc/AUC hold and
  the gate thresholds on score quantiles, not probabilities); router
  .377/.588 — normal gap, low ceiling.
- **Paper sync (same session): probe-receipt sentence added to the
  trained-router paragraph in sections/system.tex +
  Figure~fig:receipt (receipt_compare.png copied to paper/figures/);
  TECHNICAL_REPORT §8b.5 extended (v5).**

### 8l — router fairness sweep ($0, 2026-08-05)

User challenge: "prove you didn't train a deliberately weak router to
flatter the probe." `router_sweep` (`router_sweep.json`): 24-config grid
— features {word, char, word+char} × min_df {1, 2} × C {.01, .1, 1, 10},
identical 5-fold OOF protocol on the same 360 calib labels.

- **Every config lands in .625–.689; best .689** (word+char, min_df=1,
  C=0.1) vs the shipped .669 (+.020, within small-n noise). Probe: .828.
- The query-surface ceiling on this data is ≈.69 by exhaustive grid,
  ≈.71 by 100× public data (8j RouterBench in-domain) — two independent
  routes to the same ceiling; the shipped config is not a tuning
  artifact. Written into the system.tex router paragraph.

### 8m — RouteLLM released checkpoints, zero-shot on our pool (~$1, 2026-08-05)

User ask: "have you compared against an actually-trained router (e.g.
RouteLLM's)?" `routellm_baseline` (`routellm_baseline.json`): the two
released preference-trained routers — `bert_gpt4_augmented` and
`mf_gpt4_augmented`, trained on ~100k GPT-4-vs-mixtral preference pairs
— scored zero-shot on all 600 labeled queries
(score = calculate_strong_win_rate, their own inference code via the
routellm package).

| router | calib AUC | test AUC | area | acc@30% |
|---|---:|---:|---:|---:|
| RouteLLM BERT | .584 | .523 | +.011 | .688 |
| RouteLLM MF | .602 | .533 | +.007 | .688 |
| our same-data TF-IDF router (8f) | .669 | .721 | +.040 | .717 |
| probe (h_prompt / L22) | .828 / .843 | .819 / .879 | +.054… | .779… |

- **The real preference-trained routers land near chance on our labels
  (test .52–.53)** — below even the 360-sample same-data router. 100k
  preference pairs of "is this hard for mixtral vs GPT-4" carry almost
  nothing about "will MiniCPM-o-4.5 fail this" — the exact mirror of our
  router's .440 transfer TO RouterBench (8j). Routing knowledge is
  model-pair-specific; it does not port across talkers in either
  direction.
- **Trap-pool blindness, again**: BERT ranks trap (.444 mean score)
  BELOW hard-knowledge (.642) and hard-math (.668); MF likewise (trap
  .274 ≤ hard .31). The 100%-fail pool looks "easy" to a router trained
  on another model's preferences — the type-shortcut failure mode
  reproduced on the strongest available external router.
- Scope: zero-shot released checkpoints (their intended deployment
  mode); retraining them on our 360 labels is the same-data condition
  8f already covers. The standardized LLMRouterBench protocol remains
  future work.
- **Same scores vs the AUDIO (TTS) labels** (user ask, apples-to-apples
  with the audio probe): BERT calib .559 / test .600; MF .603 / .581 —
  vs audio L22 probe .843 / .879 and our audio-label TF-IDF router
  .743 / .814 (8n). Context from their own paper (APGR, random=.500):
  even on RouteLLM's own bench their routers reach only .53–.62 on
  MMLU/GSM8K-style objective tasks (strong only on MT-Bench chat,
  .68–.80), so .52–.60 on our pool is consistent with their published
  profile, not an artifact of our setup.
- **Paper sync (same session): system.tex router paragraph — future-work
  clause replaced with the zero-shot RouteLLM numbers; TECHNICAL_REPORT
  §8b.5 extended.** Project total ≈ **$448**.

### 8n — audio-modality router baseline ($0, 2026-08-05)

User: "is there an audio router?" There wasn't — every router number so
far was text-modality. `router_audio` (`router_audio.json`): the 8f
recipe vs the AUDIO labels (the live gate's label set, esc rate
.408 calib / .488 test), two inputs, both with 600/600 coverage:
gold query text, and the talker's own ASR transcript (what an external
router would actually see live).

| signal (audio labels) | OOF AUC | oof acc (maj .592) | test AUC | test acc (maj .512) |
|---|---:|---:|---:|---:|
| router, gold text | .743 | .706 | .814 | .696 |
| router, self-ASR transcript | .738 | .689 | .805 | .704 |
| audio L22 probe | **.843** | **.764** | **.879** | **.800** |

- **Honest headline: on audio labels the surface router is STRONGER
  than on text labels** (.743/.814 vs .669/.721) — the audio channel
  makes hard pools fail harder and more uniformly, so failure is more
  type-correlated and the type shortcut buys more. Reported as-is.
- The probe still leads every readout: AUC +.10 OOF / +.065 test,
  accuracy +10 points (.800 vs .696). The residual is exactly what the
  surface cannot carry: instance-level knowledge state + whether THIS
  utterance was heard correctly.
- **ASR input costs the router almost nothing** (−.005/−.009 AUC vs
  gold text): its handicap is not transcription quality; it is the
  information source, plus the structural live constraint (full
  utterance + ASR needed; cannot fire mid-stream — todo.tex note).
- **Paper sync (same session): audio-router sentences added to the
  system.tex router paragraph; TECHNICAL_REPORT §8b.5 extended.**

### 8m — channel-cost trace-level decomposition: relay exonerated ($0, 2026-08-05)

User hypotheses for the blue↔green gap: (H1, Changyi) talker context too
short to hold the expert micro-turn answer; (H2) talker doesn't follow /
second-guesses the relay instruction. Both testable from
`gated_traces_v2.parquet` alone (250 escalated rows across the three
partial tiers, 108 heard-fail; ref-string containment + query↔transcript
similarity, `difflib` ratio):

- **H1 REFUTED.** Expert micro-turn answers are short: median 76 chars,
  p90 376, max 1948 (~500 tokens) — nowhere near any context limit; the
  single >1500-char answer relayed successfully (heard_ok=1).
- **H2 REFUTED (again).** Only **5/108** fails have the reference inside
  the expert answer but missing from the relay; 8b already measured
  deployed-channel comply = 1.00 (n=24). One degenerate-repetition relay
  observed (Taylor MCQ) — on an already-wrong expert answer.
- **The loss is upstream of the relay (~95%).** Fail split: **69/108
  (64%) corrupted transcript** (fails' query↔transcript sim median .809
  vs .995 on heard-ok rows; MCQ options and formulas garbled —
  hard-knowledge 39, hard-math 27 dominate) → the expert answers the
  wrong question; **26/108 (24%) clean transcript but expert wrong**
  (trap 17 — expert knowledge limit, shared with the gold arm, so partly
  not channel loss at all); 5 relay-drop; ~8 ref/judge noise (7 blank
  refs + 1 in-both-but-judged-fail; one easy-fact ref is broken —
  "first Boston Marathon finishers" keyed to "$85,000").
- Magnitude cross-check: per-escalated gap at aggressive = gold-inject
  .864 − heard .621 = .243 ≈ the five-arm ASR-distill gap (gold .82 −
  self-1best .58 = .24). The relay adds ≈ nothing on top.
- **Consequence: "better relay paradigm" / "keep the talker silent" /
  seq-length ablations target ≤5% of the loss.** The lever is what the
  expert *receives* (the question uplink), not how the answer is spoken.
  Paper sync same session: decomposition sentence added to the
  speech-channel-cost bullet in sections/live.tex.

### 8o — acc × latency joined: the Pareto tradeoff figure ($0, 2026-08-09)

User ask (Aug-3 comment): "a matrix of how much latency bought how much
acc + a Pareto-frontier figure, cherry-picking allowed." Pure join of
frozen readouts — acc + CIs from `live_dualview.json` (8h), per-arm
latency from `latency_profile.json` (8i); no re-runs. Script
`figures/pareto_latency.py` → `pareto_latency.{png,pdf}` (repo +
paper/figures).

**Marginal exchange rates (heard-acc, P50 view; per-arm table in 8i):**

| segment | Δacc | ΔP50 |
|---|---:|---:|
| never → conservative | +4.6 pts (n.s.) | +0.7 s |
| conservative → balanced | +8.3 pts | +1.2 s |
| balanced → aggressive | +10.4 pts | +0.7 s |

- **balanced→aggressive is the cheapest segment per second**, for two
  measured reasons: the marginal escalations are easier queries whose
  experts return faster (escalated-row expert mean 8.9/7.5/6.7 s across
  the tiers), and each escalation displaces a local decode that itself
  averages ~4.6 s.
- Sanctioned cherry-picks in the figure: x = P50 (typical experience);
  the P99 expert tail stays in the table + a figure footnote. Both views
  drawn — gold-inject shares x positions (rescoring changes outcomes,
  not timing). Always ceiling = asymptote (synthesized, no live
  latency).
- **Random reference deliberately NOT drawn in latency space**: random
  arms were never run live, and random@matched-rate would have slightly
  *lower* latency than the gate arms (the gate escalates harder queries
  with slower experts), so any placement from frozen data would either
  flatter the gate or require unmeasured expert latencies on
  never-escalated queries. Gate-vs-random lives in rate space
  (fig:dualview) where it is exact.
- **Paper sync (same session): exchange-rate sentence + 
  Figure~fig:pareto-latency added to the latency paragraph in
  sections/live.tex; figure copied to paper/figures/.**
- **Legend relabel (2026-08-09, after teammate + user both misread the
  two curves):** "heard-acc (honest)" / "gold-inject (channel-
  controlled)" → "deployed: expert answers the talker's transcript" /
  "counterfactual: expert answers the gold text", and the gap annotation
  now states the 8m attribution (95% upstream of the relay: corrupted
  transcripts). The misreading being corrected: green is NOT the
  always-big arm (that is the .917 asymptote) — it is the same sessions
  at the same escalation rates with escalated rows re-scored on
  gold-text expert answers; the blue↔green gap is an *uplink* property,
  not a relay/injection property (8m: 64% corrupted transcript / 24%
  expert-wrong-shared-with-gold / 5 relay drops of 108).

### 8p — wav-pool integrity audit + listening pack ($0, 2026-08-12)

Meeting follow-up ("先解决TTS的问题 — 真正听一下现在读出来的声音"):
before re-rendering anything, audit whether the frozen pool's TTS files
are physically sound, and package the worst uplink failures for human
listening. Modal scan `scan_wavs.py` (stdlib peak/lead-silence over all
of `audio_pool`) → `figures/wav_audit.json`; listening pack at
`data/listen_pack/` (14 wavs + `cases.json` gold-vs-transcript +
README), served for phone listening at
https://rhe9527--tts-listen-web.modal.run/62dc5cd9 (`listen_app.py`,
`modal deploy`; scales to zero, stop with `modal app stop tts-listen`).
Pack = 2 good cases (q0578/q0588 — rare names + version numbers
transcribed verbatim, so the channel CAN be clean) + 6 true mishears + 6
broken/not-mishearing controls. Gotcha for anyone reusing the wavs: they
were written streaming, so RIFF/data sizes are 0xFFFFFFFF and players
show a 24-day duration and refuse to seek — patch the header to the real
byte length (librosa in the pipeline is unaffected).

- **File-level TTS is fine: 1/601 wavs broken** — `q0208` is 49 s of
  pure digital silence (peak=0, streaming render failure). The talker's
  "audio appears to be completely silent" transcript on that row was
  CORRECT, not a hallucination. No quiet renders (peak<3000: 0), no
  long leading silences. → The 🌟 "TTS 没读出来" hypothesis is FALSE at
  the file level; whatever TTS contributes is pronunciation-clarity,
  not broken audio.
- **The "corrupted transcript" bucket (8m's 64%) is heterogeneous** —
  eyeballing the 132 escalated ids ranked by query↔transcript difflib
  sim, at least four species: (a) rare-entity substitution (Mustafa
  Adebayo Balogun→Mustapha Arabo Balogun; Taurek→Turek); (b) spoken-math
  loss (999 − 103 heard as "nine hundred ninety nine hundred and three"
  — the operator vanishes; plus the known TTS-of-LaTeX artifact,
  `\Omega`/`\muF` read raw); (c) **not-transcription behavior: the
  talker sometimes ANSWERS instead of transcribing** (q0213 transcript =
  "D) Mongolia"; q0237 = a full answer to a 96 s question) — this
  corrupts the uplink but is an instruction-following failure, not
  hearing. [CORRECTION same day: an earlier draft listed source-text
  mojibake as species (d) based on `��pleasure��` in console output —
  the actual characters are U+201C/201D curly quotes (normal
  typography); the `��` was this machine's GBK console failing to print
  them. No mojibake exists in the pool; q0233's garbling is ordinary
  option-content mishearing.] Also: the difflib sim metric
  overstates corruption on rows that merely spell out digits (q0169
  content-intact at sim .054) — don't use it as a corruption *rate*.
- Meeting's proposed "orange line" (uplink transcript → expert, no
  relay) needs no new runs: 8d's five-arm table IS that arm on the
  escalated subset (self-1best .58 vs gold .82, expert answers scored
  directly), and 8m bounds the relay at 5/108 fails — orange ≈ blue
  ≈ 2 pts above it; the gap is uplink. Documented here so the ask
  doesn't resurface as an experiment.
- Next actions recorded in todo.tex: human listening verdict on the
  pack (does alloy enunciate entities/operators clearly?); re-render
  the escalated-fail wavs with a newer TTS (gpt-4o-mini-tts,
  instructed enunciation) and re-run the escalated subset — if heard-acc
  moves, the dual-view gap narrows and the figure gets redrawn;
  separate (c)-species rate from true mishearing (cheap: classify 132
  transcripts).

### 8q — input-side fairness audit: the "speakable subset" curve ($0, 2026-08-12)

User challenge: the bad cases make the live acc unfair — how to remove
the drop caused by "unfair" questions? Methodological rule enforced:
**exclusion must be decidable from the INPUT alone (query text + wav
bytes), never from outcomes** — anything else is cherry-picking.
Script `figures/fair_subset.py` → `figures/fair_subset_audit.json`.

Flags (pre-registered, input-side): `latex` = formula-symbolic content
(backslash commands, `^{`/`_{`/`^x` exponents, `$..$` only when the
inside contains math operators — plain dollar amounts like "$815.50"
deliberately NOT flagged, first draft over-flagged them);
`broken_wav` = digital silence from the 8p audit. Rare entities and
hard names are deliberately NOT flags — mishearing them is the
phenomenon. Result: **22/240 test ids unfair** (12 hard-knowledge with
embedded LaTeX, 10 hard-math formula problems; q0208 among them).
[Also corrected in the same pass: no mojibake exists in the pool —
8p's species (d) was a GBK console display artifact.]

**Dual-view on the fair subset (n=218, paired bootstrap 10k, seed 42):**

| tier | esc | heard (full→fair) | channel gap (full→fair) [CI fair] |
|---|---:|---|---|
| never | 0% | .400→.440 | — |
| conservative | 12% | .446→.486 | .054→.032 [+.014,+.060] |
| balanced | 32% | .529→.569 | .108→.069 [+.037,+.106] |
| aggressive | 52% | .633→.688 | .133→.083 [+.041,+.124] |
| always (synth) | 100% | .917→.922 | — |

- **~38% of the measured channel cost was the unspeakable-content
  artifact** (aggressive gap .133→.083); the remaining .083 is still
  significant and is the honest speech-channel price (entity
  substitutions, spoken-number slips) — it must NOT be excluded.
- Clean asymmetry as the sanity check: the gold-inject ceiling barely
  moves (.917→.922) — LaTeX questions specifically destroy the SPEECH
  channel, they are not intrinsically harder for the expert.
- Flagged rows account for 17/50 aggressive escalated heard-fails
  (34%), 12/39 balanced, 7/19 conservative.
- This formalizes 8d's existing scope note ("formula-symbolic math out
  of scope for the audio arm — nobody speaks LaTeX at a voice
  assistant") into a per-id filter applied uniformly to all arms.
- Borderline case NOT excluded: q0271 ("Estimate 999 − 103") is
  speakable arithmetic; whether the TTS actually voiced the minus is a
  render-quality question → if the listening pass confirms the minus is
  unvoiced, a third input-side flag (`render_defect`, verified by ear)
  becomes legitimate. Pending the user's listening verdict.
- Paper decision PENDING (user to ratify): make the speakable-subset
  curve the headline dual-view figure with the full-pool curve in the
  appendix, criteria stated as pre-registered input-side filters.

### 8r — audio-direct-to-expert: TTS exonerated, the leak is the talker's ears (~$5, 2026-08-12)

User-approved follow-up to 8q. For all 132 unique escalated test ids,
the ORIGINAL pool wav went straight to an audio-capable expert
(`modal_expert_audio.py::audio_expert`; auto-picked **gpt-audio** — no
gpt-5.5-class audio model exists in the API today) — no MiniCPM
self-transcription anywhere in the uplink. Judged against gold query +
reference by the standard judge (identical protocol to
`expert_adequate`). Control arm `::text_control`: SAME model, SAME
questions as gold text (gpt-audio rejects text-only requests, so the
text rides with a 0.25 s silent placeholder wav the model is told to
ignore). The two arms split "gpt-audio is a weaker brain than gpt-5.5"
from "the audio channel loses content". Anti-flag measures inherited
(per-id volume cache, user= attribution, concurrency 3, region-pinned).
132/132 + 132/132 collected, zero errors. Artifacts:
`audio_expert{,_text}.parquet` on gate-data + repo `data/`; analysis
`figures/audio_uplink.py`.

**Four-arm decomposition (same 132 escalated ids, same judge):**

| pool (fair subset, n=113) | gpt-5.5 text | gpt-audio text | gpt-audio audio | brain / channel |
|---|---:|---:|---:|---|
| easy-chat (24) | .88 | .83 | .71 | +.04 / +.12 |
| easy-fact (28) | .93 | .86 | .86 | +.07 / **.00** |
| hard-knowledge (34) | .91 | .59 | .41 | +.32 / +.18 |
| hard-math (9) | 1.00 | .89 | .56 | +.11 / +.33 |
| trap (18) | .61 | .50 | .50 | +.11 / **.00** |
| ALL | .87 | .72 | .61 | +.15 / +.11 |

(Full 132-id set: .86 / .71 / .54, brain +.15 / channel +.17 — the
extra channel loss vs the fair subset is the 8q LaTeX rows.)

- **⭐ TTS EXONERATED for speakable content.** On the two pools whose
  failures drove the "TTS 没读清楚" hypothesis, a good ear loses ZERO
  crossing the audio channel: trap .50→.50, easy-fact .86→.86. Star
  cases from the same wavs MiniCPM garbled: gpt-audio heard "Estimate
  999 − 103" (the minus IS voiced — q0271 adequate) and "Mustafa
  Adebayo Balogun" verbatim (q0552 adequate). **The meeting's 🌟
  root-cause question is answered: the alloy renders carry the content;
  the deployment loss is MiniCPM's own ASR.** Re-rendering with a
  better TTS is now PREDICTED NEGATIVE for entity/operator errors.
- **The remaining .083 fair gap attribution chain is CLOSED:** relay
  ≤5% (8m) → TTS 0 on speakable pools (8r) → expert fine on gold (.87)
  → the leak is the talker's transcription step, and it is not
  prompt-fixable (8d). Fixes are model-side (better duplex ASR) or
  architecture-side (forward audio, below).
- **Audio-direct as a deployment direction: REJECTED by user (same
  day).** The backend thinker stays TEXT-BASED — reasoning frontier is
  text-first, and binding the expert to an audio-native model accepts
  a permanent brain discount (measured −.15 here: gpt-audio gold-TEXT
  .72 vs gpt-5.5 .87; net swap negative, aggressive .638 vs deployed
  .688, `audio_uplink.py`). 8r stands as a DIAGNOSTIC control only.
  The one text-backend-compatible variant (audio uplink → cloud ASR →
  gpt-5.5 text) is already lower-bounded by the 8d Whisper arm (+4pp
  overall; fixes trap entities to gold-low, cannot fix MCQ option
  walls) — judged not worth further spend.
- **Long MCQ blocks are inherently audio-hostile even for the best
  ears** (hard-knowledge channel +.18 on speakable rows): holding 10
  spoken options is the failure, not entity perception. Strengthens the
  8q scope position and the public-benchmark arm (short open
  questions, no option walls).
- Caveats: easy-chat channel +.12 = 3 rows of open-ended judge
  variance (n=24); hard-math fair n=9; silent-placeholder control is
  mildly unnatural (declared in the prompt).

Spend ≈ $5 (264 gpt-audio calls + 264 judge). Project total ≈ **$380**.

**Addendum (earlier same day, 8q) — what the remaining fair gap is made of.** The 72
fair-subset escalated heard-fail rows split by transcript cleanliness
(sim .85): **35 clean-transcript** (trap 20, easy 12 — the expert is
wrong on gold too; these fail in BOTH views so they do NOT contribute
to the gap) vs **37 dirty-transcript** = genuine uplink loss,
concentrated in hard-knowledge 23 + hard-math 9 (long MCQ option
blocks). Of the 37, the gold expert answers **34/37 correctly** → the
remaining .083 gap is almost entirely recoverable-in-principle: deliver
the question faithfully and it converts. Species check: answered-
instead-of-transcribed contributes ~nothing to the LOSS — when the
talker answers instead of transcribing it usually already knows the
answer (q0213 "D) Mongolia": heard_ok=1 in both arms); only q0237
(96 s query) is a not-transcription fail. Next lever ranked: (1)
audio-direct-to-expert (send the wav, skip self-transcription — bounds
channel-inherent loss with the best available ears, ~250 calls, and
directly separates "TTS pronounces entities lossily" from "MiniCPM's
ears are weak"); (2) better-TTS re-render; (3) already measured
NEGATIVE: robust prompt, k-best GER; Whisper ears +4pp only.

---

### 8s — external-benchmark arm: the 8-figure deliverable (~$40, 2026-08-12/13)

User request: 8 figures — {our pool fair-subset, Speech TriviaQA, Speech
Web Questions, SD-QA} × {latency↔acc, escalation↔acc}. Infra:
`modal_bench.py` (build/transcribe/bench_live/ceiling/report — the sdqa
pipeline generalized to OpenAudioBench pools; 250 q each, seed 42,
official pre-rendered audio, NOT our TTS), SD-QA got its missing tiers
via the existing `run_sdqa_live`. All sweeps ran the FROZEN L22 gate +
frozen eot thresholds — zero recalibration; expert gpt-5.5 low, cached,
concurrency ≤3 throughout. Figures `figures/{bench}_{dualview,pareto}.*`
+ `fair_{dualview,pareto_latency}.*`, numbers in `bench_figures.json` /
`fair_figures.json`; all copied to paper/figures/. Gallery:
https://rhe9527--figures-gallery-web.modal.run/62dc5cd9

**Heard-acc per arm (never/cons/bal/agg), realized esc, ceiling:**

| pool | floor | cons | bal | agg | esc (c/b/a) | gpt-5.5 gold | official |
|---|---:|---:|---:|---:|---|---:|---:|
| ours-fair (218) | .440 | .486 | .569 | .688 | 12/32/52% | .922 | — |
| striviaqa (250) | .620 | .632 | .688 | .784 | 2/19/46% | .968 | 75.5 |
| swebq (250) | .412 | .436 | .456 | .604 | 6/23/47% | .736 | 70.2 |
| sdqa (200) | .495 | .535 | .595 | .720 | 3/16/44% | .930 | — |

**Gold-inject (green) added to the external figures (user question
2026-08-13 — the ceiling parquets already carry per-id gold-text expert
verdicts, so the counterfactual costs $0):** agg-arm gold-inject
striviaqa .824 / swebq .628 / sdqa .745. Two readouts: (1) **external
channel cost is small** — agg blue↔green gap .040/.024/.025 vs our
pool's .083, exactly the 8q prediction (short open questions transcribe
cleanly; no option walls); (2) **in the channel-controlled view the
gate DOES clear random on striviaqa** (+.044 at 46% vs the gold-paired
random line) — part of the deployed-view "gate ≈ random" is channel
cost eating the selection margin, same structure as the 600-pool
dual-view.

- **The gate TRANSFERS mechanically everywhere**: frozen thresholds
  fire at sane (compressed) rates on easier pools, curves rise
  monotonically, aggressive lifts +16/+19/+22 pts over the floors.
  P50 latency stays cheap (striviaqa 1.2→2.1 s, sdqa 1.4→2.3 s).
- **⚠️ Honest headline: gate ≈ random on Speech TriviaQA** (agg .784 vs
  random-at-46% .780); above random on swebq (+.04 at 47%) and sdqa
  (+.034 at 44%), both within per-arm CI width (n=250/200 — likely
  n.s. individually). Consistent with the 5b/8f audit: on our mixed
  pool a large share of the gate's random-margin is the pool/type
  shortcut; single-type external pools remove that shortcut and expose
  the thin per-query residual. The transferability figure honestly
  shows: mechanism ports, selectivity margin is small on easy
  homogeneous pools.
- Floors sit below the official chat-mode numbers (striviaqa .620 vs
  .755) — live streaming loop + judge severity, same offline→live tax
  as the 600-pool (8h: .588→.400). WebQ ceiling is only .736 (strict
  Freebase refs cap even gpt-5.5), so the .412 floor reads relative to
  that, not to 1.0.
- SD-QA escalated-subset acc .82-.84 at bal/agg — on REAL human speech
  the escalation payload works (expert latency P50 3.1-3.4 s).
- Judge protocol identical throughout (gpt-5.4-mini, ref-anchored;
  TriviaQA refs carry accepted-alias lists).

**Router-quality decomposition (user question 2026-08-13, $0):** eot
score's AUC against the never-arm local-fail label, plus
aggressive-arm escalation precision/recall:

| pool | base-fail | eot AUC | esc | precision (lift) | recall |
|---|---:|---:|---:|---|---:|
| ours-fair | .56 | .751 | 52% | .74 (×1.33) | .69 |
| striviaqa | .38 | .676 | 46% | .51 (×1.34) | .62 |
| swebq | .59 | .759 | 47% | .78 (×1.33) | .63 |
| sdqa | .51 | .719 | 44% | .68 (×1.35) | .59 |

Per-query discrimination is real and transfers (.68-.76 OOD vs .751
in-mix; in-calibration OOF was .843) with an eerily uniform ×1.33-1.35
precision lift — the transferring component looks like a generic
difficulty/confidence signal, not pool structure. The acc margin this
buys over random (+4-6 pts, gold view) is what AUC ≈ .7 mathematically
yields at these rates. Threshold quantiles transfer conservatively
(conservative tier fires 2-6% vs 14% in-mix — score distributions
shift on easier pools). Verdict logged: mechanism trained "enough to
transfer", not "enough to be decisive"; levers = calib-pool expansion
(todo, 500-1000/pool + more families) and unsupervised per-domain
threshold rescaling; the zero-training pitch survives as-is.

Spend ≈ $40 (experts for ~600 escalated + 700 ceiling calls, low
effort + judges + ~12 H100-hours). Project total ≈ **$420**.

**SD-QA dualview redraw ($0, 2026-08-13, user call):** on the SD-QA
escalation-vs-acc figure only, the grey random-escalation reference is
dropped and a **local-only floor line at 0.495** is drawn instead, so the
curve is read between the two bounds that matter (floor it starts from,
gpt-5.5 ceiling 0.930 it walks toward). `bench_figures.py` now takes
per-bench `random_line` / `floor_line` flags; the other three pools keep
the random reference (their gate-vs-random margin is the point there).
Regenerated + copied to paper/figures/; numbers unchanged.

---

### 8t — probe v2: retrain on the deployed signal + expanded pool ⭐ (~$25, 2026-08-13)

User: "你继续训练吧" — act on the 8s router diagnosis. `modal_train.py`
fixes BOTH diagnosed gaps at once: (1) **train/deploy signal mismatch**
— v1 was fit on chat-style full-prefill `h_last` but deployed on the
streaming end-of-turn read; v2 trains directly on the eot-read hidden;
(2) **calib coverage** — +800 public-benchmark queries (TriviaQA 150,
dolly 150, SimpleQA 100, GSM8K 150, **NQ-open 150**, **ARC-Challenge
100** — two new families), 8q speakable-filtered, deduped vs the frozen
pool, seed 43 (decorrelated from the eval seed), tts-1/alloy rendered,
answered from audio by MiniCPM and judged with the standard judge.
Expansion fail-rate spectrum: trap .98 / know-open .65 / easy-fact .38
/ easy-chat .32 / hard-math .09 / ARC .09 (pooled .41 — better balanced
than the original .56). **External eval pools stayed strictly out of
training; frozen 600 untouched; artifact = `midlayer_gate_audio_v2.json`
(v1 intact).** EOT hiddens captured once for all five pools (2100
streaming replays → `eoth_{tag}.shard*.npz`), so any future refit is
CPU-only.

**Transfer (AUC vs never-arm local-fail, identical eot hiddens):**

| pool | v1 | **v2** | Δ |
|---|---:|---:|---:|
| striviaqa | .676 | **.761** | +.085 |
| swebq | .759 | **.779** | +.020 |
| sdqa (real speech) | .719 | **.775** | +.056 |
| frozen-test (in-mix) | .811 | **.860** | +.049 |

In-mix does NOT regress (the pre-registered guard) — it improves. v2
OOF on its own train set = .878 (C=.0003, n=1160).

**Ablation — what bought the gain (external mean AUC):**

| fit | mean |
|---|---:|
| v1 (full-prefill signal, 360) | .718 |
| frozen-360 only, eot signal | .741 |
| expansion-800 only, eot signal | .765 |
| **both = v2 (1160)** | **.772** |

**Both levers real, roughly equal**: signal-match +.023, data +.031.
Notably expansion-alone already beats frozen-alone by +.024 — the new
families (open-domain NQ + ARC) carry transferable difficulty signal
the original five pools did not.

**Deployment readout at a matched 30% budget** (per-domain quantile
thresholds — label-free, needs only unlabeled score history):

| pool | base fail | v1 prec/rec | **v2 prec/rec** |
|---|---:|---|---|
| striviaqa | .38 | .53 / .42 | **.63 / .49** |
| swebq | .59 | .83 / .42 | **.84 / .43** |
| sdqa | .51 | .75 / .45 | **.80 / .48** |

striviaqa precision lift ×1.39→**×1.66** over base — the pool where 8s
found "gate ≈ random" is exactly where v2 helps most.

- **⚠️ Threshold-quantile transfer is STILL broken, and v2 flips its
  sign**: at the frozen global thresholds v1 under-fires (sdqa balanced
  16% vs 30% target) while v2 over-fires (sdqa balanced 47%,
  aggressive 80%). Score distributions shift per domain; a global
  quantile cannot follow. **Deployment recommendation: per-domain
  quantile thresholds** (used in the table above — label-free, exactly
  hits the budget by construction, and the AUC gain is
  threshold-independent so it survives). The zero-training claim is
  unaffected: still no gradient through the backbone, still a linear
  read on frozen activations; the calibration set simply grew.
- Live re-run DONE — see 8u below (this bullet's "not yet done" is
  resolved).

Spend ≈ $25 (800 TTS + 800 judge + ~10 H100-hours). Project ≈ **$445**.

### 8u — v2 live re-run: all 8 figures refreshed ⭐ (~$35, 2026-08-13)

User: "重新跑曲线". 12 live sweeps (4 pools × 3 escalating tiers, 2760
sessions) with the v2 probe + **per-domain quantile thresholds**;
never-arm rows reused from the v1 runs (thr=1e9 → probe never fires →
rows are probe-independent by construction; documented in
`report(never_glob=...)`). `modal_bench.py` generalized: POOLS registry
(frozen/striviaqa/swebq/sdqa), `art_path` + `suffix` params, so v1
artifacts and traces stay untouched. Figures regenerated from
`{pool}_v2_traces.parquet`; v1 versions archived as `*_v1.{png,pdf}`.

**Fix #1 confirmed live — thresholds now hit their budgets** (the 8t
diagnosis): realized rates 15/35/55% (frozen), 15/30/50%, 15/30/50%,
15/31/50% — versus v1's 2/19/46% (striviaqa) and 3/16/44% (sdqa).

**Curves (heard-acc, v1 → v2):**

| pool | never | conservative | balanced | aggressive |
|---|---|---|---|---|
| ours-fair | .440→.436 | .486→**.500** | .569→**.573** | .688→.670 |
| striviaqa | .620→.624 | .632→**.684** | .688→**.728** | .784→**.840** |
| swebq | .412→.404 | .436→.436 | .456→**.532** | .604→.560 |
| sdqa | .495→.510 | .535→**.610** | .595→**.740** | .720→**.785** |

Raw acc is rate-confounded (v2 escalates more at conservative/balanced
because its thresholds are now correct). **Rate-normalised selectivity
(lift over the random line, channel-controlled view) is the honest
comparison:**

| pool | cons v1→v2 | bal v1→v2 | agg v1→v2 |
|---|---|---|---|
| ours-fair | +.021→+.027 | +.045→+.048 | +.081→+.083 |
| striviaqa | +.004→**+.028** | +.013→**+.053** | +.043→**+.092** |
| swebq | +.002→−.002 | −.026→**+.044** | +.063→**+.010** |
| sdqa | +.027→**+.052** | +.043→**+.125** | +.059→**+.100** |

- **8s's headline finding is overturned on striviaqa**: the pool where
  v1 was indistinguishable from random (+.004/+.013/+.043) now clears
  it at every tier (+.028/+.053/+.092) — selectivity roughly doubled
  to tripled. sdqa likewise (balanced +.043→+.125). In-mix (ours-fair)
  is unchanged, as designed.
- **⚠️ swebq is the exception and it is not clean**: balanced improves
  a lot (−.026→+.044) but aggressive DROPS (+.063→+.010) and heard-acc
  falls .604→.560. Its ceiling is only .736 (strict Freebase refs), so
  at 50% escalation the headroom left to select from is thin and the
  arm is noisy at n=250; also the only pool where v2's conservative
  lift is ≈0. Reported as-is; a rerun at larger n is the honest way to
  settle it, not a re-pick of the tier.
- Latency essentially unchanged (P50 within ±0.5 s of v1 at matched
  tiers); the gate read itself stays ~30 ms.
- Figures: `fair_{dualview,pareto_latency}` + `{bench}_{dualview,pareto}`
  ×3, all v2, in figures/ + paper/figures/; gallery redeployed
  (same URL).

Spend ≈ $35. Project total ≈ **$480**.

### 8v — VoiceBench AlpacaEval: the official-matrix row ⭐ (~$20, 2026-08-13)

User challenge: "官方 benchmark 表里没有你用的那两行". **Verified by
fetching the raw README/model-card HTML (no summarizer in the loop):
`Speech TriviaQA` 75.5, `Speech Web Questions` 70.2 and `Speech CMMLU`
59.2 DO exist — in the full Audio-Understanding table; the screenshot is
the condensed matrix, which lists only `VoiceBench AlpacaEval` 4.8 as
its speech-QA row.** So 8s/8u's anchors were right, but sourced through
a WebFetch summary (which had garbled an earlier fetch) — the raw-text
verification is now on record. Everything else in the condensed matrix
is out of scope by construction: vision rows (we run
`init_vision=False`; meeting scoped audio-only), ASR rows (measure WER,
not answering — routing cannot fix ears, and 8r scoped ASR out), speech
GENERATION rows (our loop emits text), Omni rows (audio+video).

Fourth external pool added: **VoiceBench AlpacaEval, all 199 items**,
same v2 probe + per-domain quantile thresholds, four arms.
**Scoring uses VoiceBench's own judge**: gpt-4o-mini + their
`meta_prompt_open` copied verbatim from `MatthewCYM/VoiceBench`
`api_judge.py` (1-5, bare number out) — our first pass with a
home-grown rubric gave 2.78/4.86, i.e. ~2 points below the official
scale, confirming the rubric (not the model) drives the absolute level.

| arm | esc | judge score (1-5) | gold-inject |
|---|---:|---:|---:|
| never | 0% | 3.94 | 3.94 |
| conservative | 15% | 4.08 | 4.12 |
| balanced | 30% | 4.09 | 4.16 |
| aggressive | 50% | **4.26** | 4.37 |
| always (gpt-5.5, gold text) | 100% | — | **4.96** |

- **⭐ Chat-mode control settles the fairness question (user challenge,
  same day): the SAME 199 wavs answered offline with `model.chat`
  (1024-token budget, no streaming loop, no chunked prefill, no EOT
  read), judged identically, score **4.86** — i.e. we REPRODUCE the
  official 4.8 (slightly above it).** `valpaca_chatmode.parquet`,
  `modal_bench.py::valpaca_chatmode`. Therefore the entire 3.94→4.86
  gap is OUR LIVE LOOP, not model capability and not the judge:
  paired, the live loop is worse on 115/199 queries, better on 3.
  Mechanism = answer length: chat-mode median 2186 chars vs live 820.
  The duplex/omni system prompt puts the model in *spoken-reply* mode
  (short, conversational) while AlpacaEval's rubric rewards complete
  written answers; the live 512-token cap adds truncation on top
  (27% of live answers end mid-sentence, and the worst paired case —
  chat 5 vs live 1 — is a 3030-char answer cut off mid-clause).
  Consequence for the paper: **the official 4.8 line must be labelled
  as an offline-chat-mode number, and our own chat-mode 4.86 is the
  honest capability reference; only the four live arms are comparable
  to each other.** Ruled out as explanations along the way: degenerate
  repetition (4/199 rows, removing them moves 3.94→3.96) and the
  scoring rubric (already VoiceBench's own).
- **⚠️ On this pool the gate does NOT beat random** (aggressive 4.26 vs
  random-at-50% ≈ 4.45), and the mechanism is now measured, not
  inferred: **the queries the gate selected score 3.90 on the never
  arm vs 3.98 for those it skipped — no discrimination at all**, while
  escalated rows still gain (3.90→4.71) purely because gpt-5.5 writes
  better long-form answers. So the gain is expert quality, not
  selection; random picks would harvest the same. Root cause:
  open-ended instruction following has no "the model doesn't know this
  fact" event for the probe to read — every answer is partially
  creditable and the score range is compressed (3.94→4.96 = 1.0 point
  vs 40 accuracy points on our pool). Textbook species-3 of the
  three-failure taxonomy. Reported as a NEGATIVE result.
- Latency here is higher (P50 4.6 → 9.7 s) because AlpacaEval answers
  are long-form essays; the local decode dominates, not the expert.
- Figures `valpaca_{dualview,pareto}.{png,pdf}` (figures/ +
  paper/figures/), gallery now shows 10.
- Judge-infra gotcha: gpt-4o-mini 429s under batch load silently
  produced None scores concentrated at the tail of the batch (never arm
  lost 131/199 in the first pass). Fixed with concurrency 3 + 6-step
  backoff + persisted `score_err`; all four arms now n=199.

Spend ≈ $20. Project total ≈ **$500**.

---

### 8w — judge-protocol alignment: the official numbers ARE reproducible ⭐ (~$15, 2026-08-13)

User asked for comparison models on the figures; the chat-mode control
(8v's machinery, generalized to every pool) exposed a prerequisite
problem first. Offline chat mode under OUR judge: striviaqa .684 vs
official .755, **swebq .464 vs official .702** — a 24-point hole that
no loop tax explains. Root cause: **the judge**. We now copy
OpenAudioBench's own judging verbatim from
`tasks/trivia_qa_audio.py` (gpt-4o-2024-08-06, JSON
`analysis`+`judgment`, "correct if it matches **at least one** of the
reference aliases"; `_oab_judge` / `oab_rejudge_live` in
modal_bench.py) and re-scored every arm plus both ceilings.

**Chat-mode control under each judge (n=250):**

| pool | our judge | **OAB judge** | official |
|---|---:|---:|---:|
| striviaqa | .684 | **.712** | .755 |
| swebq | .464 | **.716** | .702 |

**We reproduce the official numbers** (swebq slightly above; striviaqa
4 pts below = subsample + protocol). Our reference-anchored judge is
simply much stricter on WebQ's Freebase alias lists. **Consequence: the
"swebq ceiling is only .736 / headroom is thin" story in 8s/8u was a
judge artifact — under the official judge the ceiling is .844.**

**All arms re-scored on the official scale (v2 probe):**

| pool | never | cons | bal | agg | ceiling | official MiniCPM | Qwen3-Omni-30B | Kimi-Audio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| striviaqa | .664 | .720 | .764 | **.860** | .972 | .755 | .629 | .419 |
| swebq | .572 | .648 | .680 | **.736** | .844 | .702 | .749 | .464 |

- **⭐ The headline comparison the paper needs**: on Speech TriviaQA a
  9B duplex model + our gate at 50% escalation scores **.860 live**,
  vs **.629** for Qwen3-Omni-30B-A3B (3.3× the parameters, offline) and
  .755 for MiniCPM-o's own official offline number. Routing beats
  scaling the speech model here. On swebq we clear the official
  MiniCPM number (.736 vs .702) and land just under Qwen3-Omni (.749).
- **Loop tax is now fully accounted**: striviaqa live floor .664 vs
  our chat-mode .712 (−.048 = the streaming loop) vs official .755
  (−.043 = 250-item subsample + protocol). No unexplained gap remains.
- **v1 vs v2 on the correct scale (lift over random, gold view)**:
  striviaqa +.024/+.031/+.052 → **+.033/+.062/+.080** (v2 wins at every
  tier, confirming 8u). swebq conservative −.001 → **+.045** and
  balanced +.045 → +.035, but **aggressive +.092 → +.066** — so the
  8u "swebq aggressive regression" SURVIVES the judge correction
  (smaller: −.026 lift instead of −.044 raw acc). It remains the one
  arm where v1 selected better; n=250, not re-picked.
- Comparison-model numbers extracted from the official table by
  position-mapping the raw HTML (verified against three values in the
  user's screenshot): Kimi-Audio and Qwen3-Omni-30B-A3B-Instruct, both
  offline chat mode — figure captions must say so.

### 8x — two more external pools: Llama Questions + Reasoning QA (~$25, 2026-08-13)

Added to cover the failure species the external arms still missed.
`sllama` = OpenAudioBench Llama Questions (250 of 300, English short
factoid), `sreason` = OpenAudioBench Reasoning QA (all 202,
**Chinese**, execution-type reasoning — no official MiniCPM number
exists for it, so no official line on its figure; it doubles as a
cross-lingual transfer test since the probe is calibrated mostly on
English). Four arms each, v2 probe, per-domain thresholds — realized
rates 15/30/49% on both.

**Data-integrity bug found and fixed while building these:**
`reasoning_qa`'s CSV keys audio by `.mp3` filenames while the builder
stripped only `.wav`, so keying failed and silently **fell back to
row-order pairing** — questions would have been matched to the wrong
audio. Now: strip any extension, and the row-order fallback raises
instead of guessing. The first sreason build was discarded. (Also
added `参考答案` to the reference-column detector.)

Per-domain thresholds again show why a global quantile cannot work:
the aggressive threshold is .111 on sllama vs .603 on sdqa — the same
probe's score distribution shifts by 5× across pools.

**Results (sllama on the OAB judge, sreason on ours):**

| pool | never | cons | bal | agg | always (gpt-5.5) |
|---|---:|---:|---:|---:|---:|
| sllama | .840 | .884 | .912 | **.944** | .924 |
| sreason (zh) | .584 | .624 | .683 | **.762** | .871 |

**⭐⭐ sllama: selective escalation BEATS always-escalate (.944 > .924)
— the strongest positive result in the project so far.** Decomposed on
the aggressive arm (n=250, official judge):

| gate decision | local model alone | gpt-5.5 alone |
|---|---:|---:|
| kept local (125) | **.976** | .960 |
| escalated (125) | .704 | **.888** |

The probe cleanly split a .976 subset from a .704 subset — genuine
per-query discrimination, not a pool/type shortcut (single-type pool).
**[SUPERSEDED 2026-08-20 by §8ad: paired McNemar gives p=1.00 on the easy half — "matches", not "beats"; the robust claims are the .976/.696 split (z=6.46) and the .696→.888 lift on the hard half (p<.0001).]** And because the small model *beats* the expert on the easy half
(.976 vs .960 — gpt-5.5 over-elaborates short factoids), **"escalate
everything" is NOT an accuracy upper bound**; only a selective router
collects the max of both. This is the cleanest external evidence for
the system's premise, measured with the benchmark's own judge. Judge
noise checked: 4 rows where the relay is right and gpt-5.5 wrong vs 2
the other way — an order of magnitude below the structural effect.

sreason adds the missing failure species (execution-type reasoning) and
a cross-lingual test: the probe was calibrated almost entirely on
English yet still lifts Chinese reasoning .584→.762 at 50%. Its
official-rubric scoring (per-item 打分prompt) was NOT replicated —
numbers are on our judge, no official line drawn.

**Why the P50 latency curves zigzag (verified 2026-08-14).** sllama's
per-arm P50 is non-monotonic — never 1.52s → cons **1.19s** → bal
1.73s → agg 1.64s — and this is a median-of-a-mixture effect, not a
measurement or gate bug. The probe escalates exactly the queries whose
*local* decode is longest (escalation@balanced by never-arm answer-length
quintile: 2/12/34/46/56%; conservative's 38 escalated ids had local P50
2.52s vs 1.27s for the 212 that stayed). At 15% the local pool loses its
slowest members (local-only P50 1.52→0.98s) while only 38/250 rows pay
the ~4.3s expert path, so the overall median *falls below never*; at 30%
the escalated mass reaches the median and pushes it back up. Mean
latency is monotonic (1.65/1.75/2.37/2.38s) — the fold-back lives only
in the quantile. On sreason the P50 is monotonic (3.17→3.84s) but the
**tail improves with escalation: P90 13.25→10.94s** (mean 5.08/5.20/
4.73/5.16) — long local CoT decodes (tail >13s) get replaced by 3–4s
expert round-trips, so on reasoning pools escalation buys accuracy AND
truncates the latency tail. Same frozen probe in all arms — nothing is
trained, and no latency model exists to "train"; the x-positions are
measured wall-clock medians.

### 8y — figures refreshed with comparison models (2026-08-13)

All 14 figures regenerated: 5 external pools × {escalation↔acc,
latency↔acc} + our pool × 2 + AlpacaEval × 2. OpenAudioBench pools now
carry four reference lines on one scale (their own judge): official
MiniCPM-o 4.5, **Qwen3-Omni-30B-A3B**, **Kimi-Audio**, and our own
offline chat-mode control. Gallery (14 figures):
https://rhe9527--figures-gallery-web.modal.run/62dc5cd9

Headline for the paper: on Speech TriviaQA, **9B + gate @50% = .860
live**, vs Qwen3-Omni-30B **.629** and MiniCPM-o's own official
**.755** — routing beats scaling the speech backbone, and on
Llama Questions it also beats escalating everything.

### 8z — probe v3: RL/SFT rejected; data + multi-position features executed ⭐ (~$45, 2026-08-16)

User asked whether RL (or SFT) should train the probe. **Decision: NO
to both** (rationale recorded in `todo.tex`): (1) the gate is a
single-step decision whose BOTH counterfactuals are observable offline
(never/always arms) — that is cost-sensitive supervised classification,
and policy-gradient RL would re-derive the same Bayes classifier with
orders-of-magnitude worse sample efficiency at n≈1k; (2) SFT on the
backbone breaks the zero-training frozen-checkpoint claim, shifts the
talker's answer distribution (invalidating every measured curve), and
small-n training is already falsified in-house (8f/8s router .669 <
pool-oracle .715); (3) the binding constraint is domain shift + judge
label noise (OOF .878 vs external .76–.78), which neither touches.
Executed the two supervised levers instead (`modal_train2.py`):

**1. expansion2** — 1150 new queries, 7 families NONE in the v1 mix
(dedup vs frozen + expansion + all 6 external pools, seed 44, tts-1
alloy, MiniCPM audio answers, standard judge). Fail-rate spectrum:
easy-mathword(SVAMP) .10 / know-openbook .17 / know-mmlu .31 /
know-commonsense .32 / trap-truthful .45 / hard-multihop(HotpotQA) .79
/ know-longtail(PopQA) .84 — pooled .50, adds the high-difficulty mass
the .41 expansion1 mix lacked. Train pool now 360+800+1150 = **2310**.

**2. multi-layer capture (`eoth2_*.npz`)** — ONE streaming replay per
query over all 9 pools (3901 replays, zero missing) storing
L{14,18,22,26,30} × (eot rolling last-8-token window + user-audio-mean)
in float16 → every future probe refit is CPU-only forever. Engineering
note (caught in smoke): the streaming assistant prefill runs 1-token
forwards, so "last-8 of the final forward" degenerates to a single
token — the tail window must roll ACROSS forwards.

**Refit sweep (19 configs, 5-fold OOF on train only):** L22 is still
the best single layer (eot_last .858; matches 5d), **multi-layer
concat HURTS** (L18+22+26 .837 — regularization cost exceeds the
information), **position diversity helps**: winner =
`eot_last + eot_mean8 + user_mean @ L22` (12288-d), C=1e-4, OOF
**.864**. All three reads are online-computable at zero eot latency
(running audio mean + rolling tail + last token).

**Transfer (AUC vs never-arm local fail, externals read once):**

| fit | striviaqa | swebq | sdqa | sllama | sreason | frozen-test | ext-mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 stored artifact (sanity) | .761 | .779 | .775 | .815 | .621 | .860 | .750 |
| A v2-recipe refit (L22 last, frozen+x) | .761 | .779 | .775 | .815 | .621 | .860 | .750 |
| B data lever (A + expansion2) | .762 | .804 | .780 | .817 | .682 | .872 | .769 |
| C feature lever (winner cfg, frozen+x) | .791 | .758 | .788 | .811 | .628 | .873 | .755 |
| **D v3 = winner cfg + all data** | **.789** | **.785** | **.792** | .806 | **.683** | **.879** | **.771** |

- **Sanity anchor exact**: the re-captured L22 eot hidden reproduces
  8t's v2 numbers to 3 decimals, and refit A equals the stored
  artifact — capture + fit fully deterministic.
- **Data is the bigger lever again** (+.019 ext-mean vs +.005), same
  structure as 8t; combined +.021 (.750→.771).
- **⭐ Cross-lingual transfer jump: sreason .621→.683 (+.062)** — the
  new ENGLISH multihop/long-tail families improve CHINESE reasoning
  transfer; the difficulty signal the probe reads is language-general.
- striviaqa .761→.789 (+.028), sdqa .775→.792 (+.017); the one
  regression is sllama .815→.806 (−.009, was the best pool).
- **Guard passed with headroom**: frozen-test .860→.879 (pre-registered
  "must not regress"; selection never saw the externals).
- Honest note: B ≈ D on ext-mean (.769/.771) — if deployment ever
  wants a single-vector probe, B (plain L22 eot_last on 2310) keeps
  ~all the external gain; D wins in-mix and on striviaqa/sdqa.

Deployment artifacts: `midlayer_gate_audio_v3.json` +
`gate_v3_{pool}.json` per-domain quantile thresholds (label-free,
8t recipe). **Live 4-arm re-run with v3 NOT launched** — separate
spend decision (~$35, as 8u); AUC gains are threshold-independent.

Spend ≈ $45 (1150 TTS + 1150 answers + 1150 judge + 3901 replays ≈ 14
H100-h). Project ≈ **$490**.

**8z-live — v3 live re-run, all 14 figures refreshed (~$55, 2026-08-16).**
User: "图刷新一下". `modal_bench.py::bench_live` got a v3 path (rolling
last-8 tail across forwards + running user-audio mean at L22, concat →
same `Probe.score`; v1/v2 path untouched); `make_thresholds3` scale bug
fixed (raw logit → sigmoid, Probe.score convention) and regenerated.
21 sequential sweeps (7 pools × 3 escalating tiers, 4773 sessions,
expert concurrency ≤3 throughout), never arms + ceilings reused
(probe-independent); reports re-judged everything fresh, OAB pools
re-scored with the official judge. Realized rates on budget externally
(15/30/50); frozen overshot to 16/35/61 (calib-quantile → test drift).

**Live v2 → v3 per arm** (OAB pools = official judge; others = ours):

| pool | judge | never | cons v2→v3 | bal v2→v3 | agg v2→v3 |
|---|---|---:|---|---|---|
| striviaqa | official | .664 | .720→.732 | .764→**.800** | .860→.860 |
| swebq | official | .57 | .648→.628 | .680→.680 | .736→.732 |
| sllama | official | .84 | .884→**.904** | .912→.904 | .944→**.948** |
| sdqa | ours | .51 | .610→.600 | .740→.720 | .785→**.795** |
| sreason (zh) | ours | .59 | .624→**.653** | .683→**.713** | .762→**.772** |
| frozen (full) | ours | .39 | .467→.483 | .533→.554 | .621→.596 |
| valpaca (1-5) | VoiceBench | 3.99 | 3.96→3.96 | 4.23→4.23 | 4.26→**4.35** |

(never-arm deltas across versions are ±.005–.013 = pure judge re-run
variance; same trace rows.)

- **Gains land exactly where the offline AUC gained**: striviaqa
  balanced +.036 (the probe's biggest AUC jump pool among OAB),
  **sreason uniform +.010–.030 — the cross-lingual offline finding
  (+.062 AUC) survives live**; sllama conservative +.020.
- **sllama headline strengthens: selective @50% = .948 > always-escalate
  .928** (both re-judged same run) — the project's strongest positive
  result confirmed with the better probe.
- swebq/sdqa ≈ flat (within judge noise; both were probe-flat in the
  offline table too: swebq +.006, sdqa +.017 AUC).
- ⚠️ **[SUPERSEDED 2026-08-20 by §8ad — re-mixing the same measured outcomes at exactly 50% gives .600, i.e. +.004: the overshoot is a COST bug, not an accuracy bug, and the .621→.596 delta is 1 paired SE = noise.]** Original text: frozen aggressive .621→.596 at esc .61 (vs .50) — the
  overshoot pushes ~26 extra math/LaTeX-heavy queries through the
  transcript-tax channel (8d: ASR-distill −.31 on knowledge entities);
  balanced/conservative arms improved (+.021/+.016). Fair-subset curve
  (fair_dualview): heard .422→.633 @ 57%, gold-inject .761.
- valpaca stays a negative result: agg 4.35 < random@50% ≈4.45
  (species-3 open-ended pool — no retrieval-failure event to read).

All 14 figures regenerated from `{pool}_v3_traces.parquet` /
`valpaca_v3_scored.parquet` (v2 versions archived as `*_v2.{png,pdf}`,
jsons refreshed), copied into `paper/figures/`, gallery redeployed
(same URL). Live-loop deployment artifacts now v3 end-to-end
(`gate_v3_{pool}.json`, sigmoid-scale thresholds).

Spend ≈ $55 (4773 live sessions ≈ 11 H100-h + ~1.6k expert calls
mostly cache-hits + ~7k judge calls). Project ≈ **$545**.

### 8ab — Jisen review pass: kink mechanism, latency decomposition, matched-rate precision, random-pareto lines ($0 local, 2026-08-18)

Jisen's figure numbers = the 14-figure gallery order (fair ×2,
striviaqa ×2, swebq ×2, sllama ×2, sreason ×2, sdqa ×2, valpaca ×2).
Everything below computed locally from the existing v3 traces — no
model calls, no re-runs.

**图4 (striviaqa_pareto) "small kink" is mechanism, not noise — and a
seed-rerun would NOT remove it.** The conservative arm's P50 (1.186 s)
sits LEFT of never (1.227 s). Decomposition: the 38 queries the probe
escalated at 15% had never-arm local P50 **1.711 s** vs **1.198 s**
for the 212 it kept — the probe preferentially escalates the slowest
local decodes (same median-of-a-mixture effect as sllama, §8x), so the
kept-local median falls (0.937 s) faster than the 3.8 s expert path
can pull the mixture back up. Statistically the −41 ms dip is far
inside noise (query-bootstrap CI on the P50 difference: [−285, +254]
ms) — but because the mechanism is deterministic, 3–5 seeds would
average toward the same left-fold, not away from it. If a definitive
check is ever wanted: rerun never+conservative only, ~$6/seed.

**Addendum (2026-08-24, user asked for the interpretation, not the
assumption): the kink's behavioral story, from the traces.** Local
answer length and decode time are the same variable (r = .97), and the
small model's signature failure mode on striviaqa is
**uncertainty-as-verbosity**: when it doesn't know, it hedges at
length. The 38 conservative-escalated queries vs the 212 kept, all
measured in the NEVER arm (no escalation anywhere): local P50 1.69 s
vs 1.18 s, median answer 267 vs 172 chars, local accuracy **.32 vs
.73**. Canonical example striviaqa0192 ("Where does Dame Edna Everage
come from"): locally the talker rambles 921 chars for 5.8 s about the
character not being real and gets it WRONG; the escalated path returns
"Moonee Ponds, Melbourne" in 3.1 s total — faster AND correct,
because the expert round-trip is flat ~3 s while the local ramble is
5-9 s. (Two more of the same shape: striviaqa0074 Aung San Suu Kyi
5.1→3.9 s, striviaqa0221 Rastafari 4.5→3.6 s.) The probe reads this
BEFORE the answer exists — the hidden state carries the hesitation the
verbosity would later express. Honest counterpart: the kept-pool's 27%
errors are FAST confident wrongs (striviaqa: "Joe Gargery → Our Mutual
Friend", 0.6 s, no hedging signature) — the species a 15%-budget gate
misses and the reason the curve keeps rising toward 50%. Also
clarified for the record: figure error bars are query-resampling
bootstrap from ONE live run per arm, not repeated runs; the fold's
SIGN replicates deterministically (identical probe scores → identical
escalated set), only its depth is within the noise band.

**Addendum 2 (same day, user challenge: "text models don't hedge? prove
it's not MiniCPM-specific"). The wrong-answers-are-longer signature is
cross-model.** Effect size = P(len_wrong > len_right), .50 = null,
never/local answers only:

| model | striviaqa | swebq | sllama |
|---|---:|---:|---:|
| MiniCPM-o 9B (speech) | .62 | .53 | .77 |
| NVDA VoiceChat 11B (other family) | .56 | .48 | .64 |
| gpt-5.5 (frontier TEXT, gold input) | .56 (n=8) | .47 | .67 (n=18) |

Three readouts: (1) NOT MiniCPM-specific — an architecturally
unrelated duplex model replicates the direction, attenuated because
its trained voice style is terse (median 52-82 chars vs MiniCPM's
156-400). Quantified as a pre-registered prediction (user follow-up
2026-08-24): fold depth is fueled by the answer-length TAIL the probe
can remove — MiniCPM striviaqa p90-p10 spread 550 chars ≈ the slow
decile costing +2.7 s over the median (at its measured 6.2 ms/char),
NVDA only 97 chars ≈ +0.4 s — so if the live loop is ever ported to
NVDA we predict NOT merely a shallower fold but most likely NONE:
with every local decode under ~1 s and the expert RTT ~3 s,
escalation is a net time-add on every query and the latency curve
should be plainly monotonic. (Caveat: NVDA per-query decode time is
batch-contaminated; the 0.4 s assumes same-order decode speed.)
**VERIFIED 2026-08-24 (user: "那你跑呗", ~$0.2):** the batch-timing
problem dissolves for a speech-native duplex model — its deployed
answer latency is the FRAME CLOCK (1 text token = 1 LM frame = 80 ms),
so per-query local latency = token count x 0.08 s, immune to batching.
`modal_nvda.py::dump_scores` refit the winner probe (calib=frozen 600)
for per-query NVDA scores + exact Nemotron token counts; expert path =
per-query measured gpt-5.5 RTT + relay at frame rate. Same top-r
re-mix arithmetic on both models (`figures/nvda_fold_test.py`):
**NVDA strictly monotonic on both pools** (sllama 0.96→2.79 s over
0-60%, striviaqa 1.36→3.65 s; no dip anywhere), while the SAME
arithmetic on MiniCPM reproduces its dip (sllama 1.50→1.45) — a
same-math positive control. Mechanism as predicted: NVDA local P90 =
2.0 s < expert ~4 s, so escalation is a net time-add on every query.
The prediction is now an observation (under the stated frame-clock
convention; a full live port would add turn-take offsets, which are
constant and cannot create a fold).

**Addendum 6 (2026-08-24, user: "sreason 拐弯最深——是小模型的链比大模型
长吗?", $0 from the expert cache).** `modal_bench.py::expert_usage`
pulled the cached gpt-5.5 usage for every escalated query
(completion_tokens INCLUDES hidden reasoning). sreason escalated 101:
expert total chain P50 = **144 tok** (P90 548) of which only **57
visible chars** return; RTT P50 3.67 s; throughput ≈ 39 tok/s — i.e.
NOT faster per token than the local H100 stream. The fold's depth
decomposes into three factors, in order: (1) **spoken vs hidden** —
the local model's chain IS its answer and every token enters the
clock (P50 4.38 s, tail 13 s+), the expert's chain is hidden and only
the RTT lands; (2) **effort=low is a governor** — the expert's chain
is capped short by design (144 tok median), so its path is a ~3.7 s
flat top; higher effort would lengthen it and shallow the fold — a
system knob, not model talent; (3) the user's "small model's chain is
longer" holds in TIME mostly via the TAIL: median chain lengths are
same-order (~197 chars vs 144 tok), but the local chain blows up on
hard problems while the governed expert cannot — and the probe selects
exactly the hard ones. sreason folds deepest because reasoning pools
have the fattest removable local mass (whole chains spoken aloud)
against a governed flat expert path; also explains 8x's old
observation that sreason's P90 IMPROVES with escalation (13 s chains
swapped for 3.7 s round-trips). Comparison rows: sllama expert 64 tok
/ 2.39 s, striviaqa 69 tok / 2.71 s.

**Addendum 7 (2026-08-24, user: "NVDA 11B 所以拐弯该更深?", $0).**
Two category errors corrected with one measured dissociation. (1) The
11B includes the ears (FastConformer) and mouth (TTS); NVDA's LLM
backbone is Nemotron Nano 9B — same brain size as MiniCPM's. (2)
Chain length is a TRAINING-STYLE property, not a size property — and
the style is not free. Frozen pool, same queries, same judge, by
failure species: **hard-math MiniCPM .517 (median answer 1145 chars —
the full spoken chain) vs NVDA .217 (284 chars) = −30 points;
hard-knowledge .200 vs .183 ≈ tie.** Perfect dissociation: a terse
duplex model loses exactly the compute that emitting the chain buys
(serial steps via tokens), while on pure-knowledge failures the chain
never mattered. NVDA has no fold (verified, nvda_fold_test) not
because it is stronger but because it traded away spoken reasoning —
fold-free and math-broken are the same design choice.

**Addendum 8 (2026-08-24) — Jisen Q4 ("why is gain-over-random visible
only on some pools, e.g. fig-5 swebq") now has its post-investigation
answer.** Visible gain = probe selectivity x headroom x LABEL FIDELITY
− channel cost, over measurement noise; selectivity itself is nearly
pool-invariant (AUC .70-.81 everywhere, including on NVDA with 600
generic calib queries — which also first-order kills "the probe
trained more on some pools": zero-recalibration transfer, the
most-in-family pool striviaqa shows no advantage, precision lift is an
eerily uniform x1.33-1.35; second-order true via calib width, 8z's
sreason +.062). "Base model trained more" is real but acts as
arithmetic: high floor -> precision cap = base-fail/rate (sllama .32,
we sit at 95% of it). Synthetic-data bias has no referent (public
benchmarks only). What actually suppresses fig-5: swebq's labels
half-measure the Freebase grading, not the model — the
hedging/entropy signature is null there for ALL THREE models
(P=.47-.53) and even gpt-5.5 on gold text scores only .864, so a
large share of "wrong" is judge-protocol wrong that no state-reading
probe can predict; plus the deployed-view channel tax (gold view
clears random, 8s) and the 8ad noise floor (per-arm ±.02-.03 vs
+.035-.066 excesses — consistent in sign across pools/tiers,
significant on the low-variance instruments). (Also corrects
a user misconception worth guarding in the paper: the chain must be
emitted because tokens ARE the serial compute feedback path, not
because of GPU memory.) (2) TEXT models hedge too — gpt-5.5's own wrong answers on
sllama run ~2x longer; the claim is capability-relative (each model
hedges at ITS boundary), and the kink only needs the asymmetry that
TriviaQA sits inside the 9B's boundary (84 wrongs) but barely
intersects gpt-5.5's (8 wrongs), which is why the expert path is flat
~3 s. (3) swebq is null for ALL three models — its enumerative
Freebase-style answers pin length to question format and its strict
judge decouples "wrong" from "uncertain": neither evidence nor
counterexample. The kink itself does not require hedging universality
— only decode-time ∝ answer-length (an autoregressive identity,
r=.97) plus the probe selecting long-answer queries; a non-hedging
model just folds less.

**Addendum 3 (2026-08-24, user: "用实验说服我,不要 claim" — four
questions, all answered from the traces, $0).**

*Causation, alternatives killed (striviaqa n=250).* (A) "probe keys on
verbosity/style": corr(score, len) = +.05, and in len ~ wrong + score
+ audio the score coefficient conditioned on wrongness is −.08 ≈ 0 —
the probe selects WRONGNESS (corr .48); length rides along only
through wrong (β +.26). (B) "longer audio → longer answers":
corr(audio, len) = .03, β = +.04 — dead. (C) noise — 8ad. (D)
"hedging is just 'long' renamed": explicit hedge phrasings appear in
32% of wrong vs 7% of right answers, and WITHIN length bands hedge
still predicts wrong (short band 1.00 vs .27) — a real textual
behavior. Strongest piece is the NEGATIVE CONTROL: the probe selects
wrong equally everywhere (corr(esc, wrong) = .26-.32 in every pool),
but sdqa's wrongs are short and confident (P(wrong longer) = .48,
corr(esc, len) = −.12) and sdqa has NO fold (+0.20 s). Same probe,
same selection behavior, fold only where wrong happens to be slow —
so fold = (probe catches wrong) × (pool's wrongs are slow), and
"probe directly picks slow queries" is excluded. Self-correction: the
fold has TWO fuel lines — hedging on retrieval pools, and LONG CoT on
execution pools (sreason: char-length signature masked by CJK density,
but decode time directly: escalated 4.38 s vs kept 3.04 s).

*Router effectiveness on the phenomenon + why exactly one fold.*
Escalated-set local-wrong fraction (striviaqa, base .34): cons 68% /
bal 64% / agg 54% (lift x2.0/1.9/1.6); coverage of the pool's 84
wrongs: 31% / 57% / 80%. The single fold is quantile arithmetic:
local P50 falls monotonically (1.23→0.94→0.87→1.02 s) while the
~3.3-3.8 s expert mass share grows; at 15% the median still sits in
the (faster) local mass (1.19), by 30% it must cross expert mass
(1.37), 50% → 1.96. One left-fold then monotonic rise is the necessary
shape of a mixture median.

*Which benchmarks show it.* Fold requires (i) probe catches wrong
(all pools) AND (ii) wrong/hard is slow. Fuel present: sllama (.77
hedging, −.35 s), sreason (CoT, −.41 s), striviaqa (.62, −.04 s).
Fuel absent: swebq (enumerative format pins length, +.39 s), sdqa
(short confident wrongs, +.20 s), valpaca (everyone long). Predicts
NVDA (terse style) folds nowhere.

*Anti-hedge prompting (user Q4).* Testable for ~$10 (250-query never
arm, system-prompt line "If unsure, say 'I don't know' in five words
or fewer"). Expected honestly: hedging is symptom not cause — it
converts slow-wrong to fast-wrong (latency win, accuracy ~neutral,
trust possibly worse); AND it shifts the hidden-state distribution
under the frozen probe (8t/8z saw 5x score-scale shifts across
domains), so thresholds need re-quantiling — a new deployment
configuration, not a free patch. 8d's robust-prompt (transcription)
was negative but is a different behavior. Three readouts if run:
Δwrong-answer length, Δacc, Δprobe AUC/score drift. Awaiting go/no-go.

**Addendum 4 (2026-08-24, user: "答错但不 hedging 的情况存在吗?怎么验证
不确定→hedging?") — the verification FAILED and found something better.**
Design: the probe's end-of-turn read happens BEFORE generation, so it
is a temporally-prior proxy for the internal state; if internal
uncertainty causes hedging, the pre-answer score should predict which
failures will hedge. It does not: within wrong answers, AUC(score →
subsequent hedging) = **.488**, chance. And confident errors are the
MAJORITY: 57/84 wrongs (68%) carry no hedge phrasing and decode fast
(1.13 s). So the 8ab narrative link "uncertainty → hedging" is
RETRACTED at the individual level. What replaces it is stronger:
**the probe catches both error species identically** — hedged wrongs
score .634 / AUC .786 / 81% escalated @50%; confident wrongs score
.663 / AUC .800 / 81% — i.e. the L22 state carries the
failure-is-coming signal even when the surface text is confident:
verbal confidence ≠ internal state, and the gate does NOT depend on
hedging at all. Hedging's true correlate is SLOWNESS regardless of
correctness (hedged answers 3.3-3.4 s whether right or wrong; 12
right-but-hedged exist). Revised causal graph: internal
will-fail state → wrongness (probe reads this, both species); hedging
= an independent surface style bound to verbosity/latency; the two
correlate at the group level (32% vs 7%) without sharing the
probe-readable state. The kink story survives (the escalated set is
wrong-enriched, and wrongs contain the slow hedged subset) but the
anthropomorphic "the probe reads the hesitation" must be written as
"the probe reads the coming failure".

**Addendum 5 (2026-08-24, user go: "token 级熵轨迹做一下", ~$6, two
H100 passes).** `modal_bench.py::entropy_replay`: 93 striviaqa queries
in four behavior groups (hedged-wrong 27 / confident-wrong 27 / right
27 / hedged-right 12), replayed through the exact bench streaming path,
capturing per-step full-vocab entropy + P(terminator) via an lm_head
hook. Figure `entropy_traj.{png,pdf}`. The user's hypothesized chain
(retrieval failure → entropy up → EOS suppressed → hedging) is now
MEASURED at token level:

| group | first-5 entropy | traj median | steps | P(stop) at sentence boundaries |
|---|---:|---:|---:|---:|
| hedged-wrong | **.57** | .48 | 89 | **.0024** |
| confident-wrong | .50 | .32 | 34 | .0147 |
| right | .31 | .14 | 45 | **.083** |
| hedged-right | .36 | **.52** | 96 | .0001 |

(1) Trigger confirmed: hedged errors open at ~1.8x the entropy of
correct answers (P=.70) and wander high all the way. (2) EOS
suppression confirmed and large: at mid-answer sentence boundaries the
terminator carries **35x less probability** in hedged errors than in
correct answers — "keeps talking" is literally visible in the stop
token. (3) The two error species split at token level: confident
wrongs are SHORT (34 steps), low-trajectory-entropy (.32) — the output
distribution is deceived by the false fact, which is precisely why
entropy-based signals cap at AUC ~.70 while the L22 probe reaches .80
on both species. (4) Bonus: hedged-RIGHT answers open LOW (.36, like
correct) but wander highest and longest — early entropy tracks the
retrieval state, late entropy tracks the rambling style; the two
formerly-confounded quantities separate. Engineering footnote:
MiniCPM's actual terminator is token id **151704 and it is absent from
generation_config** (first run measured the wrong stop set and read
~1e-7 everywhere; recovered from the argmax tails).

**Case-study figure (user request: "从错题簿找一个典型,用 trace 说明拐弯
怎么出现"):** `kink_case_study.{png,pdf}` (gallery 图21) walks one real
sllama query through both worlds with measured milliseconds only —
sllama0164 "How many gurus are there in Sikhism?" (ref: Ten). Probe
OFF: 487-char ramble ("only one Guru… However… to avoid confusion"),
3.10 s, wrong. Probe ON (cons arm): running chunk scores .71/.67/.66,
21 ms end-of-turn read .631 ≥ .513 → escalate before a single answer
token exists; gpt-5.5 1.68 s + relay 0.62 s = 2.32 s, correct —
0.78 s faster AND right. Pool inset: the 38 such queries (local P50
2.38 s) leave the local queue, the remaining 212 drop to 0.94 s, arm
median 1.52 → 1.17 s = the fig-8 left-fold. **Web version (user: "更偏 demo
的"): `/cases` on the demo app —
https://rhe9527--gate-demo-web.modal.run/62dc5cd9/cases — six curated
real queries side-by-side, probe-OFF (never arm) vs probe-ON (the tier
that actually escalated/kept each), verbatim answers with hedge
phrasings highlighted, judge verdicts, probe-score-vs-threshold line,
latency segment bars. Covers the four fates: 2x escalate-faster-AND-
fix (sllama0164 3.10s wrong->2.32s right; Dame Edna 5.77->3.13), 1x
both-right-but-faster (5.07->3.87), 1x pay-latency-for-accuracy (Shema
2.49 wrong->6.86 right), 1x correctly-kept (score .049), 1x
confident-wrong missed at 15%/caught at 50% (Joe Gargery). Linked from
the main demo header.**

**图8 (sllama_pareto) latency zigzag decomposed → new figure
`sllama_latency_decomp.{png,pdf}`.** Two panels: (1) the fold lives
only in the median — the MEAN is monotonic (1.65/1.81/1.96/2.06 s vs
P50 1.52/1.17/1.60/1.42 s); (2) kept-local P50 falls 1.52→0.96→0.86→
0.43 s as the probe strips slow decodes while escalated rows pay a
flat ~3 s expert round-trip. The "weird" latency is a *positive*
property: on this pool escalation buys accuracy AND (at the median)
speed; the honest way to "remove the latency artifact" in a paper
figure is to show the mean alongside the median, not to re-measure.

**Escalation precision, v2→v3 at matched 50% rate** (rank never-arm
fail labels by the aggressive arm's live eot_score, take top half —
avoids the realized-rate confound; note the never-arm rows in the v3
parquets carry reused v2-era scores, so aggressive-arm scores are the
only valid v3 read):

| pool | v2 prec@50 | v3 prec@50 | base-fail | cap = base/rate |
|---|---:|---:|---:|---:|
| frozen | .858 | **.867** | .62 | 1.00 |
| striviaqa | .520 | **.544** | .34 | .68 |
| swebq | .576 | **.608** | .43 | .86 |
| sllama | .296 | **.304** | .16 | **.32** |
| sdqa | .720 | .720 | .48 | .96 |
| sreason | .525 | **.535** | .41 | .82 |

Key reframe for "can you push precision above 74%": precision at a
fixed escalation rate is CAPPED at base-fail/rate. The .74 Jisen
remembers was the v1 ours-fair receipt (base .56, esc 52% → cap ≈1);
on sllama the cap is .32 and v3's .304 is **95% of the theoretical
maximum** (recall .93). The honest dial is (a) AUC (v2→v3: OOF
.860→.879, ext-mean .750→.771) and (b) escalating at ≈ the base-fail
rate instead of a fixed 50%. Remaining levers unchanged from 8z:
calibration-pool width (the bigger lever, public families only),
judge-label denoising, per-domain threshold drift (the 8z-live
overshoot), asymmetric cost-sensitive thresholds.

**Random-escalation reference added to ALL pareto figures** (Jisen:
"所有图都可以加上 random escalation，不用重测"). Acc = the dualview
random line (pairs with the gold view); x = simulated P50 of a random
mixture at rate r — per-id local latency from the never arm, per-id
escalated latency where any arm escalated that id, pool-draw
otherwise (400 sims × 21 rates, seed-42 rng continuation). Patched
into bench_figures.py, fair_figures.py, valpaca_figures.py; all 14+1
figures regenerated and copied to paper/figures/. This supersedes the
2026-08-13 "no random line on sdqa" call for the PARETO view only
(dualview keeps the floor+ceiling design). Headline: on striviaqa and
sllama the gated curve now visibly dominates random in BOTH axes —
random needs ~3.4 s at the median to reach the ceiling striviaqa
region the gate reaches at 2.0 s, and on sllama random@50% sits at
~2.2 s/.88 vs the gate's 1.42 s/.948. Label bug fixed in the same
pass: figure subtitles said "probe v2" while VER="_v3" — now derived
from VER. `pareto_latency.py` (the frozen-pool paper figure) reads
pre-aggregated JSONs with no per-query rows, so its random line needs
a small rebuild from frozen_v3_traces — left as a todo.tex note.

**NVIDIA NemotronLabs-VoiceChat-11B scoped (Jisen #3).** HF card
verified live: 11B end-to-end **full-duplex** speech model — Fast
Conformer speech encoder + Nemotron Nano v2 9B (hybrid
Mamba/Transformer) + NVIDIA TTS decoder; in/out = user audio 16 kHz +
text → agent text + 22.05 kHz audio + user transcription; OpenMDW
1.1 license; vLLM + NeMo offline scripts + streaming WebSocket
container; claims VoiceBench #2 and Full-Duplex-Bench 1.0 #2 among
open FD models (0.82 smooth turn-taking, 448 ms). This is exactly the
§9 pre-registered prediction test ("a new open-weight full-duplex
model should show the late-layer text-input cliff"). Caveats before
committing spend: hybrid Mamba backbone means the L22-style layer
sweep must be redone from scratch (layer semantics differ; SSM states
vs attention residuals), and hidden-state hooks need the NeMo/HF
path, not the vLLM container. Plan (pre-registered order): (1) cliff
replication — text-vs-duplex layer×position sweep, predict a
late-layer cliff; (2) probe calibration on the same public calib
pool + frozen-methodology 4-arm live curve on the same 5 external
pools/judges → the transferability figure Jisen wants; (3) Anthony
trains/fine-tunes the NVDA model, we run the identical gate harness
on his checkpoints as the training-vs-routing ablation.

---

### 8ac — NVDA NemotronLabs-VoiceChat-11B: the second-duplex-family test executed (~$40, 2026-08-18/19)

User: "开始训练 nvda 的模型，结果放到 modal 界面" — interpreted (and
stated up front) as: train OUR probe on the new model's hidden states;
fine-tuning the model itself stays Anthony's ablation. Infra:
`modal_nvda.py` (download / smoke / run_answers / judge / fit), NeMo
Speech branch `nemotron-labs-voicechat`, 44 GB combined safetensors on
a new `nvda-weights` volume; same wavs, same judge (`escalate.
judge_many`), same recipe as eoth2/v3.

**Engineering receipts.** (1) Backbone = Nemotron Nano v2 9B, 56
NemotronHBlocks (27 Mamba2 / 4 attn / 25 MLP), reached at
`stt_model.llm.layers`; turn-taking is the agent text channel emitting
BOS/EOS (no separate head). (2) NeMo forces **cacheless** inference
for Nemotron (full prefix re-run per 80 ms frame) — so the FINAL
frame's forward contains every position, and one hook capture yields
the whole eot window + user-audio mean for free. (3) fp32 B=1 was
99.5 s/query; bf16 on the stt stack + length-bucketed batch-8 →
**4.3 s/query** with answers intact (23×). Frozen-pool math audio
(up to 3 min) OOMed fixed batches; adaptive batching (B × longest-wav
budget) recovered all 80. (4) Default system prompt makes the model
greet first — replaced with a QA prompt; answers carry `<$..$>/<|..|>`
timing markers — stripped before judging. Loading is 17.7 min/container
(fp32 key-by-key safetensors read).

**Floors (never-arm fail rate, our judge, offline replay):** frozen
.798, striviaqa .676, swebq .720, sllama .332, sdqa .690 — the 9B
Nemotron backbone is much weaker on knowledge retrieval than MiniCPM-o
(striviaqa local acc .32 vs .62 same judge). **Boundary finding:
sreason (Chinese) fail = 1.000 — VoiceChat is English-only** (Chinese
audio → fluent unrelated English hallucinations). The cross-lingual
transfer result has no analog on this model; pool skipped (zero label
variance).

**⭐ The §9 pre-registered test passes on first execution:**

- Layer sweep (eot_last, calib = frozen 600 only): mid-band peak
  L30-34 = **.714**, endpoints .693/.682 — the mid-band-readable
  structure replicates on a hybrid Mamba architecture.
- Same three reads @ L34: OOF .714 → .761 → **.790** — the same
  feature-stacking gains as MiniCPM.
- External transfer AUC: striviaqa **.781**, swebq **.793**, sdqa
  .754, sllama .701 — the MiniCPM v3 band (.79-.81) reached with a
  quarter of the calibration data.

Figures `nvda_layer_sweep` + `nvda_transfer` (图15/16) added to the
gallery and paper/figures. NOT yet done (next spend decisions): live
streaming 4-arm curve (needs the duplex loop ported to NeMo), calib
expansion to the full 2310, Anthony's fine-tuned checkpoints as the
training-vs-routing ablation.

### 8ad — noise audit + two superseded attributions + the stale live figures ($0 local, 2026-08-20)

Started as "fix the 8z-live threshold overshoot". The fix worked and
then falsified its own premise, which cascaded into an audit.

**Reconstruction method (new, $0).** Three properties of the live
sweeps make a CONTINUOUS rate-accuracy curve recoverable from the
existing traces: (1) the three gated arms carry bit-identical probe
scores (max spread 0.00000) so "top-r by score" is unambiguous; (2)
the tiers are perfectly NESTED (cons subset bal subset agg, 0
violations in 6 pools); (3) one query per session, so a query's
outcome does not depend on the arm's rate. Therefore for any r <=
agg-rate every selected id has a measured escalated outcome and every
other id has a measured local outcome (never arm). `figures/
rate_curve.py` -> `data/rate_curves.json`. Self-check: reconstruction
vs the measured arms deviates by .011 mean absolute (max .029),
consistent with the replication noise measured below.

**⚠️ SUPERSEDED #1 — the 8z-live overshoot attribution.** RESULTS 8z-live
attributed frozen aggressive .621→.596 to the calib-quantile threshold
firing at .613 instead of .50 ("pushes ~26 extra math/LaTeX queries
through the transcript-tax channel"). Re-mixing the same measured
outcomes at exactly 50% gives **.600 — i.e. +.004, nothing**. The rate
error is a COST bug, not an accuracy bug: correcting it removes 11.3%
of the expert calls at equal accuracy. `data/
gate_v3_thresholds_corrected.json` carries label-free corrected
quantiles for all 6 pools (only frozen drifts; the externals already
hit budget at 15/30/50).

**⚠️ SUPERSEDED #2 — "the small model BEATS the expert on the easy
half" (8x/8y).** Paired McNemar on the sllama aggressive arm: kept-local
half n=125, local .976 vs expert .968, discordant 2 vs 1, **p = 1.00 —
statistically indistinguishable, not "beats"**. The robust form of the
claim is (a) the probe's split is real and large — local acc .976
(kept) vs .696 (escalated), **z = 6.46**; (b) the expert's advantage is
concentrated entirely in the escalated half (.696→.888, discordant 27
vs 3, **p < .0001**); (c) therefore always-escalate spends 2× the
expert calls to buy nothing measurable over selective. The headline
"selective .948 > always .928" itself is 6-vs-1 discordant, **p =
.125** — directionally right, underpowered because both sit near
ceiling. Paper wording must move from "beats" to "matches at half the
cost".

**The replication-noise floor (why both corrections were needed).**
Same query, same audio, kept LOCAL in two different arms — the judge
verdict flips **2.3-18.8%** of the time (frozen .155, sdqa .188,
sreason .169, swebq .160, striviaqa .074, sllama .023). Repeat
ESCALATION flips 0.7-10.6%. Implied paired SE on an arm-vs-arm
accuracy delta: **.009-.028**. Against that floor, **all 18 live v2→v3
deltas are non-significant** (McNemar p = .16-1.00; the largest,
striviaqa balanced +.036, gives p = .16). This does not touch the
offline AUC gains (a much tighter statistic on the full score
distribution) — but the live-curve "confirmations" of them, including
sreason's cross-lingual +.010-.030, were over-read. Figure:
`noise_audit.{png,pdf}`.

**⚠️ Bug found while fixing the figures: the paper's two main LIVE
figures were two probe generations stale.** `live_dualview.json` /
`latency_profile.json` are written by `modal_stream.py::live_dualview`
off `gated_traces_v2.parquet` — that file is the streaming-loop-v2 /
probe-**v1** sweep (json mtime 2026-07-30). The v2 (8u) and v3
(8z-live) re-runs wrote `frozen_v{2,3}_traces.parquet` and only the
external-bench figures were refreshed, so fig:live and fig:pareto
still showed v1 arms. Rebuilt locally at $0 by `figures/
live_v3_figures.py` (v1 archived as `*_v1.{json,png,pdf}`):

| view | v1 (shown until today) | v3 (correct) |
|---|---|---|
| rates | 0/14/35/55% | 0/16/35/61% |
| heard | .400/.446/.529/.633 | .383/.483/.554/.596 |
| gold-inject | .400/.500/.637/.767 | .383/.525/.654/.771 |
| P50 latency | 2.02/2.76/4.00/4.69 s | 2.02/2.61/3.53/4.44 s |
| channel cost @agg | −.133 | **−.175** |

The v3 deployed curve is FLATTER and the channel cost LARGER: at
bal/agg the heard curve now sits below the gold-paired random line —
i.e. on this mixed pool the speech-channel tax (.175) exceeds the
selection margin, and only the channel-controlled (gold) view clears
random (+.061 at 61%). That is the honest headline for the frozen
pool and it strengthens the case for the audio-direct-to-expert lever.
`pareto_latency.py` also got the random-escalation curve it was
missing (8ab todo) and its hard-coded P99 text is now read from the
json (17.8→32.7 s, was 30.4).

**Consequence for "can we improve further".** Our measurement
precision on a 200-250-query pool (±.02-.03 per arm) is now the
binding constraint: any lever worth less than ~3 points is
undetectable in a single live sweep. Ranked next steps: (1)
audio-direct-to-expert — gold-inject says .175 sits in that channel on
this pool, far above the noise floor; (2) evaluate on paired/
variance-reduced statistics (AUC, matched-rate precision, the
reconstruction curve) rather than arm accuracy; (3) only then spend on
bigger n.

### 8ae — cloud-ASR uplink: the first lever in weeks that clears the noise floor (~$8, 2026-08-20)

User said "go" on the channel lever. Checking first stopped a bad
spend: **audio-direct-to-expert was already run (8r) and REJECTED by
the user the same day** — an audio-native expert costs -.15 of brain,
and a model-list check today confirms the premise still holds (audio
family = gpt-audio / gpt-audio-1.5 / gpt-audio-mini; **still no
gpt-5.5-class audio model**). So that arm was NOT re-run.

What had only been LOWER-bounded is the variant that keeps the frontier
TEXT brain: audio uplink -> hosted ASR -> gpt-5.5. 8d bounded it with
`openai/whisper-large-v3` (open weights, run locally) at +4pp. The
hosted frontier ASRs did not exist then. `modal_uplink2.py`, all 147
escalated frozen-pool ids, `gpt-transcribe` (auto-picked), same expert
protocol (`escalate.ask_expert`) and same judge (`escalate.judge_many`):

| arm | what the expert reads | acc (n=147) |
|---|---|---:|
| A deployed | MiniCPM's own self-transcript | .585 |
| **B cloud-ASR** | **gpt-transcribe of the same wav** | **.694** |
| C ceiling | the gold question text | .871 |

**B-A = +.109, McNemar p = .007** (24 rescued vs 8 broken) — ~5x the
paired SE (8ad), i.e. the first change in weeks that is unambiguously
real rather than noise, and **2.7x the whisper-large-v3 bound** the
lever was previously written off with. It recovers **38%** of the
gold gap; C-B = +.177 (p<.0001) remains.

Per-pool (n): easy-chat 28 **.679 -> .964**, hard-knowledge 50
.440 -> .560, trap 20 .550 -> .600, easy-fact 32 .906 -> .938,
**hard-math 17 .294 -> .294 (+.000, gold 1.000)**. The math wall is
exactly the 8q prediction: spoken LaTeX is lossy at the SOURCE, so no
ASR can recover it — the fix there is input-side (don't speak formulas)
not uplink-side. (easy-chat's B .964 > C .893 is open-ended judge
variance at n=28, not a real ASR-beats-gold effect.)

Deployment consequence: this is architecture-compatible with the
2026-08-12 text-backend decision — one extra ASR call per ESCALATED
turn only (~15-50% of turns depending on tier), the talker's own
transcript stops being load-bearing, and the expert keeps its frontier
brain. Not yet measured: the ASR call's added latency (the arm was run
offline; the call is a single short-audio request, but it belongs on
the critical path between EOT and the expert call, partially maskable
by the stall). Next: re-run one live 4-arm sweep with the uplink in the
loop to get the end-to-end curve + latency, and check whether the
+.109 survives on the external pools.

### 8af — interactive demo app (2026-08-20)

`demo_app.py` -> **https://rhe9527--gate-demo-web.modal.run/62dc5cd9**
Two modes, one probe ON/OFF switch, live metric tiles and an event log.

- **Replay ($0, scales to zero).** All 4773 measured sessions across 6
  pools. Flipping the probe OFF is not a simulation: 8ad established
  the tiers are nested with bit-identical scores, so the OFF view is
  the never-arm's MEASURED outcome for the same query. Shows the real
  per-chunk probe trace against the real per-domain threshold
  (`gate_v3_{pool}.json`), both answers, the judge verdict and the
  measured timings.
- **Live (opt-in H100, ~5 s warm / ~1 min cold).** Type anything ->
  tts-1/alloy (same voice as the frozen pool) -> MiniCPM streams it in
  1 s chunks -> v3 probe reads L22 at end-of-turn (rolling last-8 +
  running user-audio mean, byte-identical to `bench_live`) -> frozen
  threshold decides -> talker answers, or gpt-5.5 does and the talker
  relays it under a stall.

Verified end-to-end on the user's own test question. "What is NVDA
trading at right now?", balanced tier: 2.05 s of audio, 3 chunks,
running P(fail) .776/.676/.693, **end-of-turn read 54 ms -> P(fail)
.807 >= .680 -> ESCALATE**; expert 2.68 s, stall 37 ms, relay 2.33 s,
total 5.07 s. Same question probe OFF: local answer in 1.53 s. (Both
refuse honestly here — the talker also knows it lacks real-time data —
which makes it a good latency-cost illustration but a poor accuracy
one; the app ships three example questions including a long-tail fact
the talker got wrong in the sweep and the gate rescued.)

**Microphone input (user: "这个 demo 要让我能说话的").** The talker is a
speech model, so typing + TTS was a stand-in. The page now records from
the browser (MediaRecorder -> webm/opus), the container transcodes with
ffmpeg to 16 kHz mono and streams THAT into the duplex loop — no TTS
anywhere on this path. One consequence is scientifically useful: with
real speech there is no gold text, so the escalation uplink MUST be a
transcript, and the demo uses the 8ae hosted-ASR uplink (the arm
measured +.109 the same day) and shows the reader exactly what the
expert was told.

Both branches verified with real payloads built off the volume's own
audio (SD-QA human speech and a frozen-pool wav, re-encoded to
webm/opus so the request is byte-shaped like the browser's):

| | local branch (SD-QA, real human voice) | escalation branch (q0225, 48.7 s spoken MCQ) |
|---|---|---|
| end-of-turn read | 40 ms, P(fail) **.105** | 21 ms, P(fail) **.867** |
| gate | < .680 -> keep local | >= .680 -> **escalate** |
| outcome | correct answer in 1.6 s | ASR heard the full question -> gpt-5.5 "B. +7.3 J/mol" -> talker relayed it |
| total | 1.6 s | 13.2 s |

**This also closes 8ae's open latency question with a first datapoint:
the hosted-ASR call cost 4.81 s on the critical path** for a 48.7 s
clip (expert total 12.8 s, of which the talker's stall covered only
0.1 s). Short queries will pay far less, but the uplink is not free and
the stall phrase does not hide it — a live 4-arm sweep with the uplink
in the loop is still the number that matters.

Engineering notes for whoever redeploys: a live turn far exceeds the
web proxy's synchronous window, so the endpoint `spawn`s and the page
polls; `demo_app.py` imports `modal_app` at module level, so BOTH
images must mount `modal_app.py` or the web container dies before
serving (cost one confusing hang); and the mic needs HTTPS, which the
Modal URL already provides.

### 8ag — demo v2: continuous voice + GPU readiness gating (2026-08-21)

User feedback on 8af, both points valid: (1) a record-button is not a
voice conversation — they want to just TALK; (2) the mic must not be
clickable while the GPU is cold. Rebuilt the live path:

- **Resident GPU class** (`Voice`, modal.cls): the model loads once in
  `@enter` (~12-30 s off the warm volume), the browser's WebSocket
  lands on the same container, `scaledown_window=420`. `/ready` cannot
  return before `@enter` finishes, so the mic button being enabled IS
  the readiness proof — the page polls it with a visible elapsed
  counter and keeps the button disabled until then.
- **Continuous voice**: the page streams 16 kHz int16 PCM continuously
  (ScriptProcessor; no start/stop per turn). Server-side energy VAD
  (speech ≥0.2 s, then 1.25 s silence) ends the turn; then the same
  primitives as bench_live run: per-1s-chunk probe scores (streamed to
  the page live as you speak), the end-of-turn L22 read, the frozen
  threshold, local answer or 8ae-uplink escalation — then it resets
  and listens again. Multi-turn on one socket.
- Typed questions now go through the same warm container (`/say`) —
  no more per-turn cold model load.

Verified end-to-end with browser-shaped PCM streams built from the
volume's own audio: turn 0 (SD-QA human speech) P(fail) .126 -> local,
correct, 3.5 s of speech; turn 1 (spoken thermodynamics MCQ) P(fail)
.755 >= .680 -> escalated, hosted ASR transcript -> gpt-5.5 -> relay;
both turns on one socket, session survives into a third listen state.

Debug receipts (each cost a failed round): (1) the GPU image must
carry fastapi — the container crash-looped importing the in-container
ASGI app while /ready timed out silently for 8 min; the image now
replicates modal_app's proven MiniCPM spec verbatim + fastapi, because
Modal forbids stacking layers on an image that ends in add_local_dir.
(2) During cold start Modal serves /ready as a 303 long-poll redirect
chain — both the page and any client must POLL with short timeouts,
not follow one long request. (3) A WS upgrade against a cold container
times out at the proxy — always /ready first, then connect (the page
already did; the first test didn't). (4) VAD at 0.9 s cut a long
spoken question at a thinking pause -> 1.25 s. (5) A client that stops
streaming after its audio can strand the VAD one frame short of EOT
forever — a real mic never stops sending, and the test now mimics
that (background silence frames until the turn lands).

### 8ah — the "it just keeps listening" bug: two real defects, both invisible to clean-audio tests (2026-08-21)

User tried the voice demo: mic streams, model never answers, probe
on/off irrelevant. Root-caused to TWO independent defects, each of
which alone produces exactly that symptom, and NEITHER of which the
8ag test could catch because it streamed clean TTS/SD-QA audio and
stopped sending after each clip:

1. **Fixed VAD threshold vs real microphones.** The 8ag VAD used an
   absolute RMS threshold (0.010) tuned on clean wavs. Real mics have
   a noise floor and browser AGC pumps quiet passages, so silence
   never accumulates (or quiet speech never triggers) and the turn
   never ends. Fix: adaptive noise floor — EMA down 0.10 / up 0.02,
   up-adaptation only outside speech so long utterances don't erode
   their own threshold; speech = rms > max(.005, floor x 3.5). A
   first fix used instant-min tracking down and a single digitally
   silent frame (TTS inter-sentence zeros) collapsed the floor,
   making steady noise read as speech forever — hence the EMA.
2. **The post-answer drain ate the stream.** After each answer the
   server discarded backlogged frames "until a 0.05 s receive gap".
   A real mic never pauses, so the drain never exited and swallowed
   every subsequent utterance. Deleted outright: backlogged frames
   just flow through the VAD (quiet settles the floor, speech starts
   the next turn).

Also added, so this class of bug is diagnosable from the page instead
of by proxy: a live VAD readout (level / adaptive threshold / speech
state / silence progress, streamed every ~0.5 s) and an "I'm done
talking" button that forces end-of-turn if the detector misjudges —
the demo can no longer dead-end silently.

Regression test rebuilt to be mic-shaped (`_ws_test.py`): steady
rms .008 noise over EVERYTHING including the speech, continuous
frames with no gaps, immediate next utterance after an answer, plus a
manual-eot scenario. All three turns pass on one socket; turn 1's
eot read came out .671 vs the .680 threshold (fired last run at
.755) — the boundary sensitivity is the 8ad noise band doing exactly
what it says, and the local answer it kept was on track anyway.

Meta-lesson for the paper's demo section: every one of 8ag's five
receipts came from testing with idealized inputs; both 8ah defects
were only reachable with mic-shaped input. Test the transducer you
ship, not the files you have.

### 8ai — anti-hedge prompt arm: suppresses the behavior, costs 9.6 points, probe unmoved (~$5, 2026-08-24)

User: "先测4". `bench_live` gained a `sys_suffix` param (default empty
= byte-identical); `run_nohedge` appends to the stock omni persona:
"If you are not sure of the answer, say only 'I am not sure.' in five
words or fewer. Never explain your uncertainty or give background;
answer in one short sentence." Never arm, striviaqa 250, v3 probe
artifact, judged both scales; all comparisons paired on the same ids.

**1. The behavior is fully suppressible by prompt.** Median answer
181 -> 42 chars; WRONG answers 258 -> 38; local decode P50 1.20 ->
0.36 s and P90 3.31 -> **0.71 s** — the latency tail (the kink's
fuel) is annihilated, exactly as the 8ab mechanism predicts.

**2. It costs real accuracy: .664 -> .568 official judge (−.096,
McNemar 10 vs 34, p < .001)** (ours judge −.044, p = .15 — the OAB
judge rewards the context the short answers dropped). Decomposition of
the 34 right->wrong flips: **13 are explicit abstentions** ("I am not
sure") on questions the model previously got RIGHT while rambling —
its verbal self-assessment false-abstains at 43% (13 of 30
abstentions were on known items); **21 are information loss** (terse
answer drops the alias/context the judge needed, or a different
answer surfaces).

**3. The probe does not care about the persona.** First read looked
like drift (paired r = .45) — that was the 8ad scale artifact (stored
never-arm scores are v2-era). Same-generation comparison (v3
aggressive-arm scores vs v3 nohedge scores, same 250 ids): paired
**r = .95**, medians .468 vs .470, AUC .796 vs .813, virtual
escalation at the frozen corrected thresholds 15/30/50 -> 19/30/50.
**The L22 read is style-invariant: suppressing the hedging TEXT does
not touch the internal uncertainty STATE** — the strongest evidence
yet that the probe reads the state, not the style (and it makes the
probe strictly better calibrated than the model's own verbal
abstention: AUC .81 vs a 43% false-abstain rate at n=30).

**Verdict on Q4:** prompting can kill the symptom but it is a bad
trade — −9.6 points bought back latency the router already recovers
surgically (escalation FIXES the uncertain queries instead of
shortening them; the gate needs no persona change, no threshold
re-quantiling, no new deployment claim). The gate IS the correct
anti-hedge. Spend ≈ $5.

---






### 8aj — demo audio-out: the talker SPEAKS (lifts milestone shortcut #3) (~$3, 2026-08-24)

User (looking at the live demo): "我以为真实的情况应该是audio进,audio出"
— correct; text-only output was the recorded milestone shortcut. Lifted
today, faithfully (no OpenAI TTS dub — the talker's own head):

**TTS smoke (`modal_tts_smoke.py`), first init_tts=True run of the
project.** MiniCPM-o 4.5's talker head + stepaudio2 Token2wav vocoder
work under the pinned stack (torch 2.8 / transformers 4.51): load
16.5 s + vocoder 10.5 s, VRAM 23.1 GB (peak 26.9 — H100 fine). Local
answer path: first audio chunk **1.39 s** after generation start,
15.0 s speech. Relay path: first audio **0.91 s**, relays the injected
expert answer verbatim. Gotchas found: (1) `Token2wav` needs a one-time
voice-prompt cache (`init_token2wav_cache`, official
`system_ref_audio.wav`) or streaming_generate crashes on
`token2wav_cache[...]`; (2) `reset_session()` silently wipes that cache
— every per-turn reset must pass `reset_token2wav_cache=False`.

**Demo wiring (`demo_app.py`, deployed).** `init_tts=True` at load; both
answer paths generate with `generate_audio=True` and stream ~1 s 24 kHz
PCM chunks over the WS as base64 events (typed /say returns a full
`answer_wav`); the stall is a canned filler in the talker's OWN voice,
teacher-forced once at container start (audio == STALL text guaranteed),
played the moment the gate fires — exactly the `tts_filler` deployment
semantics. **The probe context is byte-identical**: `<|tts_bos|>` enters
at generation time, AFTER the end-of-turn read, and the vocoder voice
prompt never touches the LLM context, so every frozen threshold stays
valid. Cold start 28.7 s (was ~18).

Live verification (typed /say, real GPU turns): "What is NVIDIA stock
trading at right now?" scored **.693 ≥ .680** and escalated (the user's
earlier voice render scored .624 — same question straddles the balanced
threshold render-to-render: boundary behavior, the motivation for 8ak);
first relay audio 847 ms, 15.8 s spoken. Regiomontanus (aggressive)
escalated .568 ≥ .400, relay spoken 7.4 s, first audio 661 ms. Also
live-confirmed: the no-browsing gpt-5.5 expert CANNOT answer the NVDA
question either — it relays a polite refusal, so escalating real-time
queries buys nothing until the expert gets a web tool (open decision).

New metric surfaced in the demo table: "first answer audio" — the
audio-out analogue of answer_ms; text-complete latency is unchanged and
stays the paper's conservative metric.

---



### 8ak — probe v4: real-time-data awareness via FreshQA (+ web-search expert for the demo) (~$12, 2026-08-24/25)

User (from the demo): the NVDA stock-price question should route to the
expert — "我要求你训练一下这个能力". The class was genuinely untrained:
every v3 train family is static knowledge, and "what is NVDA trading at"
sat ON the balanced threshold, flipping with TTS render (voice .624 <
.680; typed render .693 ≥ .680). Boundary behavior, not a decision.

**Data (`modal_fresh.py`; no-selfmade-datasets respected).** FreshQA
(freshllms/freshqa sheet of 2026-04-21, public benchmark with fact_type
labels): `fresh_fast` = fast-changing & !false_premise (153) labeled
escalate=1 A PRIORI — the one deliberate deviation from measured labels,
because the current value is unknowable at any cutoff (a judge against a
stale gold would only add noise to a label certain by construction);
`fresh_never` = never-changing & !false_premise (150) through the
standard answer+judge path (fail-rate .47) as in-family controls so the
probe cannot just learn "FreshQA phrasing = fire". 30+30 held out.

**Refit (v3 recipe, feature config frozen to L22
eot_last+eot_mean+user_mean).** n=2553 (2310 v3 + 243 fresh train),
C=1e-4, OOF AUC .876. Both pre-registered guards pass:
- frozen-test AUC **.877 vs v3 .879** (no regression);
- fast-heldout fire-rate **.80/.93/1.0** (cons/bal/agg) vs v3
  .57/.80/1.0 — conservative +.23, balanced +.13 — while never-heldout
  stays at v3 levels (.07/.17/.60 vs .07/.17/.53).

**Calibration lesson (one iteration).** First refit quantiled the tier
budgets over the AUGMENTED train mix — 123 extra all-positive rows
pushed balanced .680 → .737, and the typed NVDA render (.706) STILL
stayed local: the new capability was being eaten by its own threshold
shift. Fix: budgets are deployment-mix quantiles, so they are taken over
the v3-mix OOF rows only (fresh rows train the direction, not the
budget). Thresholds land at .860/.680/.402 ≈ v3's — same budget
semantics, new direction. Demo switched to v4 (v3 artifact untouched).

**Expert side (user bug report, same session).** Escalating a real-time
query bought nothing: the no-tools gpt-5.5 relayed a polite refusal
(live-confirmed twice). `escalate.ask_expert_web` added — Responses API
with `web_search`, DEMO-ONLY (the measured eval arms keep the tool-free
ask_expert and its caches untouched), falling back to the no-tools
expert on error. User's live voice turn under v4: ASR heard "What is the
stock price of Nvidia today?", probe .906 ≥ threshold, ESCALATED.

**Live verification (fresh container, recalibrated v4).** "What is the
stock price of Nvidia today?" (typed /say, balanced): probe **.829 ≥
.680** → escalated; web-search expert **3.13 s** → "**$208.48 USD,
down 2.93%**" — a real price for the first time; talker relays and
SPEAKS it (first audio 1.00 s, total 10.2 s). Static control "capital
of France": probe **.015** → local, correct, instant. Ops gotcha:
after `modal deploy`, warm old-version containers keep serving the old
artifact/code for minutes — `modal app stop gate-demo --yes` +
redeploy forces fresh containers (two verify rounds hit stale ones).

---



### 8al — soft barge-in: talk over the talker and it yields (~$3, 2026-08-25)

User: full-duplex should accept interruption ("STOP" while the model
speaks) — and correctly diagnosed as untested: is it MiniCPM or echo?
Neither. Our turn-based simplex pipeline structurally could not hear an
interruption: `_answer` ran synchronously while the WS receive loop was
blocked, and the post-answer drain (an 8ah fix) then DISCARDED everything
said during the answer. MiniCPM's native duplex machinery (listen/speak
head, break_event) sits unused in the weights; echo is irrelevant while
nobody is listening.

**Implementation (soft barge-in — no duplex head, no probe change).**
(1) `_gen_speak`/`_answer` take an abort Event: generation stops at chunk
granularity, the expert wait polls every 100 ms, the relay is skipped if
aborted; partial turns return `interrupted: true`. (2) The WS answer
phase becomes a watch loop: mic frames are read WHILE the model answers;
sustained loud speech (>= 0.4 s above 5x adaptive floor — deliberately
stricter than the turn VAD so speaker echo through imperfect AEC cannot
self-interrupt) or the manual button sets abort and pushes an
`interrupt` event. (3) The interrupting audio seeds the NEXT turn
(speech state pre-armed), so what you said while cutting it off is what
it answers. (4) Frontend tracks scheduled AudioBufferSources and
cancels them on `interrupt` — playback dies instantly.

**Verification (`_ws_barge.py`, mic-shaped protocol).** Two scenarios,
both PASS on the deployed app: (a) barge-in during a LOCAL spoken answer
— "grasshoppers" answer cut mid-sentence, turn marked interrupted, the
interjected "How many died in the Columbia?" became turn 2, escalated
.878, expert 6.3 s, relayed and spoken; (b) barge-in during the
ESCALATED expert wait — abort 62 ms after the stall, no relay, partial
turn returned. Bonus finding from the first (accidental) run: a > 1.25 s
pause INSIDE a long question makes the VAD fire early and the model
start answering — the question's own continuation then triggers
barge-in and the remainder becomes the next turn: graceful recovery
from early end-pointing, which is the honest duplex-ish behavior for
the turn-based loop.

Native duplex integration (MiniCPMODuplex per-chunk listen/speak) stays
future work: the probe is calibrated on simplex streaming states, and
the L22 read's validity under the duplex loop is an open experiment.

---



### 8am — backchannel-aware barge-in: "ok" keeps the floor, "stop" takes it (~$2, 2026-08-25)

Two user reports against 8al, both correct. (1) "STOP" still ignored:
the 0.4 s continuous-loudness criterion zeroed on the plosive closures
inside the word itself (s-T-o-P), the browser AEC's double-talk
suppression shrank the barge-in signal server-side, and — the biggest
window — generation finishes BEFORE playback does, so for the last
seconds of every answer the server was back in "listening" with no
barge-in logic at all while the browser kept playing. (2) The energy-only
trigger had no semantics: backchannels ("em", "ok", "嗯") would cut the
talker off — the exact thing MiniCPM's (unused) native duplex
listen/speak head is trained NOT to do.

**Fix: two-stage duck → classify → resume/commit (demo-grade
approximation of the ls-head).** Speech ≥0.25 s over the talker →
playback volume ducks to 12% (client GainNode; server event) but
generation CONTINUES. Short burst ends (≤1.2 s + 0.45 s quiet) → the
burst goes through the 8ae ASR (~1 s) and a token-wise en+zh lexicon:
all-backchannel → volume restored, floor kept, zero loss (generation
never paused); anything else → commit: playback killed, generation
aborted, the burst seeds the next turn. Sustained speech >1.2 s →
commit immediately, no ASR. ASR failure defaults to commit (one wrong
stop is recoverable; an ignored "STOP" is the original bug). Server
criterion is now leaky (4x floor, 0.3 s, plosive-proof); the client
covers the playback-outlives-generation window locally with the same
duck-then-decide shape.

**Verification (`_ws_barge.py` rewritten, TTS-rendered probes).**
Scenario A: "Okay." (0.47 s) over the talker → duck → resume (ASR heard
"Okay.") → answer completes uninterrupted. Scenario B: "Stop!" (0.44 s)
→ duck → interrupt (heard "Stop.") → turn marked interrupted. Both PASS
on the deployed app, first attempt.

Honest limits: ~1.5 s duck-dip before a backchannel verdict (the native
ls-head would not blink); lexicon+ASR is a semantic approximation —
"wait, what?" commits (correct) but so would an enthusiastic "no way!"
(arguable). True duplex integration remains the future-work line.

---



### 8an — stats hardening D2: the +0.012 gate-vs-oracle residual is n.s. ($0, 2026-08-25)

Deadline triage (RTCA @ NeurIPS 2026, due 8-29 AoE) found system.tex:21
claiming the in-distribution "+0.012 area (~2 points at balanced)" over
the pool oracle with no test — the exact claim the 7-29 entry flagged
("is +0.012 even significant at n=240?"). Now tested:
`scripts/07_bootstrap_oracle.py`, local CPU, inputs pulled from gate-data
(calib_features / gate_config / eval_expert / eval_paraphrase). The script
reproduces eval_assemble's trapezoid-area machinery exactly — point
estimates gate **+0.0541** / oracle **+0.0422** / delta **+0.0119** match
the 7-29 numbers to the fourth digit. Paired bootstrap (resample the 240
test queries, B=10^4, seed 42; oracle pool-rates stay fixed from calib —
they are the router's definition, not test data): delta 95% CI
**[−0.003, +0.027]**, P(delta≤0)=.066, **two-sided p=.131 — NOT
significant**. system.tex:21 softened to "statistically on par with the
pool oracle in distribution"; the section's argument (the gate's case is
transfer/LOPO, not the in-distribution margin) already rested on the right
leg and is now strictly consistent. This was the only untested-delta claim
of its class in the paper (D2 of the 8-25 todo triage; grep confirmed no
other +0.012 reference).

---



### 8ao — TODO clearance pass: v3 gold-inject gaps recomputed, appendix assembled ($0, 2026-08-25)

Paper-side cleanup to zero in-text todonotes before the peer meeting.
The one new measurement: **external gold-inject gaps recomputed on the
v3 traces** (`{pool}_v3_traces.parquet` + `{pool}_ceiling.parquet`,
local CPU; escalated rows re-scored with the ceiling gold-text expert
outcome under each family's own judge). Aggressive arm:
striviaqa **+.056**, swebq **+.036**, sdqa **+.035** (v2-sweep values
quoted before: .040/.024/.025 — same order, still 3–5× below the
internal pool's .175). Striviaqa gold-view random-line clearance at
aggressive: gate .916 vs random-gold .816 = **+.100** (was +.044 on
v2). transfer.tex updated with the v3 numbers.

Also this pass: appendix.tex assembled (full-pool dualview/pareto,
noise_audit, six family figure pairs, judge + p(True) prompts verbatim
from src/escalate.py / modal_app.py, reproducibility statement);
fig:roc regenerated with p(True) curves (see 79eac42); Limitations
refreshed (three stale items corrected: threshold calibration now
label-free-quantile-solved, second-family test executed, conservative
tier now significant under v3) + input-side scope item + future-work
paragraph (folds the todo [F] list); trace-decomposition sentence
scoped to the first-generation sweep rather than re-run. todo.tex is
now process-only (compress to 8pp / feedback / submission mechanics).

---



### 8ap — the 8-page cut: full restructure per external review ($0, 2026-08-25)

GPT-based review verdict accepted in full: "~2.5 papers in one", main
text ~25pp → must be 8. Executed same day:

- **One thesis**: mid-layer competence signal → pre-token escalation →
  live duplex system → OOD speech transfer. 7 contribution bullets → 3
  (representation / modality / system). ~15 claims → 4.
- **Title**: "Zero-Training" dropped (the probe IS trained — LR on
  hidden states); now "Reading Competence from Full-Duplex Speech
  Models for Real-Time Escalation".
- **Structure**: intro (hero system figure `hero_system.py`, replaces
  the synthetic concept fig) / related 0.7pp / setup 0.7pp / NEW
  signal.tex (merged rq1+duplex+speech cores, one combined signal
  table, layer_sweep as the core figure) / NEW gate.tex (fork timing +
  offline tradeoff + thinking table) / live.tex 6pp→1.2pp (loop +
  fair_dualview + channel-tax sentence) / transfer 1.2pp (tab:transfer
  + anchors + negatives) / discussion 0.6pp (taxonomy para + 6-line
  limitations + 5-line conclusion). rq1_text/duplex/speech/system .tex
  deleted (git history).
- **Appendix A–L absorbs**: roster, extended signal analyses (lopo/
  sweep/audioptrue/controls/quadrants/SD-QA/deployed-probe), Nemotron,
  router audit, fork/overlap, live details (tables, prefetch,
  injection framings, uplink diagnostics, FalseQA/FreshQA, demo),
  latency fold, full-pool + family figures, prompts, repro.
- **Language**: no v1/v2/v3/v4 or artifact filenames in main text
  (mechanistic probe vs deployed probe); causal claims softened
  ("consistent with late-layer specialization for turn control", "the
  competence signal also predicts relay compliance", "generalizes to a
  second architecturally distinct duplex family" — not "as a class").
- **Result**: 24pp total, main text ends within page 8, 0 unresolved
  refs, compiles clean. fig:tradeoff moved to appendix to close the
  last 2-line overflow.

---

### 8aq — duplex-validation sweep: the gate under the talker ⭐ (~$4, 2026-08-26)

Answers the external reviewer's #1 objection ("the native duplex loop is
never engaged") with the cheapest credible experiment: replay the frozen
live-sweep 240 through the DEPLOYED demo voice loop (talker head ON,
spoken answers, mic open during speech) and check the frozen v4 probe.

**Protocol** (`_ws_duplex.py` → gate-data `duplex_sweep/{clean,overlap}.jsonl`,
analysis `_duplex_analyze.py`):
- probe_on=0: eot scores computed + logged per turn, nothing escalates —
  isolates gate validity from routing outcomes.
- clean arm: pool TTS wav + mic noise (rms .008) streamed at 4×, VAD EOT,
  talker answers aloud, abort after first audible chunk. 239/240, ~7 s/q.
- overlap arm: easy warmup (score<.25, local-correct, answer>5 s) puts the
  talker mid-answer; target query spoken OVER it at gain 1.6 → duck →
  interrupt → barge-in speech seeds the next turn → EOT read. barged on
  237/239 pairs, 0 false backchannel-resumes, ~16 s/pair.
- labels: frozen local-failure (max heard_ok over mode=local rows across
  tiers); same 239 joined ids for all three regimes.

**Headline: the frozen read survives the spoken loop, including barge-in
turns.**

| regime | AUC [95% CI] | ΔAUC vs headless (paired) |
|---|---|---|
| headless (frozen_v3_traces) | .866 [.817,.909] | — |
| clean voice loop | .871 [.824,.914] | +.005 [−.025,+.035] |
| overlap/barge-in | .856 [.803,.901] | −.009 [−.036,+.016] |
| overlap, barged-only n=237 | .854 [.805,.902] | — |

- per-query score stability across regimes r=.90–.92; balanced-threshold
  decision agreement .887 (clean) / .912 (overlap).
- eot_read_ms p50 24–26, p95 ≤30 — matches the paper's ~30 ms claim on
  the deployed stack.
- **EOT→first-audible (client wall): p50 ≈ .58 s both arms** — the gate
  decision precedes audible commitment by >0.5 s. This partially recovers
  P2(c) (first-audio timing), though under probe-off replay only.
- server-side turn.first_audio_ms came back null (deployed build predates
  the field?) — client wall clock used instead; do not chase.

**Scope honesty** (stated in app:duplexval): the loop keeps per-turn
session control — mic stays open while the talker speaks and interrupts
seed the next read, but incoming audio is NOT prefilled during
generation. "Concurrent-with-generation read" stays future work.

**Paper**: live.tex closing ¶ "Gate validity with the talker on",
app:duplexval, intro scope sentence updated, Limitations (vi) narrowed.

---



### 2.1 public pools ✅ (2026-07-07)

`build_public_queries` → **400 queries**: `hard-math` 150 (GSM8K test tail 100 +
MATH-500 50), `hard-knowledge` 150 (MMLU-Pro; **GPQA 401-gated** with no HF token
→ gracefully topped up from MMLU-Pro), `easy-fact` 100 (TriviaQA
unfiltered.nocontext). Dataset-loading + formatting code validated end-to-end.
Remaining 200 (easy-chat 150 + trap 50) are Claude-generated → need the secret.

Pure helpers (`src/queries.py`) unit-tested locally: MCQ formatting, GSM8K
reference extraction, and stratified 60/40 split (deterministic, seed 42) all pass.

### 8ar — TTS-on balanced re-run: EOT→first-audible measured (~$4, 2026-08-26)

P2(c) closure: the reviewer asked for time-to-first-audio; the v3 sweep was
text-out (8aj's shortcut), so nothing was recomputable offline. `bench_live`
gained a `tts` flag (default off, off-path byte-identical): init_tts +
Token2wav (`system_ref_audio` voice cache, `reset_token2wav_cache=False`
per turn — the 8aj gotchas), stall teacher-forced once into a canned buffer
in the talker's own voice, both answer paths through a `gen_speak` port of
demo_app's loop. Balanced arm re-run on the frozen 240 (4 workers, ~20 min
wall, expert answers all cache hits).

**Gate decision unchanged**: eot read p50 21 ms / p99 32 ms; the same
84/240 queries fire (escalation-rate agreement with the text-out arm exact;
<|tts_bos|> enters after the read, thresholds untouched).

**EOT → first audible audio** (n=240):
| path | p50 | p95 | p99 |
|---|---:|---:|---:|
| local answer (n=156) | .552 s | .599 | .644 |
| escalated = canned stall onset (n=84) | .022 s | .028 | .029 |
| pooled | .543 s | .597 | .639 |
| escalated: expert CONTENT audible | 6.43 s | 26.0 | 34.0 |

The headline inversion: **the escalated path reaches first audio 25×
faster than the local path** — the canned stall (3.2–4.1 s of speech)
plays the moment the gate fires, covering the head of the expert wait;
relay first PCM lands p50 .66 s after the expert answer returns. Landed:
live.tex (one sentence in the live-result paragraph) + app:livedetail
("Time to first audible audio"). Artifacts:
`data/frozen_v3tts_traces.jsonl.balanced.shard{0..3}` (+ smoke files),
`scripts/10_tts_latency.py`. Caveats: single run per query (same ±.02–.03
replication discipline as the sweep); spoken answers run long on math
(spoken_s p95 146 s — the loop speaks the whole chain of thought, a
separate UX problem, not a latency one).

### 8as — presentation pass: percent numbers, vector figures, fun finding ($0, 2026-08-26)

Meeting feedback executed, no new experiments. (1) **Number format**:
main text accuracies/rates -> percent (42.2% -> 63.3% etc.), AUC/probe
scores/thresholds -> 0.xx; appendix uniformly 0.xx. While converting,
fixed three silently-truncated never-arm values in the main transfer
table (WebQ .570 -> 57.2, SD-QA .510 -> 51.5, Reasoning .590 -> 58.9,
verified against data/*_v3_live.json + RESULTS 8w/8z tables; avg/delta
columns recomputed). (2) **All figures now embed as vector PDF** —
verified via build log, zero PNGs left in main.pdf. Redrew the five
PNG-only matplotlib figures from volume data with paper-size fonts/
legends (figures/paper_pdf_redraw.py; layer_sweep is now a 2-panel
LOPO-hard-math last/mean view with an L22 marker, replacing the 6-panel
diagnostic grid; accuracy axes in percent); teaser_v2 got a lossless
PDF wrap. Source data fetched to figures/_voldata/ (gitignored).
(3) **Fun finding into the paper**: 8al's accidental early-endpoint ->
self-barge-in -> next-turn recovery is now a short Discussion paragraph
pointing at app:duplexval. Build check: References starts on page 9 =
main text still fits the 8-page budget.

### 8at — NVDA paper-grid completion: official-judge re-mix on every transfer pool ⭐ (~$12, 2026-08-26)

User: "凡是论文表格里的 benchmark，都用在 MiniCPM 上训练过的 probe 套到
NVDA 的模型上重新跑一遍" — executed as the 8ac frozen-recipe probe (same
features/recipe/600-query calib, refit on Nemotron hiddens; raw weights
cannot cross hidden spaces) evaluated with tab:transfer's per-pool
protocol on every pool in the table. New infra: `modal_nvda_ext.py`
(rejudge_local / expert_fill / judge_experts / vb_local_valpaca /
dump_valpaca), `modal_nvda.py` POOLS += valpaca,
`figures/nvda_remix.py` + `nvda_remix_fig.py`.

**Protocol.** Offline re-mix (8ad's arithmetic, accuracy version):
top-r by NVDA probe score takes the expert outcome, the rest keep
NVDA's local answer. Judges aligned with tab:transfer per pool: the
three OAB pools re-judged with the OFFICIAL gpt-4o OAB judge (local
floors move ours→official: striviaqa .324→.352, swebq .280→.376,
sllama .668→.705), SD-QA ours, AlpacaEval VoiceBench 1-5. Expert
outcomes: 575 measured gpt-5.5 answers from the v3 escalated arms +
574 fresh fills on the same heard-transcript channel (gpt-5.5 low,
expert_cache), all 1,149 judged uniformly on the gold query (A
protocol = relay-free; B variant substitutes the measured live relay
outcome where one exists — tier points shift ≤ .032, the 100% point
≤ .040). AlpacaEval GPU pass: 199 official wavs through
NemotronVoiceChat with hidden capture; VB-judged local mean 3.548.

**⭐ All five pools clear matched-rate random (permutation p ≤ .0003).**
A-protocol table (see `data/nvda_remix.json`, figure `nvda_remix`):

| pool (judge) | 0% | 15% | 30% | 50% | 100% | AUC (official label) | rnd@50% |
|---|---|---|---|---|---|---|---|
| striviaqa (OAB) | .356 | .498 | .615 | .741 | .955 | .775 | .656 |
| swebq (OAB) | .376 | .480 | .588 | .696 | .828 | .771 | .602 |
| sllama (OAB) | .705 | .786 | .838 | .876 | .932 | .720 | .818 |
| sdqa (ours) | .310 | .425 | .545 | .675 | .890 | .754 | .600 |
| valpaca (VB 1-5) | 3.55 | 3.83 | 4.13 | 4.46 | 4.92 | — | 4.23 |

Probe split decisive everywhere — local accuracy escalated-half vs
kept-half: .187/.524 (striviaqa), .168/.584 (swebq), .581/.829
(sllama), .160/.460 (sdqa), VB 3.01/4.08 (valpaca).

**The paper's honest negatives do NOT replicate on this family.**
WebQ, SD-QA and AlpacaEval — all flat-vs-random on MiniCPM — are
unambiguous wins here. Direct confirmation of Addendum 8's
decomposition: selectivity is pool-invariant (AUC .72–.78 under
official labels), what changed is HEADROOM (floors .31–.38 vs
MiniCPM's .51–.66) and, on AlpacaEval, failure SPECIES — the terse
English-only model fails open-ended queries hard (probe-selected half
VB 3.01), where MiniCPM fails them soft (style/verbosity, invisible
to a wrongness detector, floor already 3.99).

**Caveats.** (1) Offline re-mix, not a live streaming run — the NeMo
live-loop port stays future work; on MiniCPM 8ad measured re-mix vs
live deviation at .011 mean. Why the port is non-trivial (now also in
app:nvda): the released NeMo path is CACHELESS (full-prefix re-run
per 80 ms frame, O(T²) — the 8ac receipt), so real-time needs
stateful streaming (Mamba-2 state is O(1)/frame in principle) wired
into the duplex frame protocol; a reference-code gap, not an
architecture one. (2) The expert hears MiniCPM's heard
transcript (NVDA has no offline ASR relay of its own); the A protocol
excludes the relay-back tax. (3) Dropped ids: striviaqa 3, sllama 16
(OAB judge unparsable). (4) sreason skipped — English-only model,
fail = 1.000, zero label variance (8ac boundary finding stands).

Files: `data/nvda_remix.json`, `nvda_expert_outcomes.parquet`,
`nvda_{striviaqa,swebq,sllama}.parquet` (+oab_ok col),
`nvda_valpaca.parquet`, `nvda_scores_valpaca.parquet`,
`nvda_expert_fill.parquet`, `figures/nvda_remix.{png,pdf}` (also in
paper/figures). Paper: app:nvda extended (tier table + figure),
transfer honest-negatives paragraph updated; NVDA block promoted
into the MAIN table tab:transfer (2026-08-27: always-local /
gate 15-30-50 / ceiling / AUC rows over the four available pools,
avg+Delta over its own floor, zh cell ---). Cost: ~25 min H100 +
~2.3k API calls (574 gpt-5.5 low + ~1.9k judge) ≈ $12.

### 8au — measured always-escalate arms on all seven pools (~$8, 2026-08-27)

User challenge on Fig 3 ("why don't the curves reach 100%?") -> the
100% points were synthesized (gold ceiling), never measured through the
deployed heard channel. Ran tier=always (thr=-1e9, built into
bench_live) on frozen + striviaqa/swebq/sllama/sreason/sdqa/valpaca,
sequentially (expert concurrency <=3); fast because the expert cache
already held ~half the transcripts. New `always_append` in
modal_bench.py folds the shards in INCREMENTALLY — judges only the new
rows (bench_report re-judges everything, which would wobble published
arms by the +-2-3pt judge noise; verified old arm numbers byte-stable
after the merge).

Results (heard, official judge where applicable): frozen full-pool
.617 / fair-subset .665 [.601,.729] P50 4.2s vs gated@57% .633 and
gold .922 — the last 43 points of escalation buy 3.2 points; the
uplink channel binds, not selection. External: striviaqa oab .880,
swebq .808, sllama .936 (**below selective .948 — the 8w offline
finding now holds in the live loop**), sreason .837, sdqa .885,
valpaca 4.76. Transfer table gained an "always-escalate (measured)"
row (avg 86.9, delta +23.3 vs floor); live.tex + Llama-Q claim
updated; all dualview/pareto/fair figures now end at a real measured
100% point (ARMS += always in the four figure scripts; pareto's
"synthesized" annotation removed). Commit e43ac72.

### 8as — concurrent-prefill probe validation + in-regime refit (~$10, 2026-08-27)

The tier the paper left open, now measured. Harness (modal_duplex_concurrent.py):
time-division interleave through the streaming API — while the talker is
mid-answer, each 1 s chunk of the target query prefills into the SAME session
between generation chunks (positions self-consistent: the decode loop re-derives
them from cache length), EOT read fires while generation is active (238/240
full coverage). Carrier answer teacher-forced — under sampling the duplex model
EOSes within ~2 s of user audio appearing in context (trained turn-yielding,
made visible by interleave; three failed smoke designs document it).

**Frozen v3 probe, concurrent state (n=240, labels = local-fail as 8aq):**
AUC .869 (headless surrogate) → .758 [.693,.816]; paired dAUC −.109
[−.158,−.064]. Scores drift up (mean .479→.714; frozen balanced thr fire rate
.367→.754); label-free requantiling recovers decision agreement only to .750.
The ONE shift on the transfer ladder that breaks the frozen read.

**In-regime refit (v2 scale): the signal is intact.** Same 12288-d linear read
refit on calib-360 concurrent features (existing escalate_label, C=3e-4, OOF
.775) → test-240 concurrent **AUC .818 [.761,.871]** — more than half the gap
recovered at a quarter of the deployed probe's calibration data. Read: L22
competence direction persists under concurrent listen/speak; calibration must
follow the regime. Run1-vs-run2 concurrent score corr .895; feature-pipeline
sanity: frozen weights on run-2 features reproduce .768 ≈ run-1's .758.

Caveat: interleave via the public streaming API (turn headers mid-generation,
forced carrier) — measures the concurrent STATE, not the official duplex
serving format. Artifacts: gate-data volume frozen_concurrent_traces.jsonl.shard*
(scores run), frozen_conc_{calib,test}_{traces,feats}.* (feature runs);
local copies in data/; scripts/11_concurrent_auc.py, 12_concurrent_refit.py,
figures/concurrent_refit.json. Paper: app:duplexval third arm, Limitations (vi),
intro scope clause. NOTE: landed while the parallel session's restructure
(Table 3, % scale, teaser_v2) is uncommitted — page rebalance owed after merge.

---

## Phase 8aw — why does the final layer invert? (advisor request, 2026-08-27)

济森 8-27: "最好能分析出来,为什么最后一层不行,比如我们假设到后面 Full Duplex
在考虑 Listen/Speak"。Consulted gpt-5.5 (`modal_askgpt.py` → `askgpt_interp.md`,
7 proposed experiments), ran the four that need no new GPU capture, on the
Phase-5d all-layer dumps. Scripts 14–17; data/interp_{lastlayer,controls,
subspace,reliance}.json. Probe protocol identical to `layer_sweep_report`
(reproduces L22 0.931 exactly, final layer 0.378 vs published 0.366).

### The Listen/Speak hypothesis does NOT survive in its simple form

Four candidates ruled out (paper Table `tab:mech`):

| candidate | test | result |
|---|---|---|
| prepare-to-speak state | speak-mode template prefill (5d ttstpl) | .362 vs .366 — unchanged |
| modality (audio/text) axis | cos(mean audio−text dir, w_final) | .013; chance = 1/√4096 = .016 |
| late layers add more shortcut | pool-identity decoding by depth | .88–.96 at ALL depths, duplex AND backbone |
| representation rewritten late | CKA vs backbone, standardized | .82→.78 flat (raw CKA .80→.47 is an artifact) |
| massive-activation coords | project out top-k such axes | .35–.40, no rescue (random ctrl .37) |
| query-type directions | project out pool-mean dirs | .34–.45, no rescue |

⚠️ **The raw-CKA collapse is a massive-activation artifact.** Residual streams
here carry coords of |h|~250 vs ~1 typical; they dominate the Gram matrix and
drive CKA to exactly 1.0 at some layers. Per-feature standardization removes
the effect. Any future CKA in this project must standardize.

### What does separate duplex from backbone (paper Fig `fig:mech`)

Non-destructive: split the trained probe's held-out score into the part in the
layer's top-5 PCs (estimated on TRAINING rows only) and the residual,
s = wᵀPx + wᵀ(I−P)x.

| model | residual-component LOPO math, mid → final | score var in dominant subspace, final |
|---|---|---|
| **MiniCPM-o 4.5 (duplex)** | **0.93 → 0.27** | **0.39** (≤0.18 at all earlier depths) |
| Qwen3-8B (its backbone) | 0.92 → 0.86 | 0.05 |
| **MiniCPM-o 2.6 (duplex)** | **0.81 → 0.52** | 0.16 |
| Qwen2.5-7B (raw) | 0.73 → 0.71 | 0.14 |

**Both duplex models lose the distributed read at the output; neither backbone
does** — that replicates. The *concentration* (var share .39) is o4.5-only; o2.6
matches its own backbone, so it is reported as a property of the stronger duplex
model, not a law.

### Explicitly NOT claimed

Removing the top-5 PCs at the final layer lifts LOPO math .378 → .758 — but the
same op costs the backbone .90 → .14 and the intact L22 read .931 → .760. It is a
damaging intervention that happens to help a broken readout, **not** evidence of
a duplex-added subspace. Paper states the finding only as: duplex tuning does not
erase the competence signal, it stops delivering it in distributed form to the
final last-token position. Attribution to a specific duplex objective needs
training-time access we don't have; the practical conclusion (read mid-network)
is unaffected.

Not run (need GPU / new captures): logit-lens over control tokens, listen/speak
prompt-intervention causal test (GPT's exps 3 and 6) — the two that could
actually confirm turn-control. Listed as future work.

Paper: signal.tex §4.2 mechanism paragraph rewritten (removed the unsupported
"consistent with late-layer specialization for duplex turn control" assertion),
new appendix `app:mech` + `tab:mech` + `fig:mech` (figures/lastlayer_mech.py).

### 8at — the FULL loop in the concurrent regime: 7-pool sweep + in-regime gate (~$60, 2026-08-27)

The user's directive: make it full-duplex. Everything the paper's main table
reports for the turn-based loop, re-run end-to-end in the concurrent
interleaved regime (carrier speaking, target audio prefilled into the same
KV stream, EOT read mid-generation, then stall->expert->relay or local
answer). Gate = the 8as in-regime refit probe (gate_conc_frozen.json);
per-pool label-free quantile thresholds from each pool's own never arm
(scripts/13). 36 arms total: 7 pools x {never, 15, 30, 50, always} + internal
conservative + a TTS-on latency arm. Judged with the standard pipeline
(::report / oab_rejudge_live / valpaca_report, suffix _conclive).

**Heard accuracy (our judge), concurrent regime:**
| pool | never | @15 | @30 | @50 | always |
|---|---|---|---|---|---|
| frozen (240) | .404 | .400* | .521 | .662 | .617 |
| sllama | .780 | .796 | .808 | .864 | .856 |
| striviaqa | .492 | .520 | .620 | .692 | .880 |
| swebq | .468 | .532 | .560 | .608 | .696 |
| sdqa | .495 | .555 | .605 | .705 | .895 |
| sreason | .406 | .485 | .584 | .644 | .807 |
| valpaca (VB 1-5) | 4.36 | 4.45 | 4.53 | 4.60 | 4.76 |
*frozen conservative realized only 3% (top-quantile calib->test transfer
missed in-regime; balanced 36%/aggressive 66% fine). External realized rates
.10-.51 track nominal.

**Headlines:** every pool monotone in budget; **selective > always-escalate
on BOTH frozen (.662 vs .617) and sllama (.864 vs .856)** — the paper's
signature result reproduces in the concurrent regime; latency profile holds
(EOT read 23ms, stall onset 27-34ms, local first-audio p50 .655s vs .552
turn-based — ~+100-200ms context cost); OAB arbitration: swebq floor shift
shrinks to +4.8pp under the official judge (near the +-2-3 floor; the
our-judge +9.6 is alias/style sensitivity), striviaqa floor -16pp and
sreason -18pp are real per-pool regime costs (EN carrier x zh query worst).
valpaca flips from honest-negative to rising curve (4.36->4.60) — carrier
context helps open-ended IF; random reference TBD before claiming gain.

Artifacts: {pool}_conclive_traces.jsonl.{tier}.shard* + judged parquets +
{pool}_conclive_live.json on gate-data; frozen_conclivetts_* (latency arm);
data/conc_thresholds.json; gate_conc_frozen.json (in-regime probe artifact).
TODO Friday: in-regime probe AUC per pool (never scores x labels), speakable
subset + CIs + gold-inject views, random references, the concurrent table
(ADD, not replace), Nemotron scope sentence, page balance.

### 8ax — the two direct turn-control tests (user-approved GPU runs, same day)

`modal_interp.py` (self-contained; NB: containers mount only the entry file, so
no `from modal_app import ...` — that cost the first attempt 2 h of silent
retries). Analysis `scripts/19_turncontrol_tests.py` → `data/interp_turncontrol.json`.
Also folded Qwen2.5-Omni into the reliance figure (residual holds .75 to output,
var% flat .07 → patterns with the RAW models; decay is duplex-specific).

**Control-token logit lens** (each model's own norm+lm_head over the stored
per-layer states; control set = added/special vocab, o4.5 n=106, qwen3 n=26):

| depth | o4.5 log-mass | qwen3-8B log-mass |
|---|---:|---:|
| L0 | −23.6 | −11.3 |
| mid | −18.5 | −11.1 |
| L32 | −23.8 | −23.5 |
| **L35** | **−11.8 (rises)** | **−38.5 (falls)** |

The duplex output layer uniquely stays "ready" to emit control tokens. BUT the
fine-grained predictions fail: cos(w35, control unembeddings) max .069 vs random
.068 (null); corr(control mass, w35 score) = +.18 (weak); argmax never lands on
a control token.

**Listen/speak prompt intervention** (60 calib queries × neutral/listen/speak
suffix, both models, all-layer capture): cue direction moves the final layer of
BOTH models (relative displacement duplex .27, raw .59 — cue text is just prompt
semantics, raw moves MORE); cos(t, w35) = .019 ≈ chance .016; w35 score shift
speak−listen = +0.23 sd (L22: +0.08 sd).

**Verdict: the simple turn-control-takeover story fails its own direct tests.**
What survives: the delivery failure (8aw) + the output layer being control-biased
in its output distribution. Paper: app:mech "Direct turn-control tests" paragraph,
signal.tex clause updated ("aligns with neither the control vocabulary nor an
explicit listen/speak axis"), todo P2 updated. Cost: ~6 H100 container-minutes.

### 8ay — BrownCat's Feishu duplex list folded in (2026-08-27)

Two Feishu docs (Full Duplex Training / 训练过程) arrived as PDFs. Extracted the
starred papers, verified every arXiv id + author list against arxiv.org before
writing bib (the 6 entries I wrote from memory on 8-27 also all checked out;
fixed 3 metadata errors: wang2024fsm is NeurIPS 2024 not preprint + author order,
veluri EMNLP Main, salmonn-omni year 2025→2024).

**Two papers we were missing that a reviewer would have caught:**

1. **MoshiRAG** (Chien et al., ICML 2026, arXiv:2604.12928) — *the closest work*.
   Compact full-duplex interface + selective retrieval to a stronger knowledge
   source, no retraining, evaluated on OOD math. Differences we now state:
   their trigger is the lag between speech onset and first content word, i.e.
   read off the model's own **emerging output**; what arrives is **documents for
   the small model**, not an answer from a larger one. Ours fires pre-output from
   internal state.
2. **DuplexOmni** (Huang et al., arXiv:2606.09186) — interaction layer + pluggable
   thinking layer (Gemini-3.1-Flash-Lite in their experiments), routed by a
   trained **[THINK] control token**. This is literally the special-token version
   of our small→large handoff. Control-token vocabulary from their doc:
   [THINK] / [CUT] / [WAIT] / [PEND N S] / overlap marker.

Also added: **DuplexSLA** (2605.20755, third structured-action channel so tool
calls emit without pausing speech), **FDB-v3** (2604.04847, cited in Limitations
(v) as the harder setting our turn-based loop doesn't cover), **Ohashi et al.**
(2606.11167, RL over interaction-level rewards).

**Useful fact from doc 2 for our own framing:** the MiniCPM-o 4.5 report states
its control flow predicts a **binary listen/speak token before generating
content** (they chose it over Listen-TEXT because it decouples "whether to speak"
from "what to say"). So the checkpoint we probe *itself* carries trained-in
control tokens — related.tex now says this, which makes the probe-vs-token
contrast concrete rather than abstract.

Paper: related.tex duplex paragraph extended, new paragraph "Escalating out of a
live duplex session", gap paragraph updated to acknowledge the two live-session
escalators, discussion Limitations (v) added. 35 pp, refs still start p9 (31%
into the page — main text did not regress).

NOT done: no DuplexPO in the list (济森 mentioned it as an example, it isn't in
these two docs). The list's other items (LayerSkip, Mixture-of-Depths,
DuplexCascade, UAF, Kyutai blog) are about early-exit/cascade/front-end, not our
axis — deliberately not cited.

---

### 8az — reviewer round 3 (P0/P1): fact-corrections, matched-random for the concurrent table, scope discipline (CPU-only, $0, 2026-08-27)

Feedback arrived as P0 ("不修不要提交") + P1 ("最可能改变录用结果"). Everything
below is recomputed from **already-measured outcomes** — no model call, no GPU.

**P1.1 — the one that mattered: matched-random for tab:conclive.**
New `scripts/20_conclive_random_check.py` → `figures/conclive_random.json`.
Per pool × gated tier, at that tier's **realized** escalation count k: 20k
random id-subsets, remix = always-arm outcome on the subset / never-arm
elsewhere (paired, same convention as 18_nvda_random_check.py), plus a 10k
bootstrap CI on each measured arm.

| pool | cons. | bal. | aggr. |
|---|---|---|---|
| TriviaQA | .85 | .32 | .15 |
| WebQ | .40 | .25 | .60 |
| Llama Q. | .50 | .43 | **3e-4** |
| SD-QA | .57 | .72 | .38 |
| Reason. zh | .14 | **9e-4** | **.0024** |
| internal 240 | .98 | **.017** | **5e-5** |
| internal speakable | 1.0 | **.009** | **5e-5** |
| AlpacaEval | **.031** | .14 | .15 |

Read honestly: **internally the concurrent gate is real** (balanced +4.0 /
aggressive +11.8 points over matched random on the full pool; +4.6/+12.8
speakable), **externally the 360-row in-regime probe mostly buys budget, not
selection** — exactly what its .57–.67 external AUC predicted. The table's
"realized rates track nominal, monotone gains everywhere" story was the part
that didn't survive: two internal cells are flat (40.4→40.0, 44.5→43.6) because
the conservative threshold under-fires at 2.5%.

**P0 fixes.** (1) tab:conclive rebuilt: tier columns headed by realized rates,
`rand` row + permutation stars under every pool, non-monotone cells named in
prose, Llama selective-vs-always restricted to one judge and marked
inside the ±2–3 floor (official 90.8 vs 91.2; ours 86.4 vs 85.6 — both noise).
The whole sweep is now labeled **exploratory**. (2) prior work: SEP is *not*
purely post-hoc (it probes before generation) — sentence corrected; added
mahaut2024factual (probes = most reliable factual-confidence estimator),
chuang2024lookback, lugoloobi2025difficulty/2026failures (pre-generation success
probes), varshney2026llmrouter (prefill-activation routing). Novelty restated as
duplex-specific readout failure + text→speech transfer + live/concurrent
integration — *not* pre-generation probing per se. (3) terminology: prose says
**deployed-channel answer accuracy** (sweeps are text-out; figures keep the
"heard-acc" abbreviation and say so); the concurrent arm is called
teacher-forced interleave, explicitly not the official free-running duplex
serving path.

**P1.2/1.3/1.4.** Gate description unified (4096-d *per position*; deployed
probe = 3 positions = 12,288-d) and a new app:signal paragraph says which
calibration drives which result (360 mechanistic / 2,310 deployed / 360
in-regime concurrent — the only probe fit inside the regime it is scored in).
Test-selection history disclosed (layer + feature choice on internal
calibration-split CV, confirmed on held-out pools, **not** nested per-pool
selection; internal frozen test = pre-registered guard, not an untouched
holdout) — Limitations (v). Mechanism statement unified across
abstract/intro/§5.2: *duplex checkpoints exhibit a late distributed-read decay;
mechanism unresolved* — no more turn-control specialization anywhere.

**Declined / deferred:** nested layer re-selection inside each held-out pool
(needs a full re-sweep per pool; disclosed as a limitation instead) and
full-scale 2,310-row in-regime recalibration (GPU, would likely close most of
the external gap — the honest camera-ready item).

**Page cost:** References moved doc-line 332 → 349 (~0.35 page) after clawing
back ~13 lines by compressing live.tex/related.tex. Main text still runs onto
p9; the trim decision flagged in P0-R remains open.

### 8ba — floor-control sweep: the gate does not touch barge-in vs backchannel ⭐ (user request, 2026-08-27, ~$14 GPU + ~$3 API)

用户 8-27:"现在的模型只要 user 一开口就自动停止,这破坏了 full-duplex 对
barge-in 和 backchannel 的区别——要一个实验证明 escalation 的提升没有破坏
talker 的 full-duplex 能力。" Designed + ran the floor-control sweep the same
night: `_ws_floor.py` (sweep/report) + `_floor_analyze.py`; 416 pairs against
the deployed voice demo; data `gate-data:floor_sweep/floor.jsonl`.

**Design.** Cell = phase × stimulus × arm. Phase `ans` = overlap injected
while the talker speaks a local answer (non-firing queries, BOTH arms
probe_on=0/1 — the paired orthogonality claim); phases `stall`/`wait`/`relay`
exist only under escalation (firing queries, g1 only) and are the actual risk
surface. Stimuli: `bcs` short backchannels (Okay/Yeah/Mm-hm/嗯/好的, 0.40–0.46 s),
`bcl` long lexicon-only continuers (0.62× time-stretched to 1.33–2.08 s, probes
the ≥1.2 s sustained-commit rule), `stop` out-of-lexicon commands, `bq` =
frozen-pool query spoken over the talker. Injection at real-time pacing
(0.128 s/frame) so client latencies are honest; one ws session per pair;
40 pairs/cell (ans), 16 (escalated phases).

**Claim A — orthogonality: exact.** All four ans-phase paired deltas (g1−g0)
are +0.000 [0.000, 0.000], n=40 each — 160/160 pairs make the identical
duck/resume/interrupt decision with and without the gate. Latency paired
medians: duck ≤14 ms, resume ≤31 ms, interrupt ≤6 ms. The gate reads once at
EOT and never touches the floor state machine; measured, not just asserted.

**Claim B — escalated phases behave.** Barge-in aborts the escalated turn
40/40 (stall 16, wait 14, relay 10) and seeds the next turn 40/40; interrupt
latency p50 ≈1.39–1.40 s from overlap onset. Short backchannels keep the floor
at .55–.63 across stall/wait/relay — statistically the same as the ans-phase
rate (.60), i.e. the escalation phases add no new failure mode; a
false-interrupt during `wait` does cancel the pending expert call (the cost of
fail-toward-interrupt there).

**Diagnosis — why "一开口就停": two deterministic mechanisms, both
gate-independent.** Per-variant, per-stimulus deterministic: (1) non-lexical
hums die by ASR-empty fail-closed — 'Mm-hm.'/'嗯。' 16/16 dead (empty
transcript → not classified backchannel → interrupt), 'Okay.'/'Yeah.'/'好的。'
0/16; (2) continuers with >1.4 s sustained loudness die by the no-ASR
sustained-commit rule — 'Oh wow, right, right.' (1.83 s)/'嗯,好的,继续。'
(2.08 s) 20/20 dead at int_med ≈1.52 s < stimulus end, pause-bearing 1.33–1.37 s
variants 0/20. Aggregate false-interrupt: bcs 40%, bcl 50% — identical in both
arms. So the user-felt "auto-stop" is the harness floor policy (its two
fail-closed rules + the 12%-volume duck during the ~1.5 s resume round-trip),
NOT the escalation gate.

**Also relevant:** the July FDB run (repo root `fdb/RESULTS.md`, 2026-07-04)
already measured the checkpoint's native duplex profile on Full-Duplex-Bench
v1.0 (727 samples, official scripts): best-in-class pause handling (TOR
.125/.117), user-interruption TOR .915 @0.90 s, and **zero native
backchannels** — the lexicon floor policy exists precisely because the talker
head has no backchannel behavior of its own to preserve.

**Run hygiene (3 restarts before the clean run):** (i) pool TTS pauses >1.25 s
trip the server VAD mid-query and the query tail then reads as a barge-in —
39% contamination in launch 1; fixed by a VAD-faithful pause filter (frame RMS
+ stream noise vs 0.028, max internal silence <1.0 s) + a pre-injection
early_eot guard; (ii) a stopped run's container committed the volume after
`modal volume rm` and resurrected stale records — always confirm 0 containers
before deleting; (iii) busy-retry needed on the thr probe hello. Final run:
416/416, 0 errors, 0 early_eot, 3 no_fire, 3 no_duck.

**Landed:** app:duplexval floor-control paragraph + table (appendix),
todo P1 → DONE, refs.bib + lin2025fdb (FDB v1.0).


### 8bb — full-scale (2,310-row) in-regime recalibration: scale is real but the regime keeps a residue ⭐ (~$25 GPU, 2026-08-28)

The deadline moved to 9-10, so the "honest camera-ready item" from 8at ran
now: the paper attributed the concurrent probe's weak external AUC
(.57–.67) to its quarter-scale calibration (360 rows) without testing it.
Tested it.

**Collection.** `modal_duplex_concurrent.py` generalized (FEAT_POOLS map:
pool → query file + audio dir; `concurrent_shard` takes `audio_dir`;
frozen default unchanged). The whole v3 train mix re-collected in the
concurrent-prefill state — expansion 800 (x*) + expansion2 1,150 (y*),
joining the existing calib-360 — plus the five binary external pools
(striviaqa/swebq/sllama 250, sdqa 200, sreason 202). 3,102 rows, 8×H100
per pool sequential, 48 min wall, 0 errors, gen_active_at_eot 3,102/3,102.
Feats `frozen_conc_{exp,exp2,striviaqa,swebq,sllama,sdqa,sreason}_feats.shard0-7.npz`
(volume + data/), traces alongside; log `conc_fullscale.log`.

**Refit (`scripts/21_conc_fullscale_refit.py`).** Same recipe as
scripts/12 (LogReg, C swept, 5-fold OOF, no scaler); labels =
escalate_label from calib_features/expansion_labels/expansion2_labels
(fail rates .32/.40/.50, pooled .437); C=3e-4, train OOF .823. External
labels mirror scripts/14 auc_never (never-tier conclive outcome, pool's
own judge). Artifacts `data/gate_conc_fullscale.json`,
`figures/conc_fullscale.json`.

**Replication sanity.** The 360-row probe re-scored on the freshly
collected external features reproduces the 8at trace AUCs:
.645/.641/.674/.639/.615 vs .636/.636/.672/.636/.570 — run-to-run floor
holds (sreason wobbles most, +.045).

**Result (paired on identical features, 10k bootstrap):**

| pool | p360 | p2310 | Δ paired |
|---|---|---|---|
| internal test-240 | .818 | .817 | −.001 [−.037,+.035] |
| striviaqa | .645 | .699 | +.054 [−.015,+.123] |
| swebq | .641 | .713 | **+.073 [+.023,+.121]** |
| sllama | .674 | .755 | **+.081 [+.017,+.150]** |
| sdqa | .639 | .701 | +.062 [−.007,+.131] |
| sreason (zh) | .615 | .577 | −.039 [−.117,+.037] |

Mean paired Δ: all-5 **+.046 [.017,.076]**, En-4 **+.068 [.036,.099]** —
both clear zero. Scaling curve (fixed C, 3 seeds, stratified subsample):
external-mean .625→.646→.658→.679→.689 at n=360/720/1150/1560/2310
(internal .714→.817) — monotone, flattening.

**Reading.** Scale is real (the attribution wasn't wrong) but buys only
~a third of the distance: external mean .643→.689 vs turn-based .771
(per-pool turn-based .789/.785/.806/.792/.683). English pools land
.70–.76, still short of .785–.806; Reasoning zh — no zh in the 2,310
English calibration mix — declines within noise. So the corrected claim:
*calibration must follow the regime AND the concurrent regime carries a
genuine residual external cost that calibration scale does not pay off;
plus a language axis (zh untouched by en calibration).* Internal is
scale-saturated (.818 flat from 360 to 2310).

**Landed:** app:duplexval attribution passage rewritten with the measured
numbers; app:signal provenance sentence (now two in-regime fits);
Limitations (iv) updated ("scale helps only partly"). NOT re-run: the
conclive loop tiers with the 2310 probe (tab:conclive numbers unchanged,
still 360-probe; exploratory label stays). Bib cleanup same session:
orgad2024llms was missing Hadas Kotek (added), lin2026fdbv3 full author
list (Lin/Chen/Chen/Lee), "TODO verify" notes deleted.


### 8bc — benefit-trained refit: the expert-agnostic label leaves nothing trainable on the table (~$8 API, 2026-08-28)

P0-R q3's optional item, now that the deadline allows it. app:fixedthr
re-SCORES the fail-trained probe against the benefit label (y =
local-wrong AND expert-right) and pays 0.63–0.81 external / .840→.732
internal. Open question: label's price or probe's? Answer: the label's.

**New data.** `modal_benefit.py::train_ceiling` — gpt-5.5 (low) answers
the GOLD text of all 2,310 train queries (expert-cache deduped, 0
errors), standard judge → expert-right rate .853 →
`train_ceiling.parquet`. Benefit rate on train: .305 (fail .437 × the
expert fixing ~70% of those).

**Refit (`modal_benefit.py::benefit_refit`).** Same 12,288-d L22
3-mode read from the stored eoth2 hiddens (CPU-only, as designed),
LogReg C swept, benefit OOF .758 (C=1e-4). Scored against the deployed
v3 probe on IDENTICAL features; eval label channels mirror scripts/09
(never-tier local + ceiling expert, pool's own judge). Replication
anchors: internal fail-probe benefit-AUC .732 = app:fixedthr's number
exactly; striviaqa .782 exactly; others within .01–.03 (eoth2 re-score
vs trace-score floor).

| pool | bAUC fail-probe | bAUC benefit-probe | Δ paired |
|---|---|---|---|
| internal test | .732 | .759 | **+.027 [+.004,+.052]** |
| striviaqa | .782 | .771 | −.010 [−.043,+.022] |
| swebq | .719 | .749 | +.030 [−.012,+.072] |
| sllama | .829 | .800 | −.029 [−.071,+.008] |
| sdqa | .765 | .754 | −.010 [−.047,+.025] |
| sreason | .634 | .671 | +.037 [−.007,+.083] |

Fail-AUC cost of benefit training: internal .840→.839 (zero); external
−.01 to −.03.

**Reading.** Benefit training buys a small significant gain exactly
where calibration matches the distribution (internal +.027) and NOTHING
external — every CI spans zero, mean ≈ +.004. So the benefit gap
measured in app:fixedthr is not an objective-mismatch artifact the
paper left on the table; it is the label importing the expert's own
failure surface (which the probe never sees and cannot see). The
"label = local-wrong by design" choice is vindicated as costless.

**Landed:** app:fixedthr closing sentences; artifacts
`train_ceiling.parquet`, `benefit_refit.json` (volume), modal_benefit.py.

## Phase 8bd — native full duplex: the soft barge-in harness is retired (2026-08-31)

**Ask (user):** remove the demo's harness barge-in entirely; keep (1) the
model's NATIVE full-duplex ability unconstrained, (2) an escalation hook
at the moment the talker starts to answer, (3) native interruptibility
of the relay. Point (4) — abort the in-flight thinker when the user
speaks during the wait — explicitly deferred (memory:
duplex-demo-thinker-abort).

**What the old demo actually did** (user asked point-blank; confirmed in
code): barge-in was 100% harness — server energy-VAD (≥1.2 s sustained
loud) + duck + hosted-ASR backchannel word-list (`_classify_burst`) +
a threading.Event that stopped *consuming* `streaming_generate` chunks,
plus a client-side playback kill switch. MiniCPM ran turn-based
half-duplex; interrupting speech never entered the model's context
mid-generation.

**New app `demo_duplex.py`** (Modal `gate-demo-duplex`; page
rhe9527--gate-duplex.modal.run, ws rhe9527--gate-duplex-voice):
`model.as_duplex()` → MiniCPMODuplex; every 1 s of mic audio is
`streaming_prefill`ed into the SAME context the model generates from;
`streaming_generate` per unit → the head itself emits <|listen|> or
speaks; yields via <|turn_eos|>. No VAD, no burst ASR, no abort Event,
no duck/kill-switch (browser AEC is the only echo control). Gate: L22
hook (same tail-8 + user_mean features), scored at the listen→speak
transition = "talker commits to answering"; conc-frozen weights +
thresholds (closest regime; **native-duplex schema is uncalibrated —
scores exploratory**). Fired ⇒ gpt-5.5 (web) in a background thread
(ASR uplink of the last ≤30 s of user audio); on return the answer is
prefilled as a TEXT unit (RELAY_TMPL) into the live stream — the relay
is an ordinary duplex turn, so it is natively interruptible by
construction.

**Smoke (`_ws_native_smoke.py`, 3 arms, all pass):**
- local (q0010): pattern L×15→S×31→turn_eos; mean chunk cost 0.41 s
  (realtime holds on H100); gate read at onset .179 < .386 (no fire);
  correct answer ($31).
- barge: overlap speech at 19.2 s while talker mid-answer → model
  finishes the phrase and yields ~3.8 s later, entirely its own
  decision (4 speak-chunks after overlap onset, then turn_eos).
- escalate (q0557 Lomax v. Ortiz-Marquez): onset score .469 ≥ .386 →
  fired; thinker ASR heard the question verbatim, answered in 14.8 s;
  talker meanwhile spoke its local hallucination ("John J. Carlin"),
  ended its turn; relay unit injected → talker voiced the correction
  "…was represented by Brian T. Burgess, not John J. Carlin" —
  in-band self-correction, unprompted.

**Open knobs:** (a) probe calibration on the native schema (feature
recipe transfers, thresholds don't necessarily); (b) stall steering
during the thinker wait — v1 lets the local attempt play out and the
relay corrects; (c) the deferred point-4 thinker abort. Old demo_app.py
live path is now legacy; keep for the paper's frozen sweeps
(_ws_duplex.py latency numbers were measured against it).

### 8bd addendum — stall steering in the native stream (2026-08-31)

User caught v1 letting the talker voice its (likely wrong) local answer
during the thinker wait — paper design is a stall. Three iterations:

1. Imperative note ("stop speaking and wait") → model stopped BUT
   swallowed the later relay too (kept the floor closed). 
2. Quoted-line note ("say: 'Hmm, let me double-check…'") → the line got
   voiced ONE UNIT LATE — at relay time instead of stall time. Text-unit
   imperatives are followed with a lag; the duplex head treats them more
   like context than commands.
3. **Landed: canned stall + factual note.** The STALL line is teacher-
   forced ONCE at load via the turn-based path (talker's own voice,
   same trick as the old demo's tts_filler), PLAYED at fire time, and
   the context gets a *factual* note ("You just told the user: …").
   End-to-end smoke: fire → stall audio in 0.2 s → turn closed in
   ~3.4 s → thinker 14.1 s → relay (with one nudge retry) voiced the
   verified answer at the next onset. RELAY_NUDGE retry stays in
   (first relay unit occasionally returns an empty turn).

Latency shape for the paper table: fire→stall-audio ≈ 0.2-1 s;
fire→relay-first-audio ≈ thinker wall + ~2.5 s.

**Ops lessons:** modal deploy does NOT drain warm containers (esc4
tested stale code — stop the container or wait out scaledown_window);
a util image missing modal_app.py crash-loops silently from the
module-level `from modal_app import OPENAI` (both "stuck build"
incidents were this).

## Phase 8be — native-regime recalibration: the deployed read point is nearly free ⭐ (2026-09-01)

**Setup:** modal_native_dump.py drives MiniCPMODuplex in-process (no
pacing, generate_audio=False — TTS never touches the LLM cache), one
fresh session per query, features captured at the SAME point the
deployed demo reads: after the generate of the first listen→speak
chunk. 2310 calib rows (§8bb mix, same escalate_label parquets) + 240
test. 0 no_speak across all 2550. ~20 s/query on H100.

**Probe: near-zero regime cost at the native read point.**
internal test-240, native features:
- conc-frozen (360) transferred:    .818 [.763,.869]
- conc-fullscale (2310) transferred: .807 [.751,.859]
- native in-regime refit (2310):    .830 [.777,.878]  (Δ vs 2310
  +.023 [-.012,+.059] n.s.;  turn-based reference .877)

Regime ordering is now turn-based .877 > native .830 > concurrent —
i.e. §8bb's "residual regime cost" was substantially an artifact of the
harness-interleaved concurrent read, NOT of going live. The duplex
head's own commit-to-speak moment is a naturally clean read point (the
model listens silently through the question, then the tail-8 features
cover its first ~8 answer tokens — a gen_at_eot-style read that the
head hands us for free). Robustness: the 0.7-trained probe scores .843
on temp-0.1 features (decoding-temperature invariant).

**Thresholds do NOT transfer** (the fragile part of calibration, again):
native score distribution shifts up — balanced .386 → .6445
(conservative .8311, aggressive .3800). gate_native.json deployed to
demo_duplex.

**Talker local floor drops in-regime:** same 240 ids local-correct
turn-based .483 → native default-temp .371; temp-0.1 control recovers
to .404. So ≈⅓ decoding config, ≈8 pts genuine regime cost (chunked
TTS-templated generation). Paper tables use the deployed config (0.7);
0.1 becomes the attribution footnote. Lower floor + intact AUC ⇒ the
gate's marginal value is LARGER in the deployed regime.

**Pending:** external 5-pool native AUC (the §8bb external gap is the
one number that could still bend the narrative), floor sweep full run
(8bf), validity remix tables.

## Phase 8bf — native floor control, first full pass (2026-09-01, analysis pending)

108 trials (modal_native_floor.py, in-process, gate_native thresholds;
esc carriers fire 60/60 under the new balanced .6445). Raw per-cell:

  ans:bcl 10/12 held   ans:bcs 11/12 held   (backchannel discrimination
  WITHOUT any ASR/lexicon — the harness needed hosted ASR for this)
  ans:bq 6/12 yielded<=6 (med 1 chunk among yielders)
  ans:stop 3/12 yielded<=6  ← native head is LESS sensitive to short
  burst commands than to sustained speech; the old energy-VAD harness
  cut faster on "Stop!". Honest trade: naturalness vs reliability.
  esc:wait bcs 10/10 held; esc:relay cells CONTAMINATED by a
  classification artifact — relay turns are naturally 1-3 chunks, so
  natural completion within the yield window reads as "yielded".
  relay_done .68 among fired is likewise conflated (legit bq barge-ins
  kill relay completion by design).

TODO (9-1): phase-aware re-analysis — no-stim turn-length baseline from
the test dump's n_ans_chunks; wait-phase metrics = "responds to new
question" + "relay still completes"; relay-phase yield vs natural-end
via length-matched control. No GPU rerun needed; all in the jsonl.

### 8be external pools — the regime ordering holds everywhere (2026-09-01)

Native features, three probes (external n=200-250/pool):

  pool        conc360→nat  conc2310→nat  native-refit   Δ(nat−2310)
  striviaqa      .658          .631         .711        +.080 [+.030,+.129] ⭐
  swebq          .720          .693         .736        +.043 [-.009,+.098]
  sllama         .745          .639         .757        +.118 [+.050,+.192] ⭐
  sdqa           .632          .706         .736        +.029 [-.027,+.086]
  sreason(zh)    .578          .491         .606        +.115 [+.051,+.180] ⭐

External mean (native in-regime): .709; En-4 mean .735. Reference
chain: turn-based .771 > native .709 > concurrent .689 — the same
ordering as internal, on every pool. In-regime refit beats transferred
conc probes on 5/5 pools (3/5 significant). sreason stays the weakest
(all-English calib — the known language axis, unchanged). The §8bb
scaling+regime narrative carries over verbatim with one upgrade: the
DEPLOYED regime sits closer to the turn-based ideal than the harness
concurrent regime ever did.

### 8be validity — native gated accuracy beats matched-random on 5/6 pools (2026-09-01)

8ad remix on the native dumps: native onset scores select, native local
outcomes (deployed decoding) + cached always-arm expert outcomes mix.
figures/native_validity.json. Highlights (balanced / aggressive):

  frozen     .371→.492 (esc 24%, p=.001) / .596 (44%, p<.0001)  ceiling .667
  striviaqa  .612→.756 (26%, p<.0001)   / .872 (64%, p<.0001)  ceiling .916
  swebq      .392→.556 (31%, p=.024)    / .740 (73%, p=.008)   ceiling .816
  sllama     .760→.780 (4%, p=.031)     / .828 (24%, p=.0035)  ceiling .912
  sdqa       .440→.660 (30%, p<.0001)   / .830 (70%, p<.0001)  ceiling .895
  sreason    gate never fires (zh scores sit under the global calib
             thresholds — the language axis, now expressed as a
             fire-rate failure instead of an AUC failure). Honest
             negative, consistent with 8bb.

External fire rates drift from nominal (aggressive 64-73% vs 50%
calib) — thresholds remain the fragile part of calibration, third
regime in a row. All ingredients were cached or judged tonight; zero
live escalation spend for these tables.

### 8bf phase-aware re-analysis + latency (2026-09-01)

Natural answer length (no-stim baseline, test dump): med 8 chunks
[p25 5, p75 21]. Clean per-phase story:

- **Answer speech**: backchannel false-stop 3/24 (12.5%) within the
  6-chunk window (9/12 per kind kept talking past the whole window);
  bq (sustained question) yields ≤6 in 6/12, med 8.5 post-stim chunks;
  "Stop!" bursts only 3/12 — the native head reads sustained speech,
  not short commands. bc-vs-interrupt post-stim length p=.088 (n=48).
- **Thinker wait**: the model RESPONDS to any stim (speaks within 8
  chunks 9-10/10 all kinds). Relay survival splits cleanly: bcs 8/10
  vs stop 4/10, bq 3/10 — a NEW QUESTION during the wait derails the
  pending relay 70% of the time. This is the strongest empirical
  motivation yet for the deferred thinker-abort (memory:
  duplex-demo-thinker-abort): the head already moves on; the machinery
  should too.
- **Relay speech**: robust — relay_done 26/30 under overlap (the
  earlier "2/10 held" was the length artifact; relay turns are
  naturally 1-3 chunks).

**Latency (figures/native_latency.json):** 87% of queries get the
commit-to-speak within ±1 chunk (±1 s) of question end (med gap -1);
listen chunks cost .04 s, speak chunks ~.5 s (realtime holds);
fire→stall-audio 0.2-1 s; fire→relay-first-audio = thinker wall
(13-20 s observed) + ~2.5 s.

### 8be valpaca — the seventh pool, native (2026-09-01)

Native dump 199/199 + official VB 1-5 judge (gpt-4o-mini, verbatim
prompt): native local VB 3.81, always-arm expert 4.76. Gate fire is
suppressed (2/4/24% across tiers vs nominal 15/30/50 — same axis as
sreason, open-ended scores sit low) but the aggressive tier shows a
WEAK-significant win: VB 4.116 vs matched-random 4.033, p=.0185.
Turn-based valpaca was a clean honest negative; native is "fire-rate
suppressed with a weak aggressive-tier signal". Reported as-is —
open-ended generation remains the method's boundary, the boundary just
moved a little.

## Phase 8bg — live native escalation: remix validated; the bottleneck moved to the relay channel (2026-09-01)

Full live runs (modal_native_live.py, frozen test 240 x balanced +
aggressive, real gpt-transcribe uplink of raw audio, real gpt-5.5 web,
wait paced 1 chunk/s):

- **balanced: delivered .504 @ 23% fire — remix predicted .492 @ 24%.
  The offline remix arithmetic is validated end-to-end (+.012).**
- aggressive: delivered .537 @ 45% vs remix .596 — the 6-point gap
  decomposes entirely onto fired turns: live fired-acc .495 (agg) /
  .589 (bal) vs the .667 cached-expert ceiling. The native RELAY
  channel (text-unit steering) is the new lossy element — 89% of
  relays needed the nudge retry, and the voiced relay drops/garbles
  the expert content on a fraction of turns. Turn-based's bottleneck
  was the transcription uplink; native's is the relay. (Uplink itself
  is now clean: raw-audio gpt-transcribe, p50 0.9 s.)
- Latency (real, expert-cache warm): ASR p50 0.8-0.9 s, expert p50
  4.4-5.7 s, wait p50 5-6 chunks (p90 10-15). fire->relay-first-audio
  p50 ~7-8 s.
- Selection visibly works live: unfired subsets score .478/.573 vs
  the .371 unconditional local floor.

Deliverable framing: remix tables are the headline (validated at
balanced), live aggressive documents the relay-channel cost as the
native analogue of the uplink cost — same "channel, not selection"
decomposition as the turn-based story.

## Phase 8bh — dialogue-act gate: stop words must not escalate (2026-09-01)

**User-caught bug:** floor-management utterances ("stop", "停",
"别说了", backchannels) hit the same listen→speak commit as questions;
the failure probe is OOD there and the gate escalated "stop" to
gpt-5.5. Quantified on 194 TTS'd floor stims (stop/backchannel/ack/
filler, en+zh, 2 voices, standalone-from-silence): 0% false-fire at
balanced/conservative but **37% at aggressive (stop commands 45%)**;
live mid-conversation contexts are worse (the user hit it at
balanced — user_mean then carries prior-turn audio).

**Fix (deployed):** a second linear head on the SAME L22 read —
info-seeking vs floor-management. Training: 2310 native question
features (pos) vs the 194 stims (neg). **OOF AUC 1.0000** — the act
distinction is perfectly linearly separable at L22. Threshold at the
question-side 0.5th percentile: loses 0.52% of true escalations,
passes 0% of floor stims. Escalation condition is now
act≥thr ∧ P(fail)≥tier. The validated 8be failure-probe calibration
is untouched by design.

**Live smoke:** 6 stop/backchannel stims through the deployed demo —
act .0006-.061 → "floor turn — gate bypassed", zero escalations, model
answers floor turns naturally ("Okay."); several backchannels caused
no commit at all (no gate read — the ideal case).

Files: modal_flooract.py (stim inventory + TTS), scripts/24_act_probe.py,
data/gate_act.json, figures/act_probe.json; demo_duplex.py wires the
act term and reports P(info) in the gate event. Caveat: negatives are
standalone stims; mid-conversation floor acts are covered by the live
smoke but not yet by a systematic sweep.

## Phase 8bi — context-carrying uplink: multi-turn escalation (2026-09-01)

**User-caught bug:** "what is NVDA stock" then "what about Apple" — only
"what about Apple" is ASR'd and sent to the expert, which cannot
resolve the reference. Root cause: the PROBE reads L22 with full
context (prior turns live in the model's KV cache), but the expert
UPLINK was stateless — snapshot audio of the current turn only, and
demo_duplex reset the window every end_of_turn.

**Fix (deployed):** a rolling text dialogue `history`. Escalated turns
record `User: <asr>` + `Assistant: <expert answer>` (inside thinker,
using the resolved question); local turns record the talker's own
answer (topic carrier). At fire, the expert input becomes
"Conversation so far:\n<history[-6:]>\n\nThe user now asks (resolve
references…): <asr>". The audio uplink stays per-turn clean (window
still wiped) — context lives in resolved text, not re-ASR'd old audio.
History capped at 8 lines.

**Live smoke (aggressive tier, two real TTS turns):**
- T1 "What is Nvidia's stock price today?" → relay "Nvidia is trading
  at $220.78 today, up about 1.5%."
- T2 "What about Apple?" → relay "Apple's currently at around $193.46
  per share, down a bit from yesterday."
T2 resolved to Apple's STOCK PRICE — only possible from the threaded
history. Bug fixed. Files: demo_duplex.py, _ws_context_smoke.py.

Paper note: the measured eval pools are single-turn, so no table
changes; this is a deployment/demo capability. Recorded in
Limitations as multi-turn grounding via a stateful uplink.

## Phase 8bj — live hardening round: three regressions the user caught in ten minutes of use (2026-09-01)

Live use immediately surfaced what scripted pools cannot. Three fixes,
one theme: the fitted components' calibration distributions missed
corners of the deployment distribution.

**(a) Act threshold broke on live speech.** A real request-phrased
question scored P(info)=.9267 — below the q0.5-percentile threshold
(.9565) — was ruled a floor turn, gate bypassed, and the talker
delivered an empty promise ("Sure, I'll check that for you right
away." + turn_eos). Live mic speech shifts BOTH distributions into the
calibration gap (real questions ~.93↓, live floor turns ~.37↑ vs
calib floor max .052). Fix: gap-center threshold + two data
extensions — request-phrased positives (40 TTS stims, en+zh) and
IN-CONTEXT negatives/positives (new dump mode: each stim arrives as
the SECOND turn after a carrier Q&A, mirroring how "thanks" actually
shows up mid-conversation; the standalone-calibrated "Thanks, that's
all I needed" had escalated when it arrived in-context). Joint refit:
OOF AUC still 1.000, gap [.154,.871], thr=.5124; in-context stop
commands false-fire the FAILURE probe at .57 aggressive — the act
gate is not optional. Ops: git-bash MSYS path conversion silently
rewrote --carrier /data/... to C:/Program Files/Git/data/... —
MSYS_NO_PATHCONV=1 for modal args with absolute paths.

**(b) FreshQA awareness had regressed.** "Could you tell me the stock
price of nvidia today" scored P(fail)=.2888 (< .38) → stayed local →
empty promise again. The 8be native refit reused the v3-era 2310
labels; v4's FreshQA real-time extension never carried over. Fix:
scripts/25 = refit4 transplanted (243 fresh train rows, a-priori
labels; budgets quantiled on the core mix only). Guards: internal
test AUC .830→.833 (free); fresh-heldout fast fire .18→.45 (cons),
.52→.75 (bal), .98→1.00 (agg); never-controls 0/.06. Deployed; the
user's exact sentence now fires and relays "Nvidia is trading $220.78
USD today, up 1.52%. The intraday range … $216.33–$221.25."

**(c) Relay re-entrancy.** The relay's own speak-onset passed through
the gate branch and re-fired (empty uplink, spurious second stall
mid-relay). Fix: relay_guard — no gate fire from relay injection
until that delivery's end_of_turn.

Escalation condition now: P(info)≥.5124 ∧ P(fail)≥tier ∧ ¬thinking ∧
¬relay_guard. Meta-lesson for the paper: every regime OR DOMAIN
change re-opens calibration — the L22 signals separate perfectly each
time (act AUC 1.000 twice), but thresholds and coverage must follow
deployment. Ten minutes of live use found what 3,000 scripted rows
did not; interactive evaluation is not optional (the venue's point).

## Phase 8bk — the head has no mid-turn yield channel (2026-09-01) ⭐ mechanism

**User observation:** said "stop" at count three; the model counted to
ten. Interruptions during a committed answer only take effect "at the
last micro-turn".

**Hypothesis tested (and refuted):** the serving wrapper force-rewrites
a mid-turn sampled <|listen|> to tts_bos ("not allowed to listen"), so
maybe the head WAS reacting and the wrapper was eating the signal.
Instrumented the rewrite (modal_native_listen.py, 36 trials, TTS-on
pacing, "count slowly to thirty" carrier, stop/sustained stim at 2
chunks after onset, plus a yield mode honoring >=2 attempts as
turn_eos): **mid-turn listen attempts = 0 in every chunk of every
trial**. The rewrite is dead code; the head never samples listen
mid-turn at all.

**The real mechanism:** the head's only native stop is <|turn_eos|>,
trained to fire at answer completion points. Interruption handling is
therefore TURN-granular, not token-granular — post-stop latency is
bounded by remaining answer length (stop-stim post median ~10 chunks,
range 1-14; sustained-stim similar; and 1/12 unstimmed trials
spontaneously quit after "one", so mid-answer turn_eos is
high-variance, not a controllable channel). 8bf's "stop yields 3/12"
now has its explanation: not insensitivity — a missing channel.

Paper placement: floor-control paragraph of app:native — first
mechanism-level evidence that current full-duplex ALM interruption
operates at turn granularity. Product mitigations (client-side
playback cut, prompt steering) are harness-tier and deliberately not
deployed; the demo stays honestly native. Instrumentation
(listen_attempts telemetry + dormant allow_midturn_yield) ships in
_model_src, behavior-neutral by default.

## Phase 8bl — the user was right: serving config was masking native interruptibility ⭐⭐ (2026-09-01)

**User challenge:** "give me a stock MiniCPM demo with none of our
mechanisms — I suspect your probe-off ≠ vanilla." Built demo_vanilla.py
(bare prefill/generate loop, zero hooks, cfg=official|ours toggle)
after diffing the official pytorch-simple-demo serving stack:
top_k 20 vs our 100 (as_duplex class default), force_listen_count 3
vs 0, assistant-style system prompt vs "Streaming Omni Conversation.".
Duplex DECODE code is byte-identical (incl. the mid-turn listen
suppression, official l.3100); official's length_penalty knob is
turn-based-path only.

**Scripted A/B (count-to-thirty + "stop" 2 s into the answer, n=3):**
  official config: post-stop 2.2 / 2.0 / 2.0 s — clean mid-enumeration
                   yields, three for three
  our config:      12.9 / 4.0 / 1.0 s — one full count to thirty
**The head CAN stop mid-answer ~2 s after a stop command.** top_k=100
dilutes sampling enough that the rising turn_eos rarely gets picked;
8bf's "stop 3/12" and 8bk's "turn-granular, latency bounded by answer
length" were measured under our mis-configured serving and OVERSTATE
the limitation. 8bk's mechanism half stands (listen is never sampled
mid-turn; turn_eos is the only stop channel) — but that channel is
RESPONSIVE under the official config.

**Actions:** demo_duplex aligned to official (top_k 20, force_listen 3,
assistant prompt) and redeployed; config change = another probe-regime
micro-shift, so test-240 re-dumped under the aligned config (probe
kept if AUC holds, else recalib); floor ans-arm rerunning under
aligned config for corrected 8bf numbers; 8bk paper paragraph to be
rewritten. Control demo stays deployed for side-by-side:
rhe9527--vanilla-duplex.modal.run. Meta: the vanilla control arm
should have existed from day one — config parity with the reference
serving stack is part of "native".

### 8bl continued — corrected floor numbers, probe verdict, threshold interim (2026-09-01)

**Corrected floor-control (real ws loop, official config, n=6/stim):**
  stop:        med 2.1 s post-stim (4/6 yield ≤2.6 s; 2/6 complete)
  backchannel: med 11.7 s = natural completion; min 6.8 s; 0/6 false stop
  question:    med 5.4 s (1.7–21.8)
Native discrimination AND native responsiveness both present, zero
harness. Replaces 8bf's stop numbers (measured under wrong config AND
a low-fidelity in-process harness — floor2 rerun under official config
still failed to yield, so the in-process no-TTS/GSM-carrier harness is
NOT deployment-faithful for stop-latency claims; ws-loop measurements
are canonical from here).

**Probe under official config: weights survive, thresholds don't (4th
time).** Old-config-trained probe on official-config test-240 features:
AUC .846 (was .830 — better). Fire rates collapsed (balanced 11% vs
nominal 30) — score distribution shifted down. Also: official config
IMPROVES the local floor .371→.429 (top_k 20 helps answers, not just
stops). Interim: thresholds re-quantiled on official-cfg test scores
(cons .559 / bal .399 / agg .222), deployed. Proper: full calib mix
(2310+fresh) re-dumping under official config overnight; refit check +
final quantiles tomorrow; validity tables to be refreshed with the
.429 floor and official-config scores.
> [merge note 2026-09-01: two parallel workstreams both used the 8bk tag — above = mid-turn listen channel (agent), below = oracle-headroom accounting (collaborator). Kept both; renumber at the next log pass.]

## Phase 8bk' — oracle-headroom accounting + the margin worklist ($0 local, 2026-09-01)

User challenge on the 8be validity gallery: "gated barely clears the
random-escalation line on many pools — the probe is undertrained."
Quantified instead of argued: at each tier's realized rate r the
oracle selector's margin over matched-random is joint-free bounded —
escalate the (local-wrong, expert-right) items first, so
margin_oracle(r) = min(r, p_benefit) − r·(ceiling − floor), with
p_benefit ≥ ceiling − floor guaranteed. At the balanced tier every
pool's rate sits under that bound, so the oracle margin is EXACT:

| pool | r | gate−rand | oracle−rand | captured |
|---|---:|---:|---:|---:|
| frozen | .237 | +.050 | .167 | 30% |
| striviaqa | .264 | +.063 | .184 | 34% |
| swebq | .308 | +.033 | .177 | 19% |
| sllama | .044 | +.013 | .037 | 35% |
| sdqa | .300 | +.084 | .163 | 51% |
| sreason | — (never fires) | | | |

**Diagnosis: the gap is real but "train more" explains little of it.**
(a) Internal is scale-saturated (.818 flat 360→2310, 8bb) yet captures
only 30% — the visual closeness is the AUC≈.83 signature, not an
undertrained one. (b) The native external curve IS still rising
(.643→.709, ~+.02/doubling, native_refit.json) but extrapolates to the
same-recipe turn-based ceiling ≈.771, nowhere near oracle. (c) The two
biggest visible offenders are operating-point failures, not probe
failures: sreason fires 0% (zh scores under the en-calib thresholds —
the language axis) and conservative-tier margins are mathematically
invisible (oracle ≤.04 at r≤.07). 8bc already ruled out the label as
a lever.

**Landed (local, $0):**
- `figures/native_gallery.py`: validity small-multiples now draw the
  oracle band + per-tier captured-% annotations (floor/ceiling only,
  no new data deps); `figures/native_validity.png` regenerated. Also
  guards the floor-control fig when floor.jsonl shards are absent.
- `scripts/26_pool_thresholds.py` (CPU, needs volume feats): label-free
  per-pool quantile thresholds + a WINDOWED online-quantile tracker
  (the deployable story — no labels, no pool identity), recomputes the
  scripts/23 validity table under both → gate_native_pooled.json +
  native_validity_pooled.json. Expected: sreason fires at nominal,
  aggressive drift (.64–.73 vs .50) gone.
- `scripts/27_probe_receipt_native.py` (CPU): the 8j/8k receipt
  regenerated for the CURRENT deployed probe — the 8bj v2 recipe
  (2310 core + fresh train rows, budgets on core-mix quantiles) —
  plus external test-only rows → probe_receipt_native.json. The
  shipped probe_receipt.json was four refits stale.
- `modal_train3.py` + FEAT_POOLS/refit wiring: expansion3 (~2300 en,
  same 7 families, seed 45, deduped incl. expansion2 — the measured
  scaling lever, predicted +.02–.03 external) and expansion3zh (~355
  OpenAudioBench reasoning_qa rows the eval pool did NOT sample,
  stem-disjoint, official audio — the zh axis). Method note in the
  docstring: mixing exp3zh in makes sreason in-domain; report it as
  such or keep it out of the external-transfer probe. Est. ~$80–100
  all-in; NOT run this session (no Modal creds in the remote env).

## Phase 8bl — probe ⊕ p(True) fusion, internal half: the 5b signal is additive ($0 local, 2026-09-01)

User push: "I want probe accuracy UP — what have you actually run?"
First actually-executed experiment of the margin worklist: the
representation-layer lever (8bk item 3), run on in-repo data only
(frozen_conc_{calib,test} feats + ptrue shards + v3 labels), no GPU.
`scripts/28_ptrue_fusion.py` → `figures/ptrue_fusion.json`.

**Sanity anchor reproduces exactly**: shipped conc-360 probe on local
test-240 feats = AUC .818 (8bb's number to the third digit). 5b's calib
AUCs also reproduce (.807 pre / .899 post).

**AUC (internal test-240, logit-stacker trained on calib OOF):**

| signal | calib | test | Δ vs probe [95% CI] |
|---|---:|---:|---|
| probe (12,288-d L22) | .775 OOF | .818 | — |
| ptrue_pre SOLO (1 scalar) | .807 | .805 | — |
| ptrue_post solo | .899 | .760 | (mis-calibrates calib→test) |
| **probe ⊕ pre** | .824 | **.845** | +.027 [−.011, +.065] n.s. |
| probe ⊕ post | .896 | .794 | −.024 (post drags) |

**Margin translation (remix vs matched-random, calib-quantile thr):**
balanced probe +.073 → fusion **+.095** (~+30% relative), aggressive
+.096 → +.102, conservative +.048 → +.053. All perm p ≤ .0005.

Readings. (1) One pre-answer scalar ≈ the whole 12,288-d probe — and
they are partially COMPLEMENTARY, not redundant: the stack gains on
both AUC and margin. (2) n=240 cannot make +.027 significant; the
powered test is EXTERNAL, which is also where the complementarity
should peak (probe external mean .709; 5b: ptrue transfers per-pool
with no inversion, hard-math pre .809 where probe LOPO sat at .377,
trap pool pre .945 vs probe .328). (3) ptrue_post is out: it
mis-calibrates calib→test here, consistent with 5b's trap degradation.

**Deployability caveat (the reason the gate went probe-only, on
record):** ptrue collapses under AUDIO input on the deployed backbone
(app:audioptrue — trap p_yes .055→.556, "trap dead"), and the same
appendix records the fix, confirmed twice: repeat-then-judge on the
model's OWN transcript restores introspection. So the deployed shape
is: at onset, L22 probe (free) ⊕ one short text prefill "Would you
answer this correctly? <own transcript>" → P(Yes) → stacker. Cost per
turn: one short prefill + 1-token decode (5b: "fits the streaming
design").

**GPU half (blocked on Modal creds in this env):** (a) collect
repeat-then-judge ptrue_pre on the audio pools (calib/test + the five
external, ~$2-3 by 5b costing); (b) external fusion eval — THE
decisive number; (c) if it holds, wire the stacker into demo_duplex.

## Phase 8bm — double-talking fix: mute the condemned turn (2026-09-01)

**User report:** after fire, the talker kept voicing its own (wrong,
$530) answer while the thinker worked — stall note obeyed "loosely"
again (the head replied "'ll do better." to the note, then finished
the hallucination; third confirmed failure of in-band imperative
steering, consistent with 8bk's turn-granular behavior channel).

**Fix (deployed + verified):** output-side mute of exactly the
condemned turn. After fire, the turn the gate judged likely-wrong
keeps GENERATING (context true, perception intact, natively
interruptible — user audio still prefills, barge-in unchanged) but its
audio/text are not emitted. Mute lifts at that turn's natural
turn_eos or at relay injection; unvoiced text is excluded from the
dialogue history (history records only what the user heard). Verified:
fire → 1-chunk onset fragment → stall → "muted 8 chunks" → relay
$220.78 as the sole audible answer; follow-up "Thanks" stays a floor
turn. Classification: an output-channel edit (same class as the canned
stall) — no perception/generation/interruption capability is
constrained; probe-off leaves the path nonexistent.

## Phase 8bn — the margin worklist runs on Modal: RTJ fusion external half + thresholds + receipt (~$10 GPU + ~$2 API, 2026-09-01)

User provided Modal creds ("用我的modal来跑") — every stage that was
blocked in 8bk'/8bl executed this session.

### Per-pool thresholds + receipt (the 8bk' CPU stages, volume feats
were already local)

- `scripts/26`: label-free per-pool quantiles fix both recorded
  operating-point failures — sreason fires .14/.30/.52 at the three
  tiers (was 0/0/.03 under global thresholds; zh scores just sit lower)
  and the aggressive-tier global drift (.65–.90 realized vs .50
  nominal) disappears. The WINDOWED online tracker (no labels, no pool
  identity, window=100) converges to nominal on every pool —
  `gate_native_pooled.json` + `native_validity_pooled.json`.
- `scripts/27`: the deployed-probe receipt regenerated
  (`probe_receipt_native.json`): recipe n=2553, OOF .855, internal
  test .833, external .601 (sreason) – .766 (sllama) — matches 8be to
  the third digit; the receipt is no longer four refits stale.

### Repeat-then-judge collection (modal_rtj.py, 5 external pools,
2×H100 each; internal 600 reused from 6a's asr shards)

wav → verbatim transcript (ASR_INSTR) → TEXT ptrue_pre on the
transcript (+ ptrue_pre on the original query text as a diagnostic
arm). 1,152 external rows → `rtj_{pool}.shard*.parquet`.

### ⭐ Fusion, deployed shape, native regime (scripts/29 →
figures/ptrue_fusion_native.json)

Probe branch = deployed v2 recipe verbatim (calib+exp+exp2+fresh-train,
C=3e-4); stacker = LR on the 360 calib rows [logit(probe_oof),
−logit(p_yes_rtj)], coefs [.86, .30].

| pool | n | probe | rtj solo | fusion | Δ [95% CI] |
|---|---:|---:|---:|---:|---|
| frozen test | 240 | .833 | .836 | **.863** | +.030 [+.008,+.052] ⭐ |
| striviaqa | 250 | .787 | .761 | **.823** | +.035 [+.002,+.071] ⭐ |
| swebq | 250 | .736 | .723 | .765 | +.029 [−.005,+.063] |
| sllama | 250 | .667 | .674 | .683 | +.016 [−.014,+.045] |
| sdqa | 200 | .788 | .742 | .799 | +.010 [−.018,+.038] |
| sreason (zh) | 202 | .615 | .586 | .603 | −.012 [−.065,+.040] |

Margins (per-pool quantile thresholds, remix vs matched-random):
striviaqa balanced +.069→+.085, aggressive +.080→+.096 (both p=0);
swebq/sllama ~flat; sdqa slightly down; sreason conservative flips
negative (−.012) — fusion should NOT be enabled for zh.

**Readings.**
1. The native + repeat-then-judge shape is STRONGER than 8bl's
   conc/text version: internal Δ+.030 with a CI that excludes zero at
   n=240 (8bl's +.027 spanned zero). First significant AUC lift over
   the deployed probe since 8bb.
2. External: positive on 4/5 pools, significant on striviaqa. Mean
   external Δ ≈ +.016.
3. **The 8bl prediction "complementarity peaks where the probe is
   weakest" is FALSIFIED.** Gains track where the RTJ signal itself is
   strong (test .836, striviaqa .761), not where the probe is weak —
   on sllama/sreason RTJ is as weak as the probe (.674/.586) and adds
   ~nothing. The txtq diagnostic says the transcript is not the loss:
   original-text ptrue ≈ RTJ ptrue on every pool (e.g. swebq .725 vs
   .723). 5b's optimistic per-pool transfer numbers (hard-math .809)
   came from pools where self-eval is genuinely informative; these
   external pools mostly aren't.
4. The zh axis stays untouched by fusion (self-eval carries no signal
   there) — that remains expansion3zh's job.

**Deployment disposition:** wiring the stacker into demo_duplex is
justified on the en side (one short prefill + 1-token decode per turn
buys +.03 internal / +.035 best-external), but HOLD until the exp3
refit lands — re-run scripts/29 with the exp3-trained probe and wire
only if fusion still adds on top.

### expansion3zh v2 — the original plan was impossible

Diagnostic rerun measured: OpenAudioBench reasoning_qa has exactly 202
rows and the sreason eval pool sampled ALL of them — the "~355 unused
rows" of the 8bk' plan do not exist. Rebuilt as MGSM zh (250,
human-translated GSM8K) + XCOPA zh (150/500, causal MC), public
benchmarks through the same tts-1/alloy pipeline as the en families
(400 rows, `queries_expansion3zh.jsonl`). Source-disjoint from sreason,
which therefore STAYS a fully external transfer pool — strictly cleaner
than the original in-domain plan. bugfixes landed: librosa image dep,
bare `xcopa` repo id → `cambridgeltl/xcopa`.

### expansion3 pipeline (in flight this session)

build ✅ 2300 en (7 families, seed 45, deduped) → TTS ✅ 2300+400 wavs
→ answers/judge/native-dump/refit: running; results recorded in the
next section when the refit lands.

### 8bl final — official-config recalibration lands (2026-09-01)

Full calib mix re-collected under the official serving config (2300
core + 242 fresh usable of 2550/303). scripts/26: train OOF .864
(was .845); testoff AUC .846; **official thresholds land back at
.830/.646/.386 — in-regime training renormalizes the score
distribution to almost exactly the original quantile positions**
(the interim test-based thresholds were a transient). Fresh-heldout
fast fire .80 balanced / .95 aggressive, never-controls .06/.50.
Deployed.

**Frozen validity under the official config (n=239): the strongest
validity table of the project.** Local floor .431, ceiling .669:
  conservative .485 @ 10% (rand .456, p=.008)
  balanced     .544 @ 22% (rand .483, p<.0001)
  aggressive   .661 @ 46% (rand .540, p<.0001)
At 46% escalation the gate reaches within .008 of the always-escalate
ceiling — it captures essentially ALL achievable gain at half the
expert cost. The official config improved every layer at once: local
floor (+6 pts), stop responsiveness (2.1 s), AND gate margin over
random (aggressive +12.1 pts vs +9.6 under the old config). External
pools re-dumping under official config for the full refreshed table.

### 8bl external AUC under official config (2026-09-02)

Official probe on official-config external features (labels: conclive
never-arm, as before):
  TriviaQA .759 (was .711)   WebQ .754 (.736)   Llama-QA .789 (.757)
  SD-QA .737 (.736)          Reasoning-zh .495 (.606 — COLLAPSED)
**En-4 mean .760 — within .011 of turn-based (.771). The English
regime cost is essentially eliminated** by official-config serving +
in-regime calibration. The language axis got WORSE: the all-English
assistant prompt + all-English calib pushes zh to chance. The
collaborator's exp3zh track is exactly the medicine; until it lands,
Reasoning-zh is reported as a config-sensitive honest negative.

### 8bl closing — official validity full table (verified) + an artifact-collision note (2026-09-02)

Official probe × official features, all seven pools
(figures/native_validity_official.json; paper tables updated):
  frozen    .431→.544@22%/.661@46% (both p<.0001; ceiling .669)
  TriviaQA  .568→.740@26%/.848@60% (p<.0001)
  WebQ      .411→.556@29% (p=.056)/.705@61% (p=.0045)
  Llama-QA  .767→.808@7%/.865@26% (p≤.0005)
  SD-QA     .455→.660@33%/.820@69% (p<.0001)
  sreason   no fire (language axis); AlpacaEval clean negative
  (agg 4.07 vs 4.04, p=.24 — the old-config weak signal did not
  replicate; VB local 3.65).
External AUC re-verified: En-4 .760 (turn-based .771), zh .495.

**Collision note:** data/gate_native.json was overwritten locally by a
collaborator scripts/25 run (train_n 5252 = +exp3/exp3zh, OLD-config
features) while the volume carried the scripts/26 official artifact —
one scripts/23 pass silently mixed the two (numbers discarded,
recomputed). Their artifact preserved as data/gate_native_v2exp3.json.
TODO for the exp3zh merge: re-dump exp3/exp3zh under the OFFICIAL
config and refit via scripts/26, else the zh fix trains on the wrong
serving distribution. Coordination needed on artifact ownership.

## Phase 8bo — expansion3 lands (default config): the scaling curve pays out on schedule (~$75 GPU + ~$8 API, 2026-09-01)

The 8bk' data lever, executed end-to-end this session: 2300 en (7
families, seed 45, deduped) + 400 zh v2 (MGSM 250 + XCOPA 150) → TTS →
turn-based answers → judge (en fail mix: longtail .838, multihop .762,
trap .567, mmlu .295, commonsense .273, openbook .175, easymath .105;
zh: mathword .284, causal .120) → native dump under the DEFAULT serving
config (en 2300 traces / 1 no_speak; zh 400 / 0).

**scripts/22 refit, 5009 rows (C=3e-4):** internal .834 — flat, as the
8bb saturation predicted. External mean **.709 → .736** — the measured
+.02-per-doubling curve continues exactly on schedule
(scaling_curve extended in figures/native_refit.json; scripts/22 now
carries the curve forward instead of dropping it). Per-pool Δ vs
conc-2310: sllama +.155 [+.085,+.228], **sreason +.169 [+.085,+.251] —
the zh axis moves for the first time** (Reasoning-zh .606 → .659 vs the
old native probe; 400 zh calib rows did what 4,600 en rows never
touched).

**scripts/25 (v2 recipe + exp3):** guards pass — internal .836, fresh
fast-fire .36/.73/1.00 (stronger than 8be's .23/.57/.98), never-heldout
.00/.00/.56. The 5,252-row artifact is preserved as
`gate_native_v2exp3.json` (see collision note below).

**scripts/26/27/29 rerun on the exp3 probe:** receipt external mean
.709 → .733 across the board (n=5252 recipe); en global-threshold
drift shrinks (aggressive realized .30–.71 vs .50–.90 before); sreason
still fires 0% at global thresholds — per-pool/windowed quantiles
remain the deployable fix (fires at nominal). Fusion still adds on top
of the stronger probe: internal .836 → .862 (+.026 [+.004,+.048], still
significant), striviaqa +.030 [+.000,+.062], swebq/sllama/sdqa positive
n.s., sreason −.021. **Wire-into-demo criterion: HALF-met** —
significant internally + 1/5 external, margin translation flat outside
striviaqa, zh negative. Recommendation: stacker behind a default-off
flag; decision deferred to the user (touches live latency).

**Track collision, resolved in-flight:** this ran concurrently with the
8bl official-config track, which deployed the 2542-row official gate
and re-dumped externals under official serving (en-4 .760, zh .495 —
"exp3zh is exactly the medicine"). My scripts/22+25 briefly overwrote
data/gate_native.json; the other session caught it, restored the
official gate, and preserved this track's probe as
gate_native_v2exp3.json. Convergence step ALREADY IN FLIGHT: exp3 +
exp3zh re-dumped under official config (tags exp3off / exp3zhoff,
8×H100 + 2×H100) → official refit with the 5,409-row core → the merged
probe (official serving ⊕ expanded en ⊕ zh axis) becomes the deployment
candidate.

## Phase 8bp — the tracks merge: official config + exp3 + zh axis = the deployment candidate (~$60 GPU, 2026-09-01)

Convergence step promised in 8bo, executed: exp3 (2300) + exp3zh (400)
re-dumped under the OFFICIAL serving config (exp3off: 2300 traces / 14
no_speak — force_listen's expected tax; exp3zhoff: 400 / 0), then
scripts/26_official_refit.py extended with the optional exp3off/
exp3zhoff parts and rerun: **5228 rows** (caliboff 358 + expoff 796 +
exp2off 1146 + exp3off 2286 + exp3zhoff 400 + fresh 242), C=3e-4,
OOF .854. Guards hold: testoff .844 (vs .846), fresh fast-fire
.39/.80/.95, never .00/.06/.50, thresholds ~unmoved
(.860/.674/.374). Deployed as data/gate_native.json.

**External AUCs, official features (conclive never labels):**

| pool | official-2542 | merged-5228 |
|---|---:|---:|
| striviaqa | .759 | **.767** |
| swebq | .754 | **.772** |
| sllama | .789 | **.815** |
| sdqa | .737 | .728 |
| sreason (zh) | **.495 (chance)** | **.605 (restored)** |

**En-4 mean .760 → .770 vs turn-based .771 — the English regime cost
is now fully eliminated. The zh medicine worked**: 400 MGSM+XCOPA rows
lift Reasoning-zh from the all-English chance collapse back to .605
(the pre-official level), while staying source-disjoint from the eval
pool.

**Official validity under the merged gate (scripts/23 off): all five
en pools significant at BOTH deployed tiers** — frozen bal .556@23% /
agg .657@48% (ceiling .669 — 98% of the achievable margin at half the
expert cost), striviaqa .704/.852, swebq .560/.734, sllama .800/.865,
sdqa .620/.810. sreason separates on AUC but global en thresholds
still barely fire (1%/13%) — the per-pool/windowed quantile remedy
(8bn) is the deployable fix; the windowed tracker is config-agnostic
(open item: demo does not yet consume it).

**Housekeeping:** the two paper tables the parallel session added
(tab:native-auc / tab:native-validity) had shell-escape corruption
(\b/\t control chars from a heredoc bug — the same class of bug hit
this session and was dodged by writing scripts to files); both
repaired, updated to the merged-gate numbers, and the stale
default-config validity prose refreshed. Lesson for both sessions: do
NOT pass backslash-bearing text through bash heredocs on this setup.

### 8bp addendum — official per-pool thresholds + PDF verify ($0, 2026-09-01)

scripts/26_pool_thresholds.py grew the same SFX arg as scripts/23; run
"off" against the merged 5228 gate on official features →
gate_native_pooled_official.json + native_validity_pooled_official.json.
sreason: global fire 0/.01/.13 → per-pool static exactly nominal
(.15/.30/.50), windowed .19/.27/.62 — the deployed gate's last
operating-point gap now has concrete numbers on the deployed config.
En pools' windowed tracker converges within ±.06 of nominal everywhere.
Paper rebuilt on Modal after the table repair: clean compile, 38 pages,
main.pdf refreshed.

## Phase 8bn — post-interrupt deafness: a silent loop death (2026-09-02)

**User report:** occasionally, after barging in (model yields), all
subsequent speech gets no response for the rest of the session.

**Diagnosis:** scripted repro (interrupt mid-answer → settle → new
question) passed 8/8 on demo AND vanilla — not a behavior, an
intermittent fault. Code audit found it: at fire,
np.concatenate(user_win) raises on an empty uplink window (fire
landing in the same-second window after end_of_turn clears user_win —
exactly the compressed timing a barge-in produces), and the exception
exited the whole chunk_loop: session stays connected, model deaf
forever. Intermittent ✓ post-interrupt ✓ total deafness ✓.

**Fix (deployed + regressed):** (1) no fire without uplink audio;
(2) concatenate guard; (3) per-iteration try/except in chunk_loop —
any future single-iteration fault logs "loop error (recovered)" and
the session continues; the whole "one exception kills the session"
failure class is closed. Regression: resume 3/3, escalation chain +
multi-turn resolution intact.

Residual (logged, not fixed): at balanced tier a follow-up question's
P(fail) sometimes under-fires (failure-probe calibration is
standalone-question only; the same in-context coverage gap 8bj fixed
for the ACT probe exists for the FAILURE probe). Next calibration
pass: add reqqx-style in-context rows to the failure-probe mix.

## Phase 8bq — the label is now the deployed answer: native-judged training labels, per-language operating points, receipt on the deployed manifest (~$10 API, $0 GPU, 2026-09-01)

**Trigger:** an external review of the 8bp gate made four points, all
verified against the repo: (1) training labels were turn-based
`sampling=False` outcomes while features came from native duplex
sampling (`streaming_generate` defaults `do_sample=True, T=0.7`,
official `top_k=20`) — and the internal test label (guard1) was the
v3-harness `heard_ok`, not native either; (2) the receipt
(`scripts/27`) read the default-config tags → n=5252 ≠ deployed 5228;
(3) Reasoning-zh is an operating-point failure first (global balanced
threshold fires 1%), AUC second; (4) the +.02/doubling curve is six
historical points across changing families/configs, not a fixed-set
subsampling — it says "coverage helps", not "another random doubling
pays +.02".

**Labels re-made from the deployed behaviour (no GPU).** The official
native dumps kept `answer_text`; `judge_native` (gpt-5.4-mini, same
JUDGE_SYSTEM) over caliboff/expoff/exp2off/exp3off/exp3zhoff/freshoff
= 5,528 rows for ~$10. Lesson: five parallel judge apps trip the org
TPM/RPM 429 limits (5,000+ rows came back as ERROR); `judge_native`
now retries ERROR rows only (real verdicts are never re-judged) with
60 s backoff, and `judge_all` runs tags sequentially in ONE detached
app. Result: 0 unjudged rows.

**Turn-based vs native labels on the same 5,228 rows:** agreement
.808; fail rate .463 (tb) → .541 (native). Asymmetric: 702 rows
tb-correct/native-wrong vs 290 the other way. Native answers that
never reached end_of_turn (2.6%) are .946 fail. Families that disagree
most: hard-knowledge .656, trap-truthful .673, easy-chat .712;
least: trap .977, know-arc .890, know-longtail .879. Disagreements are
plain hallucinations under sampling (Lamarck "army officer", Bednarik
"hockey") — questions the deterministic decode got right and the probe
was therefore being taught NOT to escalate.

**Three-way refit on official features (`scripts/30`, nested-CV C
selection, family GroupKFold, bootstrap CI; `figures/label_source_refit.json`):**

| source | n | strat OOF | family-GroupKFold OOF | test (v3 lbl) | test (native lbl) | TriviaQA | WebQ | Llama | SD-QA | zh | En-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turn-based (8bp deployed) | 5228 | .855 | .827 | .844 | .878 | .767 | .772 | .815 | .728 | .606 | .771 |
| **native (deployed now)** | 5228 | .848 | .822 | .843 | .876 | .802 | .777 | .807 | .700 | .621 | .771 |
| agree-only | 4224 | .911 | .879 | .842 | .883 | .788 | .764 | .825 | .703 | .619 | .770 |

Paired bootstrap native−tb (`scripts/30b`): test −.001 [−.024,+.022],
TriviaQA **+.035 [+.006,+.065]**, WebQ +.005, Llama −.008, SD-QA −.028
[−.068,+.010], zh +.016 [−.039,+.070]. **Verdict: label source is a
wash for ranking; it is a definitional fix, not a lever.** Nested CV
picks C=3e-4 in every fold for both sources, so the old "select C on
the reported OOF" was not inflating anything. The family-GroupKFold
OOF (.82) is the honest cross-family number; the stratified .85 leaks
family identity. "Agree-only" .91 OOF is the easy-subset illusion —
external does not move.

**Deployed (`scripts/31`, guards pass):** native labels, C=3e-4, OOF
.848; guard1 testoff .844→.843 (v3 labels) / .878→.876 (native
labels), tie; guard2 fresh-heldout fire fast .35/.67/.89, never
.00/.00/.43. Artifact now carries `label_source`, `manifest` (sha1 of
query ids / features / labels), `eot_thresholds_lang`. Previous gate
backed up as `gate_native_pre8bq.json`.

**Per-language operating point, from the TRAINING set only.** Tier
thresholds quantiled on the core-mix OOF per language: en
.906/.797/.553, zh .485/.386/.279 (global .897/.775/.514). On
Reasoning-zh the global thresholds fire .00/.02/.15; the zh thresholds
fire .17/.34/.55 (nominal .15/.30/.50) — matching what the label-free
per-pool oracle (8bn) achieved, without touching the eval pool. Demo
(`demo_duplex.py`) takes `?lang=en|zh` (UI selector) and picks the
language's thresholds; falls back to global for a pre-8bq artifact.
`scripts/23` mirrors this (sreason = zh).

**Validity under the deployed gate + language thresholds
(`figures/native_validity_official.json`):** frozen bal .565@27%
(p=.0005) / **agg .669@49% = the always-escalate ceiling** (p<1e-4);
TriviaQA .684/.852, WebQ .527 (p=.045)/.680 (p=.020), Llama .792/.841,
SD-QA .605/.775 — all five en pools significant at both tiers.
**Reasoning-zh: .525 → .609@17% (p=.0065) / .658@34% (p=.023) /
.713@55% (p=.066)** — the first time the zh pool beats matched random
under a DEPLOYABLE threshold (8bn's per-pool oracle needed the eval
pool's own scores).

**Receipt on the deployed manifest (`scripts/27`, off tags, manifest
check → exits if it disagrees with the artifact):** n=5228, esc .541,
OOF .848 / logloss .480 / acc .767 (majority .541); test-239 AUC .876
(native-judged labels) / logloss .456 / acc .799; budget acc
.536/.707/.795 at realized .105/.310/.523; externals .802/.777/.807/
.700/.621.

**Not done (deliberately):** (a) soft `P(fail)` from 2–3 native
samples per query — that is a GPU re-dump (~15 H100h ≈ $60–75 per
pass), not a judge cost; defer until the single-sample native labels
show a gap, which they do not. (b) zh targeted expansion to 1–2k rows
(CMMLU/C-Eval knowledge, Chinese SimpleQA long-tail, DuReader/WebQA
open QA, C3 commonsense; sample to 40–60% fail) — next data spend, ~$40–60.
(c) a fresh blind holdout — the five external pools have been used for
decisions since 8be and are a dev set in practice. (d) `no_speak` rows
(14/2300 in exp3off) still dropped from training; deployment should
escalate them by timeout, not by probe. (e) multi-turn/in-context
calibration rows for the failure probe (8bn residual) still absent.

### 8bq addendum — targeted zh expansion (1,500 rows, ~$45): the zh axis is a TASK-TYPE axis, not a language axis (2026-09-02)

**What was built:** `modal_train3.py::build_expansion4zh` — public
benchmarks only, deduped against every pool incl. expansion3zh and
sreason: zh-longtail = Chinese SimpleQA 600, zh-ceval = C-Eval val 300,
zh-cmmlu = CMMLU test 300, zh-causal = XCOPA-zh remainder 300 (C3 /
DuReader skipped: reading-comprehension questions are not self-contained
when spoken). tts-1/alloy → official native dump (8 H100 workers, 1,500
traces, 0 no_speak) → `judge_native` (native labels only; no turn-based
pass). Native fail .578 (longtail .825 / ceval .520 / cmmlu .513 /
causal .207). Side finding: 34% of zh-causal answers come back in
ENGLISH (language switch) yet are judged adequate — a UX defect the
adequacy label does not see.

**Ablation (`scripts/32`, official feats, native labels, C=3e-4,
paired bootstrap vs the 5,228 base):**

| train | n | test (native lbl) | En-4 | SD-QA | Reasoning-zh |
|---|---:|---:|---:|---:|---:|
| base (8bq deployed) | 5228 | .876 | .771 | .700 | .621 |
| +zh4 | 6728 | .866 (−.010 [−.024,+.005]) | .777 | .727 (+.027 [−.003,+.059]) | **.627 (+.006 [−.026,+.040])** |
| zh-only | 1900 | .758 (−.118) | .700 | .663 | .591 (−.030) |

**Reasoning-zh does not move.** And the flip side, the deployed 5,228
gate scored on exp4zh as a HELD-OUT pool: **AUC .811 [.789,.833]**
(longtail .719, ceval .730, cmmlu .642, causal .693). The probe reads
Chinese long-tail facts, knowledge MC and causal commonsense about as
well as the English pools (.77–.82); what it cannot read is
OpenAudioBench Reasoning-QA (multi-step logic/arithmetic word
problems). The "zh .605" number in the paper was a task-type gap
wearing a language label. zh-only training is worse everywhere
(en mass helps zh).

**Operating point is composition-dependent (`scripts/33`).** With the
zh slice at 1,900 rows, the "per-language quantile" moves from the
exp3zh mix (balanced .386) to .749 (all-zh) — on sreason that fires 5%
instead of 34%. Per-family balanced quantiles fire .13/.24/.38 (cons
p=.003, bal p=.062); the exp3zh-only rule now over-fires (61% at
balanced) because the +zh4 model shifted sreason's scores. A static
per-language threshold is only as good as the match between the
language slice's family mix and the deployment stream; the
composition-free answer remains the online windowed quantile (8bn),
which the demo still does not implement.

**Decision: NOT deployed.** No ranking gain on the target pool, a
−.010 internal test drift, and a destabilized zh operating point.
`gate_native.json` stays the 8bq 5,228 native-label gate (backup copy
`gate_native_8bq_5228.json` is identical). exp4zh (feats + native
labels on the volume and locally) becomes the second zh eval pool
(AUC only — no expert arm was run).

**Cost:** TTS ~$3, native dump ~8 H100h ≈ $35, judge ~$3.

**Lesson for the next data dollar:** stop buying "more Chinese". Buy
multi-step reasoning coverage in BOTH languages (GSM8K-style word
problems are already .10–.31 fail = few positives; harder: MATH-500
tail, LogiQA-zh/en, BBH-style, CLUE/C3-free logic sets), and multi-turn
/ in-context rows. And implement the windowed tracker in the demo.


## Phase 8bs — NVDA-recipe cross-check: ceiling test, re-run stability, pre-answer read cost (~$4 GPU + ~$0.5 API, 2026-09-02)

**Context.** The coauthor's probe report on NVIDIA-NemotronLabs-VoiceChat-11B
(v1 19 Aug / v2 / v3) was checked against the deployed MiniCPM recipe.
Their three v2 levers are already in ours: 4× calibration data (ours
600→5,228), the read at the model's own commit-to-speak moment (our
onset-chunk read, 8be), and the act head (`gate_act.json`, 8bh). Their
v3-arch (average of three layers + the commit frame) raised every
calibration metric but gave the *same accuracy at fixed escalation
budgets* (±.005), so it is not replicated here — it would cost a full
native re-dump (~15 H100h) for a change they themselves would not swap
the demo for. Their "not worth repeating" list (hyper-parameter and
head-family sweeps, label re-judging) retires receipt §5-2.
What transfers is the v3 **diagnostics**; three were run.

**1. Ceiling test on the deployed gate (`scripts/33_ceiling_native.py`, $0).**
Same loader/CV as `scripts/31` (native labels, C=3e-4, 18 pools). A
pool-prior scorer (each row scored by its pool's fold-train failure
rate) vs the probe; "within" = AUC over same-pool pairs only.

| | prior | probe | gap | within (pooled) | within (macro) | recall@15/30/50% probe | prior | random |
|---|---:|---:|---:|---:|---:|---|---|---|
| train OOF (5,228) | .786 | .848 | +.062 | .743 | .731 | .265/.506/.749 | .245/.475/.713 | .149/.299/.496 |
| frozen test (239, deployed w/b) | .767 | .876 | +.109 | .820 | .812 | .265/.500/.765 | .257/.471/.699 | .140/.272/.493 |

Same shape as the n=600 audit (Phase 5b: prior .678 / probe .822 /
within .742) and as NVDA (+.07 of .820 per-question). In distribution
the type prior does most of the budget-level work (recall@30% .475 vs
.506); the per-question component is what the single-source external
pools measure directly (.70–.81). Strongest within-pool signal:
know-longtail .863 (n=741) — the "knows it doesn't know long-tail
facts" finding again; weakest: know-commonsense .622, trap .635.

**2. Re-run stability (`scripts/34_rerun_stability.py`; `testoff2` =
second official-config native dump of frozen test, deployed sampling
T=0.7, own gpt-5.4-mini label; ~$3 GPU).** 238 queries spoke in both
runs (1 no_speak each).

| | value |
|---|---:|
| answer text identical (exact / normalized) | .017 / .088 |
| native label flip run1→run2 | **.143** (fail .567 → .584) |
| onset chunk identical / mean abs diff | .836 / 2.1 chunks |
| feature cosine run1·run2 (p05) | .950 (.886) — parts: eot_last .957, eot_mean8 .895, user_mean .996 |
| score Pearson / Spearman | .949 / .942 |
| tier decision agreement (en thr) cons/bal/agg | .941 / .912 / .941 |
| AUC run1 feats vs lab1 / lab2 | .876 / .848 |
| AUC run2 feats vs lab1 / lab2 | .887 / .872 |
| AUC vs the 204 agree-only labels (run1 / run2) | .909 / .927 |

Reading: sampling rewrites 98% of answer texts and flips 14% of
outcomes, but the commit-chunk read moves little (score r .95,
91–94% same decisions; NVDA reported 99%). Each run's features
predict the *other* run's outcome about as well as their own — the
probe reads a property of the question-in-context, not the fate of
one sampled answer. The agree-only AUC (.91–.93) bounds how much of
the reported .876 is single-sample label noise (~.04); it is the
cheap version of the soft-P(fail) item in 8bq §7.3(a).

**3. Pre-answer ("causal") read cost.** `modal_native_dump.py` now
also saves `X_pre` = the same L22 read taken *before* the onset
chunk's generate (no answer tokens in the tail; byte-identical `X`).
Scored with the deployed head, **no refit**: AUC .861 / .837 (lab1 /
lab2) vs the deployed read's .876 / .848 → −.011 to −.015; rank
correlation with the deployed score .91. NVDA measured −.03 with a
refit; ours is a lower bound on the cost since the head was fit on
post-generate features. The pre-answer variant needs its own
thresholds (deployed quantiles fire 0/1/20%): the first ≈8 answer
tokens shift the score distribution more than they change the ranking.

**Not done / notes.** (a) Multi-layer average (their v3-arch): skip
per their fixed-budget result. (b) A refit causal probe needs `X_pre`
for all 5,228 training rows = full re-dump; only worth it if a
pre-answer gate becomes a deployment target. (c) `exp4zhoff` (1,500
rows, dumped + judged) is present locally but not in the deployed
gate (paper: held-out Mandarin pool) — untouched by this phase.
Files: `figures/ceiling_native.json`, `figures/rerun_stability.json`,
`data/frozen_native_testoff2_*`, paper appendix native section
(new paragraph) + one sentence in `signal.tex`.

## Phase 8bq-2 — reasoning coverage + in-context rows + the windowed tracker: the probe is saturated at this read point (~$65 GPU+API, 2026-09-02)

**Three things were bought/built after the zh negative:** (1) a
bilingual MULTI-STEP REASONING pool, `build_expansion5rs` (1,319 rows:
en LogiQA 69 / MATH L4-5 no-LaTeX 200 / StrategyQA 150 / BBH 150;
zh LogiQA-zh 300 / CMATH gr4-6 250 / Ape210K 200; public only, deduped;
tts-1 → official native dump, 1 no_speak → judge_native; native fail
.595, per family: en-logiqa .710, mathhard .690, strategyqa .567, bbh
.487, zh-logiqa .743, ape210k .580, cmath .404); (2) IN-CONTEXT rows —
the frozen calib split (358) and expansion3zh (399) re-dumped with
`--carrier /data/audio_pool/q0400.wav` ("Why do people travel to
islands for a holiday?"), i.e. each query arrives as the SECOND turn
after a carrier Q&A, judged on the in-context native answer; (3) the
online windowed quantile tracker in `demo_duplex.py` (per-process,
per-language deque of the last 100 onset scores; threshold = its
(1-rate) quantile once ≥20 scores exist, static per-language
threshold before that; `?tracker=0` disables; gate messages now carry
thr_mode/thr_static/n_window). Deployed.

**Reasoning coverage does nothing for Reasoning-zh (`scripts/34`,
paired vs the 5,228 base):**

| train | n | test (native) | En-4 | SD-QA | Reasoning-zh | exp4zh held-out |
|---|---:|---:|---:|---:|---:|---:|
| base (deployed) | 5228 | .876 | .771 | .700 | .621 | .811 |
| +rs | 6546 | .867 (−.009 [−.023,+.004]) | .773 | .713 | **.620 (−.002 [−.039,+.038])** | .810 |
| +ctx | 5985 | .880 (+.004) | .772 | .713 | .627 (+.006 [−.013,+.025]) | .815 (+.004 [+.001,+.007]) |
| +rs+ctx | 7303 | .877 (+.001) | .774 | .722 (+.022 [−.001,+.044]) | .621 (+.000) | .814 |

**Why: the reasoning families are only moderately readable AT ALL.**
5-fold OOF AUC inside the reasoning pool (trained on base+rs): en-bbh
.606, strategyqa .629, mathhard .683, zh-logiqa .679, en-logiqa .750,
ape210k .753, cmath .764; all .736 [.708,.762]. Compare facts / MC /
causal families at .77–.82 (En-4, exp4zh). Whether a multi-step chain
will come out right is weakly encoded in the listen→speak commit
state — the read point precedes the reasoning. Reasoning-zh .62 is
that ceiling, not a coverage gap. (Consistent with the review's
"English ranking is at the plateau of the current read point/recipe".)

**In-context is an operating-point problem, quantified (8bn residual
closed):** same calib questions, second turn after a carrier Q&A →
native fail .517→.620 (label flips on 26% of rows), yet the deployed
gate's score shifts DOWN by .097 on average; balanced fire .252→.115,
conservative .104→.028, aggressive .459→.327. Ranking holds
(in-context AUC .861 [.815,.896]). Adding in-context rows to training
is a wash on AUC (table above). The fix is the operating point — the
windowed tracker follows the session's own score stream — plus, if a
static point is wanted, quantiling the core OOF with in-context rows
included.

**Decision: weights NOT redeployed** (no significant gain on any target
pool; internal test −.009 with +rs; zh threshold composition drifts
again, zh balanced .386→.752 under the full mix). `gate_native.json`
stays the 8bq 5,228 native-label gate. All three new sets (exp5rsoff,
calibctx, exp3zhctx: feats + native labels, local + volume) are kept
as eval/diagnostic pools; `scripts/31`/`27` accept them as optional
parts if a future refit wants them.

**Where this leaves the method.** Three targeted expansions (zh 1,500;
reasoning 1,319; in-context 757; ≈$110) and one label-definition fix
later, external ranking sits where 8bp left it (En-4 .77, zh
reasoning .62, zh facts/MC .81). The remaining headroom is not in the
12,288→1 linear read on L22 at the commit chunk; it is in (a) reading
LATER (a few answer tokens in — the tail features already do this for
~8 tokens; a mid-answer re-score would see the chain start), (b) a
richer head (feature standardization / elastic net were never tried —
cheap CPU), (c) fusion with p(True) on reasoning pools (8bl showed
+.026 internal). The next dollar should go to (a)/(c) experiments on
the existing dumps, not to more labels of the same shape.

**Cost:** TTS ~$3, dumps ~12 H100h ≈ $55, judge ~$6.

## Phase 8bq-3 — is the plateau in the head or in the features? Head form, non-linear capacity, three heads + router ($0, 2026-09-02)

`scripts/35_head_capacity.py`, deployed 5,228-row manifest, native
labels, stratified 5-fold OOF + the eval pools (`figures/head_capacity.json`).

**A. Head form / capacity on the same 12,288-d features:**

| head | OOF | test (native) | En-4 | SD-QA | Reasoning-zh | exp4zh |
|---|---:|---:|---:|---:|---:|---:|
| LR C=3e-4, raw (deployed) | **.848** | .876 | **.771** | .700 | .621 | **.811** |
| LR + StandardScaler | .818 | .838 | .705 | .689 | .613 | .750 |
| elastic net (saga, l1 .5, std) | .843 | .884 | .764 | .723 | .606 | .806 |
| MLP 256 relu, early stop, std | .815 | .819 | .702 | .660 | .627 | .734 |
| PCA-256 → HistGradientBoosting | .825 | .845 | .715 | .647 | .595 | .774 |

Non-linear heads are WORSE by .05–.07 external, not marginally. Per-
dimension standardization alone costs .07 En-4: the raw activation
scale carries signal (high-variance directions are the informative
ones; whitening promotes noise dimensions). Elastic net is a wash
(internal +.008, external −.007). Single configs, untuned — but the
gaps are far outside what tuning recovers.

**B. Three heads + router** (facts 4,054 / reason 940 / chat 234
training rows; heads = one LR each; router = 3-way LR on the same
features): router OOF accuracy **.973** — the commit-state knows the
task type — yet no head beats the single head on its own group
(facts .846 vs .847, reason .761 vs .766, chat .672 vs .720: small
groups lose more to data size than they gain from specialization).
On eval: soft-routed mix vs single, paired: test +.001, TriviaQA
−.003, WebQ −.014, Llama −.005, SD-QA **−.033 [−.060,−.007]**,
Reasoning-zh −.016 (oracle routing to the reason head would give
.644, +.023, but the learned router does not realize it), exp4zh
−.008. Verdict: multi-head is a wash-to-negative at this data size.

**Conclusion.** With label definition fixed, four data expansions
flat, head capacity flat-to-negative, and specialization flat, the
plateau is in the FEATURES: what the L22 state at the listen→speak
commit chunk encodes about the coming answer is ~fully extracted by
one linear read. Remaining levers all change what is read, not how:
(a) a later read point (mid-answer re-score, needs a dump, ~$60),
(b) other layers / multi-layer at 5k rows (needs a dump), (c) fusion
with a second signal family (p(True), 8bl +.026 internal). Not: more
labels of the same shape, bigger heads, more heads.

## Phase 8bt — the collaborator's shadow candidates on our bench: ranking moves, routing barely ($0, 2026-09-02)

`scripts/36_shadow_compare.py` → `figures/shadow_compare.json`. Live 8bq
(`0e6494c2…`, master 669c40d) vs ChangyiYang's two shadow artifacts from
`codex/probe-performance-audit` (issue #8, P2–P17): **P9** distilled student
(eot_mean8 + user_mean, 8,192-d, teacher = .5 z(two-sample semantic
entropy) + .5 z(RTJ p(True)), ridge α=100, blend .25) and **P16** alpha-1
ensemble (z(live) + 1.0 z(P9), folded to one 12,288-d dot product). Same six
official-native pools, same cached expert arms as scripts/23; exact
per-pool 15/30/50% budgets so all three scorers escalate the same count.

| scorer | ext-5 native AUC | ext-5 benefit AUC | casc @15/30/50 (exact) |
|---|---:|---:|---:|
| live 8bq | .7429 | .6767 | .620 / .691 / .764 |
| P9 distilled | .7642 (+.0213) | .6918 (+.0151) | .629 / .699 / .771 |
| P16 alpha-1 | .7580 (+.0151) | .6872 (+.0105) | .623 / .695 / .767 |

Per pool, native-AUC delta [95% paired bootstrap]: P9 — test +.003,
TriviaQA +.014, WebQ +.021, Llama +.009, **SD-QA +.035 [+.002,+.068]**,
Reasoning-zh +.028 [−.006,+.064]; P16 — test +.006, TriviaQA +.011,
**WebQ +.016 [+.003,+.030]**, Llama +.006, **SD-QA +.023 [+.005,+.042]**,
**Reasoning-zh +.019 [+.001,+.039]**. Reproduces his posted numbers to
the fourth decimal.

**Reading.** (1) Both candidates rank better than live on all five
external pools; P9 is the larger move, P16 the more uniform one. (2)
Cascade accuracy moves far less than AUC: +0.8 pt (P9) / +0.3 pt (P16)
at the balanced budget against 5–18 pt of oracle headroom — ranking gains
are not converting into routing gains yet. (3) SD-QA (real speech) is the
only pool where both are individually reliable, consistent with the
semantic-entropy teacher helping where ASR-like noise dominates. (4) Does
not change the deployment call: his prospective sets P15–P17 (text
benchmarks through TTS, robustness evidence only) show the gain is
source-sensitive; next evidence = live shadow traffic (P11 wiring, PR
pending), not more offline rescoring. Consistent with 8bq-3: the lever is
a second signal family, and its distilled form is worth ~+.02 ranking.
Artifact page: "Shadow Gate Bench" (claude.ai artifact, 2026-09-02).

## Phase 8bu — the whole result set re-run as native full-duplex sessions; the relay was the bottleneck, not the gate (~$190 GPU + ~$25 API, 2026-09-02)

**User call:** every gallery/paper curve must come from native
full-duplex sessions; the turn-based/harness figures (1–14, 17–20) are
retired. `modal_native_bench.py` = the old conclive protocol on the
deployed stack: one fresh `MiniCPMODuplex` session per query, official
serving config (top_k=20, force_listen_count=3, friendly-assistant
prompt), the 8bq gate read at the listen→speak commit (per-language
thresholds + 8bh act gate), raw-audio gpt-transcribe uplink, gpt-5.5
web/low, wait paced 1 chunk/wall-second, real relay. Seven pools × five
arms, every arm a separate live run (no branching). Judges as in the old
figures (OAB gpt-4o / VoiceBench gpt-4o-mini / ours). Onset features
(float16) + shadow P9/P16 scores logged per row for issue #8.

**Finding 1 — the relay lost most of the routing gain.** The
talker-steering relay (RELAY_TMPL + nudge, i.e. what the demo shipped)
on TriviaQA always: expert text .960 → delivered .728 (never .628); on
the 47 balanced-tier fired turns local .28 / expert .92 / delivered .53.
99% of relays needed the nudge, 64/250 were cut under 40 chars, some
turns answered on their own ("Snow" → "frost"). WebQ always: expert .824
→ delivered .564. A/B on the same 60 TriviaQA ids (always arm): steer
.733 vs TTS relay .933 (+.20 [+.10,+.32]); TTS relay = cleaned expert
text spoken verbatim in the talker's own voice via the teacher-forcing
path the canned stall already used, plus a context note. Deployed
(demo_duplex.py `RELAY_MODE=tts`, `_synth_pcm`), all fired arms re-run.
Cleaner v2 after WebQ lists were truncated (.756 → .796 with lists kept;
no sentence cap, abbreviation-aware, 400 chars).

**Finding 2 — native curves (TTS relay), delivered accuracy @ realized
escalation rate; expert-text bound in brackets:**

| pool | never | conservative | balanced | aggressive | always [bound] |
|---|---:|---:|---:|---:|---:|
| our pool (240) | .408 | .467 @8% | .542 @25% | .629 @52% | .704 [.765] |
| TriviaQA (250) | .628 | .604 @4% | .700 @18% | .884 @56% | .960 [.964] |
| WebQ (250) | .528 | .592 @5% | .624 @22% | .752 @60% | .796 [.823] |
| Llama Q (250) | .824 | .788 @0% | .824 @4% | .856 @19% | .916 [.932] |
| SD-QA (200) | .480 | .540 @7% | .650 @27% | .825 @66% | .870 [.885] |
| Reasoning-zh (202) | .510 | .624 @20% | .663 @33% | .728 @50% | .837 [.861] |
| AlpacaEval (199, 1–5) | 3.68 | 3.63 @2% | 3.77 @5% | 4.01 @44% | 4.20 [4.96] |

Old steering relay for reference: TriviaQA bal .644@19% / agg .740@57%
/ always .728; WebQ always .564.

**Reading.** (1) Native local floors are 1–8 pts below the harness
(Reasoning-zh worst, our pool +2.5) — regime cost, consistent with
8be/8bq remix. (2) Routing gain is larger than on the harness once the
relay is lossless: aggressive − never = +3 (Llama) to +34 (SD-QA); the
zh pool now fires 20/33/50% under the per-language thresholds. (3) The
"selective > always" reversal on Llama (harness .948 > .928) does NOT
reproduce natively (.856@19% vs .916) — native floor lower, relay now
lossless, both favour always. (4) Local answers are stochastic
(top_k=20): a 0%-fire re-run moves the floor by up to 3.6 pts (Llama
conservative .788 vs never .824), so <5-pt differences are noise. (5)
AlpacaEval: always 4.20 vs expert text 4.96 — the 400-char spoken cap
removes half the open-ended content; method boundary + medium.
(6) Latency (P50, end of user speech → answer complete, incl. spoken
relay): TriviaQA 1.8/2.6/6.4/10.2 s, SD-QA 2.4/3.3/12.9/18.7 s,
Reasoning-zh 5.1/6.2/11.9/26.9 s, AlpacaEval always 39 s.

**Judge caveat:** OAB gpt-4o judge under concurrency leaves ~5–8% of
rows verdict-less (429s); `judge_pool` now re-tries verdict-less rows
and a final `_judge_sweep.sh` pass filled them. Expert-column merge bug
(wiped earlier verdicts) fixed with combine_first.

Files: modal_native_bench.py, `_run_native_bench_{A,B}.sh`,
`_judge_sweep.sh`, scripts/pull_native_bench.sh, data/native_bench/
(judged parquets), figures/native_bench_figures.py (+ native_floors.py),
figures/native_{pool}_{dualview,pareto}.png, figures/native_bench_summary.json,
gallery_app.py (harness figures retired; 26 entries). Demo:
demo_duplex.py RELAY_MODE=tts + clean_expert + `_synth_pcm`, deployed
15:20. Old-cleaner WebQ always run archived on the volume as
`old_cleaner_*`.

## Phase 8bv — probe lift by failure type: the gate sees "confident wrong", it does not see "execution" ($2 API, 2026-09-03)

`modal_failure_taxonomy.py` classified all 600 never-arm failures on the
six native QA pools (1,392 rows) with gpt-5.4-mini (structured output;
inputs: question text, gpt-transcribe transcript of the same audio,
reference, local answer, judge note) into perception / knowledge_gap /
confident_wrong / execution / quality_other; `scripts/37_failure_taxonomy_lift.py`
scores each class with the never-arm onset score at the deployed
per-language thresholds and the always-arm (TTS relay) expert outcome.

| type | n (share) | AUC vs correct | recall @cons/bal/agg | expert fixes | pts recovered @bal/@agg | recoverable |
|---|---:|---:|---:|---:|---:|---:|
| perception (misheard) | 36 (6%) | **.889** | .28/.56/.86 | .39 | 0.65/0.86 | 1.0 |
| knowledge gap (hedges) | 17 (3%) | .829 | .24/.65/.82 | .71 | 0.57/0.72 | 0.9 |
| confident wrong | 418 (70%) | .813 | .10/.37/.79 | .76 | **8.3/18.6** | 22.9 |
| execution (working slips) | 44 (7%) | **.469** | .18/.32/.57 | .59 | 0.65/1.15 | 1.9 |
| quality / other | 81 (14%) | .631 | .22/.35/.52 | .68 | 1.1/1.8 | 4.0 |

False-fire on correct rows: .02 / .08 / .31 at cons/bal/agg. Execution
failures are 35/44 from Reasoning-zh; perception 14/36 from our pool,
7 from SD-QA.

**Reading.** (1) 70% of native failures are specific wrong answers that
read as confident; the pre-answer state still ranks them at AUC .81 and
the aggressive tier catches 79% of them at a 31% false-fire — the hidden
state knows more than the surface text (cf. Orgad et al. 2024). 81% of
the routing gain (18.6 of 22.9 recoverable points at aggressive) comes
from this class. (2) Execution failures are invisible at the commit
point: AUC .47. Their 57% escalation at aggressive is the zh threshold
firing at its nominal rate, not discrimination. This is the one class a
later read point (after the first answer tokens) could move; it caps at
~1.9 points per 100 questions on these pools. (3) Perception failures
are the best-ranked (.89) but least fixable (.39): the expert gets the
raw audio through gpt-transcribe, which also mishears some of them.
(4) Caveat: the classifier cannot see the model's internal state, so a
misheard question answered fluently lands in "confident wrong"; class
labels carry judge noise; knowledge_gap has n=17.
Files: modal_failure_taxonomy.py, scripts/37_failure_taxonomy_lift.py,
data/native_bench/failure_types.parquet, figures/failure_taxonomy_lift.{png,json};
gallery 图N7.

## Phase 8bw — failure-type explainer page + headroom accounting ($0, 2026-09-03)

`scripts/38_taxonomy_examples.py` pulls per-type real examples out of the 8bv
taxonomy (each carrying the deployed gate's never-arm score on that exact row,
which tiers it fires at under the per-language thresholds, and whether the
always-arm expert fixed it) and adds a net-points ledger over the same 1,392
rows: gain = failure escalated and expert fixed it, loss = correct row
escalated and expert got it wrong.

| tier | fire rate | gain | loss | net (pts/100q) |
|---|---:|---:|---:|---:|
| conservative | 7% | +3.9 | −0.1 | **+3.7** |
| balanced | 21% | +11.2 | −0.4 | **+10.8** |
| aggressive | 49% | +23.1 | −1.2 | **+22.0** |
| always escalate | 100% | +30.8 | −2.9 | **+27.9** |

**Reading.** The aggressive tier buys 79% of the always-escalate net gain at
49% of the cost; the entire remaining headroom on these pools is 5.9 net
points, and it costs double to reach. Split by type (gross), the 7.6-point
gap is confident-wrong 4.3, quality/other 2.2, execution 0.7 — i.e. the
cheapest unclaimed block is quality/other, most of which is empty or
MAX_ANS-truncated answers that a *rule* (empty/truncated → escalate) catches
without any probe. Rows with an empty answer have no onset score at all.

Explainer page (五类定义/判定/真实例子 + 训练成果 + 剩余空间, reads both JSONs
at build): `taxonomy_app.py` →
https://rhe9527--failure-taxonomy-page-web.modal.run/7f31ac0d
Files: taxonomy_app.py, scripts/38_taxonomy_examples.py,
figures/failure_examples.json.

## Phase 8bw — read-point timeline: uncertainty unfolds with the answer (~$45 GPU + ~$12 API, 2026-09-03)

`modal_native_dump.py` now saves, per query, the same 12,288-d L22 read at
five moments of ONE generation: `X_pre` (before the onset chunk's
generate; no answer token), `X` (deployed: after it), `X_k1/2/3` (after the
1st/2nd/3rd answer chunk; padded with the last available read — 0.2 / 2.8
/ 10.6% of rows end earlier). Fresh official-config dumps of the whole
8bq training merge (`calibk expk exp2k exp3k exp3zhk freshk`, 5,236 rows
after judge) and the six eval pools (`testk …`), labels judged on the
same generation. `scripts/38_readpoint_refit.py`: one 8bq-recipe probe
per read point (L2 LR, C=3e-4, row-random 5-fold OOF).

| read | OOF | test | TriviaQA | WebQ | Llama | SD-QA | Reasoning-zh | En-4 | ext-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pre | .848 | .832 | .820 | .721 | .779 | .736 | .711 | .764 | .753 |
| onset (deployed) | .850 | .852 | .819 | .782 | .785 | .738 | .648 | .781 | .755 |
| k1 | .852 | .854 | .793 | .777 | .810 | .747 | .681 | .782 | .762 |
| k2 | .855 | .846 | .823 | .786 | .809 | .766 | **.769** | .796 | .791 |
| k3 | .857 | .858 | **.857** | .791 | .824 | .790 | .744 | **.815** | **.801** |
| onset+k1 (logit mean) | .856 | .859 | .814 | .786 | .801 | .752 | .674 | .788 | .765 |

Cascade at an exact 30% budget (local from this dump, expert from the
TTS-relay always arm): k2 vs onset — Reasoning-zh .668→.713, TriviaQA
.780→.784, others within ±1 pt. Preliminary leave-one-pool-out on the
eval pools alone (1,150 train rows) showed the same ordering (onset .714
→ k3 .765).

**Interpretation.** (1) The commit-point state encodes *retrieval*
confidence (did I hear it, do I have it, is this the right entity);
*execution* confidence does not exist yet — the slip has not happened.
After two answer chunks the trajectory is in the state and Reasoning-zh
rises .648→.769; internal test (recall-dominated) does not move. Same
pattern as text-side P(IK) < P(True) and answer-token probes (Kadavath
2022, Orgad 2024), here resolved on the streaming time axis. (2) The
"probe ceiling" of 8bq-3 is the pre-answer-read ceiling (.76–.78); k3
reaches .80 with +3 s and a partially voiced wrong answer — a product
trade, not a deployment change; the paper's pre-commitment claim
stands. (3) `pre` vs `onset`: onset includes the first spoken unit;
for English facts that unit is often the entity (WebQ +.06 for onset),
for zh / real speech it is a discourse marker (Reasoning-zh +.06 for
pre). Read-point value tracks *when the answer content enters the
token stream*. Per-language read-point choice would be eval-pool
tuning; not adopted. (4) Ranking gains still convert weakly at fixed
budgets except on zh (same lesson as 8bt / issue #8).
Files: modal_native_dump.py (X_k1..3, n_post), `_run_postread.sh`,
scripts/pull_postread.sh, scripts/38_readpoint_refit.py,
figures/readpoint_refit.{json,png}, data/frozen_native_*k_judged.parquet
(feature shards stay on the volume, ~1.2 GB). Gallery 图N8.

## Phase 8bx — two-stage gate (commit-point probe + k2 re-score on a gray band): not worth it ($0, 2026-09-03)

`scripts/39_two_stage.py` on the 8bw dumps. Policy P(r, d, f): fire the
top r(1-f)N rows by onset score at the commit; defer the next dN rows
(gray band) to the k2 probe and fire the top r·f·N - |A| of them; the
rest answer locally. d=0 is the deployed gate, d=1 is k2-only. Delivered
accuracy at exact per-pool budgets, expert = TTS-relay always arm.

| budget | one-stage | band .2 / half at k2 | band .5 / half at k2 | k2-only (all deferred) |
|---|---:|---:|---:|---:|
| 15% | .638 | .642 | .646 | .637 |
| 30% | .711 | .716 | .718 | .720 |
| 50% | .770 | — | .784 | .783 |

Per pool @30%: only Reasoning-zh moves (.668 → .708 band-half / .713
k2-only); the other five are within ±.005.

**Reading.** The k2 read's AUC advantage (+.04 ext-5) buys under one
point of mean delivered accuracy at a fixed budget, for two reasons:
(1) at a fixed budget only the ranking near the threshold matters, and
there onset and k2 agree on most rows — the k2 probe re-orders the
middle of the ranking, not the boundary; (2) the rows it does flip are
concentrated in execution failures (Reasoning-zh, +4.5 pts), which are a
small share of failures elsewhere (8bv: 7% overall). Combined with
the latency cost (20–50% of turns hear ~2 s of a possibly wrong answer
before the decision) the two-stage gate is not a deployment candidate;
the commit-point single decision stands. It is a real lever only for
reasoning-heavy traffic. Consistent with 8bt/8bw and issue #8: ranking
gains convert weakly into routing gains at fixed budgets.
Files: scripts/39_two_stage.py, figures/two_stage.{png,json}; gallery 图N9.

## Phase 8by — the last three probe levers, closed on the live rows ($0, 2026-09-03)

Three routes were still listed as "not yet refuted" after 8bq-3. All
three are now measured on the native live rows (never-arm local
outcome, TTS-relay always-arm expert outcome, exact per-pool budgets).

1. **Later read point** (8bw/8bx): ranking +.036 ext-5 AUC at k2,
   Reasoning-zh +.12; delivered accuracy at a fixed budget +0.7-0.9 pt
   mean, +4.5 on zh only; two-stage gray band no better. Analysis
   result, not a deployment lever.
2. **Second-signal fusion (P9 / P16 shadow scores)** logged on every
   live row by the bench runner. AUC on the live never arms: frozen
   .858 -> .864/.865, WebQ .732 -> .742/.749, SD-QA .784 -> .809/.815,
   Llama .703 -> .700/.691, zh .685 -> .680/.671. Delivered accuracy,
   external-4 mean: 15% .657 -> .661/.657, 30% .717 -> .717/.719,
   50% .770 -> .766/.772. Within +-.005 everywhere. Same verdict as
   8bt and as ChangyiYang's P32: ranking moves, routing does not.
3. **Empty / truncated-answer rule.** Empty answers: 8 rows in 1,392
   (7 wrong, expert fixes 6). Truncation (60-chunk cap without
   end-of-turn): frozen 29 (90% wrong, expert fixes 45%), Reasoning-zh
   26 (92% / 77%), none elsewhere. Forcing those rows to the front of
   the budget: external-5 mean +.004/+.007/+.005 at 15/30/50%; zh
   +.015; frozen -.016 at the two lower budgets because the displaced
   probe picks were more expert-fixable than the truncated rows.
   Detection is also late (the cap is reached after ~60 s of speech),
   and answer length is not an early proxy: frozen answers >=800 chars
   are 86% wrong but the expert fixes only 50% of them (formula-heavy
   problems the expert also misses). Worth ~0.5 pt, not the 2.2 pt
   the taxonomy ceiling suggested; not deployed.

**Net.** The probe-side ledger is closed: every remaining route buys
ranking, none buys more than a point of delivered accuracy at a fixed
budget. Recoverable headroom is 7.5 pt/100 at aggressive (8bv), and it
is spread across confident-wrong rows the probe already ranks near
the boundary. Gains from here come from the channel (relay, done:
+23 pt on TriviaQA always), the expert (fixable rate .39 on misheard
turns), and the judge/label floor, not the read.
