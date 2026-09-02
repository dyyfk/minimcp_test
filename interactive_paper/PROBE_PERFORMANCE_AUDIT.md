# Probe performance audit (2026-09-02)

Scope: deployed official-config native gate at `03bf0e4`, artifact
`data/gate_native.json` (`train_n=5228`, `C=3e-4`). This audit uses the
committed code and data artifacts. The large feature shards remain on the
original `gate-data` Modal volume and were not visible from the available
`liquid-ai` Modal workspace, so model refits are proposed rather than claimed.

## Bottom line

Another generic English data doubling is not the first experiment to run.
The current `.770` En-4 result is measured against legacy turn/concurrent
labels and is almost equal to the `.771` turn-based reference. That shows the
probe has learned the proxy target well; it does not show saturation on the
deployed official-native failure target.

The highest-value next step is to judge the already-collected official-native
training answers and refit on deployment-aligned labels. After that, optimize
the routing utility (`expert_ok - local_ok`) rather than only local failure.
The committed paired outcomes show 5--18.5 points of oracle routing headroom at
the deployed budgets, depending on pool and tier.

## Finding 1: training and headline AUC labels are not deployment-aligned

`scripts/26_official_refit.py` pairs official native features
(`caliboff`, `expoff`, ...) with the older turn-based `escalate_label`
parquets. The receipt and refit evaluation paths likewise use
`frozen_v3_traces` or the `*_conclive_traces` never arm for AUC. In contrast,
the native validity table uses `frozen_native_*off_judged.parquet`.

An ID-level join shows that the two targets disagree often:

| pool | n | label agreement | official-native floor | legacy floor | changed |
|---|---:|---:|---:|---:|---:|
| internal test | 240 | .821 | .429 | .483 | 43 |
| Speech TriviaQA | 250 | .752 | .568 | .504 | 62 |
| Speech WebQ | 250 | .668 | .396 | .616 | 83 |
| Llama Questions | 250 | .824 | .752 | .816 | 44 |
| SD-QA | 200 | .700 | .455 | .495 | 60 |
| Reasoning zh | 202 | .713 | .525 | .406 | 58 |

Thus 18--33% of evaluation labels change with regime/pipeline. The native
answers for training dumps are already recorded by `modal_native_dump.py`; the
missing step is to run `judge_native` on the official training tags. This is a
judge/API pass, not another GPU feature dump.

Tracking: [issue #5](https://github.com/dyyfk/minimcp_test/issues/5).

## Finding 2: local failure is not the optimal routing target

At a fixed escalation budget the desired ordering is expected gain minus
cost, not simply `P(local failure)`. A failure probe treats an expert-fixable
failure and a both-wrong failure as the same positive label, and it does not
penalize cases where escalation changes a correct local answer into an
incorrect expert answer.

Using the committed official-native local outcomes and cached always-expert
outcomes, the retrospective gain oracle has the following advantage over the
current probe at exact per-pool quantile budgets:

| pool | gap @15% | gap @30% | gap @50% | positive benefit | harmful escalation |
|---|---:|---:|---:|---:|---:|
| internal | +6.9 | +14.3 | +9.7 | 33.3% | 9.6% |
| Speech TriviaQA | +5.2 | +10.4 | +12.0 | 36.8% | 2.0% |
| Speech WebQ | +5.0 | +12.3 | +18.4 | 45.2% | 3.2% |
| Llama Questions | +5.9 | +6.7 | +4.6 | 18.4% | 2.4% |
| SD-QA | +5.0 | +11.5 | +18.5 | 47.0% | 3.0% |
| Reasoning zh | +8.9 | +17.8 | +12.9 | 32.2% | 4.0% |

The oracle is not an attainable model; it measures how much room remains in
the routing objective. The older 2310-row benefit experiment is not decisive
for the merged official regime because it predates both the current feature
distribution and the deployment-aligned labels.

## Finding 3: threshold calibration is an immediate deployment lever

The merged probe separates Reasoning zh (`AUC .605` on the legacy target), but
global English-mix thresholds fire on only 0%, 1%, and 13% of zh turns. Static
per-pool thresholds restore 15%, 30%, and 50%; the committed online-window
simulation is the deployable version. At 30%, the official-native zh cascade
accuracy rises from `.525` under the almost-never-firing global threshold to
`.649` under the per-pool threshold.

This primarily restores the intended expert budget; it does not improve
ranking. Production should maintain separate warm-started quantile trackers
for at least language buckets, otherwise a changing English/Chinese traffic
mix will move the threshold for both. Log realized rate, accuracy/quality,
language, and score drift. Fall back to the artifact threshold until a bucket
has enough observations.

## Recommended experiment order

### P0: make measurements reproducible

The deployed artifact is official/5228, while the old receipt script reads
default features and the committed receipt says 5252. The audit branch makes
the receipt derive the serving regime from the artifact, require the exact
training count and dimension, hash the gate, record block provenance, and
report official-native and legacy evaluation targets separately. It refuses
to emit a plausible receipt when those checks fail.

Tracking: [issue #6](https://github.com/dyyfk/minimcp_test/issues/6), branch
`codex/probe-performance-audit`, commit `7e28e60`.

### P1: deployment-aligned relabel/refit (highest expected value)

1. Run `judge_native` for `caliboff`, `expoff`, `exp2off`, `exp3off`,
   `exp3zhoff`, and `freshoff` against the existing traces.
2. Report disagreement with the legacy target by source, language, and
   no-speak/EOT status. Rejudge a stratified 10--20% subset three times to
   estimate stochastic label noise; use a soft failure rate if disagreement is
   material.
3. Fit matched models on identical official features:
   - current legacy-failure target;
   - official-native failure target;
   - multi-task/soft target using both labels.
4. Select hyperparameters with source-grouped or leave-one-family-out CV, not
   row-random CV alone. Keep the untouched official-native external pools as
   the final test.

Primary metrics: official-native AUC and cascade accuracy at exact 15/30/50%
budgets, with paired bootstrap intervals. Keep legacy AUC only as a transfer
diagnostic.

### P2: optimize expert gain

Collect actual expert outcomes on a stratified subset of the aligned training
rows, then compare:

- a direct gain classifier for `expert_ok > local_ok`;
- two heads for `P(local_ok)` and `P(expert_ok)`, routed by expected delta;
- the current failure score followed by a small benefit reranker.

Include a cost term for latency/API spend. Evaluate routing utility and
harmful-escalation rate, not benefit AUC alone. A staged subset is preferable
to paying for all 5228 expert calls before the target shows transfer.

### P3: cheap linear improvements on the same features

Only after fixing the target, run a controlled matrix of:

- fold-local standardization (then algebraically fold mean/std into `w,b`, so
  deployment remains one dot product);
- per-block normalization for `eot_last`, `eot_mean8`, and `user_mean`;
- source-balanced sample weights;
- PCA/SVD at 256/512/1024 dimensions plus logistic regression;
- each feature block alone and pairwise, to retest whether all 12288
  dimensions help out of source.

The current L2 logistic regression is not scale-invariant, so "no scaler" is
not a neutral choice. All preprocessing must be fit inside each CV fold.
Prefer a simple model only when its external-native interval is not worse.

### P4: cover live traffic rather than another generic doubling

The residual follow-up under-fire is already documented: the failure probe is
calibrated on standalone questions. Add carrier/in-context rows using the
existing `carrier` path, plus disfluencies, corrections, and multi-turn
references. If more data is purchased, prioritize:

1. official-native Chinese and cross-lingual rows;
2. real-speech SD-QA-like rows;
3. follow-up/in-context rows;
4. failure families with high expert-fixable rate.

Generic English rows are last: on the legacy proxy, En-4 is already at the
turn-based reference.

## Decision gates

A candidate should replace the deployed probe only if it:

- improves mean official-native external AUC by at least .015 or mean cascade
  accuracy by at least 2 points at 30% budget;
- improves at least three external pools with paired intervals that do not
  indicate a material regression;
- preserves FreshQA fast/never guards and realized budget calibration;
- reports latency and harmful-escalation rate;
- reproduces from a receipt tied to the exact gate checksum and serving config.

The repeat-then-judge fusion can remain an English-only, default-off experiment:
its current result adds about .026 internally but is negative on zh and was
evaluated through the same legacy-target path. Re-evaluate it only after P1,
because otherwise it optimizes a measurement that is not the deployed target.
