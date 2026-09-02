# Probe performance audit (2026-09-02)

Scope: official-config native gate at `03bf0e4`, artifact
`data/gate_native.json` (`train_n=5228`, `C=3e-4`), plus the author's
deployment-aligned 8-bit artifact now served by the live demo. This audit uses
the committed code and published data artifacts. The initial static audit did
not have the large feature shards. P1 and the staged P2 expert-gain experiment
were subsequently executed; their results are recorded below.

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

Both recommended target experiments have now been run. P1 confirms the label
mismatch (only 80.8% agreement on the 5228 training rows), but a matched
native-label refit does not improve external ranking or routing. P2 labels a
fixed stratified 1500-row subset with paired expert outcomes, but its direct
gain router transfers worse than the live aligned failure router. The result
does not support replacing the live artifact. The next useful experiment needs
a different representation or a richer routing model, not more labels for the
same 12288-dimensional linear head.

## P2 execution result: direct expert-gain routing does not transfer

A fixed 1500/5228 subset was sampled by source family, language, and native
failure with seed 42. The ordered ID-list SHA256 is
`0e6f5a1a6c4563bb1de1bd5410724f734513097810106613584ae438e064b9fc`.
The exact repository expert/judge protocol produced 1500 judged outcomes with
zero API errors: the expert was correct on 82.7% of rows versus 46.1% locally;
38.3% had positive gain and only 1.7% had harmful gain. The label parquet
SHA256 is
`a3ef657fec85867d8841f783c5350493287332f8145e7050d07294c79858facc`;
the selection-manifest SHA256 is
`e1674662872ebd66545e9adb5f8171dadcf6164d16e4964c8ebc3b517fa22cc9`.

Source-family-grouped five-fold CV (35 groups, with the same family kept
together across expansion rounds) selected the positive-benefit logistic head
at `C=1e-4`, with mean routing objective `+.1671` across exact 15/30/50%
budgets. On the same training rows, the frozen live aligned failure score is
better at `+.1816`. An exploratory standardized blend sweep favors zero weight
on the gain score; increasing its weight monotonically reduces the objective.
This does not support a small gain reranker on the evidence available here.

Untouched official-native pools, candidate versus the live aligned 8-bit
failure router:

| pool | n | benefit AUC aligned | benefit AUC gain | cascade delta @15/30/50% |
|---|---:|---:|---:|---:|
| internal test | 239 | .712 | .690 | +.000 / +.008 / -.025 |
| Speech TriviaQA | 250 | .747 | .678 | +.000 / -.028 / -.044 |
| Speech WebQ | 241 | .604 | .601 | +.008 / +.004 / -.008 |
| Llama Questions | 245 | .755 | .671 | -.016 / -.037 / -.008 |
| SD-QA | 200 | .678 | .631 | -.020 / -.025 / +.000 |
| Reasoning zh | 202 | .599 | .530 | -.010 / -.045 / -.020 |

Across the five external pools, mean benefit AUC falls from `.677` to `.622`.
Mean cascade accuracy changes by `-.8`, `-2.6`, and `-1.6` points at 15%, 30%,
and 50%. The gain candidate loses benefit AUC on every external pool and fails
the replacement gate. Its artifact SHA256 is
`1f9d710f70e4991787cb852175d695cb8482c2259f444409016e7b47a1be2643`;
the full result-receipt SHA256 is
`6c728061f69b1bcba96f028fc4b29854a1a490ad2c0686e82c44c3d20e6d4aaa`.

The expert is almost always correct and is rarely harmful on this staged
sample, so the positive-gain label is close to native local failure with added
noise. The observed oracle headroom therefore does not imply that the same
small-model hidden-state vector can predict expert benefit out of source. Do
not deploy this candidate or buy the remaining expert labels under the same
linear-feature hypothesis.

## P3a execution result: block pruning helps ranking slightly, not routing

The first post-P2 iteration kept the live 5228 native/policy labels fixed and
swept the three feature reads alone and pairwise, the full concatenation,
`C={1e-4,3e-4,1e-3}`, and fold-local per-dimension standardization. All 120
fits used five source-family-grouped folds (37 groups); the 1500 paired expert
outcomes were used only as an internal routing validation signal. Incremental
OpenAI cost was zero.

The frozen selection rule chose the configuration with the best routing OOF
objective among candidates within `.005` of the best native grouped-OOF AUC.
It selected raw `eot_mean8 + user_mean`, dropping `eot_last`, at `C=3e-4`:
native grouped-OOF AUC `.8127`, paired benefit OOF AUC `.6898`, and routing
objective `+.1711`. The full raw feature configuration measured `.8091`,
`.6867`, and `+.1709`; standardization did not help.

On the five untouched external pools, the pruned head changes mean native AUC
from `.7429` to `.7516` (`+.0087`) and benefit AUC from `.6767` to `.6814`
(`+.0047`). The only individually reliable gain is SD-QA native AUC
`.726 -> .751` (paired bootstrap delta `+.0250`, 95% CI
`[+.0071,+.0446]`). Mean cascade accuracy changes by approximately
`+.0000/+.0025/+.0022` at 15/30/50% budgets; every 30% pool-level interval
includes zero. Result receipt SHA256:
`aa94aa18c6343d5693a78bebc83bc693faf8ee530c6f4d218e32e1de2783ff84`.

This is a real regularization clue—the final-token read is mildly redundant
or harmful once the rolling tail and user-audio mean are present—but it does
not clear either replacement threshold. Keep it as the base configuration for
cheap nonlinear/low-rank follow-ups; do not deploy it alone.

### P3b: hidden-geometry nonlinear residual is rejected

A nested-safe low-capacity residual used only 18 per-row summary scalars:
mean, standard deviation, RMS, and maximum absolute activation for each of the
three blocks, plus pairwise cosine and RMS-distance. Source-family outer folds
fit the P3a linear base and histogram-gradient-boosted residual independently;
the sweep tested 3/7/15 leaves and residual blend weights
`0/.1/.25/.5/.75/1`.

Every nonzero residual weight reduces grouped OOF native AUC, benefit AUC, and
routing utility. The selection rule chooses residual weight zero (`.8127`,
`.6898`, `+.1711` respectively), so the external candidate is exactly P3a.
Result receipt SHA256:
`00efaf371e639a545305b73e3a7378213eb8a7b776d50c060d141d37d6f96a1f`.
Simple activation magnitude and block geometry do not explain the remaining
transfer error; stop this branch rather than increasing tree capacity.

### P3c: low-rank PCA improves OOF routing but loses externally

Five outer family-grouped folds each fit a 512-component randomized PCA on
the P3a `eot_mean8 + user_mean` features, reusing the decomposition for
128/256/512-dimensional, raw/whitened, and four-`C` logistic candidates. The
winner was raw PCA-512 (`C=1e-3`, 84.9% full-fit explained variance), with
native OOF AUC `.8105`, benefit OOF AUC `.6916`, and routing objective
`+.1724`. The latter is the best internal routing score in P3, but it does not
transfer.

Across the five external pools, mean native AUC changes `.7429 -> .7339`,
benefit AUC `.6767 -> .6656`, and cascade accuracy
`-.0012/-.0061/-.0022` at 15/30/50%. No pool has a reliable improvement.
Receipt SHA256:
`370d5a9b13eb491540a775bc26cf0d092eadd4f80840b8a714e2bffc57e270a4`.
Stop low-rank linear compression: its grouped training advantage is another
non-transferable regularization artifact.

### P4: semantic-uncertainty supervision helps, but not enough to deploy

A fixed 1,000-row subset of the staged paired set was sampled proportionally
within the original source-family, language, and native-failure strata (seed
43). For each row, three stochastic official-config native-duplex answers were
generated locally on B300 and paired with the cached deterministic answer.
Generation completed 1,000/1,000 with zero errors. A local multilingual
sentence encoder supplied connected-component entropy at three fixed cosine
thresholds plus continuous answer-dissimilarity targets; no OpenAI judge was
used. TTS cost was $1.896735 and local generation cost no OpenAI tokens.
The published bundle contains query text but not the original waveforms, so
these samples use one fixed Alloy TTS rendering. They measure semantic/model
instability under a controlled speech surface, not acoustic perturbation
uncertainty on the original recording.

Direct answer disagreement is informative on the fixed training pilot: mean
pairwise dissimilarity has native-failure AUC `.7263` and expert-benefit AUC
`.6519`. The strongest direct variant, deterministic-to-stochastic answer
dissimilarity, reaches `.7274/.6531`. This validates semantic instability as
a distinct uncertainty signal, but direct use requires three extra native
generations per request.

For a cheap single-pass surrogate, source-family-grouped outer folds predicted
the semantic targets from the P3a `eot_mean8 + user_mean` features with a
multi-output ridge sweep. The frozen winner predicts stochastic-sample
pairwise dissimilarity (`alpha=1000`) and blends it at weight `.25` with the
P3a failure score. It records native OOF AUC `.8066`, benefit OOF AUC `.6900`,
and routing objective `+.1743`, versus `.7981/.6839/+.1690` for the fold-matched
base on the same pilot rows.

On the five untouched external pools, the surrogate changes mean native AUC
from `.7429` to `.7528` (`+.0099`), benefit AUC from `.6767` to `.6826`
(`+.0058`), and cascade accuracy by `-.0004/+.0083/+.0033` at 15/30/50%.
Speech WebQ supplies the only reliable native-ranking gain: `.772 -> .803`,
paired-bootstrap delta `+.0314`, 95% CI `[+.0077,+.0561]`. Reasoning-zh loses
benefit AUC (`.599 -> .574`), and the mean 30% cascade gain remains below the
predeclared two-point replacement threshold. Do not deploy the surrogate yet.

Selection parquet SHA256: `c4a3581df4f8718886de0d404a80ed086881e8e9b2131c6e47e03e6ca11e8041`;
generated sample-stream SHA256: `2ec928629319326ea461749622908349743f36a668c3f7f0c5add7c5dc2cd320`;
semantic-label SHA256: `5adb9bc937cd7b9417be2b4bf59bc63b4a6088bfc3ea38535fbd92bee59422c6`;
result receipt SHA256: `4909e209dfb947a28b9ca4ff36db5867fee85d0b7ae62a1c2b1fcde9b3ba3f53`.
The next frozen test evaluates the direct multi-sample signal on the exact
1,138-row external intersection; that separates target quality from surrogate
fit quality.

### P5: direct semantic uncertainty clears the mean-AUC gate

The direct test generated three new answers for all 1,138 rows in the fixed
external intersection, with zero errors, then embedded and scored the four
answers exactly as in the training pilot. Candidate selection did not use the
external outcomes: source-family OOF scores compared the live three-block and
P3a two-block bases, six predeclared semantic metrics, fixed blend weights,
and a seven-scalar logistic stack. The winner is the P3a score blended equally
with deterministic-to-stochastic answer dissimilarity. On the training pilot
it records native AUC `.8124`, benefit AUC `.7002`, and routing objective
`+.1733`.

The external evaluation changes five-pool mean native AUC from `.7429` to
`.7632` (`+.0203`) and benefit AUC from `.6767` to `.6958` (`+.0191`). Mean
cascade accuracy changes by `+.0047/+.0114/+.0028` at 15/30/50%. WebQ is the
strongest pool: native AUC `.772 -> .811`, benefit AUC `.604 -> .671`, and
cascade `+.021/+.037/+.025`; its benefit-AUC delta is reliable (`+.0667`, 95%
CI `[+.0149,+.1215]`). SD-QA and Reasoning-zh also gain native AUC, while
TriviaQA and Llama Questions regress slightly. No individual native-AUC
interval excludes zero.

This clears the predeclared `+.015` five-pool mean native-AUC threshold, the
first iteration to do so. It is not yet a drop-in production replacement:
the mean 30% cascade gain is only 1.14 points, support is heterogeneous, and
the score requires three additional native generations plus sentence
embedding. Run a cached one-/two-sample latency ablation before choosing a
serving design; retain the single-pass P4 surrogate as the cheap alternative.

An audit correction occurred before the final evaluation: the first selection
parquet carried expert labels from the ceiling files rather than the exact
`tier=always` conclive files used by every preceding external comparison. The
1,138 IDs, queries, audio renders, native labels, and generated answers were
identical, but 83 expert outcomes differed. The corrected result reproduces
all live-aligned benefit baselines from P3 and is the only result reported
above. Corrected selection SHA256:
`f04677d7af5a977d8e1e4e48ab132e934d33647bcc6929a3c5e1b18798ac6669`;
sample-stream SHA256:
`de873a9ea06813ab7944745d086a244386c26e29897692acedc7ca1ed6a7bc2f`;
label SHA256:
`7308ceb11d035d02c510fa798c276d8eff04b5f7d3872dd20f13e2dbd5134ca5`;
result SHA256:
`665821747854261244a3569ed3b4f260b461d7d6e7a003de863bb2349742ca04`.

## P1 execution result: aligned labels do not improve the gate

The bundle from
[`dyyfk/minicpm-o45-native-gate-data`](https://huggingface.co/datasets/dyyfk/minicpm-o45-native-gate-data)
was downloaded without the duplicate zip and verified against every entry in
`SHA256SUMS`. The manifest SHA256 is
`11b73100bac37e7ddd91ab05a9d18b4e3e34798bf043ffd9541b06660a6abf90`;
it identifies source commit `03bf0e4`, the 12288-dimensional feature recipe,
and the official serving configuration. All 5228 declared training rows joined
to a native label (4986 core + 242 fresh train).

The author's supplied native-label artifact reports row-random 5-fold OOF AUC
`.8477`; an independent rerun reproduced `.847674`. The stricter source-grouped
5-fold sweep selected the same `C=3e-4` but measured `.8346`, a useful estimate
of the family-transfer penalty. The independent full-data coefficients and the
supplied artifact were effectively identical (cosine similarity `>.999999`),
and produced the same exact-budget cascade decisions.

Official-native evaluation, old deployed gate versus aligned refit:

| pool | n | native AUC old | native AUC aligned | delta (95% bootstrap CI) | cascade delta @15/30/50% |
|---|---:|---:|---:|---:|---:|
| internal test | 239 | .878 | .876 | -.002 [-.025, +.021] | +.000 / -.008 / +.008 |
| Speech TriviaQA | 250 | .800 | .811 | +.011 [-.018, +.042] | -.012 / -.024 / +.016 |
| Speech WebQ | 241 | .786 | .772 | -.015 [-.049, +.020] | -.021 / -.012 / -.004 |
| Llama Questions | 245 | .758 | .742 | -.015 [-.052, +.022] | -.016 / +.008 / +.000 |
| SD-QA | 200 | .763 | .726 | -.037 [-.076, +.000] | -.005 / +.000 / +.000 |
| Reasoning zh | 202 | .666 | .662 | -.004 [-.057, +.049] | +.015 / -.005 / -.020 |

Across the five external pools, mean native AUC changes from `.755` to `.743`.
Mean cascade accuracy changes by `-.8`, `-.7`, and `-.2` points at 15%, 30%,
and 50% budgets. No pool has a statistically reliable AUC improvement. This
fails the predeclared replacement gate (at least +.015 external AUC or +2
points at 30% plus broad pool-level support), so the deployed artifact should
remain unchanged. The native-label artifact is valuable as a provenance- and
calibration-correct candidate, not as a performance upgrade.

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
