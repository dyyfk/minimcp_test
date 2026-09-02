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

Later local-only iterations found that independent answer-conditioned signals
do transfer: the frozen P3a + two-sample semantic entropy + RTJ fusion improves
five-pool mean native AUC by `.0334`. It remains shadow-only. A fixed latency
benchmark shows the two semantic generations cost median `7.21s` and p90
`25.52s` before RTJ, making the current formulation too slow for synchronous
activation despite its offline ranking gain. Distilling that teacher back into
the P3a hidden features recovers a `+.0213` mean external native-AUC gain with
no extra serving pass; this is now the preferred shadow candidate.

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
embedding.

A cached latency ablation then rebuilt and reselected the candidate from only
the first one or two stochastic answers; it made no new model or API calls.
One extra answer selects entropy at cosine `.78` with blend `.25`: external
native AUC `+.0151`, benefit AUC `+.0068`, and cascade
`+.0034/+.0022/+.0078`. Two extra answers select entropy at `.70` with blend
`.25`: native AUC `+.0240`, benefit AUC `+.0162`, and cascade
`-.0008/+.0099/+.0050`. The two-sample variant is the quality/latency knee: it
beats the three-sample candidate on mean native AUC, improves four of five
pools, and has reliable native gains on both SD-QA (`+.0426`, 95% CI
`[+.0044,+.0833]`) and WebQ (`+.0491`, `[+.0234,+.0789]`). It gives up only
0.15 point of mean 30%-budget cascade accuracy versus three samples. Prefer
two samples for a direct implementation; next test threshold-band adaptive
second sampling to reduce its average generation count.

The threshold-band adaptive follow-up is rejected. It swept second-sample
fractions `0/.1/.2/.3/.5/.75/1` and selected the smallest fraction within
`.001` routing objective of the training-pilot best. The objectives are
`.1730/.1720/.1730/.1733/.1737/.1737/.1737`, so zero second samples already
falls within tolerance. This reduces exactly to the one-sample candidate and
its weak 30%-budget external gain (`+.0022`). A balanced-budget-only selector
would prefer a small gray band, but introducing a different post-hoc objective
after opening the external result is not justified. Keep the clear static
choice: one sample for minimum latency, or two for ranking quality; do not add
adaptive serving complexity on this evidence. Adaptive receipt SHA256:
`7ac935c42efa2961d489c55b4685975c9af7c468bc23816f602187e184cfece5`.

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
The one- and two-sample result SHA256 values are respectively
`9ddf2f53e35efa3e821bc659f88cc600f55c68ba9251caa38c0d22fd80691f04`
and `b9814205309778897397363f24680bb34210a45acbbe8874ee07e29b5c22815a`.

### P6: text p(True) adds ranking signal, but is a transcript ceiling

The next local iteration collected the repository's exact first-token
p(True) signal on the known query text for the 1,000-row training pilot and
1,138-row external set. All 2,138 rows completed without error; external
self-assessment averaged 52 ms per row after model load. On its own, negative
p(Yes) has `.7175` mean external native AUC. The source-grouped frozen p(True)
fusion improves native AUC by `+.0188`, benefit AUC by `+.0080`, but cascade
accuracy only by `-.0040/+.0008/+.0074` at 15/30/50%.

P(True) and two-sample semantic uncertainty are complementary. A predeclared
fixed three-way grid selects `.50` P3a score + `.25` semantic entropy + `.25`
text p(True), reaching training-pilot native AUC `.8299`, benefit AUC `.7035`,
and routing objective `+.1703`. Externally it is the strongest ranking result
so far: mean native AUC `+.0311`, benefit AUC `+.0179`, and cascade
`+.0033/+.0077/+.0070`. Native gains are reliable on SD-QA (`+.0616`, 95% CI
`[+.0137,+.1101]`) and WebQ (`+.0474`, `[+.0152,+.0812]`); TriviaQA and Llama
also improve, while Reasoning-zh regresses (`-.0130`). This supports the
roadmap's predeclared English-only p(True) guard, but does not improve the
30%-budget cascade metric over semantic sampling alone.

This is not yet a deployable RTJ result because the p(True) prompt sees the
ground-truth query text. A one-row audio-transcribe-then-judge smoke produced
an exact transcript and numerically identical p(True), but cost 25.7 seconds,
far above the 52 ms text-only diagnostic. Run a small stratified RTJ parity
and latency sample before paying that cost on all rows. Text-p(True) result
SHA256: `c759a0c273b48a26cbd0074ce42588e7ae223b502bc616479c50d2b3db6ed1ab`;
three-way result SHA256:
`bd4bbee4704bb55d560d9134c8ccdf4132969e3e7dbb6b4014232e23d598db0b`.

### P7: local repeat-then-judge validates and strengthens the candidate

A fixed 50-row parity sample first tested the complete audio-to-transcript-to-
p(True) path. Normalized transcripts exactly matched the query on 78% of rows
and had mean character similarity `.954`; RTJ and ground-truth-text p(True)
had Pearson correlation `.893`, median absolute difference `.011`, and mean
difference `.071`. Median end-to-end latency was `.80s`, mean `1.52s`, and
p90 `4.48s`; the initial 25.7-second smoke was a long-tail outlier. Some ASR
instructions elicited an answer rather than a transcript, confirming that the
text ceiling could not substitute for the actual RTJ pass.

The full fixed 1,000/1,138 train/external run then completed with zero errors
and no OpenAI calls. External latency after model load was median `.62s`, mean
`.90s`, p90 `1.56s`; RTJ/text-p(True) correlation was `.909` with mean absolute
difference `.058`. RTJ alone fused with P3a improves five-pool mean native AUC
by `+.0216`, benefit AUC by `+.0117`, and cascade accuracy by
`-.0030/+.0066/+.0105`. Native gains are reliable on SD-QA, Llama Questions,
and TriviaQA, while Reasoning-zh is nearly flat rather than suffering the text
ceiling's larger inversion.

The same predeclared three-way grid selects `.50` P3a + `.25` two-sample
semantic entropy + `.25` RTJ. This is the strongest overall candidate:
five-pool mean native AUC `+.0334`, benefit AUC `+.0202`, and cascade
`+.0005/+.0097/+.0094` at 15/30/50%. Native AUC improves on all five pools;
SD-QA gains `+.0551` (95% CI `[+.0057,+.1028]`) and WebQ `+.0435`
(`[+.0105,+.0787]`). It clears the aggregate ranking gate with broader support
than direct semantic uncertainty alone, though its extra inference cost and
sub-two-point 30% cascade gain still argue for a shadow deployment rather than
immediate replacement.

RTJ sample-stream SHA256 values: train
`be0e4a60565253021accd43ecedbbbeecd0018eb08c79ac76ff561828b494d5d`,
external
`b17b444fc5a867835690c98597115dcdb3b4838d1f933e961ed7a897c6e9fe90`.
RTJ-only result SHA256:
`94d42c74b88741c1d9b5315d662c66bd8ab0d00d743959cd483a2a084700120d`;
RTJ+semantic fusion result SHA256:
`396de32461de11111071008f1729262c6f5b21401d72ecce6a25413d0477ad51`.

The frozen scorer is exported as `data/gate_shadow_semantic_rtj.json`, SHA256
`1bba247f7565107e2ffe1f51b3ef445cd8a4287614b77d3619fe859ed16a5542`.
It contains the fitted 8,192-dimensional P3a coefficients, all three training
centers/scales and fusion weights, exact semantic/RTJ recipes, validation
receipts, fixed-set latency evidence, a per-language rolling-quantile
requirement, and required shadow logs. The artifact is explicitly marked
`shadow_only` with activation prohibited; it does not overwrite or alter
`gate_native.json`.

### P8: fixed-set latency rules out synchronous activation

The semantic generator now records per-sample and per-row elapsed time. I
reran exactly two stochastic answers on the frozen 50-row RTJ parity sample
(10 rows per external pool) using eight B300 ranks. All 50 rows completed with
zero errors; model loading and embedding time are excluded. Two-sample native
generation measured median `7.21s`, mean `10.25s`, p90 `25.52s`, and maximum
`30.27s`. The long tail is workload-dependent: SD-QA's ten rows have median
`25.02s`, while Llama Questions, TriviaQA, and WebQ medians are `5.46s`,
`5.98s`, and `5.11s`.

Adding the matched RTJ measurements gives a serial capacity estimate of median
`8.14s`, mean `11.77s`, and p90 `30.08s`. Even an optimistic three-replica
estimate—two semantic samples and RTJ run concurrently—has median `3.93s` and
p90 `22.22s`; this is an estimate from matched branch timings, not a measured
concurrent serving run. The result strengthens the existing shadow-only
decision: keep collecting scores asynchronously, but do not put the P7 fusion
on the synchronous user path. The next optimization target should be
distillation of the answer-conditioned signal into a single pass, not more
sampling or serving parallelism.

Latency receipt: `figures/shadow_latency_benchmark.json`, SHA256
`5888a5ce615bc91c00f3d0a3790f0340f88ce2cec0f2ef7a012ac28323cf5bb1`;
semantic stream SHA256
`3eece7d4811a73e1152acf11e105f83cc59e8f4d68c5c3eb03d5a4224528d78c`.

### P9: single-pass distillation clears the mean-AUC gate

To remove the synchronous multi-generation cost, I distilled the frozen
answer-conditioned teacher—equal-weight standardized two-sample semantic
entropy and RTJ uncertainty—into the existing hidden features. Selection used
only the fixed 1,000-row pilot, five source-family folds, three input block
sets, ridge `alpha={100,1000,10000}`, and fixed blend weights. The external
pools were evaluated only after this grid froze its winner.

The direct teacher is strong on the pilot (`.7882` native AUC, `.6803` benefit
AUC), and the hidden-state ridge predicts it with OOF Spearman `.6503`.
Selection chooses P3a's `eot_mean8 + user_mean` blocks, ridge `alpha=100`,
blended `.25` with the P3a score. Against the frozen live 8bq artifact, the
single-pass candidate improves five-pool mean native AUC by `+.0213`, benefit
AUC by `+.0151`, and cascade accuracy by `+.0088/+.0080/+.0072` at
15/30/50%. Native AUC improves on all five pools: SD-QA `+.0347`, Llama
`+.0091`, Reasoning-zh `+.0283`, TriviaQA `+.0136`, and WebQ `+.0209`.
SD-QA's interval excludes zero (`[+.0013,+.0683]`).

The first P9 receipt mistakenly compared the candidate against the committed
legacy-label/global-threshold artifact (SHA256 `b0bf2eaa...`) rather than the
live 8bq artifact (SHA256 `0e6494c2...`), understating the gain. The script now
fails closed unless the latter exact SHA is supplied. Candidate scores and the
training-only winner did not change; only the baseline comparison was
corrected.

This is the best latency/quality Pareto point so far: the answer-conditioned
teacher is needed only offline, and its two linear components fold into one
8,192-dimensional coefficient vector with no extra serving pass. Keep it
shadow-only until online score calibration and routing lift are observed; do
not overwrite the live gate yet. Corrected result SHA256:
`d9833b641691059d2a515788080e2b37c7349f05fa73ac070bb2bf493e4278f0`.
The exported artifact is `data/gate_shadow_distilled_semantic_rtj.json`,
SHA256 `c85e0697788b2f8ce819fc963aa68c5a5ef34e0ae59c4f58b7917ccbf848dbb0`;
its algebraic fold differs from the two-component scorer by at most `1.8e-6`.

### P10: leave-one-family-out robustness supports freezing P9

The 1,000-row teacher set spans 33 source families. Holding out each family in
turn from the ridge fit, without changing the frozen blocks, alpha, blend, or
external evaluation, gives a narrow stability envelope:

- mean native-AUC delta: minimum `+.0151`, median `+.0211`, maximum `+.0240`;
- mean benefit-AUC delta: minimum `+.0097`, median `+.0148`, maximum `+.0174`;
- mean balanced-cascade delta: minimum `+.0019`, median `+.0073`, maximum
  `+.0108`.

All 33 leave-one-family-out fits still clear the `+.015` mean native-AUC gate,
and all 33 improve native AUC on every external pool. The worst native-AUC
cases omit `know-openbook` (`+.0151`) or `know-longtail` (`+.0159`), so the
result is not driven by one small family. Robustness receipt SHA256:
`78dcc9f6bef45c81e8ef0f81d78dfd679a9097e91c1f48f245aa65fcf928a5bf`.

This supports freezing the P9 artifact for shadow evaluation. It does not
create a new independent test set: all robustness fits are still scored on the
same five external pools. Stop offline candidate tuning on those pools; the
next acceptance evidence must come from new source-disjoint data or live
shadow traffic.

### P11: inactive demo shadow wiring

`demo_duplex.py` now packages and loads the frozen distilled artifact, checks
that it is explicitly `shadow_only` with `activation_prohibited: true`, and
computes its score from `eot_mean8 + user_mean` beside the live score. The
shadow value is emitted in existing websocket score/gate events for
observation. It has no threshold and is not referenced by the `fired`
expression, so it cannot change escalation behavior. The live artifact,
threshold selection, dialogue-act guard, and expert path are unchanged. This
is code-level plumbing only; it has not been deployed.

### P12: executable shadow-safety contract

`src/test_distilled_shadow.py` makes the P11 safety boundary executable. It
checks the artifact's shadow/activation flags, lack of deployable thresholds,
frozen 8,192-dimensional feature recipe, and compatibility with the production
pure-Python `Probe`. It also parses `demo_duplex.py` and fails if any shadow
name or attribute enters the sole live `fired` assignment, while requiring the
observational websocket fields to remain present. The original 28 gate tests
and all 11 distilled-shadow checks pass.

### P13: fail-closed shadow acceptance receipt

`scripts/52_analyze_distilled_shadow.py` consumes only new shadow JSONL and
the frozen artifact. It requires unique IDs and complete language, live score,
distilled score, latency, realized action, local outcome, and expert outcome
fields; it rejects unsafe artifacts, incomplete rows, duplicates, non-binary
outcomes, and undersized samples. The receipt reports native/benefit AUC,
per-language exact-budget cascade accuracy and harmful escalation, observed
policy accuracy, latency, score drift, and hashes of both inputs. It has no
activation path. A synthetic end-to-end smoke passes.

### P14: one-shot FreshQA heldout is directionally positive

After freezing P9--P10, I opened the 60 feature-bearing FreshQA heldout rows
for the first and only time (30 stable `fresh_never`, 30 fast-changing
`fresh_fast`). These rows were excluded from the 5,228-row base fit, the
1,000-row teacher fit, and all prior external evaluations. The evaluator pins
both artifact SHA256 values and permits no selection or retuning.

On actual official-native failures, AUC changes `.8193 -> .8400` (`+.0207`,
95% bootstrap CI `[-.0278,+.0802]`). On the deployed policy target, which
forces all fast-changing rows positive, AUC changes `.8727 -> .8975`
(`+.0248`, CI `[-.0229,+.0852]`). Exact 15/30/50% failure precision and recall
are tied because the two rankings overlap on 93--97% of rows at this small
sample size. The result is supportive and directionally consistent with P9,
but not individually significant; FreshQA family overlap and `n=60` mean it
does not replace a new source-disjoint shadow set.

Receipt SHA256:
`1bf1ee5ac2da315192f6b2b0a782047ee33598c522b7422d7f20bb57dddfddd5`.

### P15: prospective source-disjoint validation frozen

Before generating any audio, native answer, hidden feature, or label, I froze
400 new public validation rows: 200 from WinoGrande-debiased validation and
200 from SciQ validation, seed 45. Neither source appears among the 33 teacher
families; normalized source questions have zero exact overlap with all bundled
training/evaluation query metadata. SciQ answer positions are deterministically
permuted to avoid a fixed correct option.

Ordered ID-list SHA256:
`301bdd18e3e03e54a2969f2eaecf3350e4c5102b5fcdc0fd423715cabeb52968`;
canonical content SHA256:
`9295e1a47bd493af7f14dfb35d4bf395ab3d58fafd17fd465b58efb3b2f61f9a`.
The 60,104 rendered characters imply `$0.90156` at the recorded TTS rate.
The candidate and comparison protocol stayed frozen; this set is validation
only and was not used for refitting or candidate selection. TTS completed
400/400 with no errors. Official-native inference then completed 400/400 with
no errors, no no-speak cases, no empty answers, no missing end-of-turn markers,
and finite 12,288-dimensional features for every row.

The exact repository ref-anchored judge marked 68.5% of answers adequate
(31.5% failure). On the full prospective set, live-to-candidate native-failure
AUC changes `.7961 -> .8059`: `+.0098`, bootstrap 95% CI
`[-.0062,+.0258]`. Exact-budget failure precision changes `+.0500`, `+.0083`,
and `-.0050` at 15%, 30%, and 50%, respectively; ranking agreement is
92.0--94.5%.

The source split is not broad enough for activation:

| frozen source | n | failure rate | live AUC | candidate AUC | delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| SciQ | 200 | .095 | .7962 | .7839 | -.0122 [-.0669,+.0449] |
| WinoGrande | 200 | .535 | .5672 | .5945 | +.0273 [-.0087,+.0638] |

Thus P15 is directionally positive in aggregate and at the two lower budgets,
but it fails both predeclared prospective conditions: the aggregate interval
includes zero and SciQ regresses. The distilled candidate remains
shadow-only/activation-prohibited; the live gate remains unchanged. Result
receipt SHA256:
`6d642b9eca72bdc54f2dee2187beb1909ad4fbcc895dbe57be59f3f179579794`.

Judging used 122,934 input and 31,086 output tokens with zero errors. At the
published standard `gpt-5.4-mini` rates checked 2026-09-02, that is `$0.23209`;
conservative cumulative OpenAI spend is `$28.95545 / $500`.

### P16: conservative ensemble transfers broadly and significantly

After P15 became development data, a fixed grid selected the smallest blend
that retained at least `+.015` mean AUC on the five old external pools while
improving both P15 sources. The winner, `z(live) + 1.0*z(distilled)`, folds
exactly into one 12,288-dimensional dot product (maximum algebraic discrepancy
`4.4e-15`). Its frozen shadow-only artifact SHA256 is
`4d75a506a59e6206e4687a6b3630b1dae54034f902d9136112689c9091eeec15`.

Only after freezing that artifact, I selected 450 untouched rows from three
additional repositories absent from the teacher families: BoolQ, HellaSwag,
and QASC (150 each). IDs/content were frozen before TTS/native/judge outputs;
the selection SHA256 is
`39d18e50d08c5f2c85e23a343c43a914f4791574ee91e1a70a3ee8b2d714c40c`.
Official-native capture completed 450/450 with finite features and no inference
or no-speak errors; one answer reached the 60-chunk cap without an end marker
and remained subject to the same strict judge.

| frozen source | n | failure rate | live AUC | ensemble AUC | delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| BoolQ | 150 | .180 | .6763 | .6841 | +.0078 [-.0078,+.0256] |
| HellaSwag | 150 | .367 | .5860 | .6115 | +.0255 [+.0019,+.0496] |
| QASC | 150 | .427 | .6770 | .6864 | +.0094 [-.0111,+.0301] |

The macro source delta is `+.01424`, source-stratified bootstrap 95% CI
`[+.00280,+.02635]`; the pooled delta is `+.01325`, CI
`[+.00323,+.02307]`. All three sources improve and both aggregate intervals
exclude zero. Exact global failure precision changes by `+4.41`, `+1.48`, and
`-0.44` points at 15/30/50% budgets. Thus the ensemble clears breadth and
statistical-support gates, but narrowly misses the original `+.015` macro AUC
replacement threshold. It remains shadow-only and the live gate is unchanged.
Result receipt SHA256:
`8a55fdc35fb376a23f560eeadaf65643fe44fe873f5087701ad8859bf6f67806`.

P16 TTS cost `$1.56219`; judge usage 147,854 input and 33,259 output tokens,
cost `$0.26056` at the published standard rates. Conservative cumulative
OpenAI spend is `$30.77819 / $500`.

### P17: stronger blend is not source-robust

After P16 became development-only, alpha 2 from the original blend grid
improved all P16 sources and moved their macro delta to `+.01775`. I froze the
corresponding one-pass artifact (SHA256
`a26829f06561e34cc957e437138ea9b03571c8a487e37ee22e91001ce5541be5`)
before selecting a third independent 450-row set: balanced SNLI, SST-2, and WiC
validation samples, with zero exact overlap against bundled queries, P15, or
P16. Official-native capture and judging both completed 450/450 with zero
errors, no no-speak cases, and complete end-of-turn answers.

| frozen source | n | failure rate | live AUC | alpha-2 AUC | delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| SNLI | 150 | .300 | .6193 | .6552 | +.0360 [-.0040,+.0788] |
| SST-2 | 150 | .153 | .8370 | .7925 | -.0445 [-.0874,-.0044] |
| WiC | 150 | .333 | .5838 | .6364 | +.0526 [+.0166,+.0922] |

The macro source delta is `+.01469`, source-stratified CI
`[-.00841,+.03906]`; pooled AUC changes by `+.01715`, CI
`[+.00117,+.03377]`. The pooled result is misleading for a global activation
decision: SST-2 has a statistically reliable regression that offsets strong
WiC and SNLI improvements. P17 therefore fails breadth, statistical macro
support, and the `+.015` macro replacement gate. Do not activate alpha 2.

This ends coefficient-only global blend escalation: increasing distilled
weight predictably improves several reasoning/word-sense sources but degrades
sentiment, and repeated global-alpha testing would overfit the prospective
sets. The alpha-1 P16 artifact remains the safest shadow candidate, not a live
replacement. P17 receipt SHA256:
`a091bb86477e4270eef43960252e3a9f726f319232ab428beb9b7ffca68c3db0`.

P17 TTS cost `$1.24167`; judge usage 139,290 input and 33,813 output tokens,
cost `$0.25663`. Conservative cumulative OpenAI spend is
`$32.27649 / $500`.

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

## Controlled multi-turn follow-up

P19 replayed 120 P16 targets after completed cross-source carrier turns. The
standalone-to-context score correlation was only `.473` for live and `.494`
for P16. P16's contextual macro AUC delta versus live was effectively zero
(`-.00019`, source-stratified 95% CI `[-.05894,+.06123]`) and failed source
breadth. This establishes that standalone ranking cannot be assumed to
transfer into an accumulated dialogue state.

P20 then tested a zero-forward context correction frozen on P19:
`current_P16 - 0.5 * prior_P16`. On an independent 120-session P17 replay,
119 targets produced scoreable onset features and one produced no onset. The
correction's macro AUC delta was only `+.00311` (95% CI
`[-.05081,+.05902]`), with SNLI `-.02344`, SST-2 `+.02165`, and WiC
`+.01111`; pooled AUC fell `.63246→.60965`. The prior-score subtraction is
therefore rejected. It adds no forward pass, but its apparent P19 gain did not
transfer. Keep P16 shadow-only and require organic first-turn/follow-up logs
before any activation decision.

P21 tested whether the representation could recover when trained directly on
the contextual distribution. It replayed 1,000 source-balanced official
targets after completed cross-pool carrier turns; 960 sessions produced
scoreable target-onset features (27 carrier non-EOT and 13 target no-onset
rows were excluded before fitting). The complete pre-output grid compared
two feature-block choices and five regularization values under target-pool
grouped cross-validation.

The selected `eot_mean8+user_mean`, `C=1e-4` probe is worse than live even on
grouped OOF: macro pool AUC `.79759→.78015` (`-.01744`, source bootstrap 95%
CI `[-.06856,+.02629]`) and pooled AUC `.86443→.80753`. Ten of 18 target
pools regress. No independent P22 set was opened because P21 failed its
training-stage screen. Direct contextual relabeling/refitting with the same
L22 linear representation is therefore rejected; the remaining gap is not
fixed by exposing this head to more unrelated prior-turn state.

P22 removed the main limitation of P19--P21: all 120 bilingual targets were
genuinely dependent on the preceding user turn. The frozen fixtures balance
English and Chinese across linked lookup, constraint selection, and state
update tasks. The repository's optional `calibctx` / `exp3zhctx` hooks were
not used because those rows are absent from both the exported 5,228-row bundle
and its deployed training recipe.

All 120 sessions reached a carrier EOT and a scoreable target onset; 117/120
target answers reached EOT. The native model failed 64/120 judged targets,
providing meaningful separation rather than a ceiling-only memory test. The
unchanged live gate outperforms P16: pooled AUC `.68583→.61942` (`-.06641`),
English `.75778→.70333`, and Chinese `.68552→.61652`. Macro pool delta is
`-.03681` with source-stratified bootstrap 95% CI `[-.08235,+.00366]`; five
of six pools regress. At exact 15/30/50% budgets, P16 pooled precision changes
by `-.0556/-.0556/-.0333`. P22 therefore rejects P16 for dependent follow-up
traffic and strengthens the case for leaving the live gate unchanged. The
fixture result remains controlled rather than organic evidence.

P23 tested the strongest remaining zero-forward hypothesis: reduce the two
accumulated-context feature blocks while preserving the current-user block.
The coefficient was frozen from P22 before opening a second independent set:
scale `eot_last` and `eot_mean8` (and the bias) by `.625`, leaving `user_mean`
unchanged. P22 development diagnostics predicted a macro pool AUC gain of
`+.02196` while keeping pooled AUC nonnegative versus live. The prospective
set comprised 120 new bilingual dependent conversations across disjoint
temporal-order, arithmetic-ledger, and seating-swap families.

All 120 sessions reached carrier EOT, target onset, and target EOT with zero
inference errors; 66/120 judged targets failed. Prospectively, however, the
context-block mix is essentially flat: pooled AUC `.61055→.61364` (`+.00309`)
and macro pool delta `+.00405`, source-stratified 95% CI
`[-.03188,+.04005]`. English improves `+.02602`, but Chinese regresses
`-.01004`; arithmetic regresses in both languages (`-.01333` English,
`-.03125` Chinese). It therefore fails the frozen macro-gain, positive-CI,
language-breadth, and minimum-pool gates. Reject it, keep the live gate
unchanged, and stop context-block coefficient tuning. The result receipt
SHA256 is
`b1917ab97285333188e11617e7e147927def9c88e7e0eb9eecfe24a5f1e8b36f`.
P23 TTS cost was `$0.295065`; judge usage was 33,514 input and 10,005 output
tokens, costing `$0.070158`. Conservative cumulative OpenAI spend is
`$34.052445 / $500`.

P24 directly tested whether the probe head was under-parameterized or merely
under-tuned. The frozen audit used nested source-family CV over all 5,228
official-native training rows and 37 source families. Five outer folds
estimated the whole selection process; each outer training split used three
inner source-family folds to choose among a standardized linear head and
residual MLPs with 32, 128, or 512 hidden units (approximately 0.41M, 1.59M,
or 6.30M parameters). The sweep also varied dropout, weight decay, learning
rate, BCE versus focal loss, and inner-only early stopping. The outer test
fold was never used to choose an epoch or hyperparameter.

Higher capacity is not the missing ingredient. The standardized linear head
has the best mean inner macro-source AUC (`.70701`); nonlinear heads are chosen
in only three of five outer folds and disagree on configuration. The complete
nested selection procedure scores `.67232` macro source AUC and `.79029`
pooled AUC, versus `.70482` and `.81270` for the prior grouped-OOF P3a linear
head: deltas `-.03250` and `-.02241`. It fails every frozen graduation rule,
so no new prospective set is opened and no larger head is packaged. This is a
zero-API-cost rejection, not evidence that training loss cannot be reduced;
it shows that added parameters do not transfer across source families at the
available sample size. Receipt SHA256:
`5a645a229889076017ec786bbbb9fd24aaa1e385d8bc6636c3571c63dc07e6a1`.
