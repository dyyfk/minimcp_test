<!-- Reviewer: Claude Fable 5.1 (thinking-xhigh, Cursor subagent, 2026-09-01). Input: NVDA_PROBE_TRAINING.md + figs + probe_lab.py. Verbatim. -->

## 1. Critique of the current training

**Methodological weaknesses**

- **Pool identity is most of the headline number (§6.1, fig 8).** A predictor that outputs only the pool fail rate scores AUC **.752** on the calibration set. Deployed OOF .820 is therefore +.068 over pool identity; mean within-pool AUC is **.713** (OOF) and **.687** (LOPO). Fig 8 also plots trap at AUC .17 (n=150, 2 negatives) while the text says ".58–.88" — say explicitly that trap is degenerate and excluded. Reviewer will demand within-pool mean and LOPO as headline numbers next to pooled OOF.
- **The deployed read is not "at commit" (§3.1, §3.4, fig 1; constraint 3).** `onset_last` is the state at commit+7 after the model has consumed 7 of its own answer tokens. My per-frame sweep (frame k ‖ user_mean, L30) shows OOF flat (.808–.816) but **external AUC rising monotonically with k** (.779 at k=0 → .797 at k=7). The transfer advantage of onset over eot/commit reads comes from reading the model's own first 560 ms of answer. That is legitimate engineering, but the claim "predicts whether the model *will* answer wrongly" should be "reads the first 560 ms of the answer", and the escalation must barge in on a model already speaking. Also `H_eot` and `H_mean` include post-commit frames for the 44 % of queries that commit early (fig 7), so none of the three features is strictly pre-commit.
- **Train/serve skew on `user_mean` (likely bug).** Offline `H_mean` = mean over `[prompt_len, t_end)` (§2.2). Online `nvda_duplex_probe.py` averages **PAD frames before commit**: it includes silent-tail frames when commit > t_end (56 % of queries) and excludes overlap frames when commit < t_end (44 %). Median gap is 3 frames so the typical error is small, but for the early-commit cluster (196 rows commit >3.2 s before audio end) the two means differ substantially. `user_mean` contributes +.004 OOF / +.004 ext in the stack (ablation: `last ‖ mean8` = .8156 / .8041), so a skew larger than that erases its value. Verify by replaying a few hundred queries through the online state machine and diffing features.
- **The −6 s cluster in fig 7 is unexplained.** ~196 rows (7.9 %) commit >3.2 s before the audio ends. Fail rate .658 vs .723 for the rest. Are these barge-ins, filler tokens ("Sure"), or marker runs that pass the ≥3-non-PAD rule? If the latter, the commit definition is wrong for 8 % of data and the onset window sits mid-utterance. Report the first 3 tokens of these rows and a robustness fit without them.
- **No-commit rows (69, §1.1) are dropped from every fit and every metric.** Silence is a deployment failure the gate cannot catch. Report their fail rate and count them as missed escalations in the selective-risk numbers.
- **Statistics (§4.3, §5.1, fig 6).** External-mean bootstrap SD is **.015** (400 reps), so .781 → .808 is ~1.8 SD and no paired test is given. OOF CV-split noise is .0004 (5 seeds); the doc's "noise at n=2,481" should quote the sampling SE (~.009), not seed jitter. "≈+.013 AUC per 1k rows" is a two-point slope; my learning curve (25/50/75/100 %: .778 / .793 / .800 / .808) is concave — the last 620 rows bought +.008. Data is *a* lever, not *the* lever.
- **§4.3 multi-layer claim did not reproduce.** Concatenating the full stack at L26/30/34 with C=1e-4 gives OOF **.821** / ext .813 in my run vs .8113 in the hill-climb. Check what `hillclimb.json` variant C actually concatenated (last-only? unscaled?).
- **Hyper-parameter selection by random 5-fold (§3.1)** rewards pool memorisation. LOPO happens to also pick C=1e-4 (3e-5 .755, 1e-4 .770, 3e-4 .769), so no damage, but the protocol should be LOPO by design. The onset-vs-eot pick was within OOF noise (.820 vs .812) and the external gap was reported afterwards — state the timeline explicitly to pre-empt a constraint-6 challenge.
- **Head family was never varied (§3.1, §6.2).** With z-scoring and C=1e-4, LR is essentially a shrunken class-centroid direction; the monotone C curve says covariance is ignored, not that the recipe is "at ceiling". A shrinkage-LDA head matches or beats it and is non-monotone in its shrinkage (below).
- **Act head (§3.3).** OOF .9999 separating 2,258 single-voice TTS questions from 196 two-voice en+zh stims is a confound red flag (voice / language / duration, not dialogue act). Rebuild stims with the same voices and test on short questions vs long backchannels.
- **Labels (§1.3, §6.5).** Fail rate .72; "fail" conflates refusals, hedges, wrong facts, garbled speech. Judge κ on identical transcripts is unmeasured. Report a fail-type breakdown; the probe may be a hedge detector.
- **Frozen `test`-240 rows are inside the n=2,481 cohort** used for sweeps and external transfer (§1.1). Fine for externals; make sure no internal-test number in the paper is produced by a model that saw them.

**What is fine:** scaler inside the CV pipeline (no fold leakage); externals scored once, cold; thresholds from OOF quantiles; labels from the model's own text channel; capture cacheless so read points are exact.

**What the doc does not show and a reviewer will ask for:** within-pool / LOPO headline metrics; CIs and paired bootstraps in the §5.1 table; precision and coverage per tier (fig 9 shows correct answers spread uniformly up to .75, so precision at the 50 % tier is low); external fire rates at fixed thresholds (§6.7); fail-type breakdown; the early-commit and no-commit rows; a learning curve.

## 2. Ranked proposals

All computations use `probe_lab.Calib` (`E_on`, `E_eot`, `M`, `y`, `meta.pool`) and `Ext.pools[t]`. Selection is always OOF or LOPO on calibration only. Ext SE ≈ .015, so single-variant gains below .01 are directional.

1. **Multi-layer score ensembling (logit averaging of per-layer heads).** For L in `[22,26,30,34,38]` (fixed a priori: the fig 3 plateau, and the only layers on disk externally): fit `deployed_clf()` on `stack(E_on, M, cal.j(L))`; score = mean over L of `decision_function / std_train`. Averaging linear heads is one linear head on the 5-layer concatenation (67,200 dims), so constraint 5 holds and export stays `w, b`. Measured: LR ext **.813**, LOPO .773, OOF .8225; ShrinkLDA ext **.816**, LOPO .780. Expected +.005–.01 ext: per-layer heads carry partly independent nuisance noise. Cost: 5 hooks online (capture already has them), 5× weights. Kill: LOPO gain <.005 over 5 seeds. Brushes constraint 5 in spirit only. No re-capture.
2. **Shrinkage-LDA head, dual/Woodbury form.** On z-scored X: `w = (S + λI)^{-1}(μ1−μ0)`, `S` = pooled within-class covariance, computed via the n×n Gram matrix (`ShrinkLDA` in my runs, ~1 s). λ by OOF on calib over `{3,10,20,30,50,100,300}` → 30 (OOF-argmax, no external peeking). Measured λ=30: OOF .8177, LOPO **.774**, ext **.811**; non-monotone in λ. Reasoning: LR at C=1e-4 ignores covariance; LDA whitens shared nuisance directions (voice, prosody, length), which is what shifts across pools — sllama (fail .33) gains most. Cost: nil; closed form, deterministic. Kill: OOF-selected λ yields LOPO < .770. Constraint 5: still linear.
3. **Add the strictly-at-commit frame to the stack.** `X = [E_on[:,j,0], E_on[:,j,-1], E_on[:,j].mean(1), M[:,j]]` at L30 (17,920 dims), same estimator. Measured: OOF .8201 (=), LOPO **.777** (+.007), ext .811 (+.003). Also gives a constraint-3-clean fallback score at t=commit (`E_on[:,j,0] ‖ M`: OOF .808, ext .779) for latency-critical use. Cost: nil; online already holds `onset[0]`. Kill: LOPO gain not replicated across seeds.
4. **Combine 1–3: 5-layer logit-average of ShrinkLDA(λ=30) on the 4-block stack.** Measured: OOF .8215, LOPO **.787**, ext **.819**; paired bootstrap vs deployed on the same external rows: +.010 ± .005 (98.6 % of reps > 0). Disclosure: I saw external numbers while iterating; the team should re-derive the selection on LOPO only. Each component is selectable on calibration alone (λ by OOF, layer set fixed, stack by LOPO). Kill: re-derived LOPO-selected config gives ext < .812.
5. **Make LOPO (or leave-one-tag-out) the selection criterion and a headline metric.** `lopo_scores` per config (14 fits, ~7 s). Report pooled OOF, mean within-pool AUC, LOPO. Not a gain by itself; it changes what future sweeps select (per-frame sweep: OOF flat while ext moves .02). Cost: nil.
6. **Fix `user_mean` to match the online path (re-capture).** Redefine `H_mean` as the mean over pre-commit PAD frames `[prompt_len, commit)`, which is what `nvda_duplex_probe.py` computes and needs no VAD. Requires re-capture (~4.3 s/query, ~4 h GPU for 3,500 queries), or drop `user_mean` from the deployed stack now at −.004 ext. Expected gain: 0 to +.005 on paper, but removes an unmeasured deployment loss. Kill: replay diff shows online and offline features match to <1 % for >95 % of queries.
7. **Re-capture per-frame pre-commit trajectories at L22–38 (needs re-capture).** Store all PAD-frame states before commit (~60 frames × 5 layers × 4,480 fp16 ≈ 2.7 MB/query). Enables: correct `user_mean`; a commit-relative "last-k-before-commit" window (a VAD-free, constraint-3-clean replacement for `H_eot`); exponentially-weighted means; frame deltas. Expected +.005–.01, mainly by making the measurement valid (§6.3). Kill: commit-relative window no better than `H_eot` under LOPO.
8. **Expand calibration toward the external shape, not more hard pools.** The learning curve is concave (+.008 for the last 620 rows) and the weakest external (sllama) has fail rate .33 while calibration has .72; fig 9 shows correct-answer scores spread uniformly. Add ~1,000 easy, open-ended factual/conversational queries the model tends to get right. Cost: ~1.2 h GPU + judge. Expected +.005–.01 ext, larger on sllama. Kill: within-pool AUC on the new pool <.65 and ext unchanged.
9. **Measure judge κ; relabel only if it is poor.** Re-judge ~300 identical transcripts 3×; if κ<.8, majority-label the calibration set. Two-pass fractional labels gave +.0001, so this mostly bounds the ceiling. Cost: API only.
10. **Report the operating points.** Precision/coverage per tier, external fire rates at fixed thresholds, selective-risk curves including no-commit rows. No AUC change; it is the deliverable a reviewer grades.

## 3. What not to do

- **Pool- or class-balanced sample weights, or projecting out pool-mean directions.** Measured: pool-class weights OOF .792 / LOPO .767 / ext .781; null-projection k=5 ext .803, k=13 .774. The base-rate signal is partly real difficulty; removing it lowers transfer.
- **More regularisation or unsupervised dimensionality reduction.** C=3e-5 ext .798; PCA-256 → LR ext .784; bagging 10× LR ext .803. The discriminative directions are low-variance; PCA and heavier shrinkage discard them, and the model is already low-variance.
- **Score-ensembling across reads (onset+eot+commit).** Ext .800 < .808; dual-read concat .807. The eot and commit reads are weaker and dilute the onset read.
- **Non-linear heads, robust scaling, tricks on the L30 stack.** MLP-64 (doc) .801; RobustScaler+clip ext .806. No gain and constraint 5 risk.
- **Selecting on external numbers, including the layer set.** External SE is .015 per variant; chasing it is noise fitting and breaks constraint 6.

## 4. My quick runs (own CPU runs, seed 42, single fit each; ext SE ≈ .015)

| variant | OOF | within-pool mean (OOF) | LOPO | ext mean (striviaqa/swebq/sllama/sdqa) |
|---|---|---|---|---|
| pool base rate as score (oracle) | .752 | — | — | — |
| **deployed onset@L30, LR C=1e-4** | .8201 | .713 | .770 | **.808** (.838/.851/.771/.774) |
| commit-frame ‖ user_mean @L30 | .808 | .701 | .756 | .779 |
| commit-frame ‖ user_mean @L38 | .809 | .697 | .756 | .778 |
| onset frame k ‖ user_mean @L30, k=0…7 | .808–.816 | — | — | .779 → .797 (monotone in k) |
| commit ‖ last ‖ mean8 ‖ user @L30 | .8201 | .716 | .777 | .811 |
| traj (commit, last−commit, mean8−user, user) | .8197 | .714 | .775 | .808 |
| halves mean(f0–3) ‖ mean(f4–7) ‖ user | .8235 | .722 | .776 | .805 |
| frames8 ‖ user (40k dims) | .8203 | .722 | — | .805 |
| last ‖ mean8 (no user_mean) | .8156 | .706 | — | .804 |
| user_mean only | .8024 | .691 | — | .757 |
| C=3e-5 / C=3e-4 | .814 / .816 | .707 / .713 | .755 / .769 | .798 / .807 |
| pool weights / pool+class weights | .818 / .792 | .714 / .714 | .773 / .767 | .802 / .781 |
| null-proj pool dirs k=5 / k=13 | .742 / .713 | .711 / .698 | .699 / .679 | .803 / .774 |
| RobustScaler + clip 5 | .8185 | .711 | .766 | .806 |
| ShrinkLDA λ=3/10/20/**30**/50/100/300 | .792/.812/.817/**.818**/.816/.811/.798 | — | .756/.772/.775/**.774**/.771/.763/.742 | .799/.811/.813/**.811**/.807/.797/.777 |
| ShrinkLDA λ=30 on 4-block stack | .8178 | .712 | .781 | .813 |
| ens LR + ShrinkLDA @L30 | .8195 | .712 | .774 | .810 |
| bag10 LR C=3e-4 | .8176 | .710 | — | .803 |
| layer-mean stack L26/30/34 | .8218 | .716 | .773 | .811 |
| logit-avg LR × 5 layers L22–38 | .8225 | .716 | .773 | .813 |
| logit-avg ShrinkLDA × 5 layers | .8201 | .712 | .780 | .816 |
| logit-avg LR × 3 layers L26/30/34 | .8231 | .717 | — | .814 |
| concat 3 layers L26/30/34, C=1e-4 / 3e-5 | .821 / .823 | .718 / .716 | — | .813 / .813 |
| logit-avg 3 reads onset@30+eot@34+commit@38 | .8202 | .719 | .769 | .800 |
| concat dual read onset@30 ‖ eot@34 | .8211 | .717 | — | .807 |
| PCA-256 → LR C=1e-2 | .8072 | .705 | — | .784 |
| **combo: ShrinkLDA(30) × 5 layers × 4-block stack** | .8215 | .716 | **.787** | **.819** (.831/.854/.794/.797); paired Δ vs deployed +.010 ± .005 |
| learning curve 25/50/75 % of calib (ext only, 3 seeds) | — | — | — | .778 / .793 / .800 |
| noise: deployed OOF over 5 seeds | .8211 ± .0004 | — | — | ext-mean bootstrap SD .015 |

Timing-only features (onset frame, n_frames, gap) reach OOF .58 but LOPO .35 — commit timing does not transfer and should not be added as a feature.
