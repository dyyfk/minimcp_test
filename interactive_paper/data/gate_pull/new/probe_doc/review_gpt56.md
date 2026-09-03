<!-- Reviewer: GPT-5.6 (Cursor subagent, 2026-09-01). Input: NVDA_PROBE_TRAINING.md + figs + probe_lab.py. Verbatim. -->

## 1. Critique of the current training

### Critical validity problems

1. **The deployed probe is not read at commit.** Section 3.1 uses `H_onset[:, J30, -1]` and the mean of all eight onset frames. Section 3.4 confirms that scoring occurs 640 ms after commit. These states can encode several generated answer tokens. This is target leakage and violates constraint 3. The only currently stored strictly-at-commit feature is `H_onset[:, j, 0]`.

2. **`H_mean` leaks future audio on 44% of queries.** Section 2.2 and Figure 7 show that 44% of commits precede `t_end`, but `H_mean` averages through `t_end`. A live gate cannot have those future positions at commit. Section 3.4’s claim that the online implementation reproduces training is therefore false for early commits.

3. **The external pools may no longer be cold.** Sections 4.2 and 5 compare several reads and recipes using external AUC, and the best external read becomes the deployed recipe. Unless onset@L30 was irrevocably selected before any external scores were viewed, this is validation-set selection. The chronology and preregistration are missing.

4. **OOF performance is selected and reported on the same folds.** Layers, reads, stacks, scaling, `C`, and 18 hill-climb variants were compared using the same five-fold OOF predictions (§4, Figures 3–5). Reporting the winner’s `.820` without nested CV is optimistically biased.

5. **Random stratified folds preserve the pool shortcut.** Figure 2 shows extreme pool base rates. Figure 8 shows within-pool AUCs of only `.58–.88`, including nearly uninformative balanced knowledge pools. Random folds test interpolation over known pools, templates, and TTS conditions—not transfer.

6. **The reported LOPO `.770` is not a clean transfer AUC.** `lopo_scores` pools scores produced by different held-pool models. Their intercepts and scales are not comparable, so cross-pool ranking can recover pool base rates. In my run, the macro-average of held-pool within-pool AUCs was only `.658`, versus pooled `.770`. LOPO should report one AUC per eligible held pool and their macro mean. Near-single-class pools cannot contribute a meaningful AUC.

7. **Threshold estimation does not match the exported model.** Section 3.2 takes quantiles of predictions from models trained on 80% of the data, then applies those thresholds to a model trained on 100%. Its score distribution can differ. Thus even the in-regime 15/30/50% fire rates are not guaranteed.

8. **Dropping no-commit queries creates conditional evaluation.** The 69 dropped rows have no defined gate behavior. A deployment review needs their failure rate and an explicit fallback policy.

### What is sound

- The model is frozen and captured through NVIDIA’s native duplex path.
- Scaling is fitted inside each CV fold. External features are not independently standardized.
- The strong L2 logistic baseline is appropriate for \(p \gg n\).
- Per-pool external AUCs are reported rather than only a pooled external AUC.
- Replay label flips are measured, and pass-2 labels are used consistently.
- The CPU harness reproduces the documented numbers exactly.

### Missing evidence a reviewer should require

- Grouped or nested CV by pool, semantic question, source, and audio variant.
- Paired bootstrap confidence intervals for Figures 3–6 and model differences.
- Macro within-pool AUC, source-held-out AUC, and worst-pool AUC.
- Risk–coverage curves and selective accuracy at the 15/30/50% budgets. Figure 9 only shows score density.
- External fire rates, threshold drift, and score calibration.
- Performance stratified by commit gap, especially early versus post-`t_end` commits.
- A duplicate/template audit across calibration files and shards.
- Judge repeatability and a human audit of borderline labels.
- Current-probe remix results. Section 5.2 only evaluates the obsolete 600-row probe.
- Natural human floor-management evaluation for the act head. Its random TTS OOF `.9999` is not persuasive.

## 2. Ranked proposals

The valid baseline is the strict commit feature, not the reported `.808` external recipe. My quick run gives `.767` external mean for strict commit@L38. Improvements should be measured from that causal baseline.

1. **Domain-diverse grouped calibration expansion** `[re-capture]`
 - **Computation:** Add 1–2k calibration questions concentrated on MCQ, multihop, long-tail, human speech, dialects, voices, and moderate noise. Capture causal commit features at `L ∈ {22,26,30,34,38}`. Keep all audio versions of one semantic question in one group. Append their arrays and labels to `Calib`.
 - **Selection:** Nested source/pool-held-out CV only. Do not inspect external labels until the recipe is frozen.
 - **Expected gain:** External `+.01–.03`; LOPO `+.01–.02`. Section 4.3’s observed slope is about `+.013` external AUC per 1k rows, though it will diminish.
 - **Cost:** GPU replay, judging, and a new untouched final benchmark.
 - **Kill:** A 400–500-query pilot fails to improve macro source-held-out AUC by `.01`.
 - **Constraints:** Brushes 2, 3, and 6; must use native duplex capture and causal reads.

2. **Causal trajectory heads with sparse score fusion** `[re-capture]`
 - **Computation:** Store an eight-frame ring buffer ending at commit, `H_pre[:, j, 0:8]`, plus running mean `H_run[:, j]` through commit. For each layer train separate heads on `h_commit`, `mean(H_pre)`, `h_commit-h_prev`, and `H_run`. Use fold-local `StandardScaler → LogisticRegression(C∈{3e-5,1e-4,3e-4})`. Fuse their cross-fitted logits with a tiny ridge meta-head over at most 20 scores.
 - **Selection:** Fully nested macro LOPO. Freeze layers, `C`, and fusion weights before external scoring.
 - **Expected gain:** `+.01–.03` over strict commit; it may recover much of the apparent post-commit advantage without answer leakage.
 - **Cost:** One calibration re-capture; negligible head cost.
 - **Kill:** Less than `.005` gain in both nested macro LOPO and worst-pool AUC.
 - **Constraints:** Directly satisfies 3; fusion remains linear and satisfies 5.

3. **Nested transfer-target model selection** `[disk now]`
 - **Computation:** Candidate inputs are `cal.E_on[:, cal.j(L), 0]` for `L={22,26,30,34,38}` and causal layer differences such as `E_on[:,j38,0]-E_on[:,j22,0]`. Search `C={3e-5,1e-4,3e-4,1e-3}`.
 - **Selection:** Outer leave-one-pool-out evaluation; inner leave-one-training-pool-out selection. Optimize the macro mean of held-pool AUCs, not pooled LOPO AUC.
 - **Expected gain:** `0–.01`; the main benefit is removing selection bias. My quick equal-layer ensemble did not help.
 - **Cost:** Tens of CPU fits.
 - **Kill:** Nested selection repeatedly returns commit@L38 alone or changes by less than `.005`.
 - **Constraints:** Uses only calibration and strict commit states; satisfies 3 and 6.

4. **Remove the pool-identity subspace** `[disk now]`
 - **Computation:** On each training fold, standardize strict-commit features, compute each calibration-pool centroid, SVD the centered centroid matrix, and project out its top `k∈{1,2,4,8}` directions. Fit logistic regression with the `C` grid above. Apply the same scaler and projection to held-out data.
 - **Selection:** Nested macro LOPO; include `k=0` as the baseline.
 - **Expected gain:** `0–.01` external/LOPO, probably with lower pooled OOF. It directly attacks Figure 8’s shortcut.
 - **Cost:** Low.
 - **Kill:** Worst-pool or macro LOPO falls by `.005`, or the selected `k` is consistently zero.
 - **Constraints:** Brushes 6 because nuisance dimensions and `k` must be calibration-selected only.

5. **Supervised low-rank linear heads** `[disk now]`
 - **Computation:** Compare:
 - `StandardScaler → PLSRegression(n_components∈{2,4,8,16,32}, scale=False)`;
 - `StandardScaler → PCA(k∈{32,64,128,256}, whiten=True) → LogisticRegression`;
 - the same PCA features with shrinkage LDA;
 - `StandardScaler → RidgeClassifier(alpha∈{100,300,1000,3000})`.
 - **Selection:** Nested macro LOPO over estimator and hyperparameters.
 - **Expected gain:** `0–.01`; PLS is the most plausible variance-reduction candidate. My single PCA-64 and ridge settings both lost performance.
 - **Cost:** Moderate CPU.
 - **Kill:** No estimator beats logistic regression by `.005` in nested macro LOPO.
 - **Constraints:** Satisfies 5. Do not run full-covariance LDA directly in 13,440 dimensions.

6. **Mild pool and class reweighting** `[disk now]`
 - **Computation:** For training-fold row \(i\), use
 \(w_i \propto (n/n_{g_i})^\alpha [n_{g_i}/(2n_{g_i,y_i})]^\beta\), with `α,β∈{0,.25,.5}`. Pass these as `logisticregression__sample_weight`.
 - **Selection:** Nested macro LOPO; include `(0,0)`.
 - **Expected gain:** `0–.008`. Full pool/class equalization is too aggressive: my quick run reduced external mean from `.808` to `.773`.
 - **Cost:** Low.
 - **Kill:** No macro-LOPO gain, or any balanced knowledge pool loses more than `.01`.
 - **Constraints:** Brushes 6; weights must never use external prevalence.

7. **Robust fold-local standardization** `[disk now]`
 - **Computation:** Replace `StandardScaler` with training-fold median/IQR scaling, clip standardized values to `[-5,5]`, then fit logistic regression over the same `C` grid. Also test 1st/99th-percentile winsorization followed by z-scoring.
 - **Selection:** Nested macro LOPO.
 - **Expected gain:** `0–.005`; useful only if a small number of activation outliers drive coefficients.
 - **Cost:** Low to moderate.
 - **Kill:** No `.003` macro-LOPO gain or poor coefficient stability across folds.
 - **Constraints:** Satisfies 5 and 6.

8. **Targeted label reliability pass** `[no GPU unless answers are replayed]`
 - **Computation:** Join pass-1/pass-2 labels through `cal.meta[['tag','id']]`. Rejudge replay disagreements and the top 10% highest cross-fitted logistic-loss rows three times. Train on soft failure probability by duplicating each row with labels 0/1 and weights `1-q_i`/`q_i`.
 - **Selection:** Decide the rejudging rule before new judgments; evaluate with grouped nested CV.
 - **Expected gain:** `<.003`; Section 1.3’s two-pass result of `+.0001` makes this low priority.
 - **Cost:** Judge calls and audit time.
 - **Kill:** Label disagreement remains below 2% and nested AUC changes by less than `.002`.
 - **Constraints:** Brushes 5 and 6; the resulting head remains logistic and only calibration labels may be used.

## 3. What not to do

1. **Do not deploy `onset_last`, `onset_mean8`, or full-audio `H_mean` at commit.** Their performance includes post-commit answer or future-audio information.

2. **Do not concatenate more raw layers or frames.** Figure 5 already shows that multi-layer concatenation loses to L30, consistent with the variance-limited regime. The MLP result gives the same warning.

3. **Do not choose recipes from the four reported external pools and continue calling them cold.** Freeze the recipe first, then use a new untouched benchmark or treat these pools as development data.

4. **Do not optimize pooled random-fold OOF or pooled LOPO.** Both reward pool identity and prevalence. Use macro held-pool AUC, worst-pool AUC, and nested selection.

5. **Do not apply blind class balancing, SMOTE, full pool/class equalization, or unregularized LDA.** AUC does not require balanced priors, synthetic hidden states lack a defensible semantics, and the quick weighting/LDA diagnostics were substantially worse.

## 4. Quick CPU experiments

These are post-hoc diagnostics, not valid grounds for selecting a future “cold” recipe.

| Variant | OOF | LOPO pooled | Held-pool macro | External mean |
|---|---:|---:|---:|---:|
| Documented baseline | .8201 | .7698 | .6582 | .8084 |
| `C=3e-4` | .8163 | .7691 | .6528 | .8069 |
| Strict commit `E_on[:,j38,0]` | .7937 | .7484 | .6457 | .7667 |
| Strict commit, equal-logit L22–38 ensemble | .7933 | .7482 | .6376 | .7622 |
| Separate-head ensemble of deployed three views | .8141 | .7601 | .6570 | .7992 |
| Full pool/class-balanced baseline | .7889 | .7655 | .6503 | .7735 |
| PCA-64, whitened, LR | .8086 | .7665 | .6432 | .7967 |
| Ridge, `alpha=1000` | .7583 | .7174 | .5943 | .7560 |

The `.770`→`.658` LOPO discrepancy is the most important diagnostic: much of the nominal transfer score comes from comparing differently scaled predictions across held-pool models, not ranking failures within an unseen pool.
