# Layer ablation of the deployed probe (review item R3)

Branch `zetian/layer-ablation-deployed-probe`, opened 2026-09-02.

## Question

The paper's headline ("a mid-network representation, rather than the
final-layer read, is the practical signal") was established with
single-position, text-input LOPO sweeps. Two facts make it untested where
it matters:

1. The audio-calibrated single-position sweep already in the repo
   (`data/layers/layer_sweep_minicpm-o45-audio.json`, Phase 5d) shows the
   final layer is **not** worse than L22 on any held-out pool under audio
   input (easy-chat .672 vs .621, easy-fact .707 vs .675, hard-knowledge
   .714 vs .696, hard-math .936 vs .901).
2. The deployed probe (`eot_last + eot_mean8 + user_mean` at L22,
   12,288-d, 2,310-row mix, C = 1e-4; `modal_train2.py::refit3`) was
   never re-fit at another depth and scored on the paper's external pools.

This experiment re-fits **the identical feature set on the identical
training rows** at every decoder layer and scores **the identical
evaluation pools with the identical labels**. The only free variable is
the layer.

## Pre-registered protocol

* Capture: `modal_train4_layers.py::eoth3_shard` — the eoth2 streaming
  replay (session reset, omni system prompt, 1 s chunks, assistant
  prefill, rolling 8-token tail across forwards, running user-audio
  mean) with hooks on all 36 decoder layers. Tags: frozen, expansion,
  expansion2 (training + internal test), striviaqa, swebq, sdqa, sllama,
  sreason (evaluation).
* Fit: per layer, `LogisticRegression(C=1e-4)` on
  `[eot_last | eot_mean8 | user_mean]` (12,288-d), training rows =
  frozen calib (360) + expansion + expansion2 with the v3 labels.
  No standardisation, no per-layer C selection in the primary analysis
  (C selected on train-OOF per layer is reported as a secondary row via
  `--with-oof`).
* Evaluation: AUC against never-arm local failure on the five external
  pools (`EXT_TRACES`, same parquets as `eval_transfer3`) and the frozen
  test split. Paired bootstrap over queries (2,000 draws, seed 42) of
  AUC(layer) − AUC(L22) per pool. Label-free per-pool quantile budgets
  (15/30/50 %) report failure recall and precision per layer.
* Sanity anchor: the shipped `midlayer_gate_audio_v3.json` weights
  applied to the new capture's L22 features must reproduce the paper's
  Table 1 AUC row (.789/.785/.792/.806/.683, frozen-test .879) within
  capture jitter; if it does not, the capture is wrong, not the model.
* Externals are read once, by `layer_ablation3`, after the capture. No
  quantity is selected on them.

## Decision rule (written before the numbers)

Let Δ = external-mean AUC(final layer L35) − AUC(L22), with the five
per-pool paired intervals.

* Δ ≤ −0.03 and at least three pools' intervals exclude zero → the
  deployed-modality evidence supports "mid-network rather than
  final-layer"; put this table in the main text next to Fig. 2.
* |Δ| < 0.03 and no pool's interval excludes zero → the final layer is
  as good on the deployed path; rewrite the abstract's last sentence and
  contribution 1 to "robust to text-pool shift and available earlier
  (57–61 % of prefill)", and keep the text-input inversion as the
  representational finding it is.
* Anything else → report the full curve and say which pools differ.

The same table gives, for free, the full depth curve of the deployed
feature set on audio input (the audio analogue of Fig. 2) and the
budget-recall counterpart to Table 1.

## Commands

```bash
# smoke (5 queries per tag, shard 99, all layers)
modal run modal_train4_layers.py::run_eoth3 --tags striviaqa,frozen --limit 5
# full capture (externals first, then the training tags)
modal run modal_train4_layers.py::run_eoth3 --tags striviaqa,swebq,sdqa,sllama,sreason,frozen
modal run modal_train4_layers.py::run_eoth3 --tags expansion,expansion2
# ablation on the volume
modal run modal_train4_layers.py::layer_ablation3 --source eoth3
# what is computable today from the five-layer eoth2 capture (no GPU)
modal run modal_train4_layers.py::layer_ablation3 --source eoth2
# receipts
modal volume get gate-data layer_ablation_eoth3.json figures/
modal volume get gate-data layer_ablation_eoth3.md figures/
```

Local alternative once shards are pulled: `scripts/64_layer_ablation.py`.

## Cost

eoth2 captured the same 3,901 replays in about 14 H100-hours; hooks on
36 layers instead of 5 add CPU-copy overhead per chunk but no extra
model forwards, so expect a similar bill (≈ $40–60 on Modal). Storage:
36 × 9 × 4096 float16 per query ≈ 2.7 MB → ≈ 10 GB on the volume.

## Status

* 2026-09-02: analysis module (`src/layer_ablation.py`) written and
  unit-tested on synthetic shards; capture module and local runner
  written; GPU capture not yet launched (needs Modal credentials in the
  running environment).

* 2026-09-03: Modal credentials available. Smoke capture (5 queries x 2
  tags, all 36 layers) reproduces the eoth2 features bit-exactly at the
  five shared layers (max |diff| = 0 on H_eot and H_mean; identical v3
  logits). Five-layer ablation from the existing eoth2 capture
  (`figures/layer_ablation_eoth2.{json,md}`; Modal run
  ap-dTBvXdkzweCHiuI35xgahj): the L22 row reproduces the paper's Table 1
  AUC row exactly (.789/.785/.792/.806/.683, frozen-test .879, external
  mean .771); L18 -0.034, L26 -0.050, L14 -0.103, L30 -0.113 on the
  external mean, with per-pool paired intervals excluding zero on 4/5
  pools for L26 and L30. Full 36-layer capture launched (run_eoth3, all
  eight tags, workers=3).

| layer | striviaqa | swebq | sdqa | sllama | sreason | frozen-test | ext mean | Δ ext mean vs L22 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L14 | 0.712* | 0.690* | 0.717* | 0.644* | 0.576* | 0.853 | 0.668 | -0.103 |
| L18 | 0.783 | 0.747* | 0.779 | 0.756* | 0.619* | 0.884 | 0.737 | -0.034 |
| L22 (deployed) | 0.789 | 0.785 | 0.792 | 0.806 | 0.683 | 0.879 | 0.771 | +0.000 |
| L26 | 0.736* | 0.725* | 0.789 | 0.755* | 0.602* | 0.873 | 0.721 | -0.050 |
| L30 | 0.671* | 0.625* | 0.745 | 0.687* | 0.560* | 0.845* | 0.658 | -0.113 |

`*` = paired-bootstrap 95% interval of AUC(layer) − AUC(L22) excludes zero on that pool.
