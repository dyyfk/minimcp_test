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

Local alternative once shards are pulled: `scripts/33_layer_ablation.py`.

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
