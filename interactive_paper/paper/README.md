# Paper draft — "When Does a Speech Model Know to Hand Off?"

Draft v0.1, generated 2026-07-24 from `TECHNICAL_REPORT.md` v3 (Phases 0–7a).
Numbers trace to `../RESULTS.md`; do not edit numbers here without a matching
RESULTS.md entry.

## Build and Overleaf synchronization

This directory is the paper source of truth. Commit the changed `.tex`,
`refs.bib`, required figures, and the newly compiled `main.pdf` together.
Build the repository PDF from this directory with `tectonic main.tex`.
Overleaf uses pdfLaTeX + BibTeX (TeX Live 2026, Normal mode); pagination can
differ between engines even when every source file is identical.

Create the lean Overleaf source package from the same checkout:

```bash
python -m pip install Pillow pypdf
python build_overleaf_zip.py /tmp/MiniMCP_RTCA_2026_Overleaf_Lite.zip
```

Import the ZIP using **New project → Existing project (.zip)**, or replace
the changed source files in an existing project after checking collaborator
edits. The exporter includes only required compilation files, preserves all
TeX/BibTeX/style bytes, and puts opaque PNG plots in lossless PDF containers.
It omits duplicate/unused assets, scripts, and the compiled reference PDF.
This reduces the source ZIP from about 17.5 MB to 2.8 MB; it does not promise
that every compile stays below Overleaf's free-plan timeout.

## Layout

- `main.tex` — preamble, title, author block, section includes
- `sections/` — one file per section; edit these
- `refs.bib` — bibliography; entries marked `TODO verify` need checking
- `figures/` — PNGs copied from `../figures/` (regenerate there, then re-copy)

## Conventions

- `\todonote{...}` marks open items inline (renders red).
- The appendix "TODO / Roadmap" section collects experiment-level TODOs and
  must be deleted before submission.
- Signal names: `\ptrue` renders p(True).

## PDF font repair (2026-09-08)

The six legacy PDF figures (`roc`, `nvda_probe_reads`, `fork_pareto`,
`timeline_live`, `sllama_latency_decomp`, and `noise_audit`) now embed matching
DejaVu TrueType fonts. Their plotted paths, text, and numerical data were
preserved. `neurips_2026.sty` was restored byte-for-byte from the
[official 2026 formatting package](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
Restoring the style changes line spacing and float placement; the compiled
paper remains 8 main-text pages and 42 pages overall.

`fix_legacy_figure_fonts.py` performs this resource-only repair if an old copy
of a figure is restored. It checks glyph metrics and leaves already-repaired
files unchanged. The historical audit records the font-repair snapshot, before later citation
edits. The repair used matplotlib 3.11.1, fonttools 4.64.0, and
pypdf 6.18.0. For future Matplotlib exports, set `pdf.fonttype = 42` before
saving to embed TrueType fonts directly.

Build from this directory with `tectonic main.tex`. Check the final PDF with
`pdffonts main.pdf`: no font should be Type 3, and every font should have
`emb` set to `yes`. See `revision_data/font_repair_audit.json` for the repair
hashes and validation results.

## Citation corrections (2026-09-08)

The appendix now distinguishes reduced detection performance from
undetectability, generated-token evidence from a pre-generation test, and
training-dynamics findings from hypotheses about frozen representations.
The RouterBench title, LLM Router author field, and Freeze-Omni version and
author list were corrected. FDB comparisons and the RouterBench subset are
explicitly scoped; published AlpacaEval performance points to Table 4.

See `revision_data/citation_audit_2026-09-08.md` for all 45 cited works and
the limits of the numerical checks. RouterBench's exact experiment and
FDB's per-sample results have not been independently recomputed.

## Internal reporting: all 240 queries, unit weights

Current manuscript results retain all 240 original Internal test IDs,
including 60 chat queries. Every query has weight 1 in accuracy and call-rate
calculations. Both model blocks use this full cohort. External pool means
retain their existing equal-pool convention.

`revision_data/internal_unweighted_summary.json` is the shared reporting
source. Its MiniCPM accuracy and call rates reproduce `native_summary.json`;
it also records fresh paired bootstrap intervals over the same archived
outcomes, original NVDA pass-3 replay results, and independently recorded
nonnegative TTFA on the same query IDs. NVDA architecture selection remains
post-selection, and accuracy and timing remain separate experiments.

From this directory, with numpy, pandas, pyarrow, and matplotlib available:

```bash
python build_internal_unweighted.py
python build_main_results.py
python build_academic_revision_figure.py
python check_internal_reporting.py
tectonic main.tex
```

The Internal builder accepts `--data-dir`, `--queries`, `--nvda-expert`, and
`--ttfa-results` for source locations. It verifies archived hashes and IDs.
Accuracy and call rate use this unweighted report; current timing values
come directly from `../ttfa_real/nonnegative/summary.json`. Historical weighted and 180-query subset summaries and scripts are
retained as analysis archives; they are not inputs to the current manuscript.

## Nonnegative server TTFA

The paper reports waiting time after scheduled input end:
`max(0, first_answer_pcm - input_end)`. Completed early responses remain
in the statistics with zero wait and are counted separately. Failed
attempts retain missing TTFA values and are excluded from timing statistics.
The raw `ttfa-v3` logs and the historical signed summary remain unchanged.

Recompute all 25 pool/arm combinations from the 5,950 recorded attempts,
then redraw the paper figure, from this directory:

```bash
python ../ttfa_real/recompute_nonnegative_ttfa.py \
  --results ../ttfa_real/results/ttfa1 --out-dir ../ttfa_real/nonnegative
python build_academic_revision_figure.py
```

The recomputation validates query IDs, attempt counts, audio endpoint
selection, and source hashes. The summary records 5,904 completed sessions,
46 failures, and 64 early responses. The Internal latency table uses all 240 Internal query IDs,
rounded to two decimals; the figure uses unrounded mean and P50 values.
These are server audio-ready statistics. Accuracy retains the judged
benchmark values; client playback is unmeasured.

## Pareto, cost, and layer-decline reporting

The Table 1 cost panel is restored with measured dollar costs
(2026-09-10; it had been briefly omitted while the dollar row was NR).
`build_main_results.py` regenerates it when its markers are present in
the table source. Timing is mean nonnegative server TTFA from the full
`ttfa1` / `ttfa-v3` run; expert use is the realized escalation rate from
the content benchmark. Internal uses 240 queries, and external entries
average the four pools equally.
The native records do not contain token or tool billing usage, so GPT
dollar costs are measured directly: `modal_expert_cost.py` replays every
escalated call's recorded uplink transcript through the identical expert
configuration (gpt-5.5, reasoning effort low, web_search, same system
prompt) and records billed usage; `build_expert_cost.py` prices it at
official rates and writes `revision_data/expert_cost_usd.json`, which
`build_main_results.py` reads. Identical transcripts are billed once and
shared; a non-escalated query costs $0.

The 3-by-2 figure's left column connects nondominated observed policies
in accuracy versus call rate. TriviaQA conservative is dominated by local.
The right column shows unconnected policy points in accuracy versus mean
TTFA, using the retained accuracy values and latest full timing run.
In the left column, thin solid lines also connect all five policy settings,
including the dominated TriviaQA conservative point; thicker lines mark
the Pareto frontier.
The convention is a comparison of reported policy summaries; metric-specific
input and recording details are documented once in the experimental setup
and appendix. P50 and tail statistics
remain in the timing summary and the Internal appendix table.

`revision_data/reporting_protocol.json` pins the metric source files and
hashes. Accuracy and call rate must not switch to timing-session outputs;
TTFA must not fall back to reconstructed timing or older cached means.
All displayed pool/arm timing values come from the same full timing run.

`python build_layer_sweep.py` reproduces Figure 2 from the five archived
layer-sweep JSON files and highlights the late decline. It records source
hashes and plotted arrays in `revision_data/layer_sweep_values.json`.
The caption says the decline may reflect specialization for duplex
interaction control and cites MiniCPM-o 4.5, Section 3.3, for its binary
Listen/Speak decision before content generation. The recorded mechanism
controls remain in the appendix. No probe refit or API calls are needed.
Both figure builders mirror the PDF/PNG assets
to `../figures/`.

## Native ablation figure (2026-09-09)

`python build_native_ablation.py` redraws the four matched AUCs in
`figures/native_feature_ablation.{pdf,png}`, the appendix routing figure
`figures/native_ablation_routing.{pdf,png}`, and the named per-pool table
`sections/native_ablation_values.tex`. The source is the unchanged recorded
summary `../figures/native_feature_ablation.json` from the 8db analysis.
The builder prints its SHA-256 and validates the equal-pool AUC averages;
it does not refit probes, rejudge answers, or run model inference.

The main plot omits the unlabeled per-pool dots and the separately trained
deployed-gate reference. Their values remain in the appendix table and text.
Routing gains use cached outcomes at a 30% budget; their pooled averages
must not be relabeled as equal-pool means. The legacy
`paper_pdf_redraw.fig_native_ablation()` entry point delegates to this
builder, which also updates the matching assets in `../figures/`.
