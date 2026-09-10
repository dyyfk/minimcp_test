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

## Accuracy and measured TTFA figure

The Internal accuracy curve retains the selected knowledge/math weights of
0.25 and weights of 1 for other categories. The main accuracy table keeps
its original mixture. The timing panels use the independent `ttfa1` run,
with signed server PCM-ready endpoints and completed-session statistics.
The timing gate is the frozen 5,228-row artifact, not the 5,221-row
no-overlap gate described for accuracy evaluation. These are not joint
accuracy-latency observations.

Rebuild and audit from the paper directory:

```bash
python build_measured_ttfa.py
python build_academic_revision_figure.py
python build_main_results.py --check
tectonic main.tex
```

`build_measured_ttfa.py` checks all 5,950 formal rows, IDs, timing arithmetic,
failures, frozen gate hashes, and recorded summaries. It generates
`revision_data/measured_ttfa.json` and `sections/measured_ttfa_table.tex`.
It reads the gate artifacts at the recorded source commit through Git.
`build_academic_revision_figure.py` checks accuracy and timing against their
respective sources, keeps all negative TTFA values, and mirrors the figure
to `../figures/`. `build_main_results.py --check` validates the unchanged
accuracy table. Historical timing audits and `native_summary.json` remain
archived; they are not sources for the current timing panels.

The primary paper changes use data from `ttfa_real/summary_ttfa1_final.json`
and formal shards. The prose report's claimed 44 failures and single GPU
type are superseded by the row audit: 46 failed attempts (44 non-commit,
one empty response, one timeout), and three recorded GPU types.

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
