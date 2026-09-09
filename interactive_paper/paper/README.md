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

## Internal figure restoration (2026-09-09)

The author requested restoring the selected weighted Internal curve first,
while leaving the main table unchanged. The figure now uses knowledge/math
weights of 0.25 and weights of 1 for the other categories, from
`revision_data/joint_reweighting_ideas.json`. Accuracy and call rate are both
weighted. All 240 queries remain included. Timing and external curves keep
their original values from `revision_data/native_summary.json`.

To rebuild only the figure:

```bash
python build_academic_revision_figure.py
```

This command validates source hashes, sample counts, plotted coordinates,
and unchanged timing. It never writes the main table. The Internal figure
and table intentionally use different weighting during this staged revision;
`build_main_results.py --check --figure-values ...` will flag that difference.
The table itself can still be checked with `python build_main_results.py --check`.

`revision_data/internal_figure_restore_audit.json` records the restored
coordinates and unchanged main-table hash. The earlier
`section4_consistency_audit.json` is a historical unweighted snapshot.
Experimental setup details and replay qualifications remain in
`sections/setup_details.tex`.
