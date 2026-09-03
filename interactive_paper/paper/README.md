# Paper draft — "When Does a Speech Model Know to Hand Off?"

Draft v0.1, generated 2026-07-24 from `TECHNICAL_REPORT.md` v3 (Phases 0–7a).
Numbers trace to `../RESULTS.md`; do not edit numbers here without a matching
RESULTS.md entry.

## Overleaf

Upload `paper.zip` (or zip this directory) via **New Project → Upload
Project** on Overleaf. Compiles with the default pdfLaTeX + BibTeX
toolchain; no custom class files.

To keep the repo and Overleaf in sync while iterating, either use Overleaf's
Git bridge (Project → Sync → Git) or re-download the zip periodically —
this directory is the source of truth for anything generated from
experiment data (tables, figures).

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
