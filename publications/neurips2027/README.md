# NeurIPS 2027 Submission

Paper infrastructure for the NeurIPS 2027 main-track submission. The full rulebook (deadlines, policies, checklist mapping, review criteria, submission strategy) lives in the vault note `notes/DLR-TomoSAR/NeurIPS 2027 Submission Guide.md`.

## Layout

- `template_kit/` — pristine official NeurIPS author kit (do not edit). Currently the NeurIPS 2026 kit, the latest published; the 2027 kit must replace it when the 2027 call for papers appears (expected February--March 2027) from `https://media.neurips.cc/Conferences/NeurIPS2027/Formatting_Instructions_For_NeurIPS_2027.zip`.
- `paper/` — the working paper source. `main.tex` includes per-section files from `sections/`, macros from `macros.tex`, references from `references.bib`, and the mandatory `checklist.tex`.

## Build

```
cd paper && make
```

Requires `tectonic` (installed via snap). Output lands in `paper/build/`, which is gitignored; only LaTeX source is committed.

## Figures

Every plot class in the repository derives from `PlotBase` (`tools/reporting/plotting.py`), which carries a class-wide figure style switch: `PlotBase.use_style("paper")` flips all subsequent rendering from the report style to the print style (9pt/8pt fonts, STIX math to match the Times body, Okabe-Ito colorblind-safe cycle, 300 DPI, and automatic upgrade of non-image plots from PNG to vector PDF; image maps stay PNG at 300 DPI). The four inference entries expose this as the `figure_style` config field (`report`/`paper`), so any pipeline report can be regenerated print-ready, e.g. `--figure_style paper`. Dedicated paper figures are created at final column size via `PlotBase.figsize(PlotBase.FULL_WIDTH)` (5.5 in, or `HALF_WIDTH` 2.65 in) and rendered into `paper/figures/` (gitignored, regenerable). Multi-panel figures are composed in LaTeX with `subcaption` from single-plot files, per the vault plotting rules.

## Kit refresh procedure (on 2027 CFP release)

1. Download the 2027 ZIP into `template_kit/`, replacing the 2026 files.
2. Copy the new `.sty` and `checklist.tex` into `paper/`, update the `\usepackage` name in `main.tex`, and port any checklist answers already written.
3. Diff the new formatting instructions and Main Track Handbook against the vault guide note and update it.

## Hard rules (one-screen summary)

- Nine content pages at submission (ten camera-ready), including all figures and tables; references, appendices, and checklist do not count. Style-file tampering (margins, fonts) is a desk rejection.
- The checklist must be included in the PDF after references and appendix; papers without it are desk rejected. Its instruction block must be deleted before submission (kept here while drafting).
- Fully anonymized at submission, including supplementary material and code; no acknowledgments, self-citations in third person.
- Submission mode is the default `\usepackage{neurips_2026}`; switch to `[main, final]` only for camera-ready and `[preprint]` only for arXiv.
- Supplementary ZIP up to 100 MB, paper PDF up to 50 MB.
- No submission of the same work to another archival venue while under review at NeurIPS.
