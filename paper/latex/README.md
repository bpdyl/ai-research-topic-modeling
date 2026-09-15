# Building the report

The submission is **two self-contained papers**. Each has its own title block,
authors, abstract, keywords, section and figure numbering, and its own reference
list numbered from `[1]`.

| Paper | Subject | References | Figures |
|---|---|---:|---:|
| **1** | Optimized Topic Modeling of AI Research Involving Nepali Authors | 22 | 6 |
| **2** | A Fuzzy Logic Controller for an Assistive-Care Flat | 13 | 26 |

## Two ways to build

| File | Produces | Upload alongside |
|---|---|---|
| `main.tex` | **Both papers in one PDF** | `task2.tex`, `appendix_task2.tex`, `figures/` |
| `task2_standalone.tex` | **Paper 2 only**, as its own PDF | `task2.tex`, `appendix_task2.tex`, `figures/` |

Paper 2's content lives entirely in `task2.tex` and `appendix_task2.tex`, which
are `\input{}` by both parents. Edit once, compile twice — the combined report
and the standalone Task 2 PDF can never disagree.

- Compiler: **pdfLaTeX** (Overleaf default)
- Class: `IEEEtran` — bundled with Overleaf, nothing to install
- Two-column conference format

## Before uploading: run the pre-flight checks

There is no TeX toolchain on the authoring machine, so these stand in for
reading a compile log:

```bash
python check_report.py                             # combined report
python check_report.py --doc task2_standalone.tex  # Paper 2 only
python check_report.py --order                     # print required bibitem order
python wordcount.py                                # prose word count vs the cap
```

`check_report.py` verifies that **each** bibliography numbers sequentially from
`[1]` in order of first citation (the lists are manual, so `\bibitem` order *is*
the numbering), that no key appears in both lists, that no `\cite` or `\ref`
dangles, that no label is duplicated, that every `\includegraphics` resolves,
that `\begin`/`\end` balance, and that no unescaped `%` silently eats a line.
Exit code is non-zero on failure.

## MATLAB figures

The three MATLAB figures in `figures/` (`matlab_fig1_membership.png`,
`matlab_fig2_ruleinference.png`, `matlab_fig3_surfaces.png`) were captured
from a real MATLAB Online run and are included with ordinary
`\includegraphics`, like every other figure. The MATLAB code needs no Fuzzy
Logic Toolbox.

To regenerate them: upload `assistive-care-flc/matlab/` to
<https://matlab.mathworks.com>, run `runAssistiveCareFIS`, and replace the
three PNGs. The Command Window also prints a PASS/FAIL cross-check against
the Python engine.

## Figures

`figures/` holds all 32 images the report uses: 6 for Paper 1 and 26 for
Paper 2, three of which are the MATLAB screenshots.

| Prefix | Source | Regenerate with |
|---|---|---|
| `fig01`–`fig14` | Paper 2 experiments | `assistive-care-flc/scripts/run_part{1,2,3}.py` |
| `fig15`–`fig23` | Paper 2 supplementary views | `make_report_figures.py` (seconds, no re-run) |
| `figCorpusByYear` etc. | Paper 1 | the Task 1 notebook |

To refresh the copies here after regenerating:

```bash
cp ../../assistive-care-flc/figures/*.png figures/
```

## Notes on the brief

- The brief asks for **one file per task**, which `task2_standalone.tex` covers
  for Task 2; use `task2_standalone.tex` or Paper 1 of
  `main.tex` for Task 1.
- Task 1 is capped at **6 pages / 4,000 words**; `wordcount.py` currently reports
  ~3,870 including the abstract. Task 2 has **no stated length limit**.
- The brief specifies **single-column** format for Task 1's paper. Both
  documents currently use IEEEtran's two-column `conference` class. Switching
  would reflow every float, so it has been left as-is — change the
  `\documentclass` line if the marker requires single-column.
