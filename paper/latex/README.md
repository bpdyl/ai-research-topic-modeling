# Overleaf build

Upload **everything in this folder** (`main.tex` + `figures/`) to a new Overleaf project.

- Compiler: **pdfLaTeX** (Overleaf default)
- Class: `IEEEtran` — bundled with Overleaf, nothing to install
- Two-column conference format

If you prefer to upload a zip: select `main.tex` and `figures/`, compress, then
Overleaf → New Project → Upload Project.

## Figures used

| File | Used in |
|---|---|
| `fig_corpus_by_year.png` | §II Dataset |
| `fig_ga_convergence.png` | §V Convergence |
| `fig_trends_ga_lda.png`  | §V Thematic evolution |

`fig_coherence_vs_k.png`, `fig_wordcloud.png` and `fig_cooccurrence.png` are included but not
currently placed — add them if you have page budget, or drop them to save space.

## Page budget

The brief allows **6 pages / 4,000 words**. Compile and check. If it overruns, cut in this order:
1. The word cloud and co-occurrence figures (if you added them)
2. §III-A preprocessing detail — the DF percentages can move to a footnote
3. The Related Work subsection — compress to one paragraph

Do **not** cut: the Social/Ethical/Legal section (2 marks), Individual Contributions (mandatory),
or the random-search control (it is the paper's most defensible result).
