# Nepal AI research topic modelling

This repository contains the code, frozen corpus, experiment evidence and papers for the
ST7085CEM Advanced Machine Learning assignment. The project compares standard LDA,
genetically optimised LDA, random-search LDA and BERTopic on AI/ML publications with
Nepal-affiliated authors (2015–2025). It also provides LLM-assisted topic labelling through
the Streamlit app in `apps/topic_labeler.py`.

## Repository layout

- `src/nrtm/` — corpus processing, models, evaluation, labelling and visualisation
- `scripts/` — numbered pipeline and figure-generation commands
- `data/processed/` — frozen corpus and reports used by the experiments
- `results/runs/` — saved configurations, metrics and experiment evidence
- `apps/` — topic-labelling application with OpenAI, Claude and Gemini adapters
- `paper/latex/` — combined and standalone LaTeX reports
- `output/pdf/` — compiled report PDFs and checksums

## Reproduce the experiments

Create an environment and install the project:

```bash
pip install -e .
```

For a full run, execute the numbered scripts in this order:

```bash
python scripts/01_build_corpus.py
python scripts/02_preprocess.py
python scripts/03_lda_baseline.py
python scripts/04_ga_lda.py
python scripts/05_bertopic.py
python scripts/06_evaluate.py
python scripts/07_temporal.py
python scripts/08_label_topics.py
python scripts/10_interpretability.py
python scripts/11_random_search.py
python scripts/09_figures.py
```

The frozen corpus in `data/processed/` is the input used by the reported results. Existing
run evidence is retained under `results/runs/`; new runs create their own timestamped
directories. Use `QUICK_MODE=True` in the notebook for a shorter verification run.

Run the checks with:

```bash
pytest
python paper/latex/check_report.py --doc main.tex
```

## Build the reports

From `paper/latex/`, compile `main.tex` for both tasks, or compile
`task1_standalone.tex` and `task2_standalone.tex` separately. Tectonic, XeLaTeX or a
standard IEEEtran LaTeX installation can be used. The report PDFs are written to
`paper/latex/build/`; packaged copies are in `output/pdf/`.

The task-1 labelling application can be started with:

```bash
python -m streamlit run apps/topic_labeler.py
```

Provider API keys are needed only for new live labelling requests. Archived evidence can be
reviewed without a provider key.
