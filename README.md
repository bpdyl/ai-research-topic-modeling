# Optimized Topic Modeling of AI Research Involving Nepali Authors

Tracking thematic evolution and trends in Nepal, 2015–2025.

**Module:** ST7085CEM — Advanced Machine Learning, Softwarica College of IT & E-Commerce
(affiliated to Coventry University) · **Task 1**

**Team:** Bibek Paudyal (250288) · Siddhartha Bhatta (250620) · Sajan Mahat (250289)

---

## What this project does

Three topic models are fitted on one shared corpus of AI/ML paper titles and abstracts, every
paper having at least one author affiliated to a Nepal-based institution:

| Role | Model |
|---|---|
| Baseline | **Standard LDA** — probabilistic, grid/coherence-selected `K` |
| **Proposed** | **GA-Optimized LDA** — a genetic algorithm searches `(K, α, β)` against a multi-objective fitness of coherence + diversity + stability |
| Benchmark | **BERTopic** — sentence-transformer embeddings → UMAP → HDBSCAN → c-TF-IDF |

They are compared on `C_v` coherence, topic diversity and human-rated interpretability. A single
global model is then used to report topic proportions across four time windows
(2015–17 · 2018–20 · 2021–23 · 2024–25), and every topic is given an LLM-drafted label
(human review pending).

A budget-matched **random-search control** is also run. It beat the genetic algorithm, so the
project does not claim evolutionary search is superior — see the paper.

---

## Repository layout

```
.
├── config/default.yaml      Single source of truth for every tunable
├── scraper/                 Data acquisition (OpenAlex + Semantic Scholar + IEEE/Springer)
├── src/nrtm/                Library code
│   ├── config.py            Config loading, seeding, run directories
│   ├── corpus/              Load, filter, validate, freeze
│   ├── preprocessing/       Clean, tokenize, represent
│   ├── models/              lda · bertopic_model · ga/
│   ├── evaluation/          coherence · diversity · stability · compare
│   ├── temporal/            Per-window topic proportions
│   ├── labeling/            LLM-assisted topic labels
│   └── viz/                 Figures
├── scripts/                 Numbered pipeline entry points — run in order
├── data/
│   ├── interim/             Intermediate artefacts (gitignored)
│   └── processed/           Frozen corpus + corpus report (committed)
├── results/runs/<RUN_ID>/   Per-run config snapshot, metrics, artefacts
├── figures/                 Figures used in the paper
├── tests/                   pytest suite
├── notebooks/               Single end-to-end notebook (portable, Colab-ready)
└── paper/                   Manuscript drafts
```

## Running it

### Option A — the notebook (portable)

`notebooks/nepal_ai_topic_modelling.ipynb` runs the whole pipeline in one file and works locally or
in Google Colab, where it clones this repository and installs its own dependencies. Set
`QUICK_MODE = False` in the setup cell to reproduce the paper's numbers (~2.5 h); leave it `True`
for a ~15 min verification run.

The notebook **orchestrates `src/nrtm/`** rather than re-implementing it, so it cannot drift from
the reported results. Change the method in the package, not in a cell.

### Option B — the scripts

```bash
pip install -e .                        # or: pip install -r requirements.txt
python scripts/01_build_corpus.py       # freeze the analysis corpus + corpus report
python scripts/02_preprocess.py         # tokens, dictionary, BoW/TF-IDF
python scripts/03_lda_baseline.py       # EXP-001
python scripts/04_ga_lda.py             # EXP-002  (~69 min)
python scripts/05_bertopic.py           # EXP-003
python scripts/06_evaluate.py           # EXP-004  + blind rating sheet
python scripts/07_temporal.py           # EXP-005
python scripts/08_label_topics.py       # EXP-006
python scripts/10_interpretability.py   # EXP-007
python scripts/11_random_search.py      # EXP-008  (~66 min, budget-matched control)
python scripts/09_figures.py
```

Each script writes a timestamped directory under `results/runs/` containing the exact config used,
the git commit, and its outputs — so any number in the paper can be traced back to the run that
produced it.

Run the tests with:

```bash
pytest
```

## Where the paper's claims come from

| Paper section | Code |
|---|---|
| Dataset description | `scraper/`, `src/nrtm/corpus/`, `scripts/01_build_corpus.py` |
| Experimental setup — preprocessing | `src/nrtm/preprocessing/`, `scripts/02_preprocess.py` |
| Methods — LDA baseline | `src/nrtm/models/lda.py`, `scripts/03_lda_baseline.py` |
| Methods — GA-Optimized LDA | `src/nrtm/models/ga/`, `scripts/04_ga_lda.py` |
| Methods — BERTopic | `src/nrtm/models/bertopic_model.py`, `scripts/05_bertopic.py` |
| Results — comparison | `src/nrtm/evaluation/`, `scripts/06_evaluate.py` |
| Results — thematic evolution | `src/nrtm/temporal/`, `scripts/07_temporal.py` |
| Results — topic labels | `src/nrtm/labeling/`, `scripts/08_label_topics.py` |
| Results — interpretability | `scripts/10_interpretability.py` |
| Results — random-search control | `scripts/11_random_search.py` |
| Figures | `src/nrtm/viz/`, `scripts/09_figures.py` |

---

## Notes for anyone running this

**Windows + gensim.** `CoherenceModel` defaults to spawning one worker process per core, and each
child re-imports the whole numpy/scipy/BLAS stack. On a 16-core machine that exhausted the system
commit limit and failed with *"the paging file is too small for this operation to complete"*.
Two guards are in place and should stay:

- `src/nrtm/__init__.py` caps BLAS thread pools at import time, before numpy loads.
- `config/default.yaml` sets `coherence_processes: 1` and `lda.multicore: false`.

Every script also needs an `if __name__ == "__main__":` guard — without it, spawned workers
re-import the module recursively.

**If imports hang, suspect the machine, not the toolchain.** During development `import
scipy.linalg` took 22 minutes and appeared to be a broken conda install. It was not: the same hang
occurred in a clean pip environment, and Windows Defender and WMI queries were hanging too. A reboot
took the identical import to 2.1 seconds. Check `python -c "import scipy.linalg"` returns in ~2 s
before starting a long run, and check whether *non-Python* system queries are also slow before
blaming the packages.

**The corpus is frozen, not live.** `scripts/01_build_corpus.py` is the only thing that may write
`data/processed/corpus_frozen.jsonl`. Every other stage reads it. Re-running the scraper does not
change the analysis corpus until that script is run again.
