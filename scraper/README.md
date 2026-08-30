# Nepal-AI-Research Scraping Engine

Builds a deduplicated dataset of AI/ML research papers (2015–2026) with at
least one Nepal-affiliated author, for the topic-modeling proposal
(ST7085CEM, Advanced Machine Learning).

## Setup

From the `scraper/` directory:

```bash
python -m venv .venv
# Windows Git Bash / macOS / Linux:
source .venv/Scripts/activate   # Windows Git Bash
# .venv/Scripts/Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Set at minimum:
- `OPENALEX_EMAIL` in `config.py`, or pass `--email` — OpenAlex's polite pool
- `SEMANTIC_SCHOLAR_API_KEY` — optional but recommended; free at
  https://www.semanticscholar.org/product/api (unauthenticated rate limits
  are tight; without a key, S2 bulk search still runs but cannot attach
  author affiliations)
- `IEEE_API_KEY` / `SPRINGER_API_KEY` — optional extra sources

## Tests (no network)

```bash
python test_pipeline_offline.py
```

## Smoke test before a full run

API field names occasionally drift from docs. Before trusting a full run:

```bash
python main.py --email you@example.com --smoke --output-dir output_smoke
```

Or the one-liner:

```bash
python -c "from sources.openalex import fetch_openalex_papers; r = fetch_openalex_papers(year_start=2023, year_end=2024, max_results=5); print(len(r)); print(r[0]['title'] if r else 'EMPTY')"
```

If that returns real records, OpenAlex is wired correctly.

## Run

```bash
python main.py --email you@example.com --target 1000
```

Useful flags:
- `--target N` — after OpenAlex, only query extra sources if unique count is still below N (default 1000). OpenAlex itself is fetched in full per time-window (it is the primary corpus).
- `--ss-api-key KEY` — Semantic Scholar key (or set in config.py)
- `--ieee-key KEY` / `--springer-key KEY` — enable those optional sources
- `--require-foreign-coauthor` — only keep papers with **both** a
  Nepal-affiliated author **and** a foreign-affiliated co-author (stricter
  than the proposal's stated inclusion rule, which only requires ≥1 Nepal
  author)
- `--resume` — merge a new run with `output/nepal_ai_papers.json` from a
  previous run (additive; does not drop already-collected papers)
- `--smoke` — tiny live run to verify API wiring
- `--max-results N` — cap each OpenAlex window (useful for debugging)

## Output

Written to `output/`:
- `nepal_ai_papers.csv` — flat table for pandas/Excel
- `nepal_ai_papers.json` — full nested structure (author lists with
  affiliations intact) for the preprocessing/modeling stage
- `checkpoint_seen_ids.json` — DOIs/titles already collected

Schema (both files):

| field | description |
|---|---|
| `paper_id` | source-specific ID (OpenAlex work ID, S2 paper ID, etc.) |
| `doi` | DOI if available |
| `title` | paper title |
| `abstract` | abstract text (reconstructed for OpenAlex, native for others) |
| `year` | publication year |
| `time_window` | which of the proposal's 4 temporal slices it falls in |
| `venue` | journal/conference name |
| `authors` | name + affiliations (+ country codes where OpenAlex supplied them) |
| `nepal_affiliations` (CSV only) | Nepal-matched affiliation strings only |
| `url` | landing page URL |
| `source` | which API it came from |
| `primary_topic` / `primary_subfield` | OpenAlex primary topic (JSON + CSV) |

## Design notes / why these sources

- **OpenAlex is primary**, not Semantic Scholar (even though the proposal
  draft lists S2 first): OpenAlex supports a server-side
  `authorships.countries:np` filter, which is a structured, reliable
  signal for "has a Nepal-affiliated author." Semantic Scholar has no
  equivalent filter. OpenAlex Concepts (`concepts.id`) are deprecated and
  frozen. The engine uses `primary_topic.subfield` 1702 (AI) and 1707
  (Computer Vision), then a keyword pass so applied AI (medical imaging,
  agricultural ML, etc.) is not lost.
- **AI-only gate:** a paper is kept only if the title or abstract uses
  AI-method vocabulary (ML, DL, NLP, LLMs, neural nets, CV, etc.).
  Domain papers about earthquakes, floods, or climate *without* those
  methods are dropped. Applied AI (`deep learning for flood mapping`) is
  kept — that is AI research.
- **IEEE Xplore / Springer are optional**: both require free API key
  registration, so they're wired up but off by default.
- **NepJOL is not implemented**: it has no public API.
- **Semantic Scholar bulk search cannot return `authors.affiliations`**
  (the API rejects that field). Queries are therefore constrained with
  named Nepali institutions, and papers must still pass the AI-method
  gate. A free S2 API key enables paper/batch enrichment with real
  affiliations.
- **`--require-foreign-coauthor` is off by default** because the
  proposal's Dataset Description only requires ≥1 Nepal-affiliated author.

## Extending

- Add more `OPENALEX_SUBFIELD_IDS` / `AI_KEYWORDS` in `config.py` if you're
  short of the target after all sources run.
- Add institution names to `nepal_institutions.py`'s `NEPAL_KEYWORDS` if
  you spot a Nepali college/university missing from the list — it affects
  the Semantic Scholar fallback filter and the `nepal_affiliations` column.
