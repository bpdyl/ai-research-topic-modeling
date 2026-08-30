"""
Central configuration for the Nepal-AI-Research scraping engine.
Edit the values below (or override via CLI flags in main.py) before running.
"""

# ── Corpus window ──────────────────────────────────────────────────
YEAR_START = 2015
YEAR_END = 2026            # inclusive
TARGET_PAPER_COUNT = 1000

# Temporal slices used later for topic-evolution tracking (matches the
# proposal's Section: Temporal Evolution Tracking). The scraper tags every
# paper with the slice it falls into so you don't have to re-bucket later.
# Proposal draft used 2024-2025 for the last window; 2026 is included here
# because the assignment asked for papers through 2026.
TIME_WINDOWS = [
    (2015, 2017),
    (2018, 2020),
    (2021, 2023),
    (2024, 2026),
]

# ── OpenAlex ────────────────────────────────────────────────────────
# OpenAlex needs no API key. Supplying an email puts you in the "polite
# pool" (faster, more reliable rate limits). Put yours here or pass
# --email on the command line.
OPENALEX_EMAIL = "your_email@example.com"
OPENALEX_BASE_URL = "https://api.openalex.org/works"

# OpenAlex Concepts are deprecated/frozen. Use the *primary* topic subfield
# (not topics.subfield, which matches any of three topics and pulls in
# earthquake/climate papers with a weak CS tag). 1706 Computer Science
# Applications is too broad and is intentionally omitted.
#   1702 Artificial Intelligence
#   1707 Computer Vision and Pattern Recognition
OPENALEX_TOPIC_FIELD = "primary_topic.subfield.id"
OPENALEX_SUBFIELD_IDS = [
    "1702",
    "1707",
]
OPENALEX_WORK_TYPES = "article|preprint|review|book-chapter"
OPENALEX_PER_PAGE = 200          # OpenAlex max page size
OPENALEX_REQUEST_DELAY = 0.2     # seconds between requests

# ── Semantic Scholar ─────────────────────────────────────────────────
# Bulk search is the only S2 endpoint that is usable without an API key.
# It does NOT support nested fields such as authors.affiliations (requesting
# them returns HTTP 400). Affiliation filtering therefore uses a Nepal-
# constrained query plus optional paper/batch enrichment when a key is set.
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SEMANTIC_SCHOLAR_API_KEY = None   # optional, register at semanticscholar.org/product/api
SEMANTIC_SCHOLAR_REQUEST_DELAY = 3.0   # unauthenticated limit is tight; raise if you hit 429s
SEMANTIC_SCHOLAR_BULK_FIELDS = (
    "title,abstract,year,venue,externalIds,url,authors"
)
SEMANTIC_SCHOLAR_BATCH_FIELDS = (
    "title,abstract,year,venue,externalIds,url,authors,authors.affiliations"
)
# Used to constrain S2 bulk search — that endpoint cannot return affiliations,
# so the query itself has to name Nepali institutions.
S2_INSTITUTION_TERMS = [
    "Tribhuvan University",
    "Kathmandu University",
    "Pokhara University",
    "Purbanchal University",
    "Pulchowk",
    "NAAMII",
    "Softwarica College",
    "Lord Buddha Education Foundation",
    "Herald College Kathmandu",
    "Nepal Academy of Science",
    "Kathmandu Engineering College",
]
# Broad AI/ML query terms used to pull candidate papers; each is run as a
# separate query and results are merged & deduped.
AI_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer vision",
    "generative AI",
    "large language model",
    "reinforcement learning",
]

# ── IEEE Xplore (optional — needs a free API key from developer.ieee.org) ──
IEEE_API_KEY = None
IEEE_BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

# ── Springer (optional — needs a free API key from dev.springernature.com) ─
SPRINGER_API_KEY = None
SPRINGER_BASE_URL = "https://api.springernature.com/meta/v2/json"

# ── Output ───────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
CHECKPOINT_FILE = "checkpoint_seen_ids.json"
CSV_FILE = "nepal_ai_papers.csv"
JSON_FILE = "nepal_ai_papers.json"

# If True, only keep papers that have >=1 Nepal-affiliated author AND
# >=1 author affiliated with a non-Nepal institution (i.e. genuine
# international co-authorship). If False (default), any paper with
# >=1 Nepal-affiliated author is kept, matching the proposal's stated
# inclusion rule.
REQUIRE_FOREIGN_COAUTHOR = False
