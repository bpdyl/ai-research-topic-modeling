"""
OpenAlex client — this is the PRIMARY source for this engine.

Why OpenAlex over Semantic Scholar as primary: OpenAlex lets you filter
server-side on `authorships.countries:np`, which is a structured signal
for "at least one Nepal-affiliated author" — far more reliable than
text-matching a free-text affiliation string. It's free, needs no API key
(just a courtesy email), and covers essentially the same corpus as
Semantic Scholar/Crossref.

Concepts (`concepts.id`) are deprecated and frozen; this client filters
on Topics subfields instead.

Docs: https://docs.openalex.org/api-entities/works
"""

import time
import logging
import requests

import config
from http_util import get_json
from ai_relevance import is_ai_paper
from nepal_institutions import any_author_foreign_affiliated, any_author_nepal_affiliated
from pipeline import coerce_year, normalize_doi

logger = logging.getLogger(__name__)


def _reconstruct_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]} to save space."""
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in range(min(positions), max(positions) + 1) if i in positions)


def build_openalex_filter(year_start, year_end, subfield_ids=None, work_types=None, keyword=None):
    """Build the OpenAlex `filter=` string. Exported so tests can lock the shape."""
    if subfield_ids is None:
        subfield_ids = config.OPENALEX_SUBFIELD_IDS
    if work_types is None:
        work_types = config.OPENALEX_WORK_TYPES
    parts = [
        "authorships.countries:np",
        f"publication_year:{year_start}-{year_end}",
    ]
    if subfield_ids:
        field = getattr(config, "OPENALEX_TOPIC_FIELD", "primary_topic.subfield.id")
        parts.append(field + ":" + "|".join(subfield_ids))
    if work_types:
        parts.append(f"type:{work_types}")
    if keyword:
        # Restrict the keyword match to title+abstract, not full text.
        parts.append(f"title_and_abstract.search:{keyword}")
    return ",".join(parts)


def _normalize_work(work):
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        insts = authorship.get("institutions") or []
        raw = authorship.get("raw_affiliation_strings") or []
        affiliations = []
        seen = set()
        for name in [i.get("display_name") for i in insts if i.get("display_name")] + list(raw):
            if name and name not in seen:
                seen.add(name)
                affiliations.append(name)
        country_codes = []
        for code in authorship.get("countries") or []:
            if code and code not in country_codes:
                country_codes.append(code)
        for inst in insts:
            code = inst.get("country_code")
            if code and code not in country_codes:
                country_codes.append(code)
        authors.append(
            {
                "name": author.get("display_name"),
                "affiliations": affiliations,
                "country_codes": country_codes,
            }
        )

    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    landing = loc.get("landing_page_url")
    topic = work.get("primary_topic") or {}
    subfield = topic.get("subfield") or {}

    return {
        "paper_id": work.get("id"),
        "doi": normalize_doi(work.get("doi")),
        "title": work.get("title") or work.get("display_name"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": coerce_year(work.get("publication_year")),
        "venue": source.get("display_name") if source else None,
        "authors": authors,
        "url": landing or work.get("id"),
        "source": "openalex",
        "primary_topic": topic.get("display_name"),
        "primary_subfield": subfield.get("display_name"),
    }


def _session(email):
    session = requests.Session()
    ua_email = email if email and "@" in email else "nepal-ai-scraper"
    session.headers.update(
        {
            "User-Agent": f"nepal-ai-scraper (mailto:{ua_email})",
            "Accept": "application/json",
        }
    )
    return session


def fetch_openalex_papers(
    concept_ids=None,  # accepted but ignored; kept so older call sites don't explode
    subfield_ids=None,
    year_start=None,
    year_end=None,
    email=None,
    per_page=None,
    max_results=None,
    keyword=None,
    require_foreign_coauthor=False,
):
    """
    Pulls works with >=1 author affiliated with Nepal (authorships.countries:np)
    within AI-related topic subfields and a year range. Uses cursor pagination.

    Returns a list of normalized paper dicts (see _normalize_work).
    """
    if concept_ids:
        logger.warning("OpenAlex concepts are deprecated; ignoring concept_ids=%s", concept_ids)

    subfield_ids = subfield_ids if subfield_ids is not None else config.OPENALEX_SUBFIELD_IDS
    year_start = year_start if year_start is not None else config.YEAR_START
    year_end = year_end if year_end is not None else config.YEAR_END
    email = email or config.OPENALEX_EMAIL
    per_page = per_page or config.OPENALEX_PER_PAGE
    if max_results:
        per_page = min(per_page, max_results)

    filter_str = build_openalex_filter(year_start, year_end, subfield_ids=subfield_ids, keyword=keyword)
    logger.info("OpenAlex filter: %s", filter_str)

    params = {
        "filter": filter_str,
        "per-page": per_page,
        "cursor": "*",
        "select": (
            "id,doi,title,display_name,abstract_inverted_index,"
            "publication_year,primary_location,authorships,primary_topic"
        ),
    }
    if email and "@" in email:
        params["mailto"] = email

    results = []
    session = _session(email)

    while True:
        data = get_json(session, config.OPENALEX_BASE_URL, params=params)
        page = data.get("results") or []
        for work in page:
            paper = _normalize_work(work)
            if not any_author_nepal_affiliated(paper.get("authors")):
                continue
            if not is_ai_paper(paper.get("title"), paper.get("abstract")):
                continue
            if require_foreign_coauthor and not any_author_foreign_affiliated(paper.get("authors")):
                continue
            results.append(paper)
            if max_results and len(results) >= max_results:
                logger.info("OpenAlex returned %d papers (capped)", len(results))
                return results

        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor or not page:
            break
        params["cursor"] = next_cursor
        time.sleep(config.OPENALEX_REQUEST_DELAY)

    logger.info("OpenAlex returned %d papers", len(results))
    return results
