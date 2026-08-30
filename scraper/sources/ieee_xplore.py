"""
IEEE Xplore client — OPTIONAL supplementary source.

Requires a free API key from https://developer.ieee.org (self-serve
registration, instant approval, ~200 calls/day on the free tier).
Set config.IEEE_API_KEY (or pass api_key=) to enable; otherwise
fetch_ieee_papers() returns an empty list and logs a skip notice.

Docs: https://developer.ieee.org/docs/read/Metadata_API_details
"""

import time
import logging
import requests

import config
from http_util import get_json
from nepal_institutions import any_author_nepal_affiliated, any_author_foreign_affiliated
from pipeline import coerce_year, normalize_doi

logger = logging.getLogger(__name__)


def _normalize_article(article):
    authors = []
    for a in (article.get("authors") or {}).get("authors", []):
        aff = a.get("affiliation")
        authors.append(
            {
                "name": a.get("full_name"),
                "affiliations": [aff] if aff else [],
                "country_codes": [],
            }
        )
    return {
        "paper_id": str(article.get("article_number") or ""),
        "doi": normalize_doi(article.get("doi")),
        "title": article.get("title"),
        "abstract": article.get("abstract") or "",
        "year": coerce_year(article.get("publication_year")),
        "venue": article.get("publication_title"),
        "authors": authors,
        "url": article.get("html_url") or article.get("pdf_url"),
        "source": "ieee_xplore",
    }


def _ai_querytext():
    terms = " OR ".join(f'"{k}"' for k in config.AI_KEYWORDS)
    return f"({terms})"


def fetch_ieee_papers(year_start=None, year_end=None, api_key=None, max_results=500, require_foreign_coauthor=False):
    api_key = api_key or config.IEEE_API_KEY
    if not api_key:
        logger.info("IEEE_API_KEY not set — skipping IEEE Xplore (get a free key at developer.ieee.org)")
        return []

    year_start = year_start if year_start is not None else config.YEAR_START
    year_end = year_end if year_end is not None else config.YEAR_END

    results = []
    start_record = 1
    page_size = 200
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "nepal-ai-scraper"})

    while len(results) < max_results:
        params = {
            "apikey": api_key,
            "format": "json",
            "max_records": min(page_size, max_results - len(results)),
            "start_record": start_record,
            "sort_order": "asc",
            "sort_field": "article_number",
            "affiliation": "Nepal",
            "querytext": _ai_querytext(),
            "start_year": year_start,
            "end_year": year_end,
            "content_type": "Journals,Conferences",
        }
        data = get_json(session, config.IEEE_BASE_URL, params=params)
        articles = data.get("articles") or []
        if not articles:
            break
        for article in articles:
            paper = _normalize_article(article)
            authors = paper.get("authors") or []
            has_aff = any((a.get("affiliations") or []) for a in authors)
            if has_aff and not any_author_nepal_affiliated(authors):
                continue
            if require_foreign_coauthor and not any_author_foreign_affiliated(authors):
                continue
            results.append(paper)
            if len(results) >= max_results:
                break

        start_record += page_size
        total = int(data.get("total_records") or 0)
        if start_record > total:
            break
        time.sleep(0.5)

    logger.info("IEEE Xplore returned %d papers", len(results))
    return results[:max_results]
