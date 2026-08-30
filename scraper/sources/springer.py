"""
Springer Nature client — OPTIONAL supplementary source.

Requires a free API key from https://dev.springernature.com (self-serve,
5000 calls/day free tier). Set config.SPRINGER_API_KEY (or pass api_key=)
to enable; otherwise fetch_springer_papers() returns an empty list.

Springer's metadata API rarely exposes structured author-country, so this
queries with affiliation:Nepal plus AI keywords. Treat results as a
supplementary pool and rely on pipeline.merge_and_dedupe().

Docs: https://dev.springernature.com/docs
"""

import time
import logging
import requests

import config
from http_util import get_json
from nepal_institutions import any_author_foreign_affiliated, is_nepal_affiliated
from pipeline import coerce_year, normalize_doi

logger = logging.getLogger(__name__)


def _affiliations_from_record(record):
    affs = []
    for key in ("affiliations", "affiliation", "organization"):
        val = record.get(key)
        if isinstance(val, str) and val:
            affs.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item:
                    affs.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("affiliation") or item.get("organization")
                    if name:
                        affs.append(name)
    return affs


def _normalize_record(record):
    record_affs = _affiliations_from_record(record)
    authors = []
    for a in record.get("creators") or []:
        extra = []
        if isinstance(a, dict):
            for key in ("affiliation", "affiliations"):
                val = a.get(key)
                if isinstance(val, str) and val:
                    extra.append(val)
                elif isinstance(val, list):
                    extra.extend(x for x in val if isinstance(x, str) and x)
        authors.append(
            {
                "name": a.get("creator") if isinstance(a, dict) else str(a),
                "affiliations": extra or list(record_affs),
                "country_codes": [],
            }
        )
    urls = record.get("url")
    url = None
    if isinstance(urls, list) and urls:
        first = urls[0]
        url = first.get("value") if isinstance(first, dict) else str(first)
    elif isinstance(urls, str):
        url = urls
    return {
        "paper_id": record.get("identifier"),
        "doi": normalize_doi(record.get("doi")),
        "title": record.get("title"),
        "abstract": record.get("abstract") or "",
        "year": coerce_year(record.get("publicationDate")),
        "venue": record.get("publicationName"),
        "authors": authors,
        "url": url,
        "source": "springer",
    }


def fetch_springer_papers(year_start=None, year_end=None, api_key=None, max_results=500, require_foreign_coauthor=False):
    api_key = api_key or config.SPRINGER_API_KEY
    if not api_key:
        logger.info("SPRINGER_API_KEY not set — skipping Springer (get a free key at dev.springernature.com)")
        return []

    year_start = year_start if year_start is not None else config.YEAR_START
    year_end = year_end if year_end is not None else config.YEAR_END

    results = []
    start = 1
    page_size = 100
    query = (
        '(keyword:"artificial intelligence" OR keyword:"machine learning" '
        'OR keyword:"deep learning" OR keyword:"neural network") AND affiliation:Nepal'
    )
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "nepal-ai-scraper"})

    while len(results) < max_results:
        params = {
            "q": f"{query} AND onlinedatefrom:{year_start}-01-01 AND onlinedateto:{year_end}-12-31",
            "api_key": api_key,
            "p": page_size,
            "s": start,
        }
        data = get_json(session, config.SPRINGER_BASE_URL, params=params)
        records = data.get("records") or []
        if not records:
            break
        for record in records:
            paper = _normalize_record(record)
            blob_affs = [aff for a in paper.get("authors") or [] for aff in (a.get("affiliations") or [])]
            if blob_affs and not any(is_nepal_affiliated([aff]) for aff in blob_affs):
                continue
            if require_foreign_coauthor and not any_author_foreign_affiliated(paper.get("authors")):
                continue
            results.append(paper)
            if len(results) >= max_results:
                break

        start += page_size
        total = int((data.get("result") or [{}])[0].get("total", 0) or 0)
        if total and start > total:
            break
        if not records:
            break
        time.sleep(0.5)

    logger.info("Springer returned %d papers", len(results))
    return results[:max_results]
