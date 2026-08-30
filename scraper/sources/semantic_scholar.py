"""
Semantic Scholar client — SECONDARY source, used to fill gaps OpenAlex
misses and to cross-validate.

The bulk-search endpoint does not support nested author affiliation fields
(requesting `authors.affiliations` returns HTTP 400). Queries therefore
constrain to Nepal/institution names server-side. When an API key is
available, paper/batch is used to attach affiliations and filter properly.
Without a key, papers are kept only if a Nepali institution is named in
the title/abstract/venue (the bare word "Nepal" is not enough — that would
include foreign papers merely about Nepal).

Docs: https://api.semanticscholar.org/api-docs/graph
"""

import time
import logging
import requests

import config
from http_util import get_json, post_json
from ai_relevance import is_ai_paper
from nepal_institutions import (
    any_author_nepal_affiliated,
    any_author_foreign_affiliated,
    mentions_nepal_institution,
)
from pipeline import coerce_year, normalize_doi

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


def _normalize_paper(paper):
    authors = []
    for a in paper.get("authors") or []:
        affs = a.get("affiliations") or []
        # S2 sometimes returns affiliations as list[str], sometimes list[dict]
        cleaned = []
        for aff in affs:
            if isinstance(aff, str) and aff:
                cleaned.append(aff)
            elif isinstance(aff, dict):
                name = aff.get("name") or aff.get("affiliation")
                if name:
                    cleaned.append(name)
        authors.append(
            {
                "name": a.get("name"),
                "affiliations": cleaned,
                "country_codes": [],
            }
        )
    external_ids = paper.get("externalIds") or {}
    return {
        "paper_id": paper.get("paperId"),
        "doi": normalize_doi(external_ids.get("DOI")),
        "title": paper.get("title"),
        "abstract": paper.get("abstract") or "",
        "year": coerce_year(paper.get("year")),
        "venue": paper.get("venue"),
        "authors": authors,
        "url": paper.get("url"),
        "source": "semantic_scholar",
    }


def _keep_paper(paper, require_foreign_coauthor):
    if not is_ai_paper(paper.get("title"), paper.get("abstract")):
        return False
    authors = paper.get("authors") or []
    has_any_aff = any((a.get("affiliations") or []) for a in authors)
    if has_any_aff:
        if not any_author_nepal_affiliated(authors):
            return False
        if require_foreign_coauthor and not any_author_foreign_affiliated(authors):
            return False
        return True
    blob = " ".join(
        part for part in (paper.get("title"), paper.get("abstract"), paper.get("venue")) if part
    )
    return mentions_nepal_institution(blob)


def _session(api_key):
    session = requests.Session()
    headers = {"Accept": "application/json", "User-Agent": "nepal-ai-scraper"}
    if api_key:
        headers["x-api-key"] = api_key
    session.headers.update(headers)
    return session


def _enrich_affiliations(session, papers):
    """Best-effort paper/batch lookup to attach author affiliations."""
    ids = [p.get("paperId") for p in papers if p.get("paperId")]
    by_id = {}
    for i in range(0, len(ids), _BATCH_SIZE):
        chunk = ids[i : i + _BATCH_SIZE]
        try:
            data = post_json(
                session,
                config.SEMANTIC_SCHOLAR_BATCH_URL,
                payload={"ids": chunk},
                params={"fields": config.SEMANTIC_SCHOLAR_BATCH_FIELDS},
            )
        except Exception:
            logger.exception("Semantic Scholar paper/batch enrich failed for %d ids", len(chunk))
            break
        rows = data if isinstance(data, list) else (data.get("data") or [])
        for row in rows:
            if row and row.get("paperId"):
                by_id[row["paperId"]] = row
        time.sleep(config.SEMANTIC_SCHOLAR_REQUEST_DELAY)
    if not by_id:
        return papers
    merged = []
    for paper in papers:
        extra = by_id.get(paper.get("paperId"))
        merged.append(extra if extra else paper)
    return merged


def fetch_semantic_scholar_papers(
    queries=None,
    year_start=None,
    year_end=None,
    api_key=None,
    require_foreign_coauthor=None,
    max_results_per_query=None,
    enrich_affiliations=None,
):
    """
    Runs each query against the bulk search endpoint, constrained to Nepal,
    then keeps papers with a Nepal-affiliated author (or a named Nepali
    institution in title/abstract when affiliations are unavailable).
    """
    queries = queries or config.AI_KEYWORDS
    year_start = year_start if year_start is not None else config.YEAR_START
    year_end = year_end if year_end is not None else config.YEAR_END
    api_key = api_key or config.SEMANTIC_SCHOLAR_API_KEY
    require_foreign_coauthor = (
        config.REQUIRE_FOREIGN_COAUTHOR if require_foreign_coauthor is None else require_foreign_coauthor
    )
    if enrich_affiliations is None:
        enrich_affiliations = bool(api_key)

    session = _session(api_key)
    all_results = []
    seen_ids = set()

    for query in queries:
        # Constrain to named Nepali institutions. Bulk search cannot return
        # author affiliations, so `+ Nepal` alone would keep foreign papers
        # that are merely about Nepal.
        inst_or = " | ".join(f'"{t}"' for t in config.S2_INSTITUTION_TERMS)
        constrained = f'("{query}") + ({inst_or})'
        params = {
            "query": constrained,
            "fields": config.SEMANTIC_SCHOLAR_BULK_FIELDS,
            "year": f"{year_start}-{year_end}",
        }
        query_count = 0
        raw_page = []
        while True:
            data = get_json(session, config.SEMANTIC_SCHOLAR_BASE_URL, params=params)
            page = data.get("data") or []
            raw_page.extend(page)
            query_count += len(page)

            if max_results_per_query and query_count >= max_results_per_query:
                raw_page = raw_page[:max_results_per_query]
                break

            token = data.get("token")
            if not token or not page:
                break
            params["token"] = token
            time.sleep(config.SEMANTIC_SCHOLAR_REQUEST_DELAY)

        if enrich_affiliations and raw_page:
            raw_page = _enrich_affiliations(session, raw_page)

        kept = 0
        for paper in raw_page:
            pid = paper.get("paperId")
            if pid and pid in seen_ids:
                continue
            norm = _normalize_paper(paper)
            if not _keep_paper(norm, require_foreign_coauthor):
                continue
            if pid:
                seen_ids.add(pid)
            all_results.append(norm)
            kept += 1

        logger.info("Semantic Scholar query %r -> %d kept (%d raw)", constrained, kept, len(raw_page))
        time.sleep(config.SEMANTIC_SCHOLAR_REQUEST_DELAY)

    logger.info("Semantic Scholar total: %d papers", len(all_results))
    return all_results
