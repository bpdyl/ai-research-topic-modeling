"""
Merge, dedupe, checkpoint, and export helpers shared across sources.

Every source module (sources/openalex.py, sources/semantic_scholar.py, ...)
returns a list of dicts with the same schema:

{
    "paper_id": str,
    "doi": str | None,
    "title": str,
    "abstract": str,
    "year": int | None,
    "venue": str | None,
    "authors": [{"name": str, "affiliations": [str], "country_codes": [str]}],
    "url": str | None,
    "source": str,
}

This module merges lists from multiple sources into one deduped set.
"""

import os
import re
import json
import csv
import logging

import config
from nepal_institutions import any_author_foreign_affiliated, is_nepal_affiliated

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_DOI_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def normalize_doi(doi):
    if not doi:
        return None
    doi = _DOI_PREFIX_RE.sub("", str(doi).strip())
    return doi.lower() or None


def coerce_year(year):
    if year is None or year == "":
        return None
    if isinstance(year, int):
        return year
    s = str(year).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def normalize_title(title):
    """Lowercase, strip punctuation/whitespace so near-identical titles
    from different sources collapse to the same dedup key."""
    if not title:
        return ""
    t = title.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def time_window_for_year(year):
    year = coerce_year(year)
    if year is None:
        return None
    for start, end in config.TIME_WINDOWS:
        if start <= year <= end:
            return f"{start}-{end}"
    return None


def nepal_affiliation_strings(paper):
    """Affiliation strings that actually look Nepal-based (for the CSV column)."""
    out = []
    seen = set()
    for author in paper.get("authors") or []:
        for aff in author.get("affiliations") or []:
            if aff and is_nepal_affiliated([aff]) and aff not in seen:
                seen.add(aff)
                out.append(aff)
    return out


def filter_require_foreign_coauthor(papers):
    kept = []
    for paper in papers:
        if any_author_foreign_affiliated(paper.get("authors") or []):
            kept.append(paper)
    logger.info("Foreign-coauthor filter: %d -> %d papers", len(papers), len(kept))
    return kept


def merge_and_dedupe(*paper_lists):
    """
    Merges any number of paper-list results, deduping first on DOI
    (when present), then on normalized title. First occurrence wins;
    later sources fill in only if a field on the first is empty.
    """
    by_doi = {}
    by_title = {}
    ordered = []

    for papers in paper_lists:
        for paper in papers:
            paper = dict(paper)
            paper["year"] = coerce_year(paper.get("year"))
            paper["doi"] = normalize_doi(paper.get("doi"))
            paper["time_window"] = time_window_for_year(paper.get("year"))

            doi = paper.get("doi")
            norm_title = normalize_title(paper.get("title"))

            existing = None
            if doi and doi in by_doi:
                existing = by_doi[doi]
            elif norm_title and norm_title in by_title:
                existing = by_title[norm_title]

            if existing:
                # Fill gaps (e.g. OpenAlex has no abstract, S2 does) without
                # overwriting data we already have.
                for field in ("abstract", "venue", "doi", "url"):
                    if not existing.get(field) and paper.get(field):
                        existing[field] = paper[field]
                if not existing.get("authors") and paper.get("authors"):
                    existing["authors"] = paper["authors"]
                sources = existing.get("sources") or [existing.get("source")]
                extra = paper.get("source")
                if extra and extra not in sources:
                    sources.append(extra)
                existing["sources"] = [s for s in sources if s]
                continue

            paper["sources"] = [paper["source"]] if paper.get("source") else []
            ordered.append(paper)
            if doi:
                by_doi[doi] = paper
            if norm_title:
                by_title[norm_title] = paper

    logger.info("Merged %d papers after dedup", len(ordered))
    return ordered


def load_checkpoint(path=None):
    path = path or os.path.join(config.OUTPUT_DIR, config.CHECKPOINT_FILE)
    if not os.path.exists(path):
        return {"seen_dois": [], "seen_titles": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_previous_export(path=None):
    path = path or os.path.join(config.OUTPUT_DIR, config.JSON_FILE)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_checkpoint(papers, path=None):
    path = path or os.path.join(config.OUTPUT_DIR, config.CHECKPOINT_FILE)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    seen_dois = [p["doi"] for p in papers if p.get("doi")]
    seen_titles = [normalize_title(p["title"]) for p in papers if p.get("title")]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seen_dois": seen_dois, "seen_titles": seen_titles}, f)


def filter_already_seen(papers, checkpoint):
    seen_dois = set(checkpoint.get("seen_dois", []))
    seen_titles = set(checkpoint.get("seen_titles", []))
    fresh = []
    for p in papers:
        doi = (p.get("doi") or "").lower()
        title = normalize_title(p.get("title"))
        if doi and doi in seen_dois:
            continue
        if title and title in seen_titles:
            continue
        fresh.append(p)
    return fresh


def export_to_json(papers, path=None):
    path = path or os.path.join(config.OUTPUT_DIR, config.JSON_FILE)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d papers to %s", len(papers), path)


def export_to_csv(papers, path=None):
    path = path or os.path.join(config.OUTPUT_DIR, config.CSV_FILE)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "paper_id", "doi", "title", "abstract", "year", "time_window",
        "venue", "authors", "nepal_affiliations", "url", "source",
        "primary_topic", "primary_subfield",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in papers:
            author_names = "; ".join(a.get("name") or "" for a in p.get("authors", []))
            writer.writerow(
                {
                    "paper_id": p.get("paper_id"),
                    "doi": p.get("doi"),
                    "title": p.get("title"),
                    "abstract": p.get("abstract"),
                    "year": p.get("year"),
                    "time_window": p.get("time_window"),
                    "venue": p.get("venue"),
                    "authors": author_names,
                    "nepal_affiliations": "; ".join(nepal_affiliation_strings(p)),
                    "url": p.get("url"),
                    "source": p.get("source"),
                    "primary_topic": p.get("primary_topic"),
                    "primary_subfield": p.get("primary_subfield"),
                }
            )
    logger.info("Wrote %d papers to %s", len(papers), path)
