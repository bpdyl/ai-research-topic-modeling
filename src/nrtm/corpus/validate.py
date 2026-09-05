"""
Corpus validation report.

The output of this module is not a debugging aid — it is the raw material for
the paper's "Problem and Dataset description" section, and the evidence behind
marking criterion 2.3 (data preparation). It deliberately reports the corpus's
weaknesses (preprint share, window imbalance) rather than only its size,
because those must be stated in Limitations either way.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Sequence

from nrtm.corpus.load import Document

# Venues that are preprint servers rather than peer-reviewed outlets.
PREPRINT_VENUE_RE = re.compile(
    r"(?i)\b(arxiv|ssrn|zenodo|research\s?square|preprints?\.org|biorxiv|medrxiv"
    r"|techrxiv|osf\b|authorea|figshare|hal\b)"
)
LEADING_ABSTRACT_RE = re.compile(r"(?i)^\s*abstract\b[:.\s-]*")
COPYRIGHT_RE = re.compile(r"(?i)(all rights reserved|©\s?\d{4}|copyright\s+\d{4})")


def _percentiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "min": s[0],
        "p25": s[n // 4],
        "median": s[n // 2],
        "p75": s[(3 * n) // 4],
        "max": s[-1],
        "mean": round(sum(s) / n, 1),
    }


def build_report(docs: Sequence[Document], cfg) -> dict[str, Any]:
    """Compute the full corpus report."""
    n = len(docs)
    if n == 0:
        return {"n_documents": 0, "error": "empty corpus"}

    windows = [f"{a}-{b}" for a, b in cfg["corpus"]["time_windows"]]

    # ── Temporal ───────────────────────────────────────────────────────
    years = Counter(d.year for d in docs)
    win_counts = Counter(d.time_window for d in docs)
    # DEC-013 requires per-window n to be reportable everywhere downstream.
    by_window = {w: win_counts.get(w, 0) for w in windows}
    unwindowed = n - sum(by_window.values())

    # ── Venue / peer-review composition (RISK-003) ─────────────────────
    preprint, no_venue, published = [], [], []
    for d in docs:
        venue = (d.venue or "").strip()
        if not venue:
            no_venue.append(d)
        elif PREPRINT_VENUE_RE.search(venue):
            preprint.append(d)
        else:
            published.append(d)

    # ── Length ─────────────────────────────────────────────────────────
    abs_words = [len((d.abstract or "").split()) for d in docs]
    text_words = [d.raw_tokens for d in docs]

    # ── Text hygiene (drives the preprocessing spec) ───────────────────
    leading_abstract = sum(1 for d in docs if LEADING_ABSTRACT_RE.match(d.abstract or ""))
    copyright_tail = sum(1 for d in docs if COPYRIGHT_RE.search(d.abstract or ""))
    non_ascii = sum(1 for d in docs if any(ord(c) > 127 for c in (d.abstract or "")))
    mostly_non_latin = 0
    for d in docs:
        letters = [c for c in (d.abstract or "") if c.isalpha()]
        if letters and sum(1 for c in letters if ord(c) > 127) / len(letters) > 0.3:
            mostly_non_latin += 1

    # ── Integrity ──────────────────────────────────────────────────────
    def _norm_title(t: str) -> str:
        t = unicodedata.normalize("NFKC", (t or "").lower())
        t = re.sub(r"[^\w\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    title_counts = Counter(_norm_title(d.title) for d in docs)
    dup_titles = sum(1 for c in title_counts.values() if c > 1)
    dup_ids = sum(1 for c in Counter(d.doc_id for d in docs).values() if c > 1)

    return {
        "n_documents": n,
        "temporal": {
            "years": dict(sorted((y, c) for y, c in years.items() if y is not None)),
            "by_window": by_window,
            "documents_outside_windows": unwindowed,
            "note": (
                "DEC-013: one global model is fitted on the whole corpus; these "
                "windows slice the REPORTING of topic proportions only. Per-window "
                "n must appear on every temporal table and figure."
            ),
        },
        "venue_composition": {
            "published": len(published),
            "preprint_server": len(preprint),
            "no_venue_recorded": len(no_venue),
            "pct_verifiably_published": round(100.0 * len(published) / n, 1),
            "top_preprint_venues": Counter(
                (d.venue or "") for d in preprint
            ).most_common(8),
            "distinct_venues": len({(d.venue or "").strip() for d in docs if (d.venue or "").strip()}),
            "note": (
                "The proposal describes the corpus as peer-reviewed and "
                "venue-verified. Report this composition honestly in the paper "
                "rather than restating an unqualified peer-review claim (RISK-003)."
            ),
        },
        "length": {
            "abstract_words": _percentiles(abs_words),
            "title_plus_abstract_words": _percentiles(text_words),
        },
        "coverage": {
            "with_doi": sum(1 for d in docs if d.doi),
            "with_venue": sum(1 for d in docs if (d.venue or "").strip()),
            "with_nepal_affiliation_string": sum(1 for d in docs if d.nepal_affiliations),
            "authors_per_paper": _percentiles([d.n_authors for d in docs]),
        },
        "text_hygiene": {
            "leading_abstract_prefix": leading_abstract,
            "copyright_boilerplate": copyright_tail,
            "contains_non_ascii": non_ascii,
            "mostly_non_latin_script": mostly_non_latin,
            "note": "These counts define the cleaning steps in preprocessing (RISK-013).",
        },
        "integrity": {
            "duplicate_normalised_titles": dup_titles,
            "duplicate_doc_ids": dup_ids,
            "empty_titles": sum(1 for d in docs if not (d.title or "").strip()),
            "empty_abstracts": sum(1 for d in docs if not (d.abstract or "").strip()),
        },
        "composition": {
            "top_primary_subfield": Counter(
                d.primary_subfield for d in docs if d.primary_subfield
            ).most_common(12),
            "top_primary_topic": Counter(
                d.primary_topic for d in docs if d.primary_topic
            ).most_common(12),
            "sources": dict(Counter(d.source for d in docs)),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    """Human-readable summary for the console and the run directory."""
    if not report.get("n_documents"):
        return "EMPTY CORPUS"

    L: list[str] = []
    L.append(f"Documents: {report['n_documents']}")

    t = report["temporal"]
    L.append("\nTime windows (reporting units, DEC-013):")
    for w, c in t["by_window"].items():
        L.append(f"  {w}: {c}")
    if t["documents_outside_windows"]:
        L.append(f"  (outside any window: {t['documents_outside_windows']})")

    v = report["venue_composition"]
    L.append(
        f"\nVenue composition: {v['published']} published, "
        f"{v['preprint_server']} preprint-server, {v['no_venue_recorded']} no venue "
        f"({v['pct_verifiably_published']}% verifiably published)"
    )
    L.append(f"  distinct venues: {v['distinct_venues']}")

    ln = report["length"]["title_plus_abstract_words"]
    L.append(
        f"\nDocument length (title+abstract, words): "
        f"min {ln['min']} / median {ln['median']} / max {ln['max']} (mean {ln['mean']})"
    )

    h = report["text_hygiene"]
    L.append(
        f"\nCleaning needed: {h['leading_abstract_prefix']} leading-'Abstract', "
        f"{h['copyright_boilerplate']} copyright tails, {h['contains_non_ascii']} non-ASCII"
    )

    i = report["integrity"]
    L.append(
        f"\nIntegrity: {i['duplicate_normalised_titles']} duplicate titles, "
        f"{i['duplicate_doc_ids']} duplicate ids, {i['empty_titles']} empty titles"
    )
    return "\n".join(L)
