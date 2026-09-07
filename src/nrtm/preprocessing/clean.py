"""
Text cleaning — the stage SHARED by both representation branches.

This is deliberately conservative. It removes artefacts of the harvesting
process (boilerplate prefixes, copyright tails, URLs) and normalises Unicode,
but it does **not** lowercase, strip punctuation or drop stop words. Those are
LDA-specific and happen in `tokenize.py`.

Why the split matters (OQ-015 / RISK-008): BERTopic embeds documents with a
sentence transformer, which depends on natural word order, casing and function
words. Feeding it the LDA token stream would cripple the benchmark model and
make the comparison at the heart of this project meaningless. One cleaning
stage, two representations.

Counts below are measured on the frozen 1,088-document corpus
(see data/processed/corpus_report.json).
"""

from __future__ import annotations

import re
import unicodedata

# "Abstract", "ABSTRACT:", "Abstract -" etc. at the very start. 71 documents.
LEADING_ABSTRACT_RE = re.compile(r"^\s*abstract\b[\s:.–—-]*", re.IGNORECASE)

# Publisher boilerplate that carries no topical signal.
COPYRIGHT_RE = re.compile(
    r"(?i)(©\s*\d{4}.*$|copyright\s+\d{4}.*$|all rights reserved.*$"
    r"|this article is licensed under.*$|creative commons.*$)"
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DOI_RE = re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Section headers publishers inline into abstracts ("Background: ... Methods: ...").
# Only stripped when they appear as a standalone labelled header.
STRUCTURED_HEADER_RE = re.compile(
    r"(?i)\b(background|objectives?|methods?|materials and methods|results?"
    r"|conclusions?|findings|purpose|aim|introduction|discussion)\s*:\s*"
)

# Parenthesised acronym glosses: "artificial intelligence (AI)".
# Removing the gloss stops phrase detection from producing BOTH
# `artificial_intelligence` and `artificial_intelligence_ai` for one concept —
# measured at 191 vs 160 documents before this was added. The acronym is not
# lost: authors who define one almost always use it standalone later, and it is
# in the protected-acronym list either way.
PAREN_ACRONYM_RE = re.compile(r"\(\s*[A-Z][A-Za-z]{1,7}s?\s*\)")

WHITESPACE_RE = re.compile(r"\s+")
# Control characters that survive harvesting and fragment tokens.
# Built numerically: escape sequences in this range are easy to corrupt.
_CONTROL_CODES = list(range(0, 9)) + [11, 12] + list(range(14, 32)) + [127]
CONTROL_TABLE = {c: " " for c in _CONTROL_CODES}


def normalise_unicode(text: str) -> str:
    """NFKC-normalise, then map the typographic characters that survive it.

    NFKC does not fold curly quotes or dashes, and those otherwise fragment
    tokens ('don’t' vs "don't"). 394 documents contain non-ASCII characters.
    Accented Latin characters in author names and loanwords are preserved —
    stripping to ASCII would corrupt them.
    """
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "-", "―": "-", "−": "-",
        "…": "...", " ": " ", "​": "", "﻿": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(
    text: str,
    strip_leading_abstract: bool = True,
    strip_copyright: bool = True,
    strip_structured_headers: bool = True,
    strip_paren_acronyms: bool = True,
) -> str:
    """Clean one document. Returns natural-language text, case and punctuation intact."""
    if not text:
        return ""

    text = normalise_unicode(text)
    text = text.translate(CONTROL_TABLE)

    if strip_copyright:
        text = COPYRIGHT_RE.sub(" ", text)

    text = URL_RE.sub(" ", text)
    text = DOI_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)

    if strip_leading_abstract:
        text = LEADING_ABSTRACT_RE.sub("", text)

    if strip_structured_headers:
        text = STRUCTURED_HEADER_RE.sub(" ", text)

    if strip_paren_acronyms:
        text = PAREN_ACRONYM_RE.sub(" ", text)

    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_documents(texts, **kwargs) -> list[str]:
    return [clean_text(t, **kwargs) for t in texts]
