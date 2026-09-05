"""
Load the scraper's export and normalise it into modelling documents.

The scraper's JSON (scraper/output/nepal_ai_papers.json) is treated as
IMMUTABLE raw input. Nothing here writes back to it.

A "document" for topic modelling is `title + ". " + abstract`. Titles carry
strong topical signal in academic text and abstracts alone lose it, so both are
used — this matches the proposal ("titles and abstracts").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Document:
    """One paper, flattened to what the modelling stages need."""

    doc_id: str
    title: str
    abstract: str
    year: int | None
    venue: str | None
    doi: str | None
    url: str | None
    source: str | None
    primary_topic: str | None
    primary_subfield: str | None
    authors: list[str] = field(default_factory=list)
    nepal_affiliations: list[str] = field(default_factory=list)
    n_authors: int = 0
    time_window: str | None = None      # recomputed from config, not trusted from raw
    text: str = ""                      # title + abstract, set in __post_init__

    def __post_init__(self) -> None:
        title = (self.title or "").strip()
        abstract = (self.abstract or "").strip()
        if title and abstract:
            # Avoid doubling the separator when the title already ends in one.
            sep = " " if title[-1] in ".?!" else ". "
            self.text = f"{title}{sep}{abstract}"
        else:
            self.text = title or abstract

    @property
    def raw_tokens(self) -> int:
        """Whitespace token count of title+abstract, before any preprocessing.
        Used only for the minimum-length filter."""
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_raw(path: str | Path) -> list[dict[str, Any]]:
    """Read the scraper's JSON export."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw corpus not found: {path}\n"
            "Expected the scraper's export. Check paths.raw_corpus in config/default.yaml."
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of papers in {path}, got {type(data).__name__}")
    return data


def _nepal_affiliation_strings(record: dict[str, Any]) -> list[str]:
    """Affiliation strings that look Nepal-based.

    Deliberately a light re-implementation rather than an import from
    scraper/: the scraper is a standalone package with its own sys.path
    conventions, and coupling the modelling code to it would make both harder
    to run. The matching here only feeds the corpus report, never inclusion —
    inclusion was already decided by the scraper (DEC-006).
    """
    out: list[str] = []
    seen: set[str] = set()
    for author in record.get("authors") or []:
        for aff in author.get("affiliations") or []:
            if aff and "nepal" in aff.lower() and aff not in seen:
                seen.add(aff)
                out.append(aff)
    return out


def to_documents(records: Iterable[dict[str, Any]]) -> list[Document]:
    """Normalise raw scraper records into Document objects."""
    docs: list[Document] = []
    for i, r in enumerate(records):
        authors = [a.get("name") or "" for a in (r.get("authors") or [])]
        docs.append(
            Document(
                # paper_id is an OpenAlex URL and is unique across the corpus
                # (verified: 0 duplicates). Fall back to an index if absent.
                doc_id=str(r.get("paper_id") or f"doc_{i:05d}"),
                title=r.get("title") or "",
                abstract=r.get("abstract") or "",
                year=r.get("year"),
                venue=r.get("venue"),
                doi=r.get("doi"),
                url=r.get("url"),
                source=r.get("source"),
                primary_topic=r.get("primary_topic"),
                primary_subfield=r.get("primary_subfield"),
                authors=[a for a in authors if a],
                nepal_affiliations=_nepal_affiliation_strings(r),
                n_authors=len(r.get("authors") or []),
            )
        )
    return docs


def write_jsonl(docs: Iterable[Document], path: str | Path) -> Path:
    """Freeze documents to JSONL — one JSON object per line.

    JSONL over a single JSON array so the frozen corpus streams, diffs
    line-by-line in git, and stays readable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Document]:
    """Read a frozen corpus back. Every downstream stage loads via this."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen corpus not found: {path}\n"
            "Run: python scripts/01_build_corpus.py"
        )
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.pop("text", None)  # recomputed in __post_init__
            docs.append(Document(**d))
    return docs
