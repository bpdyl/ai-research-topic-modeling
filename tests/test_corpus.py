"""
Tests for corpus loading and filtering.

These deliberately cover the decisions that are easy to get silently wrong:
the year boundary (DEC-012), window recomputation (the scraper's final window
was 2024-2026, ours is 2024-2025), and the length floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.corpus.load import Document, to_documents, write_jsonl, read_jsonl  # noqa: E402
from nrtm.corpus.filters import apply_filters, time_window_for_year           # noqa: E402

WINDOWS = [[2015, 2017], [2018, 2020], [2021, 2023], [2024, 2025]]

CFG = {
    "corpus": {
        "year_min": 2015,
        "year_max": 2025,
        "require_abstract": True,
        "min_raw_tokens": 30,
        "time_windows": WINDOWS,
    }
}


def _doc(**kw) -> Document:
    base = dict(
        doc_id="d1", title="A study", abstract="word " * 40, year=2022,
        venue="Journal", doi=None, url=None, source="openalex",
        primary_topic=None, primary_subfield=None,
    )
    base.update(kw)
    return Document(**base)


class TestDocument:
    def test_text_joins_title_and_abstract(self):
        d = _doc(title="Deep learning in Nepal", abstract="We train a CNN.")
        assert d.text == "Deep learning in Nepal. We train a CNN."

    def test_text_does_not_double_terminal_punctuation(self):
        d = _doc(title="Is AI useful?", abstract="Yes.")
        assert d.text == "Is AI useful? Yes."

    def test_text_falls_back_to_title_when_no_abstract(self):
        assert _doc(title="Only a title", abstract="").text == "Only a title"

    def test_raw_tokens_counts_title_and_abstract(self):
        assert _doc(title="a b", abstract="c d e").raw_tokens == 5


class TestTimeWindows:
    def test_maps_years_into_the_configured_windows(self):
        assert time_window_for_year(2016, WINDOWS) == "2015-2017"
        assert time_window_for_year(2020, WINDOWS) == "2018-2020"
        assert time_window_for_year(2025, WINDOWS) == "2024-2025"

    def test_year_outside_all_windows_is_unassigned(self):
        # 2026 is excluded by DEC-012 and must not silently land in a window.
        assert time_window_for_year(2026, WINDOWS) is None
        assert time_window_for_year(None, WINDOWS) is None


class TestFilters:
    def test_drops_years_after_the_configured_max(self):
        docs = [_doc(doc_id="keep", year=2025), _doc(doc_id="drop", year=2026)]
        kept, _ = apply_filters(docs, CFG)
        assert [d.doc_id for d in kept] == ["keep"]

    def test_keeps_both_year_boundaries(self):
        docs = [_doc(doc_id="lo", year=2015), _doc(doc_id="hi", year=2025)]
        kept, _ = apply_filters(docs, CFG)
        assert {d.doc_id for d in kept} == {"lo", "hi"}

    def test_drops_documents_without_an_abstract(self):
        docs = [_doc(doc_id="ok"), _doc(doc_id="no_abs", abstract="")]
        kept, _ = apply_filters(docs, CFG)
        assert [d.doc_id for d in kept] == ["ok"]

    def test_drops_documents_below_the_length_floor(self):
        docs = [_doc(doc_id="long"), _doc(doc_id="short", abstract="tiny abstract")]
        kept, _ = apply_filters(docs, CFG)
        assert [d.doc_id for d in kept] == ["long"]

    def test_recomputes_window_ignoring_any_stale_raw_value(self):
        # The scraper tagged 2024-25 papers as "2024-2026"; that must be replaced.
        docs = [_doc(year=2024, time_window="2024-2026")]
        kept, _ = apply_filters(docs, CFG)
        assert kept[0].time_window == "2024-2025"

    def test_report_records_a_funnel(self):
        docs = [_doc(doc_id="ok"), _doc(doc_id="drop_year", year=2026),
                _doc(doc_id="drop_abs", abstract="")]
        _, report = apply_filters(docs, CFG)
        assert report.n_in == 3
        assert report.n_out == 1
        assert [s.name for s in report.steps] == [
            "year_range", "require_abstract", "min_raw_tokens"
        ]
        assert report.to_dict()["total_dropped"] == 2


class TestRoundTrip:
    def test_frozen_corpus_survives_a_write_read_cycle(self, tmp_path):
        docs = [_doc(doc_id="a", year=2019), _doc(doc_id="b", year=2023)]
        for d in docs:
            d.time_window = time_window_for_year(d.year, WINDOWS)
        path = tmp_path / "frozen.jsonl"
        write_jsonl(docs, path)
        back = read_jsonl(path)
        assert [d.doc_id for d in back] == ["a", "b"]
        assert back[0].time_window == "2018-2020"
        assert back[0].text == docs[0].text


class TestToDocuments:
    def test_extracts_authors_and_nepal_affiliations(self):
        raw = [{
            "paper_id": "W1", "title": "T", "abstract": "A", "year": 2021,
            "authors": [
                {"name": "A. Sharma", "affiliations": ["Tribhuvan University, Kathmandu, Nepal"]},
                {"name": "J. Smith", "affiliations": ["MIT"]},
            ],
        }]
        d = to_documents(raw)[0]
        assert d.n_authors == 2
        assert d.authors == ["A. Sharma", "J. Smith"]
        assert d.nepal_affiliations == ["Tribhuvan University, Kathmandu, Nepal"]

    def test_generates_an_id_when_paper_id_is_missing(self):
        d = to_documents([{"title": "T", "abstract": "A"}])[0]
        assert d.doc_id == "doc_00000"
