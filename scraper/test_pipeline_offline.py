"""
Offline sanity tests — no network calls. Run with: python test_pipeline_offline.py
Exercises the parts of the engine that don't need internet access: abstract
reconstruction, title normalization, dedup merging, affiliation matching,
CSV nepal_affiliations, resume merge, and OpenAlex filter shape.
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_relevance import is_ai_paper
from sources.openalex import _reconstruct_abstract, build_openalex_filter
from pipeline import (
    normalize_title,
    merge_and_dedupe,
    time_window_for_year,
    nepal_affiliation_strings,
    filter_require_foreign_coauthor,
    export_to_csv,
    load_previous_export,
    normalize_doi,
    coerce_year,
)
from nepal_institutions import (
    is_nepal_affiliated,
    any_author_nepal_affiliated,
    any_author_foreign_affiliated,
    mentions_nepal_institution,
    author_is_nepal_affiliated,
)
import config

passed = 0
failed = 0


def check(label, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  OK   {label}")
    else:
        failed += 1
        print(f"  FAIL {label}")


print("Abstract reconstruction:")
inv_idx = {"Deep": [0], "learning": [1], "for": [2], "Nepali": [3], "NLP": [4]}
check("reassembles words in order", _reconstruct_abstract(inv_idx) == "Deep learning for Nepali NLP")
check("empty index -> empty string", _reconstruct_abstract({}) == "")
check("None -> empty string", _reconstruct_abstract(None) == "")
gapped = {"Deep": [0], "NLP": [4]}
check("skips missing positions rather than crashing", _reconstruct_abstract(gapped) == "Deep NLP")

print("\nTitle normalization:")
check(
    "punctuation/case/whitespace collapse to same key",
    normalize_title("GA-Optimized LDA: A Study!") == normalize_title("ga optimized lda a study"),
)
check("empty title -> empty string", normalize_title("") == "")
check("None title -> empty string", normalize_title(None) == "")

print("\nDOI / year helpers:")
check("strips doi.org prefix", normalize_doi("https://doi.org/10.1/ABC") == "10.1/abc")
check("empty doi -> None", normalize_doi("") is None)
check("year from date string", coerce_year("2022-05-01") == 2022)
check("year int passthrough", coerce_year(2019) == 2019)

print("\nNepal affiliation matching:")
check("Tribhuvan University matches", is_nepal_affiliated(["Department of CS, Tribhuvan University"]))
check("Kathmandu, Nepal matches", is_nepal_affiliated(["Some Lab, Kathmandu, Nepal"]))
check("MIT does not match", not is_nepal_affiliated(["MIT, Cambridge, MA"]))
check("empty list does not match", not is_nepal_affiliated([]))
check("Lord Buddha Education Foundation matches", is_nepal_affiliated(["Lord Buddha Education Foundation, Kathmandu"]))
check("generic IOE in Jaipur does NOT match", not is_nepal_affiliated(["Institute of Engineering", "Arya Institute of Engineering and Technology, Jaipur, Rajasthan"]))
check("IOE Pulchowk does match", is_nepal_affiliated(["Institute of Engineering", "Pulchowk Campus, Lalitpur, Nepal"]))
check(
    "Zhejiang Agriculture and Forestry University does NOT match",
    not is_nepal_affiliated([
        "Zhejiang A & F University",
        "Agriculture and Forestry University",
        "Zhejiang Agriculture and Forestry University, Hangzhou 311300, China",
    ]),
)
check("country code NP counts as Nepal author", author_is_nepal_affiliated({"name": "A", "affiliations": [], "country_codes": ["NP"]}))
check(
    "NP country code is not enough when affiliation text is foreign",
    not author_is_nepal_affiliated({
        "name": "Ranu Sewada",
        "affiliations": ["Institute of Engineering", "Computer Science Engineering Arya Institute of Engineering and Technology, Jaipur, Rajasthan"],
        "country_codes": ["NP"],
    }),
)

authors_mixed = [
    {"name": "A. Sharma", "affiliations": ["Institute of Engineering, Pulchowk Campus"], "country_codes": ["NP"]},
    {"name": "J. Smith", "affiliations": ["Stanford University"], "country_codes": ["US"]},
]
check("mixed author list -> Nepal author detected", any_author_nepal_affiliated(authors_mixed))
check("mixed author list -> foreign author detected", any_author_foreign_affiliated(authors_mixed))

authors_nepal_only = [{"name": "A. Sharma", "affiliations": ["Tribhuvan University"], "country_codes": ["NP"]}]
check("Nepal-only list -> no foreign author", not any_author_foreign_affiliated(authors_nepal_only))

authors_codes_only = [
    {"name": "A. Sharma", "affiliations": [], "country_codes": ["NP"]},
    {"name": "J. Smith", "affiliations": [], "country_codes": ["IN"]},
]
check("country codes alone detect Nepal + foreign", any_author_nepal_affiliated(authors_codes_only) and any_author_foreign_affiliated(authors_codes_only))

print("\nTitle/abstract institution fallback (S2 without affiliations):")
check("Tribhuvan in title matches", mentions_nepal_institution("A study at Tribhuvan University"))
check("bare 'Nepal' in title does NOT match", not mentions_nepal_institution("Earthquake risk mapping in Nepal"))
check("NAAMII matches", mentions_nepal_institution("Dataset released by NAAMII"))

print("\nTime windows:")
check("2016 -> 2015-2017", time_window_for_year(2016) == "2015-2017")
check("2025 -> 2024-2026", time_window_for_year(2025) == "2024-2026")
check("None year -> None", time_window_for_year(None) is None)

print("\nMerge + dedupe:")
list_a = [
    {"paper_id": "A1", "doi": "10.1/abc", "title": "GA Optimized LDA for Nepal AI", "abstract": "", "year": 2022,
     "venue": None, "authors": [], "url": None, "source": "openalex"},
]
list_b = [
    {"paper_id": "B1", "doi": "https://doi.org/10.1/abc", "title": "GA-Optimized LDA for Nepal AI!", "abstract": "Full abstract text.",
     "year": 2022, "venue": "IEEE Access", "authors": [{"name": "A. Sharma", "affiliations": []}],
     "url": "http://example.com", "source": "semantic_scholar"},
    {"paper_id": "B2", "doi": None, "title": "Unrelated Paper About Rice Yield Prediction", "abstract": "",
     "year": 2019, "venue": None, "authors": [], "url": None, "source": "semantic_scholar"},
]
merged = merge_and_dedupe(list_a, list_b)
check("dedupes same DOI across sources into one record", len(merged) == 2)
merged_target = next(p for p in merged if p["doi"] == "10.1/abc")
check("gap-filling pulls abstract from second source", merged_target["abstract"] == "Full abstract text.")
check("gap-filling pulls venue from second source", merged_target["venue"] == "IEEE Access")
check("first-source title is kept (not overwritten)", merged_target["title"] == "GA Optimized LDA for Nepal AI")
check("time_window tag applied", merged_target["time_window"] == "2021-2023")

print("\nResume merge is additive (not a drop-old-rows filter):")
previous = [
    {"paper_id": "OLD", "doi": "10.old/1", "title": "Already Collected Paper", "abstract": "x", "year": 2018,
     "venue": None, "authors": [], "url": None, "source": "openalex"},
]
new_batch = [
    {"paper_id": "NEW", "doi": "10.new/1", "title": "Fresh Paper", "abstract": "y", "year": 2024,
     "venue": None, "authors": [], "url": None, "source": "openalex"},
    {"paper_id": "DUP", "doi": "10.old/1", "title": "Already Collected Paper", "abstract": "", "year": 2018,
     "venue": None, "authors": [], "url": None, "source": "semantic_scholar"},
]
resumed = merge_and_dedupe(previous, new_batch)
check("resume keeps previous papers and adds new ones", len(resumed) == 2)
check("resume does not duplicate the old DOI", sum(1 for p in resumed if p["doi"] == "10.old/1") == 1)

print("\nCSV nepal_affiliations is Nepal-only:")
mixed_paper = {
    "paper_id": "M1", "doi": None, "title": "T", "abstract": "", "year": 2021, "time_window": "2021-2023",
    "venue": None, "url": None, "source": "openalex",
    "authors": [
        {"name": "A. Sharma", "affiliations": ["Tribhuvan University", "Stanford University"], "country_codes": ["NP", "US"]},
        {"name": "J. Smith", "affiliations": ["MIT"], "country_codes": ["US"]},
    ],
}
affs = nepal_affiliation_strings(mixed_paper)
check("keeps Tribhuvan, drops Stanford and MIT", affs == ["Tribhuvan University"])

print("\nForeign-coauthor filter:")
only_nepal = [{"authors": authors_nepal_only, "title": "n", "doi": None, "year": 2020, "source": "x"}]
mixed = [{"authors": authors_mixed, "title": "m", "doi": None, "year": 2020, "source": "x"}]
check("drops Nepal-only papers", filter_require_foreign_coauthor(only_nepal) == [])
check("keeps mixed papers", len(filter_require_foreign_coauthor(mixed)) == 1)

print("\nOpenAlex filter shape (primary AI/CV topics, not deprecated concepts):")
filt = build_openalex_filter(2015, 2026)
check("uses authorships.countries:np", "authorships.countries:np" in filt)
check("uses primary_topic.subfield.id", "primary_topic.subfield.id:" in filt)
check("does not use loose topics.subfield", "topics.subfield.id" not in filt)
check("does not use concepts.id", "concepts.id" not in filt)
check("includes AI subfield 1702", "1702" in filt)
check("includes CV subfield 1707", "1707" in filt)
check("does not include broad CS Applications 1706", "1706" not in filt)
kw = build_openalex_filter(2015, 2026, subfield_ids=[], keyword="machine learning")
check("keyword pass omits subfield constraint", "subfield" not in kw)
check("keyword pass includes title_and_abstract.search", "title_and_abstract.search:machine learning" in kw)

print("\nStrict AI-method filter:")
check(
    "deep learning for floods is AI research",
    is_ai_paper("Deep learning for flood mapping in the Koshi basin", "We train a CNN on satellite imagery."),
)
check(
    "Gorkha earthquake observations are not AI",
    not is_ai_paper(
        "Strong-Motion Observations of the M 7.8 Gorkha, Nepal, Earthquake Sequence",
        "We present accelerometer recordings and peak ground acceleration.",
    ),
)
check(
    "solar PV dust study is not AI",
    not is_ai_paper("Dust accumulation effects on efficiency of solar PV modules", "A case study of Kathmandu."),
)
check("BERT NLP paper is AI", is_ai_paper("BERT for Nepali named entity recognition", ""))
check("LLM paper is AI", is_ai_paper("Evaluating large language models on Nepali QA", ""))
check("title abbreviation AI counts", is_ai_paper("An AI-based diagnostic system for TB", ""))

print("\nSemantic Scholar bulk fields are actually supported:")
check("bulk fields omit authors.affiliations", "affiliations" not in config.SEMANTIC_SCHOLAR_BULK_FIELDS)

print("\nload_previous_export on missing file:")
check("missing JSON -> empty list", load_previous_export(os.path.join(tempfile.gettempdir(), "no-such-nepal-ai.json")) == [])

print("\nCSV export writes Nepal-only affiliation column:")
buf_dir = tempfile.mkdtemp()
csv_path = os.path.join(buf_dir, "t.csv")
export_to_csv([mixed_paper], path=csv_path)
with open(csv_path, encoding="utf-8") as f:
    row = next(csv.DictReader(f))
check("CSV nepal_affiliations excludes Stanford/MIT", row["nepal_affiliations"] == "Tribhuvan University")
check("CSV authors is semicolon-joined names", "A. Sharma" in row["authors"] and "J. Smith" in row["authors"])

print(f"\n{passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
