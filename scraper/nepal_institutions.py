"""
Heuristic matching for 'is this author affiliated with a Nepali institution?'

Used as a fallback / cross-check whenever a source doesn't give us a clean
structured country code (OpenAlex does; Semantic Scholar and most others
only give a free-text affiliation string, so we match against this list).

Extend NEPAL_KEYWORDS with any institutions relevant to your corpus.
"""

import re

NEPAL_KEYWORDS = [
    # Country name itself (safe for affiliation strings, too noisy for titles)
    "nepal",
    # Public universities
    "tribhuvan university",
    "kathmandu university",
    "pokhara university",
    "purbanchal university",
    "mid-western university",
    "mid western university nepal",
    "far-western university",
    "far western university",
    "nepal open university",
    "lumbini buddhist university",
    "gandaki university",
    "madan bhandari university",
    "rajarshi janak university",
    # Engineering campuses under Institute of Engineering (IOE)
    "institute of engineering, pulchowk",
    "institute of engineering, nepal",
    "ioe pulchowk",
    "pulchowk campus",
    "thapathali campus",
    "purwanchal campus",
    "paschimanchal campus",
    "western regional campus, pokhara",
    # Research institutes
    "naamii",
    "nepal applied mathematics and informatics institute",
    "nepal academy of science and technology",
    "nepal agricultural research council",
    "nast nepal",
    # Private colleges commonly producing CS/AI research
    "softwarica college",
    "islington college",
    "herald college kathmandu",
    "the british college",
    "lord buddha education foundation",
    "kathmandu engineering college",
    "nepal college of information technology",
    "ncit nepal",
    "khwopa college",
    "kantipur engineering college",
    "himalaya college of engineering",
    "advanced college of engineering and management",
    "deerwalk",
    "nepal engineering college",
    "samriddhi college",
]

_CITY_AFFILIATION_KEYWORDS = [
    "kathmandu, nepal",
    "lalitpur, nepal",
    "pokhara, nepal",
    "biratnagar, nepal",
    "dharan, nepal",
    "bharatpur, nepal",
    "khumaltar, nepal",
    "dhulikhel, nepal",
]

NEPAL_KEYWORDS.extend(_CITY_AFFILIATION_KEYWORDS)

# Bare country / city-country strings are useful in affiliation lines but
# match too many "papers about Nepal" in titles/abstracts.
_TITLE_UNSAFE = {
    "nepal",
    "kathmandu, nepal",
    "lalitpur, nepal",
    "pokhara, nepal",
    "biratnagar, nepal",
    "dharan, nepal",
    "bharatpur, nepal",
    "khumaltar, nepal",
    "dhulikhel, nepal",
}

_PATTERN = re.compile("|".join(re.escape(k) for k in NEPAL_KEYWORDS), re.IGNORECASE)
_INSTITUTION_PATTERN = re.compile(
    "|".join(re.escape(k) for k in NEPAL_KEYWORDS if k not in _TITLE_UNSAFE),
    re.IGNORECASE,
)


_GENERIC_PHRASE_RE = re.compile(
    r"institute of engineering|agriculture and forestry university",
    re.IGNORECASE,
)
_NEPAL_CONTEXT_RE = re.compile(
    r"nepal|pulchowk|tribhuvan|kathmandu|lalitpur|pokhara|dhulikhel|khumaltar|chitwan|rampur",
    re.IGNORECASE,
)


def is_nepal_affiliated(affiliation_strings):
    """
    affiliation_strings: iterable of free-text affiliation strings for a
    single author (or a list of authors' affiliation lists flattened).
    Returns True if any string matches a known Nepal keyword.
    """
    if not affiliation_strings:
        return False
    parts = [s for s in affiliation_strings if s]
    for s in parts:
        if _PATTERN.search(s):
            return True
    # "Institute of Engineering" exists in many countries. Only accept it
    # when the same author's affiliation blob is Nepal-flavoured.
    combined = " ".join(parts)
    if _GENERIC_PHRASE_RE.search(combined) and _NEPAL_CONTEXT_RE.search(combined):
        return True
    return False


def mentions_nepal_institution(text):
    """True if text names a Nepali institution (not merely the word 'Nepal')."""
    if not text:
        return False
    return bool(_INSTITUTION_PATTERN.search(text))


def author_is_nepal_affiliated(author):
    """
    Single author dict. Prefers structured country codes (OpenAlex), then
    free-text affiliation strings.
    """
    if not author:
        return False
    affs = author.get("affiliations") or []
    if is_nepal_affiliated(affs):
        return True
    # Trust a bare NP country code only when there is no affiliation text
    # to contradict it. OpenAlex sometimes tags non-Nepal "Institute of
    # Engineering" campuses as NP; those rows have foreign address lines.
    codes = [str(c).upper() for c in (author.get("country_codes") or []) if c]
    if "NP" in codes and not affs:
        return True
    return False


def any_author_nepal_affiliated(authors):
    """
    authors: list of author dicts, each expected to optionally have an
    'affiliations' key (list[str]) and/or 'country_codes'.
    """
    return any(author_is_nepal_affiliated(a) for a in (authors or []))


def any_author_foreign_affiliated(authors):
    """
    True if at least one author looks non-Nepal (has affiliation/country data
    and is not Nepal-affiliated). Authors with no affiliation data are ignored.
    """
    for author in authors or []:
        affs = author.get("affiliations") or []
        codes = [str(c).upper() for c in (author.get("country_codes") or []) if c]
        if not affs and not codes:
            continue
        if not author_is_nepal_affiliated(author):
            return True
    return False
