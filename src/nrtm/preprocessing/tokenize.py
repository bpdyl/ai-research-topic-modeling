"""
Tokenisation for the LDA / GA-LDA branch.

BERTopic does NOT use this module — it consumes cleaned natural text from
`clean.py`. See OQ-015 / RISK-008.

Pipeline: lowercase -> tokenise -> drop numbers/short tokens -> drop stop words
-> lemmatise -> (phrase detection, in `represent.py`).

**Acronym preservation** is an explicit proposal commitment ("preserving
critical technical acronyms and localized terms"). Three things would
otherwise destroy them:

1. the minimum-token-length filter removes `ai`, `ml`, `cv`, `nn` (2 chars);
2. the WordNet lemmatiser maps `gans` -> `gan` but also `cnns` -> `cnns` and,
   worse, `ai` -> `ai` only by luck — it is not acronym-aware;
3. `nlp`/`llm` are absent from any stop-word list but are short enough to be
   caught by naive filters.

Protected tokens bypass the length filter and the lemmatiser entirely.

Known limitation, worth one line in the paper: matching is case-insensitive
after lowercasing, so a lowercase English word that collides with a protected
acronym is also kept. In this corpus the realistic collision is `cv`
(computer vision / curriculum vitae / cross-validation). Given the corpus is
AI research, the AI reading dominates; the ambiguity is acknowledged rather
than silently resolved.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Words: letters only, with optional internal apostrophe. Hyphenated compounds
# are split so phrase detection can rejoin the ones that matter
# ("deep-learning" -> deep, learning -> deep_learning); this keeps the
# vocabulary from fragmenting on inconsistent hyphenation.
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Plural acronyms map to their singular so `cnns` and `cnn` are one type.
_ACRONYM_PLURAL_RE = re.compile(r"^([a-z]{2,6})s$")


@lru_cache(maxsize=1)
def _wordnet_lemmatiser():
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _nltk_stopwords() -> frozenset[str]:
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


def build_stopwords(domain_stopwords=(), keep=()) -> frozenset[str]:
    """NLTK English stop words plus the corpus-specific list from config.

    `keep` (the protected acronyms) is subtracted last, so a domain stop word
    can never shadow a protected acronym.
    """
    words = set(_nltk_stopwords()) | {w.lower() for w in domain_stopwords}
    return frozenset(words - {k.lower() for k in keep})


def lemmatise(token: str) -> str:
    """WordNet lemmatisation, noun then verb.

    Two passes because the lemmatiser defaults to nouns and would leave verb
    forms untouched ('trained' stays 'trained'). POS tagging would be more
    accurate but costs a tagger download and a large speed penalty across
    ~900 GA fitness evaluations; noun-then-verb is the standard compromise.
    """
    lem = _wordnet_lemmatiser()
    return lem.lemmatize(lem.lemmatize(token, pos="n"), pos="v")


def tokenise_document(
    text: str,
    stopwords: frozenset[str],
    protected: frozenset[str],
    min_token_length: int = 3,
) -> list[str]:
    """Tokenise one cleaned document into LDA-ready terms."""
    if not text:
        return []

    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        # Normalise plural acronyms (cnns -> cnn) before the protected check.
        m = _ACRONYM_PLURAL_RE.match(raw)
        if m and m.group(1) in protected:
            raw = m.group(1)

        if raw in protected:
            tokens.append(raw)          # bypasses length filter and lemmatiser
            continue

        if len(raw) < min_token_length:
            continue
        if raw in stopwords:
            continue

        lemma = lemmatise(raw)
        # Lemmatisation can shorten a token below the floor or into a stop word
        # ('studies' -> 'study' is fine; 'was' -> 'be' is not wanted).
        if len(lemma) < min_token_length or lemma in stopwords:
            continue
        tokens.append(lemma)

    return tokens


def tokenise_documents(
    texts,
    domain_stopwords=(),
    protected_acronyms=(),
    min_token_length: int = 3,
) -> list[list[str]]:
    protected = frozenset(a.lower() for a in protected_acronyms)
    stops = build_stopwords(domain_stopwords, keep=protected)
    return [
        tokenise_document(t, stops, protected, min_token_length) for t in texts
    ]
