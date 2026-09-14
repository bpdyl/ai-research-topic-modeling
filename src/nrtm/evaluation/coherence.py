"""
Topic coherence.

Deliberately operates on **topic word lists** (`list[list[str]]`) rather than on
a fitted model object. LDA, GA-LDA and BERTopic produce entirely different model
types; reducing each to its top-N words per topic is what makes the three
comparable at all, and it is the representation the human interpretability
rating uses too.

`processes=1` everywhere — this is not a preference. gensim's CoherenceModel
defaults to spawning one worker per core; on Windows each child re-imports the
whole scipy/BLAS stack, and on this machine that exhausted the system commit
limit. The GA runs on the order of 900 coherence evaluations, so the default
would not survive. See DEC-015.
"""

from __future__ import annotations

from typing import Sequence

# c_v and c_npmi score against a sliding window over the tokenised texts;
# u_mass scores against document co-occurrence counts in the BoW corpus.
_NEEDS_TEXTS = {"c_v", "c_npmi", "c_uci"}
_NEEDS_CORPUS = {"u_mass"}


def coherence_score(
    topics: Sequence[Sequence[str]],
    dictionary,
    texts: Sequence[Sequence[str]] | None = None,
    corpus=None,
    measure: str = "c_v",
    processes: int = 1,
    top_n: int | None = None,
) -> float:
    """Coherence of a set of topics under one measure.

    `topics` is a list of top-word lists. Words absent from `dictionary` are
    dropped: gensim raises rather than skipping them, which matters for
    BERTopic, whose c-TF-IDF vocabulary is not the LDA dictionary.
    """
    from gensim.models import CoherenceModel

    vocab = set(dictionary.token2id)
    cleaned: list[list[str]] = []
    for topic in topics:
        words = [w for w in topic if w in vocab]
        if top_n:
            words = words[:top_n]
        # A topic needs >= 2 words for any pairwise coherence measure.
        if len(words) >= 2:
            cleaned.append(words)

    if not cleaned:
        return float("nan")

    kwargs = dict(topics=cleaned, dictionary=dictionary, coherence=measure, processes=processes)
    if measure in _NEEDS_TEXTS:
        if texts is None:
            raise ValueError(f"coherence '{measure}' requires `texts`")
        kwargs["texts"] = texts
    elif measure in _NEEDS_CORPUS:
        if corpus is None:
            raise ValueError(f"coherence '{measure}' requires `corpus`")
        kwargs["corpus"] = corpus
    else:
        raise ValueError(f"unknown coherence measure: {measure}")

    return float(CoherenceModel(**kwargs).get_coherence())


def all_coherences(
    topics: Sequence[Sequence[str]],
    dictionary,
    texts=None,
    corpus=None,
    measures: Sequence[str] = ("c_v", "c_npmi", "u_mass"),
    processes: int = 1,
    top_n: int | None = None,
) -> dict[str, float]:
    """Every configured coherence measure at once.

    Reporting more than one matters here: the GA optimises `c_v`, so quoting
    only `c_v` as evidence that GA-LDA wins is circular. `c_npmi` and `u_mass`
    are measures the GA never saw (RISK-007).
    """
    out: dict[str, float] = {}
    for m in measures:
        try:
            out[m] = coherence_score(
                topics, dictionary, texts=texts, corpus=corpus,
                measure=m, processes=processes, top_n=top_n,
            )
        except Exception as exc:  # a failing measure must not kill a long run
            out[m] = float("nan")
            out[f"{m}_error"] = str(exc)[:200]
    return out
