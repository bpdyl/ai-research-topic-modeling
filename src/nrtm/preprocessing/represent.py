"""
Representation building — where the two branches diverge.

    cleaned text ──┬─► tokenise ─► phrases ─► Dictionary ─► BoW ─► TF-IDF   (LDA, GA-LDA)
                   └─► cleaned natural sentences                            (BERTopic)

The BERTopic branch is intentionally almost a pass-through. A sentence
transformer is trained on natural language: it needs word order, casing and
function words. Handing it lemmatised, stop-word-stripped bags of words would
degrade it badly and rig the comparison this project exists to make
(OQ-015 / RISK-008).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def build_phrases(
    token_docs: Sequence[Sequence[str]],
    bigrams: bool = True,
    trigrams: bool = True,
    min_count: int = 10,
    threshold: float = 20.0,
):
    """Detect multi-word terms and fold them into single tokens.

    'deep learning' -> 'deep_learning'. Matters here because the corpus's
    signal lives in compound technical terms ('machine_learning',
    'neural_network', 'random_forest'); leaving them split scatters one
    concept across several topics.

    Returns (transformed_docs, phraser_models) — the models are returned so
    the same transformation can be re-applied to any held-out text.
    """
    from gensim.models.phrases import Phrases, Phraser

    docs = [list(d) for d in token_docs]
    models = []

    if bigrams:
        bigram = Phraser(Phrases(docs, min_count=min_count, threshold=threshold))
        docs = [bigram[d] for d in docs]
        models.append(bigram)

        if trigrams:
            # Trigrams are found by running bigram detection over already
            # bigrammed text ('convolutional_neural' + 'network').
            trigram = Phraser(Phrases(docs, min_count=min_count, threshold=threshold))
            docs = [trigram[d] for d in docs]
            models.append(trigram)

    return docs, models


def build_dictionary(token_docs, no_below: int = 5, no_above: float = 0.5, keep_n: int = 50000):
    """Build the gensim Dictionary and apply frequency filtering.

    `no_below=5` drops terms appearing in fewer than 5 documents — mostly
    typos, author surnames and one-off jargon that add dimensions without
    adding topics. `no_above=0.5` drops terms in more than half the corpus,
    which carry no discriminative signal.
    """
    from gensim.corpora import Dictionary

    dictionary = Dictionary(token_docs)
    before = len(dictionary)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    return dictionary, before


def build_bow(token_docs, dictionary):
    return [dictionary.doc2bow(doc) for doc in token_docs]


def build_tfidf(bow_corpus):
    """TF-IDF model + transformed corpus.

    The proposal commits to 'Bag-of-Words / TF-IDF generation'. Note that LDA
    is fitted on the BoW counts, not TF-IDF — LDA's generative model assumes
    integer counts, and feeding it fractional TF-IDF weights is a common
    methodological error. TF-IDF is retained for the exploratory term analysis
    and as a documented part of the VSM stage.
    """
    from gensim.models import TfidfModel

    model = TfidfModel(bow_corpus)
    return model, [model[doc] for doc in bow_corpus]


def write_jsonl_rows(rows, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}\nRun: python scripts/02_preprocess.py")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def vocabulary_stats(token_docs, dictionary=None) -> dict:
    """Numbers for the paper's Experimental Setup section.

    When `dictionary` is supplied, `most_common` is restricted to terms that
    actually survived `filter_extremes` — i.e. what the model really sees.
    Reporting pre-filter counts is misleading: the most frequent raw term in
    this corpus ("model", 59.8% DF) is removed by `no_above` and never reaches
    LDA at all.
    """
    from collections import Counter

    lengths = [len(d) for d in token_docs]
    types = Counter()
    for d in token_docs:
        types.update(set(d))

    if dictionary is not None:
        in_vocab = set(dictionary.token2id)
        ranked = Counter({t: c for t, c in types.items() if t in in_vocab})
    else:
        ranked = types

    stats = {
        "n_documents": len(token_docs),
        "total_tokens": sum(lengths),
        "unique_types": len(types),
        "tokens_per_doc": {
            "min": min(lengths) if lengths else 0,
            "median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
        "empty_documents": sum(1 for n in lengths if n == 0),
        # Document frequency, not raw count — the useful quantity for judging
        # whether a term is discriminative.
        "most_common": ranked.most_common(25),
    }
    if dictionary is not None:
        stats["dictionary_size"] = len(dictionary)
        stats["most_common_note"] = "document frequency, restricted to in-dictionary terms"
    return stats
