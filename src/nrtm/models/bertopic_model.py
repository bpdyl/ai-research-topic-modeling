"""
BERTopic — the modern embedding-based benchmark.

Pipeline: sentence-transformer embeddings -> UMAP -> HDBSCAN -> class-based
TF-IDF for topic representation.

Three things here exist specifically to keep the comparison against LDA honest
(RISK-008):

1. **Input.** BERTopic consumes the natural-sentence branch from Phase 4, not
   the LDA token stream. A sentence transformer depends on word order, casing
   and function words; feeding it lemmatised bags of words would cripple the
   benchmark and make the project's central comparison meaningless.

2. **Seeds.** UMAP is stochastic. `random_state` is set explicitly, and note
   that doing so forces UMAP single-threaded — slower, but the alternative is
   an unreproducible result.

3. **Outliers.** HDBSCAN assigns unclustered documents to topic -1. Those
   documents are *not* modelled, so a BERTopic run covering 60% of the corpus
   is not comparable to an LDA run covering 100% unless the coverage is
   reported. `fit_bertopic` always returns the outlier count.
"""

from __future__ import annotations

from typing import Sequence


def build_embeddings(documents: Sequence[str], model_name: str, show_progress: bool = True):
    """Encode documents once.

    Embeddings are deterministic, so they are computed once and reused across
    stability refits — only UMAP and HDBSCAN are stochastic. This turns a
    3-seed stability estimate from 3 encodings into 1.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(list(documents), show_progress_bar=show_progress)


def fit_bertopic(
    documents: Sequence[str],
    embeddings=None,
    embedding_model: str = "all-MiniLM-L6-v2",
    min_topic_size: int = 15,
    umap_kwargs: dict | None = None,
    hdbscan_kwargs: dict | None = None,
    stopwords: Sequence[str] | None = None,
    seed: int = 42,
):
    """Fit BERTopic and return (model, topics, probabilities, info).

    A `CountVectorizer` with the same stop-word list used for LDA is passed in
    deliberately. BERTopic's default vectoriser keeps English stop words, so
    its c-TF-IDF topic words would be padded with "the", "of", "and" while
    LDA's were not — the coherence comparison would then measure preprocessing,
    not modelling.
    """
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    umap_kwargs = dict(umap_kwargs or {})
    hdbscan_kwargs = dict(hdbscan_kwargs or {})
    umap_kwargs.setdefault("random_state", seed)

    umap_model = UMAP(
        n_neighbors=umap_kwargs.get("n_neighbors", 15),
        n_components=umap_kwargs.get("n_components", 5),
        min_dist=umap_kwargs.get("min_dist", 0.0),
        metric=umap_kwargs.get("metric", "cosine"),
        random_state=umap_kwargs["random_state"],
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_kwargs.get("min_cluster_size", min_topic_size),
        metric=hdbscan_kwargs.get("metric", "euclidean"),
        cluster_selection_method=hdbscan_kwargs.get("cluster_selection_method", "eom"),
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=list(stopwords) if stopwords else "english",
        ngram_range=(1, 2),
        min_df=2,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(list(documents), embeddings=embeddings)

    n_outliers = sum(1 for t in topics if t == -1)
    info = {
        "n_topics": len([t for t in set(topics) if t != -1]),
        "n_outliers": n_outliers,
        "outlier_fraction": n_outliers / len(topics) if topics else float("nan"),
        "n_documents": len(topics),
    }
    return topic_model, topics, probs, info


def bertopic_top_words(topic_model, top_n: int = 10) -> list[list[str]]:
    """Top-N words per topic, excluding the outlier topic.

    Same shape as `nrtm.models.lda.topic_top_words`, which is what lets the
    shared evaluation metrics score all three model families identically.
    Topic -1 is excluded: it is a residual bucket, not a discovered theme, and
    including it would flatter diversity with an incoherent grab-bag.
    """
    out = []
    for topic_id in sorted(t for t in topic_model.get_topics() if t != -1):
        words = topic_model.get_topic(topic_id)
        if not words:
            continue
        out.append([w for w, _score in words[:top_n]])
    return out


def bertopic_doc_topic_matrix(topics: Sequence[int], n_topics: int):
    """Hard document x topic assignment matrix.

    BERTopic gives a hard assignment, not a distribution like LDA. Rows for
    outlier documents are all-zero, so they contribute nothing to the temporal
    proportions rather than being silently reassigned. That asymmetry with LDA
    must be stated wherever the two are compared.
    """
    import numpy as np

    out = np.zeros((len(topics), n_topics), dtype=float)
    for i, t in enumerate(topics):
        if t is not None and t >= 0 and t < n_topics:
            out[i, t] = 1.0
    return out


def vocabulary_overlap(topics: Sequence[Sequence[str]], dictionary) -> dict:
    """How much of BERTopic's topic vocabulary exists in the LDA dictionary.

    Coherence is scored against the LDA reference corpus for all three models,
    so words absent from that dictionary are dropped. If BERTopic loses a large
    share of its words, its coherence score is computed on a different basis
    from LDA's and the comparison needs a caveat. Measuring this is what makes
    that judgement possible instead of assumed.
    """
    vocab = set(dictionary.token2id)
    total = sum(len(t) for t in topics)
    found = sum(1 for t in topics for w in t if w in vocab)
    return {
        "topic_words_total": total,
        "topic_words_in_dictionary": found,
        "coverage": found / total if total else float("nan"),
    }
