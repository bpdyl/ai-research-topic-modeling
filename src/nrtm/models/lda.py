"""
LDA fitting and topic extraction.

Used by both the baseline (Phase 5) and the GA (Phase 6) — the GA's fitness
function calls `fit_lda` with candidate `(K, alpha, eta)` values, so this module
is on the hot path for roughly 900 fits. Keep it cheap and side-effect free.
"""

from __future__ import annotations

from typing import Sequence


def fit_lda(
    corpus,
    dictionary,
    num_topics: int,
    seed: int = 42,
    passes: int = 10,
    iterations: int = 100,
    chunksize: int = 500,
    alpha="symmetric",
    eta=None,
    multicore: bool = False,
):
    """Fit an LDA model on a BoW corpus.

    `multicore` defaults to False on purpose. `LdaMulticore` spawns worker
    processes that each re-import the scipy/BLAS stack; combined with the GA's
    repeated fitting that exhausted the system commit limit during development.
    Single-process is slower per fit but survives the full run (DEC-015).

    LDA is fitted on **integer BoW counts**, never TF-IDF: the generative model
    assumes counts, and fractional weights break that assumption (DEC-017).
    """
    from gensim.models import LdaModel, LdaMulticore

    kwargs = dict(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        chunksize=chunksize,
        alpha=alpha,
        random_state=seed,
        eval_every=None,      # perplexity logging every chunk is pure overhead
    )
    if eta is not None:
        kwargs["eta"] = eta

    if multicore:
        kwargs.pop("random_state", None)      # LdaMulticore has no random_state
        return LdaMulticore(workers=2, **kwargs)
    return LdaModel(**kwargs)


def topic_top_words(model, top_n: int = 10) -> list[list[str]]:
    """Top-N words per topic, as plain strings.

    This is the lingua franca of the whole evaluation stage: coherence,
    diversity, stability, the LLM labelling prompt and the human rating sheet
    all consume exactly this shape, for all three model families.
    """
    return [
        [word for word, _weight in model.show_topic(topic_id, topn=top_n)]
        for topic_id in range(model.num_topics)
    ]


def topic_top_words_with_weights(model, top_n: int = 10) -> list[list[tuple[str, float]]]:
    """Top-N (word, probability) pairs per topic, for tables and figures."""
    return [
        [(w, float(p)) for w, p in model.show_topic(tid, topn=top_n)]
        for tid in range(model.num_topics)
    ]


def doc_topic_matrix(model, corpus, num_topics: int | None = None):
    """Dense document x topic proportion matrix.

    `minimum_probability=0` is essential: gensim otherwise omits low-probability
    topics, producing ragged rows that silently break the temporal analysis's
    per-window means (DEC-013).
    """
    import numpy as np

    k = num_topics or model.num_topics
    out = np.zeros((len(corpus), k), dtype=float)
    for i, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            out[i, topic_id] = prob
    return out


def perplexity(model, corpus) -> float:
    """Per-word held-out perplexity (lower is better).

    Reported as a secondary check only. Perplexity is known to correlate poorly
    — sometimes negatively — with human judgements of topic quality
    (Chang et al., 2009), which is precisely why this project's primary metrics
    are coherence, diversity and interpretability rather than likelihood.
    """
    import numpy as np

    return float(np.exp2(-model.log_perplexity(corpus)))


def sweep_num_topics(
    corpus,
    dictionary,
    texts: Sequence[Sequence[str]],
    k_values: Sequence[int],
    seed: int = 42,
    top_n: int = 10,
    coherence_measures: Sequence[str] = ("c_v",),
    processes: int = 1,
    progress: bool = True,
    **fit_kwargs,
) -> list[dict]:
    """Fit LDA at each K and score it. Returns one record per K."""
    from nrtm.evaluation.coherence import all_coherences
    from nrtm.evaluation.diversity import topic_diversity, mean_pairwise_jaccard

    results = []
    for k in k_values:
        model = fit_lda(corpus, dictionary, num_topics=k, seed=seed, **fit_kwargs)
        topics = topic_top_words(model, top_n=top_n)

        row = {"k": k}
        row.update(all_coherences(
            topics, dictionary, texts=texts, corpus=corpus,
            measures=coherence_measures, processes=processes,
        ))
        row["diversity"] = topic_diversity(topics, top_n=top_n)
        row["mean_pairwise_jaccard"] = mean_pairwise_jaccard(topics, top_n=top_n)
        row["perplexity"] = perplexity(model, corpus)
        results.append(row)

        if progress:
            primary = coherence_measures[0]
            print(f"  K={k:>3}  {primary}={row.get(primary, float('nan')):.4f}"
                  f"  diversity={row['diversity']:.3f}"
                  f"  perplexity={row['perplexity']:.1f}")
    return results
