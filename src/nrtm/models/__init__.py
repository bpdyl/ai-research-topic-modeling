"""Topic models: LDA baseline, GA-optimised LDA, BERTopic benchmark."""

from nrtm.models.lda import (
    fit_lda,
    topic_top_words,
    topic_top_words_with_weights,
    doc_topic_matrix,
    perplexity,
    sweep_num_topics,
)

__all__ = [
    "fit_lda", "topic_top_words", "topic_top_words_with_weights",
    "doc_topic_matrix", "perplexity", "sweep_num_topics",
]
