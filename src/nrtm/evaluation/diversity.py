"""
Topic diversity and redundancy.

Diversity is the proportion of unique words across the union of all topics'
top-N word lists (Dieng et al., 2020). 1.0 means no topic shares a word with
any other; low values mean the model has rediscovered the same theme several
times.

It is reported alongside coherence because the two trade off: a model can score
very high coherence by producing K near-identical topics of tightly co-occurring
words. Coherence alone would call that a success. This is exactly the failure
mode the GA could be driven into if fitness were coherence-only, which is why
the proposal's fitness is multi-objective (DEC-017, OQ-013).
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence


def topic_diversity(topics: Sequence[Sequence[str]], top_n: int = 25) -> float:
    """Proportion of unique words across all topics' top-N lists."""
    if not topics:
        return float("nan")
    words = [w for t in topics for w in list(t)[:top_n]]
    if not words:
        return float("nan")
    return len(set(words)) / len(words)


def mean_pairwise_jaccard(topics: Sequence[Sequence[str]], top_n: int = 25) -> float:
    """Mean Jaccard overlap between every pair of topics.

    The complement view of diversity: 0.0 means fully disjoint topics.
    Reported because it localises the problem — diversity gives one number for
    the whole model, while a high pairwise mean with high diversity points at a
    few duplicated topics rather than uniform mush.
    """
    sets = [set(list(t)[:top_n]) for t in topics if t]
    if len(sets) < 2:
        return float("nan")
    scores = []
    for a, b in combinations(sets, 2):
        union = a | b
        scores.append(len(a & b) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def redundant_topic_pairs(
    topics: Sequence[Sequence[str]], top_n: int = 10, threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Topic pairs sharing more than `threshold` of their top-N words.

    Useful in the write-up: naming the specific duplicated topics is more
    informative than reporting a single diversity figure.
    """
    sets = [set(list(t)[:top_n]) for t in topics]
    out = []
    for i, j in combinations(range(len(sets)), 2):
        union = sets[i] | sets[j]
        if not union:
            continue
        score = len(sets[i] & sets[j]) / len(union)
        if score > threshold:
            out.append((i, j, round(score, 3)))
    return sorted(out, key=lambda x: -x[2])
