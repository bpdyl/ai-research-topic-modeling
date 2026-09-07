"""
Topic stability across random seeds.

Resolves OQ-011. The proposal uses "stability" in both the GA fitness and the
evaluation metrics but never defines it. Definition adopted here:

    Fit the same configuration under m different random seeds. For each pair of
    runs, match topics one-to-one by maximising total Jaccard overlap of their
    top-N word sets (Hungarian assignment), and take the mean matched overlap.
    Stability is the mean over all pairs.

Range [0, 1]; 1.0 means every seed recovers the same topics.

Why this definition. LDA's variational inference is seed-dependent, and
Agrawal, Fu & Menzies (2018) — the paper the proposal cites to justify using a
GA at all — frame LDA's core weakness as exactly this *instability*. Measuring
seed agreement therefore tests the proposal's own stated motivation rather than
inventing a new criterion.

Why Hungarian matching rather than comparing topic k to topic k: topic indices
are arbitrary. Two runs can recover identical topic sets in different orders,
and a naive index-wise comparison would score that near zero.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def matched_jaccard(
    topics_a: Sequence[Sequence[str]],
    topics_b: Sequence[Sequence[str]],
    top_n: int = 10,
) -> float:
    """Mean Jaccard overlap after optimal one-to-one topic matching.

    When the two runs have different topic counts, the smaller set is matched
    into the larger and the score is scaled by the size ratio — otherwise a
    model with 5 topics could score 1.0 against one with 40 simply by matching
    its five best.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    A = [set(list(t)[:top_n]) for t in topics_a if t]
    B = [set(list(t)[:top_n]) for t in topics_b if t]
    if not A or not B:
        return float("nan")

    sim = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            sim[i, j] = _jaccard(a, b)

    # linear_sum_assignment minimises, so negate to maximise overlap.
    rows, cols = linear_sum_assignment(-sim)
    matched = float(sim[rows, cols].mean())

    n_matched = len(rows)
    larger = max(len(A), len(B))
    return matched * (n_matched / larger)


def topic_stability(
    fit_and_extract: Callable[[int], Sequence[Sequence[str]]],
    seeds: Sequence[int],
    top_n: int = 10,
) -> dict:
    """Stability of a configuration across seeds.

    `fit_and_extract(seed) -> topics` keeps this module independent of any
    particular model, so the same function serves LDA, GA-LDA and BERTopic.
    """
    if len(seeds) < 2:
        raise ValueError("stability needs at least 2 seeds")

    runs = [fit_and_extract(s) for s in seeds]
    scores = [
        matched_jaccard(a, b, top_n=top_n)
        for a, b in combinations(runs, 2)
    ]
    valid = [s for s in scores if s == s]  # drop NaN
    return {
        "stability": sum(valid) / len(valid) if valid else float("nan"),
        "pairwise": [round(s, 4) for s in scores],
        "n_seeds": len(seeds),
        "seeds": list(seeds),
    }
