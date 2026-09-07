"""
Inter-rater agreement.

The proposal promises an interpretability score plus an agreement statistic. A
single rater cannot supply the latter: agreement is a property of a *set* of
independent raters, so with one rater the number simply does not exist.

Krippendorff's alpha is used rather than Fleiss' kappa or Cohen's kappa because
the rating scale is **ordinal** (1-5). Kappa treats a 1-vs-5 disagreement as
identical to a 4-vs-5 disagreement, which is wrong here: two raters who differ
by one point broadly agree, and two who differ by four do not. Krippendorff's
alpha with an ordinal difference function encodes that.

Interpretation follows the usual convention: alpha >= 0.800 is reliable,
0.667-0.800 permits tentative conclusions, below 0.667 is unreliable.
"""

from __future__ import annotations

from typing import Sequence


def _ordinal_delta(a: int, b: int, ranks: Sequence[int], counts: dict) -> float:
    """Ordinal difference function for Krippendorff's alpha.

    The distance between two ordinal categories depends on how many
    observations lie between them, not merely on their numeric labels.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    between = [g for g in ranks if lo <= g <= hi]
    total = sum(counts[g] for g in between)
    return (total - (counts[lo] + counts[hi]) / 2.0) ** 2


def krippendorff_alpha_ordinal(ratings: Sequence[Sequence[int | None]]) -> float:
    """Krippendorff's alpha for ordinal data.

    `ratings` is raters x items; use None for a missing rating.

    Implemented directly rather than pulled from a package: the coefficient is
    short, the ordinal variant is the part most third-party implementations get
    wrong or omit, and a marker can check this against the published definition.
    """
    n_raters = len(ratings)
    n_items = len(ratings[0]) if n_raters else 0

    # Units with at least two ratings; single-rated units carry no information
    # about agreement and are excluded, as the coefficient requires.
    units = []
    for j in range(n_items):
        vals = [ratings[i][j] for i in range(n_raters) if ratings[i][j] is not None]
        if len(vals) >= 2:
            units.append(vals)
    if not units:
        return float("nan")

    counts: dict[int, int] = {}
    for vals in units:
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
    ranks = sorted(counts)
    n_total = sum(counts.values())

    if len(ranks) == 1:
        return 1.0  # every rater gave the same value to everything

    # Observed disagreement
    d_o = 0.0
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        pair_sum = 0.0
        for x in range(m):
            for y in range(m):
                if x != y:
                    pair_sum += _ordinal_delta(vals[x], vals[y], ranks, counts)
        d_o += pair_sum / (m - 1)
    d_o /= n_total

    # Expected disagreement
    d_e = 0.0
    for a in ranks:
        for b in ranks:
            if a != b:
                d_e += counts[a] * counts[b] * _ordinal_delta(a, b, ranks, counts)
    d_e /= n_total * (n_total - 1)

    return 1.0 - (d_o / d_e) if d_e else float("nan")


def percent_agreement(ratings: Sequence[Sequence[int | None]], tolerance: int = 0) -> float:
    """Proportion of rater pairs agreeing within `tolerance` points.

    Reported alongside alpha because alpha is hard to read intuitively.
    `tolerance=1` answers "how often do raters land within one point", which is
    the question a reader actually has about a 1-5 scale.
    """
    from itertools import combinations

    n_raters = len(ratings)
    n_items = len(ratings[0]) if n_raters else 0
    agree = total = 0
    for j in range(n_items):
        vals = [(i, ratings[i][j]) for i in range(n_raters) if ratings[i][j] is not None]
        for (_, a), (_, b) in combinations(vals, 2):
            total += 1
            if abs(a - b) <= tolerance:
                agree += 1
    return agree / total if total else float("nan")


def interpret_alpha(alpha: float) -> str:
    if alpha != alpha:
        return "not computable"
    if alpha >= 0.800:
        return "reliable"
    if alpha >= 0.667:
        return "tentative conclusions only"
    return "unreliable"
