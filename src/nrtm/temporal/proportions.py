"""
Thematic evolution over time.

Implements DEC-013: **one global model, per-window topic proportions**. The
proposal's stage 5 asked for the best model to be re-fitted per window, but the
2015-2017 window holds 8 documents — no topic model fitted on 8 documents is
defensible. Fitting once on the whole corpus and reporting how topic mass
redistributes across windows answers the same research question (RQ2) without
that failure.

Two rules the proposal does not state but the data forces:

1. **Normalise within window.** 63% of the corpus sits in 2024-2025. Raw topic
   mass per window would therefore track window size, not thematic change.
   Proportions are normalised so each window sums to 1 and the quantity plotted
   is "share of research attention in that period".

2. **Report n on every output.** With n=8 in the first window, a proportion is
   an estimate from eight documents. Any table or figure that hides that is
   misleading, so `n` travels with the numbers rather than being a footnote.
"""

from __future__ import annotations

from typing import Sequence


def window_proportions(
    doc_topics,
    windows: Sequence[str],
    doc_windows: Sequence[str],
    normalize_within_window: bool = True,
):
    """Mean topic proportion per time window.

    `doc_topics` is documents x topics. For LDA these are soft proportions; for
    BERTopic they are one-hot rows, with outlier documents all-zero so they
    contribute nothing rather than being silently reassigned.

    Returns (matrix windows x topics, counts per window).
    """
    import numpy as np

    doc_topics = np.asarray(doc_topics, dtype=float)
    n_topics = doc_topics.shape[1]
    out = np.zeros((len(windows), n_topics), dtype=float)
    counts = []

    for w_i, window in enumerate(windows):
        idx = [i for i, w in enumerate(doc_windows) if w == window]
        counts.append(len(idx))
        if not idx:
            out[w_i, :] = np.nan
            continue

        block = doc_topics[idx, :]
        # Documents with no assigned mass (BERTopic outliers) are dropped from
        # the mean rather than counted as zeros, which would dilute every topic.
        mass = block.sum(axis=1)
        active = block[mass > 0]
        if len(active) == 0:
            out[w_i, :] = np.nan
            continue

        mean = active.mean(axis=0)
        if normalize_within_window and mean.sum() > 0:
            mean = mean / mean.sum()
        out[w_i, :] = mean

    return out, counts


def topic_trends(proportions, windows: Sequence[str], counts: Sequence[int],
                 min_docs: int = 30) -> list[dict]:
    """Classify each topic as rising, declining or stable.

    The slope is computed over windows that meet `min_docs`. Windows below that
    threshold are excluded from the fit but still reported, because a trend line
    anchored on 8 documents is not a trend — it is noise with a direction.
    """
    import numpy as np

    props = np.asarray(proportions, dtype=float)
    usable = [i for i, c in enumerate(counts) if c >= min_docs]
    excluded = [windows[i] for i, c in enumerate(counts) if c < min_docs]

    trends = []
    for t in range(props.shape[1]):
        series = props[:, t]
        if len(usable) >= 2:
            x = np.array(usable, dtype=float)
            y = series[usable]
            slope = float(np.polyfit(x, y, 1)[0]) if not np.isnan(y).any() else float("nan")
        else:
            slope = float("nan")

        first, last = series[usable[0]], series[usable[-1]] if usable else (np.nan, np.nan)
        change = float(last - first) if usable else float("nan")

        if slope != slope:
            label = "unknown"
        elif abs(change) < 0.01:
            label = "stable"
        else:
            label = "rising" if change > 0 else "declining"

        trends.append({
            "topic": t,
            "slope": round(slope, 6) if slope == slope else None,
            "first_window_share": round(float(series[usable[0]]), 4) if usable else None,
            "last_window_share": round(float(series[usable[-1]]), 4) if usable else None,
            "absolute_change": round(change, 4) if change == change else None,
            "trend": label,
            "by_window": {w: (round(float(v), 4) if v == v else None)
                          for w, v in zip(windows, series)},
        })

    return sorted(trends, key=lambda d: -(d["absolute_change"] or 0)), excluded
