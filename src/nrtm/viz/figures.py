"""
Figures for the paper.

House style is set once here so every figure in the paper matches. Marking
criterion 5.2 asks for results "clearly presented, with appropriate
visualisations", and inconsistent figures read as carelessness.

Every figure is saved at 300 dpi and must remain legible in greyscale — papers
get printed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

FIGSIZE = (7.0, 4.2)
DPI = 300


def _style():
    import matplotlib
    matplotlib.use("Agg")           # no display on this machine; must precede pyplot
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    return plt


def plot_coherence_vs_k(
    rows: Sequence[dict],
    out_path: str | Path,
    primary: str = "c_v",
    secondary: str | None = "diversity",
    chosen_k: int | None = None,
    title: str = "LDA topic coherence and diversity by number of topics",
) -> Path:
    """Coherence-vs-K curve — the figure that justifies the chosen K.

    Diversity is overlaid on a second axis because the two must be read
    together: coherence often keeps climbing as K grows while topics start
    duplicating, and only the pair reveals that.
    """
    plt = _style()
    ks = [r["k"] for r in rows]
    ys = [r.get(primary, float("nan")) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(ks, ys, marker="o", color="#B5533C", label=f"coherence ({primary})")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel(f"Coherence ({primary})", color="#B5533C")
    ax.tick_params(axis="y", labelcolor="#B5533C")
    ax.set_xticks(ks)

    if chosen_k is not None:
        ax.axvline(chosen_k, color="#444444", linestyle="--", linewidth=1)
        ax.annotate(f"selected K = {chosen_k}", xy=(chosen_k, max(ys)),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=9, color="#444444")

    if secondary:
        ax2 = ax.twinx()
        ax2.plot(ks, [r.get(secondary, float("nan")) for r in rows],
                 marker="s", linestyle=":", color="#4A6670", label=secondary)
        ax2.set_ylabel("Topic diversity", color="#4A6670")
        ax2.tick_params(axis="y", labelcolor="#4A6670")
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)

    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ga_convergence(
    history: Sequence[dict],
    out_path: str | Path,
    baseline_cv: float | None = None,
    title: str = "GA convergence: fitness by generation",
) -> Path:
    """Convergence plot — best and mean fitness per generation.

    Mean is shown alongside best because best alone can look like healthy
    convergence while the population has actually collapsed onto one genome.
    The gap between the two curves is the visible evidence of diversity.

    The baseline's C_v is drawn on the right axis when available, so the figure
    answers the question the paper actually asks: did the GA beat grid search?
    """
    plt = _style()
    gens = [h["generation"] for h in history]

    fig, ax = plt.subplots()
    ax.plot(gens, [h["best_fitness"] for h in history],
            marker="o", color="#B5533C", label="best fitness")
    ax.plot(gens, [h["mean_fitness"] for h in history],
            marker="s", linestyle="--", color="#4A6670", label="population mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (weighted sum)")
    ax.set_xticks(gens)
    ax.legend(loc="lower right")

    if baseline_cv is not None:
        ax2 = ax.twinx()
        ax2.plot(gens, [h["best_c_v"] for h in history],
                 marker="^", linestyle=":", color="#7A8450", label="best $C_v$")
        ax2.axhline(baseline_cv, color="#999999", linestyle="-.", linewidth=1)
        ax2.annotate(f"grid-search baseline $C_v$ = {baseline_cv:.4f}",
                     xy=(gens[0], baseline_cv), xytext=(2, 4),
                     textcoords="offset points", fontsize=8, color="#666666")
        ax2.set_ylabel("Coherence ($C_v$)", color="#7A8450")
        ax2.tick_params(axis="y", labelcolor="#7A8450")
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax2.legend(loc="center right", fontsize=8)

    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_topic_trends(
    proportions,
    windows: Sequence[str],
    counts: Sequence[int],
    labels: Sequence[str],
    out_path: str | Path,
    top_n: int = 8,
    title: str = "Thematic evolution of AI research involving Nepali authors",
) -> Path:
    """Topic share by time window.

    Per-window document counts are printed on the x-axis tick labels, not
    relegated to a caption. The first window holds 8 documents; a reader who
    cannot see that will over-read its trend line (DEC-013).

    Only the `top_n` topics by absolute change are drawn — plotting all of them
    produces an unreadable tangle, and the ones that did not move are precisely
    the ones with nothing to show.
    """
    import numpy as np

    plt = _style()
    props = np.asarray(proportions, dtype=float)

    change = np.nan_to_num(np.abs(props[-1, :] - props[0, :]))
    order = np.argsort(-change)[:top_n]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(windows))

    for rank, t in enumerate(order):
        ax.plot(x, props[:, t], marker="o", linewidth=1.8,
                color=cmap(rank % 10), label=labels[t])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n(n={c})" for w, c in zip(windows, counts)])
    ax.set_xlabel("Time window")
    ax.set_ylabel("Share of topic mass within window")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_wordcloud(token_docs: Sequence[Sequence[str]], out_path: str | Path,
                   max_words: int = 120,
                   title: str = "Most frequent terms in the corpus") -> Path:
    """Word cloud over the preprocessed corpus.

    Frequency here is *document* frequency, not raw count: a term repeated 40
    times in one paper should not outrank one appearing in 40 papers. Raw-count
    clouds routinely mislead for exactly that reason.
    """
    from collections import Counter
    from wordcloud import WordCloud

    plt = _style()
    df = Counter()
    for doc in token_docs:
        df.update(set(doc))

    wc = WordCloud(width=1400, height=750, background_color="white",
                   colormap="copper", max_words=max_words,
                   prefer_horizontal=0.95, random_state=42)
    wc.generate_from_frequencies(dict(df))

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_cooccurrence_network(token_docs: Sequence[Sequence[str]], out_path: str | Path,
                              top_terms: int = 35, min_edge: int = 25,
                              title: str = "Term co-occurrence network") -> Path:
    """Co-occurrence network over the most frequent terms.

    Layout is a deterministic circular arrangement ordered by degree rather
    than a force-directed one: spring layouts are seed-dependent and would make
    the figure irreproducible, which matters for a submitted paper.
    """
    from collections import Counter
    from itertools import combinations
    import numpy as np

    plt = _style()

    df = Counter()
    for doc in token_docs:
        df.update(set(doc))
    terms = [t for t, _ in df.most_common(top_terms)]
    index = {t: i for i, t in enumerate(terms)}

    edges = Counter()
    for doc in token_docs:
        present = sorted({t for t in doc if t in index})
        for a, b in combinations(present, 2):
            edges[(a, b)] += 1
    edges = {k: v for k, v in edges.items() if v >= min_edge}

    degree = Counter()
    for (a, b), w in edges.items():
        degree[a] += w
        degree[b] += w
    ordered = sorted(terms, key=lambda t: -degree.get(t, 0))
    angles = np.linspace(0, 2 * np.pi, len(ordered), endpoint=False)
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(ordered, angles)}

    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    if edges:
        max_w = max(edges.values())
        for (a, b), w in edges.items():
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            ax.plot([x1, x2], [y1, y2], color="#B5533C",
                    alpha=0.10 + 0.5 * (w / max_w), linewidth=0.4 + 2.2 * (w / max_w),
                    zorder=1)
    for t in ordered:
        x, y = pos[t]
        ax.scatter([x], [y], s=18 + 90 * (degree.get(t, 0) / max(degree.values() or [1])),
                   color="#4A6670", zorder=2)
        ax.annotate(t, (x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7.5, zorder=3)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title(f"{title}\n(top {top_terms} terms, edges >= {min_edge} co-occurrences)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_corpus_by_year(years: Sequence[int], out_path: str | Path,
                        title: str = "Corpus composition by publication year") -> Path:
    """Publication-year histogram.

    Belongs in the paper's Dataset section rather than an appendix: 63% of the
    corpus falls in the final window, and every temporal claim has to be read
    against that skew.
    """
    from collections import Counter

    plt = _style()
    counts = Counter(y for y in years if y)
    xs = sorted(counts)
    fig, ax = plt.subplots()
    ax.bar(xs, [counts[x] for x in xs], color="#B5533C", width=0.7)
    for x in xs:
        ax.annotate(str(counts[x]), (x, counts[x]), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Documents")
    ax.set_xticks(xs)
    ax.set_title(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
