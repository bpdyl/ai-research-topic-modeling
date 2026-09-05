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
