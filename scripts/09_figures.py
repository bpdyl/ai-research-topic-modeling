"""
Step 09 — Exploratory and corpus figures.

Produces the descriptive figures the paper's Dataset and Experimental Setup
sections need. The model-specific figures (coherence-vs-K, GA convergence,
thematic trends) are emitted by their own scripts so they stay tied to the run
that produced them.

Usage:
    python scripts/09_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed        # noqa: E402
from nrtm.viz.figures import (                                # noqa: E402
    plot_wordcloud, plot_cooccurrence_network, plot_corpus_by_year,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Corpus and exploratory figures")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run = new_run(cfg, tag="figures")
    print(f"Run: {run.run_id}\n")

    tokens = [json.loads(l) for l in open(cfg.path("tokens"), encoding="utf-8") if l.strip()]
    frozen = [json.loads(l) for l in open(cfg.path("frozen_corpus"), encoding="utf-8") if l.strip()]
    token_docs = [r["tokens"] for r in tokens]
    years = [r["year"] for r in frozen]
    print(f"{len(token_docs)} documents\n")

    figs_dir = cfg.path("figures_dir")
    figs_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    made = []
    for name, fn, kwargs in (
        ("fig_corpus_by_year.png", plot_corpus_by_year, {"years": years}),
        ("fig_wordcloud.png", plot_wordcloud, {"token_docs": token_docs}),
        ("fig_cooccurrence.png", plot_cooccurrence_network, {"token_docs": token_docs}),
    ):
        out = fn(out_path=run.file(name), **kwargs)
        shutil.copy(out, figs_dir / name)
        made.append(name)
        print(f"  {name}")

    run.write_json("figures.json", {"figures": made, "n_documents": len(token_docs)})
    root = run.path.parents[2]
    print(f"\nFigures -> {figs_dir.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
