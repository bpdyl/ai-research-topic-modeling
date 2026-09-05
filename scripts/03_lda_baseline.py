"""
Step 03 — Standard LDA baseline (EXP-001).

Sweeps K, selects the best by C_v coherence, refits at that K, and records
topics plus every metric. This is the baseline GA-optimised LDA must beat, so
its selection procedure has to be defensible: K is chosen by a documented
criterion over a stated grid, not by eye.

The comparison the paper makes is therefore GA-search vs grid-search, which is
the honest framing — not GA vs an arbitrary hand-picked K.

Usage:
    python scripts/03_lda_baseline.py
    python scripts/03_lda_baseline.py --k-min 5 --k-max 20 --k-step 5   # quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed                       # noqa: E402
from nrtm.models.lda import (                                                # noqa: E402
    fit_lda, sweep_num_topics, topic_top_words, topic_top_words_with_weights,
    doc_topic_matrix, perplexity,
)
from nrtm.evaluation.coherence import all_coherences                         # noqa: E402
from nrtm.evaluation.diversity import (                                      # noqa: E402
    topic_diversity, mean_pairwise_jaccard, redundant_topic_pairs,
)
from nrtm.viz.figures import plot_coherence_vs_k                             # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit the standard LDA baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--k-min", type=int, default=None)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--k-step", type=int, default=None)
    return p.parse_args()


def load_inputs(cfg):
    """Frozen tokens + dictionary + BoW. Fails loudly if Phase 4 has not run."""
    import json as _json
    from gensim.corpora import Dictionary, MmCorpus

    tokens_path = cfg.path("tokens")
    if not tokens_path.exists():
        raise SystemExit(f"Missing {tokens_path}\nRun: python scripts/02_preprocess.py")

    rows = [_json.loads(l) for l in open(tokens_path, encoding="utf-8") if l.strip()]
    texts = [r["tokens"] for r in rows]
    dictionary = Dictionary.load(str(cfg.path("dictionary")))
    corpus = list(MmCorpus(str(cfg.path("bow_corpus"))))
    return rows, texts, dictionary, corpus


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    lcfg = cfg["lda"]
    ecfg = cfg["evaluation"]
    processes = cfg.get("coherence_processes", 1)

    run = new_run(cfg, tag="lda_baseline")
    print(f"Run: {run.run_id}\n")

    rows, texts, dictionary, corpus = load_inputs(cfg)
    print(f"Corpus: {len(corpus)} documents, {len(dictionary)} dictionary terms\n")

    k_min = args.k_min if args.k_min is not None else lcfg["k_min"]
    k_max = args.k_max if args.k_max is not None else lcfg["k_max"]
    k_step = args.k_step if args.k_step is not None else lcfg["k_step"]
    k_values = list(range(k_min, k_max + 1, k_step))

    fit_kwargs = dict(
        passes=lcfg["passes"],
        iterations=lcfg["iterations"],
        chunksize=lcfg["chunksize"],
        alpha=lcfg["alpha"],
        eta=lcfg.get("eta"),
        multicore=lcfg.get("multicore", False),
    )

    print(f"Sweeping K over {k_values} (seed={seed}, passes={lcfg['passes']})")
    t0 = time.time()
    sweep = sweep_num_topics(
        corpus, dictionary, texts,
        k_values=k_values, seed=seed,
        top_n=ecfg["top_n_words"],
        coherence_measures=("c_v",),   # sweep on the primary only; full set at the end
        processes=processes,
        **fit_kwargs,
    )
    sweep_secs = time.time() - t0
    print(f"\nSweep finished in {sweep_secs:.0f}s")

    # ── Select K ───────────────────────────────────────────────────────
    best = max(sweep, key=lambda r: (r["c_v"] if r["c_v"] == r["c_v"] else -1))
    best_k = best["k"]
    print(f"\nSelected K = {best_k} (highest C_v = {best['c_v']:.4f})")

    # Honest caveat: if the best K sits at the edge of the grid, the true
    # optimum may lie outside it. Say so rather than quietly reporting the edge.
    if best_k in (k_values[0], k_values[-1]):
        print(f"  NOTE: K={best_k} is at the edge of the search grid "
              f"[{k_values[0]}, {k_values[-1]}] — the optimum may lie beyond it. "
              "The GA searches a continuous range and is not bounded this way.")

    # ── Refit at the selected K and score fully ────────────────────────
    print("\nRefitting at the selected K and scoring on all measures...")
    model = fit_lda(corpus, dictionary, num_topics=best_k, seed=seed, **fit_kwargs)
    topics = topic_top_words(model, top_n=ecfg["top_n_words"])

    metrics = all_coherences(
        topics, dictionary, texts=texts, corpus=corpus,
        measures=tuple(ecfg["coherence_measures"]), processes=processes,
    )
    metrics["diversity"] = topic_diversity(topics, top_n=ecfg["diversity_top_n"])
    metrics["mean_pairwise_jaccard"] = mean_pairwise_jaccard(topics, top_n=ecfg["diversity_top_n"])
    metrics["perplexity"] = perplexity(model, corpus)
    metrics["num_topics"] = best_k

    print("\nBaseline metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<24} {v:.4f}")
        else:
            print(f"  {k:<24} {v}")

    redundant = redundant_topic_pairs(topics, top_n=ecfg["top_n_words"], threshold=0.5)
    if redundant:
        print(f"\n  {len(redundant)} near-duplicate topic pair(s) "
              f"(>50% shared top-{ecfg['top_n_words']} words): {redundant[:5]}")
    else:
        print(f"\n  No near-duplicate topic pairs above 50% overlap.")

    print(f"\nTopics (top {ecfg['top_n_words']} words):")
    for i, t in enumerate(topics):
        print(f"  {i:>2}: {', '.join(t)}")

    # ── Persist ────────────────────────────────────────────────────────
    doc_topics = doc_topic_matrix(model, corpus, num_topics=best_k)
    model.save(str(run.file("lda_model")))

    import numpy as np
    np.save(run.file("doc_topics.npy"), doc_topics)

    payload = {
        "experiment": "EXP-001",
        "run_id": run.run_id,
        "seed": seed,
        "k_grid": k_values,
        "selected_k": best_k,
        "selected_k_at_grid_edge": best_k in (k_values[0], k_values[-1]),
        "sweep_seconds": round(sweep_secs, 1),
        "fit_kwargs": {k: v for k, v in fit_kwargs.items()},
        "sweep": sweep,
        "metrics": metrics,
        "topics": topics,
        "topics_with_weights": [
            [[w, round(p, 5)] for w, p in t]
            for t in topic_top_words_with_weights(model, top_n=ecfg["top_n_words"])
        ],
        "redundant_topic_pairs": redundant,
        "n_documents": len(corpus),
        "dictionary_size": len(dictionary),
    }
    run.write_json("lda_baseline.json", payload)

    fig = plot_coherence_vs_k(
        sweep, run.file("coherence_vs_k.png"), primary="c_v",
        secondary="diversity", chosen_k=best_k,
    )
    # A copy in figures/ for the paper; run/ keeps the provenance copy.
    import shutil
    figures_dir = cfg.path("figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig, figures_dir / "fig_coherence_vs_k.png")

    root = run.path.parents[2]
    print(f"\nRun artefacts -> {run.path.relative_to(root)}")
    print(f"Figure        -> {(figures_dir / 'fig_coherence_vs_k.png').relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
