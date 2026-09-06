"""
Step 04 — GA-Optimized LDA (EXP-002). The project's proposed model.

A genetic algorithm searches LDA's (K, alpha, eta) against a weighted-sum
fitness of coherence, diversity and stability. The result is compared against
the grid-searched baseline from EXP-001.

Cost warning: fitness requires `ga.stability_seeds` full LDA fits per unique
genome. Use --smoke first to measure the per-evaluation cost on this machine
before committing to a full run.

Usage:
    python scripts/04_ga_lda.py --smoke        # tiny run, verifies the loop
    python scripts/04_ga_lda.py                # full run from config
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed                       # noqa: E402
from nrtm.models.ga import FitnessEvaluator, SearchSpace, run_ga             # noqa: E402
from nrtm.models.lda import (                                                # noqa: E402
    fit_lda, topic_top_words, topic_top_words_with_weights, doc_topic_matrix, perplexity,
)
from nrtm.evaluation.coherence import all_coherences                         # noqa: E402
from nrtm.evaluation.diversity import (                                      # noqa: E402
    topic_diversity, mean_pairwise_jaccard, redundant_topic_pairs,
)
from nrtm.viz.figures import plot_ga_convergence                             # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA-optimised LDA")
    p.add_argument("--config", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run (pop 4, 2 generations, 2 seeds) to measure cost")
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    return p.parse_args()


def load_inputs(cfg):
    from gensim.corpora import Dictionary, MmCorpus

    tokens_path = cfg.path("tokens")
    if not tokens_path.exists():
        raise SystemExit(f"Missing {tokens_path}\nRun: python scripts/02_preprocess.py")
    rows = [json.loads(l) for l in open(tokens_path, encoding="utf-8") if l.strip()]
    texts = [r["tokens"] for r in rows]
    dictionary = Dictionary.load(str(cfg.path("dictionary")))
    corpus = list(MmCorpus(str(cfg.path("bow_corpus"))))
    return texts, dictionary, corpus


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    gcfg, lcfg, ecfg = cfg["ga"], cfg["lda"], cfg["evaluation"]
    processes = cfg.get("coherence_processes", 1)

    population_size = args.population or gcfg["population_size"]
    generations = args.generations or gcfg["generations"]
    stability_seeds = [seed + i for i in range(gcfg["stability_seeds"])]

    if args.smoke:
        population_size, generations = 4, 2
        stability_seeds = [seed, seed + 1]
        print("SMOKE MODE: pop=4, generations=2, 2 stability seeds\n")

    run = new_run(cfg, tag="ga_lda_smoke" if args.smoke else "ga_lda")
    print(f"Run: {run.run_id}\n")

    texts, dictionary, corpus = load_inputs(cfg)
    print(f"Corpus: {len(corpus)} documents, {len(dictionary)} dictionary terms")

    space = SearchSpace.from_config(gcfg)
    print(f"Search space: K [{space.k_min}, {space.k_max}], "
          f"alpha [{space.alpha_min}, {space.alpha_max}], "
          f"eta [{space.eta_min}, {space.eta_max}]")
    print(f"GA: pop={population_size}, generations={generations}, "
          f"crossover={gcfg['crossover_rate']}, mutation={gcfg['mutation_rate']}, "
          f"elitism={gcfg['elitism']}, tournament={gcfg['tournament_size']}")
    print(f"Fitness weights: {gcfg['fitness_weights']}")
    print(f"Stability seeds: {stability_seeds}")
    print(f"Worst case: {population_size * generations} evaluations x "
          f"{len(stability_seeds)} LDA fits = "
          f"{population_size * generations * len(stability_seeds)} fits "
          f"(fewer with caching)\n")

    fit_kwargs = dict(
        passes=lcfg["passes"], iterations=lcfg["iterations"],
        chunksize=lcfg["chunksize"], multicore=lcfg.get("multicore", False),
    )

    evaluator = FitnessEvaluator(
        corpus=corpus, dictionary=dictionary, texts=texts,
        weights=gcfg["fitness_weights"],
        stability_seeds=stability_seeds,
        top_n=ecfg["top_n_words"],
        diversity_top_n=ecfg["diversity_top_n"],
        coherence_processes=processes,
        fit_kwargs=fit_kwargs,
        trace_path=run.file("fitness_trace.jsonl"),
    )

    t0 = time.time()
    result = run_ga(
        evaluator, space,
        population_size=population_size,
        generations=generations,
        crossover_rate=gcfg["crossover_rate"],
        mutation_rate=gcfg["mutation_rate"],
        n_elite=gcfg["elitism"],
        tournament_size=gcfg["tournament_size"],
        seed=seed,
    )
    print(f"\nGA finished in {result.total_seconds:.0f}s")
    print(f"Evaluator: {result.evaluator_stats}")

    best = result.best.genome
    print(f"\nBest genome: K={best.k}, alpha={best.alpha:.4f}, eta={best.eta:.4f}")
    print(f"  fitness={result.best.fitness:.4f}  c_v={result.best.c_v:.4f}  "
          f"diversity={result.best.diversity:.4f}  stability={result.best.stability:.4f}")

    # ── Refit the winner and score on every measure ────────────────────
    print("\nRefitting the best genome and scoring on all measures...")
    model = fit_lda(corpus, dictionary, num_topics=best.k, seed=seed,
                    alpha=best.alpha, eta=best.eta, **fit_kwargs)
    topics = topic_top_words(model, top_n=ecfg["top_n_words"])

    metrics = all_coherences(
        topics, dictionary, texts=texts, corpus=corpus,
        measures=tuple(ecfg["coherence_measures"]), processes=processes,
    )
    metrics["diversity"] = topic_diversity(topics, top_n=ecfg["diversity_top_n"])
    metrics["mean_pairwise_jaccard"] = mean_pairwise_jaccard(topics, top_n=ecfg["diversity_top_n"])
    metrics["perplexity"] = perplexity(model, corpus)
    metrics["num_topics"] = best.k
    metrics["stability"] = result.best.stability

    print("\nGA-LDA metrics:")
    for k, v in metrics.items():
        print(f"  {k:<24} {v:.4f}" if isinstance(v, float) else f"  {k:<24} {v}")

    # ── Compare against the EXP-001 baseline ───────────────────────────
    baseline = None
    runs_dir = cfg.path("results_dir")
    candidates = sorted(runs_dir.glob("*_lda_baseline/lda_baseline.json"))
    if candidates:
        with open(candidates[-1], encoding="utf-8") as fh:
            baseline = json.load(fh)
        b = baseline["metrics"]
        print(f"\nvs baseline ({candidates[-1].parent.name}):")
        for key in ("c_v", "c_npmi", "u_mass", "diversity", "num_topics"):
            if key in b and key in metrics:
                delta = metrics[key] - b[key]
                arrow = "better" if delta > 0 else ("worse" if delta < 0 else "same")
                if key in ("u_mass",):
                    arrow = "better" if delta > 0 else "worse"
                print(f"  {key:<14} baseline {b[key]:>9.4f}  ->  GA {metrics[key]:>9.4f}  "
                      f"({delta:+.4f}, {arrow})")
    else:
        print("\nNo EXP-001 baseline found to compare against.")

    redundant = redundant_topic_pairs(topics, top_n=ecfg["top_n_words"], threshold=0.5)
    print(f"\nNear-duplicate topic pairs (>50% overlap): {len(redundant)}")

    print(f"\nTopics (top {ecfg['top_n_words']} words):")
    for i, t in enumerate(topics):
        print(f"  {i:>2}: {', '.join(t)}")

    # ── Persist ────────────────────────────────────────────────────────
    import numpy as np
    model.save(str(run.file("ga_lda_model")))
    np.save(run.file("doc_topics.npy"), doc_topic_matrix(model, corpus, num_topics=best.k))

    payload = {
        "experiment": "EXP-002",
        "run_id": run.run_id,
        "smoke": args.smoke,
        "seed": seed,
        "search_space": {
            "k": [space.k_min, space.k_max],
            "alpha": [space.alpha_min, space.alpha_max],
            "eta": [space.eta_min, space.eta_max],
        },
        "ga_parameters": {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": gcfg["crossover_rate"],
            "mutation_rate": gcfg["mutation_rate"],
            "elitism": gcfg["elitism"],
            "tournament_size": gcfg["tournament_size"],
            "fitness_weights": gcfg["fitness_weights"],
            "stability_seeds": stability_seeds,
        },
        "best_genome": best.to_dict(),
        "best_fitness": result.best.fitness,
        "metrics": metrics,
        "topics": topics,
        "topics_with_weights": [
            [[w, round(p, 5)] for w, p in t]
            for t in topic_top_words_with_weights(model, top_n=ecfg["top_n_words"])
        ],
        "redundant_topic_pairs": redundant,
        "history": [h.to_dict() for h in result.history],
        "evaluator_stats": result.evaluator_stats,
        "total_seconds": round(result.total_seconds, 1),
        "baseline_run": candidates[-1].parent.name if candidates else None,
        "baseline_metrics": baseline["metrics"] if baseline else None,
    }
    run.write_json("ga_lda.json", payload)

    fig = plot_ga_convergence(
        [h.to_dict() for h in result.history], run.file("ga_convergence.png"),
        baseline_cv=baseline["metrics"]["c_v"] if baseline else None,
    )
    if not args.smoke:
        import shutil
        figures_dir = cfg.path("figures_dir")
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(fig, figures_dir / "fig_ga_convergence.png")

    root = run.path.parents[2]
    print(f"\nRun artefacts -> {run.path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
