"""
Step 11 — Random-search control (EXP-008).

The single most important missing experiment. EXP-002 showed GA-LDA beating a
grid-searched baseline, but part of that advantage is simply that the GA
searched a continuous 3-D space against an 8-point grid. Without this control
the honest claim is only "finer search helps" — not "evolutionary search wins".

This runs random search over the identical space, with the identical fitness
function, stability seeds and LDA settings, at the **same budget** as the GA
(matched on unique fitness evaluations, since the GA's cache means its 300
scheduled evaluations cost only 183 real ones).

Everything differs from EXP-002 in exactly one respect: how candidates are
proposed. Any difference in outcome is therefore attributable to the search
strategy and nothing else.

Usage:
    python scripts/11_random_search.py                # budget auto-matched to the GA
    python scripts/11_random_search.py --budget 183
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.config import load_config, new_run, set_seed                    # noqa: E402
from nrtm.models.ga import FitnessEvaluator, SearchSpace                  # noqa: E402
from nrtm.models.lda import (                                             # noqa: E402
    fit_lda, topic_top_words, perplexity,
)
from nrtm.evaluation.coherence import all_coherences                      # noqa: E402
from nrtm.evaluation.diversity import topic_diversity, mean_pairwise_jaccard  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-search control for the GA")
    p.add_argument("--config", default=None)
    p.add_argument("--budget", type=int, default=None,
                   help="Unique evaluations. Defaults to the GA run's actual count.")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def ga_budget(results_dir: Path) -> tuple[int, dict] | tuple[None, None]:
    """Recover the GA's real cost so the control is matched, not guessed."""
    hits = sorted(h for h in glob.glob(str(results_dir / "*_ga_lda" / "ga_lda.json"))
                  if "smoke" not in h)
    if not hits:
        return None, None
    payload = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    return payload["evaluator_stats"]["unique_evaluations"], payload


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    gcfg, lcfg, ecfg = cfg["ga"], cfg["lda"], cfg["evaluation"]
    processes = cfg.get("coherence_processes", 1)

    results_dir = cfg.path("results_dir")
    matched, ga_payload = ga_budget(results_dir)
    budget = args.budget or matched or 183
    if args.smoke:
        budget = 4

    run = new_run(cfg, tag="random_search_smoke" if args.smoke else "random_search")
    print(f"Run: {run.run_id}\n")

    from gensim.corpora import Dictionary, MmCorpus
    rows = [json.loads(l) for l in open(cfg.path("tokens"), encoding="utf-8") if l.strip()]
    texts = [r["tokens"] for r in rows]
    dictionary = Dictionary.load(str(cfg.path("dictionary")))
    corpus = list(MmCorpus(str(cfg.path("bow_corpus"))))

    space = SearchSpace.from_config(gcfg)
    stability_seeds = [seed + i for i in range(gcfg["stability_seeds"])]

    print(f"Corpus: {len(corpus)} documents, {len(dictionary)} terms")
    print(f"Search space: K [{space.k_min}, {space.k_max}], "
          f"alpha [{space.alpha_min}, {space.alpha_max}], eta [{space.eta_min}, {space.eta_max}]")
    print(f"Budget: {budget} unique evaluations"
          + (f"  (matched to GA run {ga_payload['run_id']})" if matched and not args.budget else ""))
    print(f"Fitness weights: {gcfg['fitness_weights']}  (identical to EXP-002)")
    print(f"Stability seeds: {stability_seeds}\n")

    fit_kwargs = dict(
        passes=lcfg["passes"], iterations=lcfg["iterations"],
        chunksize=lcfg["chunksize"], multicore=lcfg.get("multicore", False),
    )
    evaluator = FitnessEvaluator(
        corpus=corpus, dictionary=dictionary, texts=texts,
        weights=gcfg["fitness_weights"], stability_seeds=stability_seeds,
        top_n=ecfg["top_n_words"], diversity_top_n=ecfg["diversity_top_n"],
        coherence_processes=processes, fit_kwargs=fit_kwargs,
        trace_path=run.file("fitness_trace.jsonl"),
    )

    # Same log-uniform sampler the GA uses to seed generation 0, so the two
    # strategies draw their candidates from an identical distribution.
    rng = random.Random(seed)
    history, best = [], None
    t0 = time.time()

    for i in range(budget):
        genome = space.random_genome(rng)
        result = evaluator.evaluate(genome)
        if best is None or result.fitness > best.fitness:
            best = result
        history.append({
            "evaluation": i,
            "fitness": round(result.fitness, 6),
            "best_so_far": round(best.fitness, 6),
            "k": genome.k,
            "c_v": round(result.c_v, 6),
        })
        if (i + 1) % 20 == 0 or i == budget - 1:
            print(f"  {i+1:>4}/{budget}  best={best.fitness:.4f}  "
                  f"K={best.genome.k} alpha={best.genome.alpha:.3f} eta={best.genome.eta:.3f}  "
                  f"[c_v={best.c_v:.4f}]  {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"\nRandom search finished in {elapsed:.0f}s")
    print(f"Evaluator: {evaluator.stats()}")

    g = best.genome
    print(f"\nBest genome: K={g.k}, alpha={g.alpha:.4f}, eta={g.eta:.4f}")
    print(f"  fitness={best.fitness:.4f}  c_v={best.c_v:.4f}  "
          f"diversity={best.diversity:.4f}  stability={best.stability:.4f}")

    print("\nRefitting the winner and scoring on all measures...")
    model = fit_lda(corpus, dictionary, num_topics=g.k, seed=seed,
                    alpha=g.alpha, eta=g.eta, **fit_kwargs)
    topics = topic_top_words(model, top_n=ecfg["top_n_words"])
    metrics = all_coherences(topics, dictionary, texts=texts, corpus=corpus,
                             measures=tuple(ecfg["coherence_measures"]), processes=processes)
    metrics["diversity"] = topic_diversity(topics, top_n=ecfg["diversity_top_n"])
    metrics["mean_pairwise_jaccard"] = mean_pairwise_jaccard(topics, top_n=ecfg["diversity_top_n"])
    metrics["perplexity"] = perplexity(model, corpus)
    metrics["num_topics"] = g.k
    metrics["stability"] = best.stability

    print("\nRandom-search LDA metrics:")
    for k, v in metrics.items():
        print(f"  {k:<24} {v:.4f}" if isinstance(v, float) else f"  {k:<24} {v}")

    # ── The comparison this experiment exists for ──────────────────────
    if ga_payload:
        gm = ga_payload["metrics"]
        gb = ga_payload["best_genome"]
        print("\n" + "=" * 70)
        print("GA vs RANDOM SEARCH at equal budget")
        print("=" * 70)
        print(f"{'':<22}{'Random':>14}{'GA':>14}{'GA advantage':>16}")
        print("-" * 66)
        print(f"{'best fitness':<22}{best.fitness:>14.4f}"
              f"{ga_payload['best_fitness']:>14.4f}"
              f"{ga_payload['best_fitness'] - best.fitness:>+16.4f}")
        for key in ("c_v", "c_npmi", "u_mass", "diversity", "stability", "num_topics"):
            if key in gm and key in metrics:
                print(f"{key:<22}{metrics[key]:>14.4f}{gm[key]:>14.4f}"
                      f"{gm[key] - metrics[key]:>+16.4f}")
        print(f"{'K / alpha / eta':<22}"
              f"{f'{g.k}/{g.alpha:.3f}/{g.eta:.3f}':>14}"
              f"{f'{gb['k']}/{gb['alpha']:.3f}/{gb['eta']:.3f}':>14}")
        print("-" * 66)
        delta = ga_payload["best_fitness"] - best.fitness
        if delta > 0.005:
            print("READING: the GA found a materially better solution at equal budget. The "
                  "evolutionary mechanism, not merely a finer search, is doing work.")
        elif delta > 0:
            print("READING: the GA is ahead but only marginally. Most of its advantage over the "
                  "grid baseline is attributable to searching a continuous space, not to evolution. "
                  "State this plainly in the paper.")
        else:
            print("READING: random search matched or beat the GA at equal budget. The paper must "
                  "NOT claim evolutionary search is superior; the honest claim is that finer "
                  "search over (K, alpha, eta) helps regardless of strategy.")

    run.write_json("random_search.json", {
        "experiment": "EXP-008",
        "run_id": run.run_id,
        "seed": seed,
        "budget_unique_evaluations": budget,
        "budget_matched_to": ga_payload["run_id"] if ga_payload else None,
        "search_space": {"k": [space.k_min, space.k_max],
                         "alpha": [space.alpha_min, space.alpha_max],
                         "eta": [space.eta_min, space.eta_max]},
        "fitness_weights": gcfg["fitness_weights"],
        "best_genome": g.to_dict(),
        "best_fitness": best.fitness,
        "metrics": metrics,
        "topics": topics,
        "history": history,
        "evaluator_stats": evaluator.stats(),
        "elapsed_seconds": round(elapsed, 1),
        "ga_comparison": {
            "ga_run": ga_payload["run_id"] if ga_payload else None,
            "ga_best_fitness": ga_payload["best_fitness"] if ga_payload else None,
            "ga_metrics": ga_payload["metrics"] if ga_payload else None,
        },
    })

    root = run.path.parents[2]
    print(f"\nRun artefacts -> {run.path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
