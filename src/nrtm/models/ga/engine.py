"""
The GA loop.

Generational with elitism. Records per-generation best/mean fitness so the
convergence plot the paper needs can be drawn from the run alone.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Sequence

from nrtm.models.ga.chromosome import Genome, SearchSpace, random_population
from nrtm.models.ga.fitness import FitnessEvaluator, FitnessResult
from nrtm.models.ga.operators import next_generation


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_genome: dict
    best_c_v: float
    best_diversity: float
    best_stability: float
    seconds: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class GAResult:
    best: FitnessResult
    history: list[GenerationRecord] = field(default_factory=list)
    evaluator_stats: dict = field(default_factory=dict)
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "best": self.best.to_dict(),
            "history": [h.to_dict() for h in self.history],
            "evaluator_stats": self.evaluator_stats,
            "total_seconds": round(self.total_seconds, 1),
        }


def run_ga(
    evaluator: FitnessEvaluator,
    space: SearchSpace,
    population_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    n_elite: int,
    tournament_size: int,
    seed: int = 42,
    progress: bool = True,
) -> GAResult:
    """Run the GA and return the best genome plus the full history."""
    rng = random.Random(seed)
    t_start = time.time()

    population = random_population(space, population_size, rng)
    best_overall: FitnessResult | None = None
    history: list[GenerationRecord] = []

    for gen in range(generations):
        t_gen = time.time()
        results = [evaluator.evaluate(g) for g in population]
        fitnesses = [r.fitness for r in results]

        gen_best = max(results, key=lambda r: r.fitness)
        if best_overall is None or gen_best.fitness > best_overall.fitness:
            best_overall = gen_best

        record = GenerationRecord(
            generation=gen,
            best_fitness=round(gen_best.fitness, 6),
            mean_fitness=round(sum(fitnesses) / len(fitnesses), 6),
            best_genome=gen_best.genome.to_dict(),
            best_c_v=round(gen_best.c_v, 6),
            best_diversity=round(gen_best.diversity, 6),
            best_stability=round(gen_best.stability, 6),
            seconds=round(time.time() - t_gen, 1),
        )
        history.append(record)

        if progress:
            g = gen_best.genome
            print(f"  gen {gen:>2}  best={record.best_fitness:.4f}  "
                  f"mean={record.mean_fitness:.4f}  "
                  f"K={g.k} alpha={g.alpha:.3f} eta={g.eta:.3f}  "
                  f"[c_v={gen_best.c_v:.4f} div={gen_best.diversity:.3f} "
                  f"stab={gen_best.stability:.3f}]  {record.seconds:.0f}s")

        # No offspring after the final scoring pass.
        if gen < generations - 1:
            population = next_generation(
                population, fitnesses, space, rng,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                n_elite=n_elite,
                tournament_size=tournament_size,
            )

    return GAResult(
        best=best_overall,
        history=history,
        evaluator_stats=evaluator.stats(),
        total_seconds=time.time() - t_start,
    )
