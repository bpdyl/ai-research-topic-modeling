"""
Multi-objective fitness for GA-optimised LDA.

Resolves OQ-013 with a **weighted sum**:

    fitness = w_c * norm(C_v) + w_d * diversity + w_s * stability

Why a weighted sum rather than a Pareto front (NSGA-II). A Pareto approach is
more faithful to the phrase "multi-objective" and would give a nice front to
plot, but it returns a *set* of non-dominated solutions with no single winner —
and this project needs one GA-LDA model to put in a three-way comparison table
against standard LDA and BERTopic. The weights are reported, and a small
sensitivity table over alternative weightings goes in the paper so the choice
is visible rather than hidden.

C_v is min-max normalised into [0, 1] against a stated reference band, so all
three components live on the same scale. Without that, coherence (~0.40-0.50)
and diversity (~0.80) would contribute unequally regardless of the weights.

Every evaluation is cached on the rounded genome and appended to a JSONL trace
on disk. Both matter: a GA re-proposes near-duplicate genomes constantly, and a
crash three hours into a run must not lose the history the convergence plot
needs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from nrtm.models.ga.chromosome import Genome

# C_v is bounded [0, 1] in principle but in practice lives in a much narrower
# band. Normalising against the observed band rather than [0, 1] keeps the
# weighting meaningful. Floor/ceiling are stated constants, not tuned.
CV_FLOOR = 0.30
CV_CEIL = 0.70


def normalise_cv(cv: float, floor: float = CV_FLOOR, ceil: float = CV_CEIL) -> float:
    if cv != cv:          # NaN
        return 0.0
    return max(0.0, min(1.0, (cv - floor) / (ceil - floor)))


@dataclass
class FitnessResult:
    genome: Genome
    fitness: float
    c_v: float
    diversity: float
    stability: float
    seconds: float
    cached: bool = False

    def to_dict(self) -> dict:
        d = self.genome.to_dict()
        d.update(
            fitness=round(self.fitness, 6),
            c_v=round(self.c_v, 6),
            diversity=round(self.diversity, 6),
            stability=round(self.stability, 6),
            seconds=round(self.seconds, 2),
            cached=self.cached,
        )
        return d


@dataclass
class FitnessEvaluator:
    """Evaluates genomes, with caching and an on-disk trace."""

    corpus: object
    dictionary: object
    texts: Sequence[Sequence[str]]
    weights: dict
    stability_seeds: Sequence[int]
    top_n: int = 10
    diversity_top_n: int = 25
    coherence_processes: int = 1
    fit_kwargs: dict = field(default_factory=dict)
    trace_path: Path | None = None
    cache: dict = field(default_factory=dict)
    n_evaluations: int = 0
    n_cache_hits: int = 0
    n_lda_fits: int = 0

    def _topics_for_seed(self, genome: Genome, seed: int) -> list[list[str]]:
        from nrtm.models.lda import fit_lda, topic_top_words

        model = fit_lda(
            self.corpus, self.dictionary, num_topics=genome.k, seed=seed,
            alpha=genome.alpha, eta=genome.eta, **self.fit_kwargs,
        )
        self.n_lda_fits += 1
        return topic_top_words(model, top_n=max(self.top_n, self.diversity_top_n))

    def evaluate(self, genome: Genome) -> FitnessResult:
        from nrtm.evaluation.coherence import coherence_score
        from nrtm.evaluation.diversity import topic_diversity
        from nrtm.evaluation.stability import matched_jaccard
        from itertools import combinations

        key = genome.key()
        if key in self.cache:
            self.n_cache_hits += 1
            cached = self.cache[key]
            return FitnessResult(
                genome=genome, fitness=cached.fitness, c_v=cached.c_v,
                diversity=cached.diversity, stability=cached.stability,
                seconds=0.0, cached=True,
            )

        t0 = time.time()

        # One fit per stability seed; the first seed also supplies the topics
        # scored for coherence and diversity. Reusing it avoids a redundant fit.
        runs = [self._topics_for_seed(genome, s) for s in self.stability_seeds]
        primary = runs[0]

        c_v = coherence_score(
            [t[: self.top_n] for t in primary], self.dictionary,
            texts=self.texts, measure="c_v",
            processes=self.coherence_processes,
        )
        diversity = topic_diversity(primary, top_n=self.diversity_top_n)

        if len(runs) >= 2:
            pair_scores = [
                matched_jaccard(a, b, top_n=self.top_n)
                for a, b in combinations(runs, 2)
            ]
            valid = [s for s in pair_scores if s == s]
            stability = sum(valid) / len(valid) if valid else 0.0
        else:
            stability = float("nan")

        fitness = (
            self.weights["coherence"] * normalise_cv(c_v)
            + self.weights["diversity"] * (diversity if diversity == diversity else 0.0)
            + self.weights["stability"] * (stability if stability == stability else 0.0)
        )

        result = FitnessResult(
            genome=genome, fitness=fitness, c_v=c_v, diversity=diversity,
            stability=stability, seconds=time.time() - t0,
        )
        self.cache[key] = result
        self.n_evaluations += 1

        if self.trace_path is not None:
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result.to_dict()) + "\n")

        return result

    def stats(self) -> dict:
        return {
            "unique_evaluations": self.n_evaluations,
            "cache_hits": self.n_cache_hits,
            "lda_fits": self.n_lda_fits,
            "cache_size": len(self.cache),
        }
