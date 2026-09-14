"""
Chromosome encoding for the GA.

The gene is LDA's hyperparameter triple:

    K      number of topics          integer,   [k_min, k_max]
    alpha  document-topic prior      real,      [alpha_min, alpha_max]
    eta    topic-word prior          real,      [eta_min, eta_max]

**Real-valued encoding, not binary.** Binary encoding is the textbook default
and the marked sample in the archive used it, but it is the wrong tool here:
alpha and eta are continuous and span two orders of magnitude, so a binary
string would impose an arbitrary discretisation on exactly the dimensions the
GA exists to search finely. K is handled as an integer gene and rounded on
decode. This is a deliberate departure worth one sentence in the paper.

Note on terminology: the proposal writes the priors as (K, alpha, beta).
gensim calls the topic-word prior `eta`. They are the same parameter; `eta` is
used in code to match the library, `beta` in prose to match the proposal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Genome:
    """One candidate LDA configuration."""

    k: int
    alpha: float
    eta: float

    def key(self, precision: int = 4) -> tuple:
        """Cache key. Rounded so genomes that differ only past the 4th decimal
        are treated as identical — a GA re-proposes near-duplicates constantly,
        and each cache hit saves `stability_seeds` full LDA fits."""
        return (self.k, round(self.alpha, precision), round(self.eta, precision))

    def to_dict(self) -> dict:
        return {"k": self.k, "alpha": round(self.alpha, 6), "eta": round(self.eta, 6)}


@dataclass(frozen=True)
class SearchSpace:
    k_min: int
    k_max: int
    alpha_min: float
    alpha_max: float
    eta_min: float
    eta_max: float

    @classmethod
    def from_config(cls, ga_cfg) -> "SearchSpace":
        s = ga_cfg["search_space"]
        return cls(
            k_min=int(s["k"][0]), k_max=int(s["k"][1]),
            alpha_min=float(s["alpha"][0]), alpha_max=float(s["alpha"][1]),
            eta_min=float(s["eta"][0]), eta_max=float(s["eta"][1]),
        )

    def clamp(self, g: Genome) -> Genome:
        """Keep a genome inside the search space after crossover/mutation."""
        return Genome(
            k=max(self.k_min, min(self.k_max, int(round(g.k)))),
            alpha=max(self.alpha_min, min(self.alpha_max, g.alpha)),
            eta=max(self.eta_min, min(self.eta_max, g.eta)),
        )

    def random_genome(self, rng: random.Random) -> Genome:
        """Uniform sample.

        alpha and eta are sampled **log-uniformly**. They span 0.01-1.0, two
        orders of magnitude; uniform sampling would put 90% of the initial
        population above 0.1 and barely explore the sparse-prior region, which
        is where topic models usually want to be.
        """
        import math

        def log_uniform(lo: float, hi: float) -> float:
            return math.exp(rng.uniform(math.log(lo), math.log(hi)))

        return Genome(
            k=rng.randint(self.k_min, self.k_max),
            alpha=log_uniform(self.alpha_min, self.alpha_max),
            eta=log_uniform(self.eta_min, self.eta_max),
        )


def random_population(space: SearchSpace, size: int, rng: random.Random) -> list[Genome]:
    return [space.random_genome(rng) for _ in range(size)]
