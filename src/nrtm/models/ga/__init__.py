"""Genetic-algorithm optimisation of LDA hyperparameters (K, alpha, eta)."""

from nrtm.models.ga.chromosome import Genome, SearchSpace, random_population
from nrtm.models.ga.operators import (
    tournament_selection, blend_crossover, gaussian_mutation, elitism, next_generation,
)
from nrtm.models.ga.fitness import FitnessEvaluator, FitnessResult, normalise_cv
from nrtm.models.ga.engine import run_ga, GAResult, GenerationRecord

__all__ = [
    "Genome", "SearchSpace", "random_population",
    "tournament_selection", "blend_crossover", "gaussian_mutation", "elitism",
    "next_generation", "FitnessEvaluator", "FitnessResult", "normalise_cv",
    "run_ga", "GAResult", "GenerationRecord",
]
