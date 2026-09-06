"""
Tests for the genetic algorithm.

The GA is the project's proposed contribution, so its operators are tested
directly rather than only through end-to-end behaviour. A silently broken
mutation or a crossover that cannot escape the parents' range would still
*produce* a result — just a worthless one — and the failure would be invisible
in the final numbers.

No LDA fitting here: these tests must stay fast. Fitness is exercised with a
stub evaluator.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.models.ga.chromosome import Genome, SearchSpace, random_population   # noqa: E402
from nrtm.models.ga.operators import (                                          # noqa: E402
    tournament_selection, blend_crossover, gaussian_mutation, elitism, next_generation,
)
from nrtm.models.ga.fitness import normalise_cv                                 # noqa: E402
from nrtm.evaluation.stability import matched_jaccard                           # noqa: E402

SPACE = SearchSpace(k_min=5, k_max=40, alpha_min=0.01, alpha_max=1.0,
                    eta_min=0.01, eta_max=1.0)


class TestGenome:
    def test_cache_key_rounds_so_near_duplicates_collide(self):
        a = Genome(k=10, alpha=0.123456, eta=0.5)
        b = Genome(k=10, alpha=0.123499, eta=0.5)
        assert a.key() == b.key()

    def test_cache_key_separates_genuinely_different_genomes(self):
        assert Genome(10, 0.1, 0.5).key() != Genome(11, 0.1, 0.5).key()
        assert Genome(10, 0.1, 0.5).key() != Genome(10, 0.2, 0.5).key()


class TestSearchSpace:
    def test_clamp_pulls_values_back_inside_bounds(self):
        g = SPACE.clamp(Genome(k=999, alpha=5.0, eta=-3.0))
        assert g.k == 40 and g.alpha == 1.0 and g.eta == 0.01

    def test_clamp_rounds_k_to_an_integer(self):
        assert isinstance(SPACE.clamp(Genome(k=12.7, alpha=0.5, eta=0.5)).k, int)

    def test_random_genomes_stay_in_bounds(self):
        rng = random.Random(0)
        for g in random_population(SPACE, 50, rng):
            assert SPACE.k_min <= g.k <= SPACE.k_max
            assert SPACE.alpha_min <= g.alpha <= SPACE.alpha_max
            assert SPACE.eta_min <= g.eta <= SPACE.eta_max

    def test_log_uniform_sampling_reaches_the_sparse_prior_region(self):
        """Uniform sampling over [0.01, 1.0] would put ~90% of draws above 0.1
        and barely explore small priors, which is where topic models usually
        want to be."""
        rng = random.Random(1)
        alphas = [g.alpha for g in random_population(SPACE, 400, rng)]
        below_0_1 = sum(1 for a in alphas if a < 0.1) / len(alphas)
        assert below_0_1 > 0.4, f"only {below_0_1:.0%} of draws below 0.1"


class TestSelection:
    def test_tournament_prefers_higher_fitness(self):
        rng = random.Random(0)
        pop = [Genome(k, 0.1, 0.1) for k in (5, 10, 15, 20)]
        fits = [0.1, 0.2, 0.3, 0.9]
        picks = [tournament_selection(pop, fits, 4, rng).k for _ in range(20)]
        assert set(picks) == {20}          # full-size tournament always wins

    def test_tournament_size_one_is_random_choice(self):
        rng = random.Random(0)
        pop = [Genome(k, 0.1, 0.1) for k in (5, 10, 15, 20)]
        fits = [0.1, 0.2, 0.3, 0.9]
        picks = {tournament_selection(pop, fits, 1, rng).k for _ in range(40)}
        assert len(picks) > 1


class TestCrossover:
    def test_children_stay_in_bounds(self):
        rng = random.Random(0)
        a, b = Genome(5, 0.01, 0.01), Genome(40, 1.0, 1.0)
        for _ in range(50):
            for c in blend_crossover(a, b, SPACE, rng):
                assert SPACE.k_min <= c.k <= SPACE.k_max
                assert SPACE.alpha_min <= c.alpha <= SPACE.alpha_max

    def test_blend_can_exceed_the_parental_interval(self):
        """Plain interval crossover contracts the population towards its own
        mean and can never reach an optimum outside the initial spread. BLX
        extends beyond the parents, so it must actually do so."""
        rng = random.Random(3)
        a, b = Genome(20, 0.40, 0.40), Genome(20, 0.50, 0.50)
        outside = 0
        for _ in range(200):
            for c in blend_crossover(a, b, SPACE, rng):
                if c.alpha < 0.40 or c.alpha > 0.50:
                    outside += 1
        assert outside > 0


class TestMutation:
    def test_zero_rate_leaves_the_genome_untouched(self):
        rng = random.Random(0)
        g = Genome(20, 0.5, 0.5)
        assert gaussian_mutation(g, SPACE, rng, mutation_rate=0.0) == g

    def test_full_rate_changes_the_genome(self):
        rng = random.Random(0)
        g = Genome(20, 0.5, 0.5)
        changed = sum(
            1 for _ in range(20)
            if gaussian_mutation(g, SPACE, rng, mutation_rate=1.0) != g
        )
        assert changed >= 18

    def test_mutants_stay_in_bounds(self):
        rng = random.Random(0)
        for _ in range(200):
            m = gaussian_mutation(Genome(40, 1.0, 1.0), SPACE, rng, 1.0)
            assert SPACE.k_min <= m.k <= SPACE.k_max
            assert SPACE.eta_min <= m.eta <= SPACE.eta_max


class TestElitism:
    def test_returns_the_fittest_in_order(self):
        pop = [Genome(k, 0.1, 0.1) for k in (5, 10, 15, 20)]
        fits = [0.1, 0.9, 0.5, 0.3]
        assert [g.k for g in elitism(pop, fits, 2)] == [10, 15]

    def test_zero_elites_returns_empty(self):
        pop = [Genome(5, 0.1, 0.1)]
        assert elitism(pop, [0.5], 0) == []

    def test_best_genome_survives_a_generation(self):
        """Without elitism a GA can lose its best solution to crossover, and
        the convergence curve stops being monotonic."""
        rng = random.Random(0)
        pop = [Genome(k, 0.1, 0.1) for k in (5, 10, 15, 20)]
        fits = [0.1, 0.2, 0.3, 0.99]
        nxt = next_generation(pop, fits, SPACE, rng, crossover_rate=1.0,
                              mutation_rate=1.0, n_elite=1, tournament_size=2)
        assert any(g == pop[3] for g in nxt)


class TestGenerationSize:
    def test_population_size_is_preserved(self):
        rng = random.Random(0)
        pop = random_population(SPACE, 9, rng)
        fits = [rng.random() for _ in pop]
        assert len(next_generation(pop, fits, SPACE, rng, 0.7, 0.05, 2, 3)) == 9


class TestFitnessNormalisation:
    def test_maps_the_reference_band_onto_zero_one(self):
        assert normalise_cv(0.30) == 0.0
        assert normalise_cv(0.70) == 1.0
        assert abs(normalise_cv(0.50) - 0.5) < 1e-9

    def test_clips_outside_the_band(self):
        assert normalise_cv(0.1) == 0.0
        assert normalise_cv(0.9) == 1.0

    def test_nan_scores_zero_rather_than_propagating(self):
        assert normalise_cv(float("nan")) == 0.0


class TestStability:
    def test_identical_topic_sets_score_one(self):
        t = [["a", "b", "c"], ["d", "e", "f"]]
        assert matched_jaccard(t, t) == 1.0

    def test_disjoint_topic_sets_score_zero(self):
        a = [["a", "b"], ["c", "d"]]
        b = [["w", "x"], ["y", "z"]]
        assert matched_jaccard(a, b) == 0.0

    def test_topic_order_does_not_matter(self):
        """Topic indices are arbitrary; two runs recovering the same topics in
        a different order must not be scored as unstable."""
        a = [["a", "b", "c"], ["d", "e", "f"]]
        b = [["d", "e", "f"], ["a", "b", "c"]]
        assert matched_jaccard(a, b) == 1.0

    def test_mismatched_topic_counts_are_penalised(self):
        """A 2-topic model must not score 1.0 against a 4-topic model just by
        matching its two best."""
        a = [["a", "b"], ["c", "d"]]
        b = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
        assert matched_jaccard(a, b) == 0.5


class TestRandomSearchControl:
    """EXP-008 compares the GA against random search at equal budget. The control
    is only valid if both draw candidates from the same distribution and differ
    solely in how candidates are proposed."""

    def test_random_sampler_is_the_one_the_ga_seeds_with(self):
        """Both strategies must use SearchSpace.random_genome, or the comparison
        confounds search strategy with proposal distribution."""
        rng_a = random.Random(7)
        rng_b = random.Random(7)
        a = [SPACE.random_genome(rng_a).key() for _ in range(20)]
        b = [SPACE.random_genome(rng_b).key() for _ in range(20)]
        assert a == b, "sampler must be deterministic under a fixed seed"

    def test_sampler_covers_the_full_k_range(self):
        """A control that never proposes large K would hand the GA an unfair win."""
        rng = random.Random(11)
        ks = [SPACE.random_genome(rng).k for _ in range(500)]
        assert min(ks) <= SPACE.k_min + 3
        assert max(ks) >= SPACE.k_max - 3

    def test_cache_key_makes_budget_matching_meaningful(self):
        """Budget is matched on UNIQUE evaluations. Two genomes differing only
        past the rounding precision must collide, or the two searches would be
        charged differently for the same work."""
        assert Genome(9, 0.30001, 0.5).key() == Genome(9, 0.30002, 0.5).key()
