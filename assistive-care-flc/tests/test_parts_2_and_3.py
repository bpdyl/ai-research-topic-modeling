"""Properties of the Part 2 genetic algorithm and the Part 3 benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from acflc import build_controller
from acflc import benchmarks as bm
from acflc import optimisers as op
from acflc.dataset import make_dataset, normalised_rmse, reference_policy
from acflc.ga import GAConfig, MembershipGA


@pytest.fixture(scope="module")
def rig():
    flc = build_controller()
    data = make_dataset(flc, n=120, seed=1)
    return flc, data, MembershipGA(flc, data, GAConfig(population=12,
                                                       generations=4, seed=0))


# ------------------------------------------------------------------- Part 2

def test_chromosome_length_is_648_bits(rig):
    """The figure Part 2 must state, checked against the encoding itself."""
    _flc, _data, ga = rig
    assert ga.n_genes == 81
    assert ga.n_bits == 648
    chrom = ga.encode(ga.flc.get_params())
    assert chrom.shape == (648,)
    assert set(np.unique(chrom)) <= {0, 1}


def test_encode_decode_round_trips_within_quantisation(rig):
    """8 bits per gene resolves span/255; the round trip must stay inside it."""
    flc, _data, ga = rig
    p = flc.get_params()
    back = ga.decode(ga.encode(p))
    tol = (ga.hi - ga.lo) / 255.0 / 2.0 + 1e-9
    assert np.all(np.abs(back - p) <= tol)


def test_decode_always_yields_a_usable_controller(rig):
    """Any chromosome at all must decode to a controller that evaluates.

    This is what the sorting repair operator buys. Without it a random
    chromosome raises on the first out-of-order triangle and the GA cannot
    even score its initial population.
    """
    flc, _data, ga = rig
    rng = np.random.default_rng(3)
    X = np.array([[22.0, 3.0, 40.0, 0.0], [31.0, 9.0, 95.0, -4.0]])
    for _ in range(25):
        chrom = rng.integers(0, 2, ga.n_bits, dtype=np.uint8)
        candidate = flc.with_params(ga.decode(chrom))
        Y = candidate.evaluate(X)
        assert Y.shape == (2, 2)


def test_decoded_parameters_stay_inside_their_universes(rig):
    """A gene can never place a breakpoint outside its variable's range."""
    _flc, _data, ga = rig
    rng = np.random.default_rng(11)
    pop = rng.integers(0, 2, (16, ga.n_bits), dtype=np.uint8)
    vals = ga.decode(pop)
    assert (vals >= ga.lo - 1e-9).all()
    assert (vals <= ga.hi + 1e-9).all()


def test_elitism_makes_best_fitness_non_decreasing(rig):
    """With elitism the best-so-far curve must never fall."""
    _flc, _data, ga = rig
    res = ga.run(seed_with_expert=True)
    hist = np.array(res.history_best)
    assert (np.diff(hist) >= -1e-15).all(), "best fitness fell between generations"


def test_seeding_means_the_ga_never_loses_to_the_expert(rig):
    """Seeding the population with the hand design makes this a guarantee."""
    flc, data, ga = rig
    spans = np.array([flc.outputs[n].hi - flc.outputs[n].lo
                      for n in flc.output_names])
    base, _ = normalised_rmse(data.y_train, flc.evaluate(data.X_train), spans)
    res = ga.run(seed_with_expert=True)
    # allow one quantisation step of slack: the expert parameters are encoded
    # onto the 8-bit grid before they enter the population
    assert res.best_nrmse <= base + 5e-3


def test_nan_predictions_are_charged_not_dropped():
    """A controller with holes must not score better by having them."""
    spans = np.array([200.0, 100.0])
    y = np.zeros((4, 2))
    good = np.zeros((4, 2))
    holed = np.zeros((4, 2))
    holed[0, 0] = np.nan
    e_good, _ = normalised_rmse(y, good, spans)
    e_holed, _ = normalised_rmse(y, holed, spans)
    assert e_holed > e_good


def test_reference_policy_is_deterministic_and_bounded():
    """The declared synthetic target must respect its own stated limits."""
    rng = np.random.default_rng(0)
    X = rng.uniform([14, 0, 0, -5], [34, 10, 100, 5], size=(500, 4))
    y1, y2 = reference_policy(X), reference_policy(X)
    np.testing.assert_array_equal(y1, y2)
    assert (np.abs(y1[:, 0]) <= 85.0 + 1e-9).all()
    assert (y1[:, 1] >= 8.0 - 1e-9).all() and (y1[:, 1] <= 95.0 + 1e-9).all()


# ------------------------------------------------------------------- Part 3

@pytest.mark.parametrize("key,expected", [("F6", 390.0), ("F9", -330.0)])
@pytest.mark.parametrize("dim", [2, 10])
def test_benchmark_attains_its_documented_optimum(key, expected, dim):
    """f(o) must equal the suite's bias term exactly, or the shift is wrong."""
    fn = bm.make(key, dim)
    assert fn.optimum == expected
    assert abs(float(fn(fn.shift[None, :])[0]) - expected) < 1e-9


@pytest.mark.parametrize("key", ["F6", "F9"])
def test_shift_is_reproducible_and_inside_the_domain(key):
    fn1, fn2 = bm.make(key, 10), bm.make(key, 10)
    np.testing.assert_array_equal(fn1.shift, fn2.shift)
    assert (fn1.shift > fn1.lo).all() and (fn1.shift < fn1.hi).all()


def test_budget_is_enforced_exactly():
    """No algorithm may spend more evaluations than it is given.

    The whole Part 3 comparison rests on this, so it is asserted rather than
    assumed -- a population method that overruns by one generation would get a
    silent advantage over simulated annealing.
    """
    fn = bm.make("F9", 2)
    lo, hi = fn.bounds()
    for name, (solver, _p) in op.ALGORITHMS.items():
        obj = bm.BudgetedObjective(fn, 1000)
        solver(obj, lo, hi, np.random.default_rng(0))
        assert obj.used == 1000, f"{name} spent {obj.used} of 1000 evaluations"


def test_every_optimiser_improves_on_random_sampling():
    """A sanity floor: each algorithm must beat the best of a random sample."""
    fn = bm.make("F9", 10)
    lo, hi = fn.bounds()
    rng = np.random.default_rng(0)
    random_best = fn(rng.uniform(lo, hi, size=(4000, 10))).min()

    for name, (solver, _p) in op.ALGORITHMS.items():
        obj = bm.BudgetedObjective(fn, 4000)
        solver(obj, lo, hi, np.random.default_rng(0))
        assert obj.best < random_best, (
            f"{name} ({obj.best:.3f}) did not beat random sampling "
            f"({random_best:.3f}) at equal budget")


def test_convergence_trace_is_monotone_non_increasing():
    """Best-so-far can only go down; the graphs depend on it."""
    fn = bm.make("F6", 2)
    lo, hi = fn.bounds()
    obj = bm.BudgetedObjective(fn, 2000)
    op.particle_swarm(obj, lo, hi, np.random.default_rng(0))
    _grid, curve = obj.convergence()
    assert (np.diff(curve) <= 1e-12).all()
