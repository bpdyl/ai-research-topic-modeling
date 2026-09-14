"""Behavioural properties of the assistive-care controller.

These are not smoke tests. Each one pins a property the report claims, so that
a later edit to the membership functions or the rule base cannot quietly
falsify a sentence in the write-up.
"""

from __future__ import annotations

import numpy as np
import pytest

from acflc import build_controller


@pytest.fixture(scope="module")
def flc():
    return build_controller()


@pytest.fixture(scope="module")
def sweep(flc):
    """A dense grid over the whole input space, and the controller's response."""
    grids = [np.linspace(flc.inputs[n].lo, flc.inputs[n].hi, k)
             for n, k in zip(flc.input_names, (41, 15, 9, 15))]
    X = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 4)
    return X, flc.evaluate(X)


def test_structure_matches_reported_sizing(flc):
    """The numbers quoted in Part 1 and Part 2 of the report."""
    assert len(flc.rules) == 54
    assert sum(1 for r in flc.rules if r.consequent[0] == "hvac") == 45
    assert sum(1 for r in flc.rules if r.consequent[0] == "dimmer") == 9
    assert flc.n_params == 81
    assert flc.n_params * 8 == 648      # chromosome length, Part 2


def test_rule_base_is_complete(flc, sweep):
    """Every reachable input must produce a defined control action.

    An incomplete rule base shows up as a NaN: no rule fires, the aggregated
    set is empty and the centroid is undefined. In a care setting that is the
    actuator holding its last value with no one aware, so it is worth an
    assertion rather than a spot check.
    """
    _, Y = sweep
    assert np.isfinite(Y).all(), (
        f"{int((~np.isfinite(Y)).any(axis=1).sum())} input combinations "
        f"produced no control action"
    )


def test_no_redundant_rules(flc, sweep):
    """Every rule must fire somewhere, or it is dead weight in the rule base."""
    X, _ = sweep
    strength = flc._firing(X)
    dead = [i for i in range(len(flc.rules)) if strength[:, i].max() == 0]
    assert not dead, f"rules never activated: {[flc.rules[i].text() for i in dead]}"


def test_never_heats_a_hot_room_or_cools_a_cold_one(flc, sweep):
    """The safety property the clamped FAM modifier exists to guarantee.

    Activity and preference adjust thermal demand but must never overrule the
    measured temperature. Without the inner clamp in `flat.py` a Comfortable
    room with a resting occupant who prefers warmth commanded full heat; this
    test is what stops that regressing.
    """
    X, Y = sweep
    hot, cold = X[:, 0] >= 28.0, X[:, 0] <= 17.0
    assert (Y[hot, 0] < 0).all(), (
        f"heating commanded in a room at >=28C (max {Y[hot, 0].max():.2f})")
    assert (Y[cold, 0] > 0).all(), (
        f"cooling commanded in a room at <=17C (min {Y[cold, 0].min():.2f})")


#: Largest local rise in the HVAC command as the room warms, as a percentage
#: of the output span. See `test_hvac_is_monotone_in_temperature` for why this
#: is not zero, and section 5 of the report, which quotes this figure.
MONOTONICITY_TOLERANCE_PCT = 0.5


def test_hvac_is_monotone_in_temperature(flc):
    """Hotter room, never meaningfully more heat -- for all other inputs.

    The command falls overall, strictly, in every slice. It is not, however,
    *pointwise* monotone, and the reason is worth stating because it is a
    property of Mamdani inference rather than a defect in this rule base.

    Where the demand scale saturates, two adjacent temperature sets map to the
    same consequent -- `Cold + Resting + Neutral` and `Cool + Resting + Neutral`
    both give HeatHigh. The activation of that consequent is then
    max(mu_Cold, mu_Cool), which is V-shaped through the crossover at about
    17.9 degrees: mu_Cold is falling, mu_Cool is rising, and past the crossover
    the aggregate activation climbs again. The crisp output follows it back up.

    The excursion is about 0.6% of the output span, an order of magnitude
    below the resolution of any real HVAC actuator, so it is bounded and
    declared here rather than engineered away -- removing it would mean giving
    up either the saturating end sets or the overlap that keeps the rule base
    complete, both of which are worth more than 0.6%.
    """
    temps = np.linspace(flc.inputs["room_temp"].lo, flc.inputs["room_temp"].hi, 401)
    span = flc.outputs["hvac"].hi - flc.outputs["hvac"].lo
    worst = 0.0

    for activity in (0.0, 5.0, 10.0):
        for preference in (-5.0, 0.0, 5.0):
            X = np.column_stack([
                temps,
                np.full_like(temps, activity),
                np.full_like(temps, 50.0),
                np.full_like(temps, preference),
            ])
            hvac = flc.evaluate(X)[:, 0]

            # the overall trend must be strictly downward, with real authority
            assert hvac[0] - hvac[-1] > 0.5 * span, (
                f"HVAC barely responded to temperature at activity={activity}, "
                f"preference={preference}: {hvac[0]:.1f} -> {hvac[-1]:.1f}")

            rise = max(0.0, float(np.diff(hvac).max()))
            worst = max(worst, 100.0 * rise / span)

    assert worst < MONOTONICITY_TOLERANCE_PCT, (
        f"local non-monotonicity grew to {worst:.3f}% of the output span, "
        f"above the {MONOTONICITY_TOLERANCE_PCT}% the report declares")


def test_lamp_keeps_a_safety_floor_below_bright_daylight(flc):
    """The fall-risk override: the lamp is never fully off unless it is bright.

    This is a deliberate design decision recorded in `flat.py`, trading a
    little energy for orientation lighting, and it is exactly the sort of
    choice that gets silently optimised away later.
    """
    for daylight in (0.0, 15.0, 40.0, 55.0):
        X = np.array([[22.0, a, daylight, 0.0] for a in np.linspace(0, 10, 21)])
        dimmer = flc.evaluate(X)[:, 1]
        assert dimmer.min() > 5.0, (
            f"lamp fell to {dimmer.min():.2f}% at daylight={daylight}")


def test_parameter_vector_round_trips(flc):
    """`with_params(get_params())` must reproduce the controller exactly.

    Part 2's GA depends on this being lossless in both directions.
    """
    p = flc.get_params()
    assert p.size == flc.n_params
    rebuilt = flc.with_params(p)
    np.testing.assert_allclose(rebuilt.get_params(), p, atol=0)

    X = np.array([[22.0, 3.0, 40.0, 0.0], [30.0, 8.0, 90.0, -3.0]])
    np.testing.assert_allclose(rebuilt.evaluate(X), flc.evaluate(X), atol=0)


def test_out_of_order_parameters_are_repaired_not_rejected(flc):
    """A GA proposing a scrambled MF must yield a valid controller.

    The repair operator sorts each membership function's parameters. Without
    it the genetic algorithm would spend most of its budget generating
    individuals that raise rather than individuals that are merely bad.
    """
    rng = np.random.default_rng(0)
    p = flc.get_params()
    scrambled = p + rng.normal(0, 8.0, p.shape)
    rebuilt = flc.with_params(scrambled)          # must not raise
    Y = rebuilt.evaluate(np.array([[22.0, 3.0, 40.0, 0.0]]))
    assert Y.shape == (1, 2)
