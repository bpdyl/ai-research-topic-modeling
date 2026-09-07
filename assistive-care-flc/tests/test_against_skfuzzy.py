"""Validate the hand-written Mamdani engine against scikit-fuzzy.

The engine in `acflc.controller` was written rather than imported, for the
reasons given in that module's docstring. This test is the price of that
choice: the same controller is rebuilt with scikit-fuzzy's `control` API and
the two are required to agree on the crisp outputs.

scikit-fuzzy is a test-only dependency. It is deliberately *not* in the
runtime requirements -- it is an independent oracle, and an oracle that shares
code with the thing it checks is not independent.

This mirrors the cross-check in the Lab 3 barometer exercise, where the
hand-computed rule strengths were checked against MATLAB's `evalfis`.
"""

from __future__ import annotations

import numpy as np
import pytest

from acflc import build_controller
from acflc.flat import ACTIVITY, DAYLIGHT, DIMMER, HVAC, PREFERENCE, ROOM_TEMP

skfuzzy = pytest.importorskip("skfuzzy", reason="oracle library not installed")
from skfuzzy import control as ctrl  # noqa: E402


def _add_mfs(target, var):
    """Copy one of our Variables onto a scikit-fuzzy Antecedent/Consequent."""
    for mf in var.mfs:
        fn = skfuzzy.trimf if mf.kind == "trimf" else skfuzzy.trapmf
        target[mf.name] = fn(target.universe, list(mf.params))


def _build_skfuzzy_twin(flc):
    """The same FIS, expressed entirely through scikit-fuzzy."""
    ants, cons = {}, {}
    for var in (ROOM_TEMP, ACTIVITY, DAYLIGHT, PREFERENCE):
        a = ctrl.Antecedent(var.universe, var.name)
        _add_mfs(a, var)
        ants[var.name] = a
    for var in (HVAC, DIMMER):
        c = ctrl.Consequent(var.universe, var.name, defuzzify_method="centroid")
        _add_mfs(c, var)
        cons[var.name] = c

    rules = []
    for rule in flc.rules:
        term = None
        for var_name, mf_name in rule.antecedents.items():
            piece = ants[var_name][mf_name]
            term = piece if term is None else (term & piece)
        out_var, out_mf = rule.consequent
        rules.append(ctrl.Rule(term, cons[out_var][out_mf]))

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


# A fixed pseudo-random sweep of the input space. Seeded so a failure is
# reproducible, and wide enough that it exercises corners as well as the
# comfortable middle where every controller looks the same.
def _sample_inputs(n, seed=20260906):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(ROOM_TEMP.lo, ROOM_TEMP.hi, n),
        rng.uniform(ACTIVITY.lo, ACTIVITY.hi, n),
        rng.uniform(DAYLIGHT.lo, DAYLIGHT.hi, n),
        rng.uniform(PREFERENCE.lo, PREFERENCE.hi, n),
    ])


def test_membership_shapes_match_skfuzzy():
    """Our trimf/trapmf must be the library's, pointwise."""
    for var in (ROOM_TEMP, ACTIVITY, DAYLIGHT, PREFERENCE, HVAC, DIMMER):
        u = var.universe
        for mf in var.mfs:
            fn = skfuzzy.trimf if mf.kind == "trimf" else skfuzzy.trapmf
            np.testing.assert_allclose(
                mf(u), fn(u, np.array(mf.params, dtype=float)),
                atol=1e-12, err_msg=f"{var.name}.{mf.name}",
            )


def test_rule_activations_match_skfuzzy_exactly():
    """The fuzzy reasoning proper must agree to floating-point precision.

    Everything up to and including aggregation is exact arithmetic in both
    implementations -- fuzzification, the min t-norm across antecedents, and
    the max across rules sharing a consequent. Only the final integration
    involves any approximation. So the activation level of every consequent
    set is required to match at 1e-12, with no tolerance for hand-waving; if
    the rule logic diverges, it diverges here.
    """
    flc = build_controller()
    twin = _build_skfuzzy_twin(flc)

    for row in _sample_inputs(40):
        for i, name in enumerate(flc.input_names):
            twin.input[name] = row[i]
        twin.compute()
        tr = flc.trace(**dict(zip(flc.input_names, row)))

        for out_name in flc.output_names:
            var = [v for v in twin.ctrl.consequents if v.label == out_name][0]
            theirs = {lbl: float(t.membership_value[twin])
                      for lbl, t in var.terms.items()}
            for mf_name in flc.outputs[out_name].mf_names:
                a = tr.clipped[out_name].get(mf_name, 0.0)
                b = theirs.get(mf_name, 0.0)
                assert abs(a - b) < 1e-12, (
                    f"{out_name}.{mf_name} at {dict(zip(flc.input_names, row))}: "
                    f"ours={a!r} skfuzzy={b!r}"
                )


def test_defuzzification_converges_to_skfuzzy():
    """Crisp outputs agree in the limit of a fine output universe.

    The two engines discretise the aggregated set differently. scikit-fuzzy
    upsamples the universe to insert the exact points where each membership
    function crosses its activation level, so its clipping corners land on
    grid points; this engine integrates a fixed uniform grid, which straddles
    them. Both are approximating the same continuous centre of gravity, and
    the residual is discretisation error, not a difference of logic -- which
    this test establishes by refining the grid and requiring the gap to fall.

    Anchoring on convergence rather than on a single loose tolerance is the
    point: a fixed 5e-4 threshold at 501 points would also be passed by an
    engine that was quietly wrong in a way that did not shrink with h.
    """
    from dataclasses import replace as _replace

    import acflc.flat as flat_mod
    from acflc.controller import FuzzyController
    from acflc.flat import lighting_rules, thermal_rules

    rules = thermal_rules() + lighting_rules()
    X = _sample_inputs(15, seed=99)
    gaps = []

    for n_points in (501, 2001, 8001):
        hv = _replace(HVAC, n_points=n_points)
        dm = _replace(DIMMER, n_points=n_points)
        flc = FuzzyController([ROOM_TEMP, ACTIVITY, DAYLIGHT, PREFERENCE],
                              [hv, dm], rules)
        # the twin must be built on the same universes to be comparable
        orig = flat_mod.HVAC, flat_mod.DIMMER
        flat_mod.HVAC, flat_mod.DIMMER = hv, dm
        globals()["HVAC"], globals()["DIMMER"] = hv, dm
        try:
            twin = _build_skfuzzy_twin(flc)
            ours = flc.evaluate(X)
            worst = 0.0
            for k, row in enumerate(X):
                for i, name in enumerate(flc.input_names):
                    twin.input[name] = row[i]
                twin.compute()
                for j, name in enumerate(flc.output_names):
                    assert np.isfinite(ours[k, j]), f"sample {k}: {name} was NaN"
                    worst = max(worst, abs(ours[k, j] - twin.output[name]))
            gaps.append(worst)
        finally:
            flat_mod.HVAC, flat_mod.DIMMER = orig
            globals()["HVAC"], globals()["DIMMER"] = orig

    assert gaps[0] < 5e-3, f"disagreement too large even at 501 points: {gaps[0]:.2e}"
    assert gaps[-1] < gaps[0] / 5, (
        f"gap did not shrink with the grid: {gaps} -- that points at a genuine "
        f"difference in the inference, not discretisation"
    )
    assert gaps[-1] < 1e-4, f"did not converge: {gaps[-1]:.2e}"


def test_traced_firing_strengths_are_consistent():
    """The trace's per-rule strengths must reproduce the batch computation."""
    flc = build_controller()
    X = _sample_inputs(12, seed=7)
    for row in X:
        kw = dict(zip(flc.input_names, row))
        tr = flc.trace(**kw)
        batch = flc.evaluate(row[None, :])[0]
        for j, name in enumerate(flc.output_names):
            assert abs(tr.outputs[name] - batch[j]) < 1e-12
        # every rule reported active must genuinely have positive strength
        assert all(tr.firing[r] > 0 for r in tr.active_rules)
        assert np.count_nonzero(tr.firing) == len(tr.active_rules)
