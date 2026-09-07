"""A Mamdani fuzzy inference engine, vectorised over samples.

Why this is written here rather than taken from a library
--------------------------------------------------------
Two hard requirements drove it:

1. **Part 2 needs speed.** The genetic algorithm rebuilds the whole rule base
   with new membership-function parameters once per individual per generation
   and scores it over the entire training set. scikit-fuzzy's `ControlSystem`
   rebuilds a graph on construction and simulates one sample at a time, which
   puts a 100 x 150 x 200 search out of reach. Here a whole dataset is pushed
   through in a handful of array operations.
2. **Part 1 needs the intermediate quantities.** The output-behaviour analysis
   is marked on rule activation, controller output and control surfaces. That
   means per-rule firing strengths, the clipped consequent sets and the
   aggregated set all have to be inspectable, and a library that returns only
   the defuzzified scalar hides exactly what has to be shown.

The engine is validated against scikit-fuzzy in `tests/test_against_skfuzzy.py`
so that the speed and transparency are not bought with correctness.

Inference pipeline (Mamdani, as specified in Part 1)
----------------------------------------------------
    fuzzification -> rule firing (AND = min) -> implication (min, i.e. clipping)
    -> aggregation (max) -> defuzzification (centroid)

One algebraic shortcut is used, and it is exact rather than an approximation.
With min-implication and max-aggregation the aggregated set is

    agg(y) = max_r  min( alpha_r , mu_C(r)(y) )

Rules that share a consequent set C can be collapsed first, because

    max over r in R_C of min(alpha_r, mu_C(y))
        = min( max over r in R_C of alpha_r , mu_C(y) )

so only one clipped curve per *consequent set* is ever built, not one per rule.
With 45 thermal rules over 5 HVAC sets that is a ninefold saving, and the
result is identical to clipping every rule separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .membership import Variable


@dataclass(frozen=True)
class Rule:
    """One IF-THEN rule.

    `antecedents` maps an input variable name to one of its MF names. A rule
    need not mention every input: the lighting rules constrain only daylight
    and activity, and say nothing about temperature or preference. Silence
    means "do not care", not "zero".
    """

    antecedents: dict
    consequent: tuple  # (output variable name, MF name)
    weight: float = 1.0
    note: str = ""

    def text(self) -> str:
        ants = " AND ".join(f"{v} is {m}" for v, m in self.antecedents.items())
        ov, om = self.consequent
        s = f"IF {ants} THEN {ov} is {om}"
        return s if self.weight == 1.0 else f"{s} (w={self.weight:g})"


@dataclass
class InferenceTrace:
    """Everything the inference did for one crisp input vector.

    This is the object the Part 1 write-up is built from: it carries the
    fuzzified inputs, every rule's firing strength, the clipped consequent
    sets, the aggregated set and the final crisp value, so each stage of the
    pipeline can be tabulated or plotted exactly as the engine computed it.
    """

    crisp_inputs: dict
    memberships: dict          # input name -> {mf name: mu}
    firing: np.ndarray         # (n_rules,) strength after weighting
    active_rules: list         # indices of rules with strength > 0
    clipped: dict              # output name -> {mf name: alpha}
    aggregated: dict           # output name -> curve over the universe
    outputs: dict              # output name -> crisp value

    def activation_table(self, rules, top=None):
        """Rules sorted by firing strength, strongest first."""
        order = sorted(self.active_rules, key=lambda r: -self.firing[r])
        if top is not None:
            order = order[:top]
        return [(r, rules[r].text(), float(self.firing[r])) for r in order]


class FuzzyController:
    """A Mamdani FIS over named input and output variables."""

    def __init__(self, inputs, outputs, rules, name="FLC"):
        self.name = name
        self.inputs = {v.name: v for v in inputs}
        self.outputs = {v.name: v for v in outputs}
        self.rules = list(rules)
        self._compile()

    # ------------------------------------------------------------------ setup

    def _compile(self):
        """Resolve rule names to integer indices once, up front.

        Doing this at construction turns every later inference into pure array
        indexing, and it fails loudly here if a rule names a set that does not
        exist -- far better than silently never firing.
        """
        self.input_names = list(self.inputs)
        self.output_names = list(self.outputs)

        self._ante = []       # per rule: [(input idx, mf idx), ...]
        self._cons = []       # per rule: (output idx, mf idx)
        for rule in self.rules:
            ants = []
            for var_name, mf_name in rule.antecedents.items():
                if var_name not in self.inputs:
                    raise KeyError(f"rule refers to unknown input {var_name!r}")
                ants.append(
                    (self.input_names.index(var_name),
                     self.inputs[var_name].index_of(mf_name))
                )
            if not ants:
                raise ValueError(f"rule has no antecedents: {rule}")
            self._ante.append(ants)

            ov, om = rule.consequent
            if ov not in self.outputs:
                raise KeyError(f"rule refers to unknown output {ov!r}")
            self._cons.append(
                (self.output_names.index(ov), self.outputs[ov].index_of(om))
            )

        # Group rule indices by the consequent set they share -- the collapse
        # described in the module docstring.
        self._groups = {}
        for r, key in enumerate(self._cons):
            self._groups.setdefault(key, []).append(r)

        # Cache each output's universe and sampled MFs; these are fixed for a
        # given parameter set and are touched on every single inference.
        self._universe = [self.outputs[n].universe for n in self.output_names]
        self._out_mfs = [self.outputs[n].mf_matrix() for n in self.output_names]

    @property
    def n_params(self) -> int:
        """Total tunable MF parameters -- Part 2's chromosome sizing."""
        return (sum(v.n_params for v in self.inputs.values())
                + sum(v.n_params for v in self.outputs.values()))

    def param_layout(self):
        """(variable name, role, n_params) per variable, in chromosome order."""
        rows = [(n, "input", self.inputs[n].n_params) for n in self.input_names]
        rows += [(n, "output", self.outputs[n].n_params) for n in self.output_names]
        return rows

    def get_params(self) -> np.ndarray:
        parts = [self.inputs[n].get_params() for n in self.input_names]
        parts += [self.outputs[n].get_params() for n in self.output_names]
        return np.concatenate(parts)

    def with_params(self, vec) -> "FuzzyController":
        """A copy of this controller with new MF parameters, rules unchanged.

        Part 2 tunes the membership functions only; the rule base is expert
        knowledge and is held fixed, so the linguistic meaning of the
        controller survives optimisation.
        """
        vec = np.asarray(vec, dtype=float)
        if vec.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {vec.size}")
        ins, outs, i = [], [], 0
        for n in self.input_names:
            k = self.inputs[n].n_params
            ins.append(self.inputs[n].with_params(vec[i:i + k]))
            i += k
        for n in self.output_names:
            k = self.outputs[n].n_params
            outs.append(self.outputs[n].with_params(vec[i:i + k]))
            i += k
        return FuzzyController(ins, outs, self.rules, name=self.name)

    # -------------------------------------------------------------- inference

    def _firing(self, X: np.ndarray) -> np.ndarray:
        """Firing strength of every rule for every sample -> (N, n_rules).

        AND is the minimum t-norm, applied across a rule's antecedents. Rules
        are weighted afterwards; every rule here carries weight 1, but the
        mechanism is kept because it is part of the Mamdani specification.
        """
        mu = [self.inputs[n].fuzzify(X[:, i])
              for i, n in enumerate(self.input_names)]
        N = X.shape[0]
        strength = np.empty((N, len(self.rules)))
        for r, ants in enumerate(self._ante):
            cols = [mu[vi][:, mi] for vi, mi in ants]
            strength[:, r] = cols[0] if len(cols) == 1 else np.min(cols, axis=0)
            if self.rules[r].weight != 1.0:
                strength[:, r] *= self.rules[r].weight
        return strength

    def _aggregate(self, strength: np.ndarray, oi: int) -> np.ndarray:
        """Aggregated fuzzy set for output `oi` -> (N, n_points)."""
        n_mfs = self._out_mfs[oi].shape[0]
        N = strength.shape[0]
        alpha = np.zeros((N, n_mfs))
        for (o, m), rule_ids in self._groups.items():
            if o == oi:
                alpha[:, m] = strength[:, rule_ids].max(axis=1)
        # min-implication (clip) then max-aggregation, broadcast over the grid
        clipped = np.minimum(alpha[:, :, None], self._out_mfs[oi][None, :, :])
        return clipped.max(axis=1)

    @staticmethod
    def _centroid(agg: np.ndarray, universe: np.ndarray) -> np.ndarray:
        """Centre-of-gravity defuzzification, by exact trapezoidal integration.

        The aggregated set is piecewise linear between grid points, so its
        first moment and its area can both be integrated exactly rather than
        approximated. Over one segment [x1, x2] with heights y1, y2:

            area        = (dx/2) (y1 + y2)
            moment.area = (dx/2) [ x1 (y1 + y2) + dx (y1 + 2 y2) / 3 ]

        and COG is the ratio of the sums. Writing the moment pre-multiplied by
        the area avoids a division that would be 0/0 on any flat-zero segment,
        of which there are many.

        This is what MATLAB's `defuzz(..., 'centroid')` and scikit-fuzzy both
        compute, and matching them is what makes the cross-check in
        `tests/test_against_skfuzzy.py` a real test rather than a comparison of
        two different formulas. The cruder discrete form is kept separately as
        `_centroid_discrete` because that is the one a person can evaluate by
        hand, and the report's worked example uses it.

        Where a sample activates no rule at all the aggregated set is
        identically zero and the centroid is undefined. That is a real
        condition, not a numerical artefact, so it is reported as NaN rather
        than silently defaulting to the middle of the universe -- a controller
        that cannot decide should say so, and the rule-base completeness check
        in `scripts/run_part1.py` looks for exactly this.
        """
        x1, x2 = universe[:-1], universe[1:]
        dx = x2 - x1
        y1, y2 = agg[:, :-1], agg[:, 1:]

        area = 0.5 * dx * (y1 + y2)
        moment_area = 0.5 * dx * (x1 * (y1 + y2) + dx * (y1 + 2.0 * y2) / 3.0)

        total = area.sum(axis=1)
        out = np.full(total.shape, np.nan)
        ok = total > 0
        out[ok] = moment_area.sum(axis=1)[ok] / total[ok]
        return out

    @staticmethod
    def _centroid_discrete(agg: np.ndarray, universe: np.ndarray) -> np.ndarray:
        """COG as the discrete weighted sum, sum(mu.y) / sum(mu).

        The textbook hand-calculation form, and the one used in the worked
        example in the report. It carries an O(h) bias at the edges of the
        aggregated set relative to the trapezoidal integral above; the size of
        that gap on this controller is quantified in `scripts/run_part1.py`
        rather than left as an assertion.
        """
        mass = agg.sum(axis=1)
        out = np.full(mass.shape, np.nan)
        ok = mass > 0
        out[ok] = (agg[ok] * universe[None, :]).sum(axis=1) / mass[ok]
        return out

    def evaluate(self, X) -> np.ndarray:
        """Crisp outputs for a batch of crisp inputs.

        X is (N, n_inputs) in `input_names` order; returns (N, n_outputs) in
        `output_names` order.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} inputs {self.input_names}, "
                f"got {X.shape[1]}"
            )
        strength = self._firing(X)
        return np.stack(
            [self._centroid(self._aggregate(strength, oi), self._universe[oi])
             for oi in range(len(self.output_names))],
            axis=1,
        )

    def __call__(self, **kwargs) -> dict:
        """Single-sample convenience form: `flc(room_temp=27, ...)`."""
        X = np.array([[float(kwargs[n]) for n in self.input_names]])
        y = self.evaluate(X)[0]
        return dict(zip(self.output_names, y))

    def trace(self, **kwargs) -> InferenceTrace:
        """Run one inference and keep every intermediate quantity."""
        X = np.array([[float(kwargs[n]) for n in self.input_names]])
        strength = self._firing(X)[0]

        memberships = {}
        for i, n in enumerate(self.input_names):
            var = self.inputs[n]
            memberships[n] = dict(zip(var.mf_names, var.fuzzify(X[0, i])))

        clipped, aggregated, outputs = {}, {}, {}
        for oi, n in enumerate(self.output_names):
            var = self.outputs[n]
            alpha = {}
            for (o, m), rule_ids in self._groups.items():
                if o == oi:
                    a = float(strength[rule_ids].max())
                    if a > 0:
                        alpha[var.mf_names[m]] = a
            clipped[n] = alpha
            agg = self._aggregate(strength[None, :], oi)
            aggregated[n] = agg[0]
            outputs[n] = float(self._centroid(agg, self._universe[oi])[0])

        return InferenceTrace(
            crisp_inputs={n: float(X[0, i]) for i, n in enumerate(self.input_names)},
            memberships=memberships,
            firing=strength,
            active_rules=[int(r) for r in np.flatnonzero(strength > 0)],
            clipped=clipped,
            aggregated=aggregated,
            outputs=outputs,
        )

    def control_surface(self, x_var, y_var, out_var, fixed, n=61):
        """Sample the input-output surface over two inputs, others held fixed.

        Returns (xs, ys, Z) with Z shaped (n, n) and indexed Z[j, i] for
        xs[i], ys[j], which is the orientation matplotlib's `plot_surface`
        expects from `meshgrid` defaults.
        """
        vx, vy = self.inputs[x_var], self.inputs[y_var]
        xs = np.linspace(vx.lo, vx.hi, n)
        ys = np.linspace(vy.lo, vy.hi, n)
        XX, YY = np.meshgrid(xs, ys)

        X = np.empty((XX.size, len(self.input_names)))
        for i, name in enumerate(self.input_names):
            if name == x_var:
                X[:, i] = XX.ravel()
            elif name == y_var:
                X[:, i] = YY.ravel()
            else:
                if name not in fixed:
                    raise KeyError(f"no fixed value supplied for input {name!r}")
                X[:, i] = float(fixed[name])

        Z = self.evaluate(X)[:, self.output_names.index(out_var)]
        return xs, ys, Z.reshape(XX.shape)
