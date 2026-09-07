"""Membership-function primitives and the fuzzy-variable container.

The shapes implemented here (triangular and trapezoidal) follow the same
definitions used by MATLAB's `trimf`/`trapmf` and by scikit-fuzzy, so that the
engine in `controller.py` can be validated against an independent
implementation (see `tests/test_against_skfuzzy.py`).

Only two shapes are used across the whole controller. Both are piecewise
linear, so evaluating one costs a handful of floating-point operations, and
every parameter is a point on the variable's own physical axis -- a
temperature in degrees Celsius, a dimmer level in percent. That keeps the
parameters readable to a domain expert (an occupational therapist can be asked
"is 19 degrees the top of Cold?") and it keeps them directly tunable by the
genetic algorithm in Part 2, which searches over exactly these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def trimf(x: np.ndarray, params) -> np.ndarray:
    """Triangular membership function with feet at a, c and apex at b.

    Degenerate forms are permitted: a == b gives a vertical left edge and
    b == c a vertical right edge, which is how a triangle is made to saturate
    at the end of a universe.
    """
    a, b, c = (float(v) for v in params)
    if not a <= b <= c:
        raise ValueError(f"trimf parameters must be ascending, got {params}")
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    if a != b:
        idx = (a < x) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)
    if b != c:
        idx = (b < x) & (x < c)
        y[idx] = (c - x[idx]) / (c - b)
    y[x == b] = 1.0
    return y


def trapmf(x: np.ndarray, params) -> np.ndarray:
    """Trapezoidal membership function: feet at a, d and a plateau over [b, c].

    Used at the ends of every universe. A trapezoid with a == b saturates: any
    temperature at or below the plateau's right shoulder is *fully* Cold, which
    is the physically correct reading -- 14 degrees and 8 degrees are not
    meaningfully different to an occupant, and both should command full heat.
    """
    a, b, c, d = (float(v) for v in params)
    if not a <= b <= c <= d:
        raise ValueError(f"trapmf parameters must be ascending, got {params}")
    x = np.asarray(x, dtype=float)
    y = np.ones_like(x)

    idx = x <= b
    y[idx] = trimf(x[idx], (a, b, b))
    idx = x >= c
    y[idx] = trimf(x[idx], (c, c, d))
    y[x < a] = 0.0
    y[x > d] = 0.0
    return y


_SHAPES = {"trimf": trimf, "trapmf": trapmf}
_ARITY = {"trimf": 3, "trapmf": 4}


@dataclass(frozen=True)
class MF:
    """One named fuzzy set on one variable."""

    name: str
    kind: str
    params: tuple

    def __post_init__(self):
        if self.kind not in _SHAPES:
            raise ValueError(f"unknown MF shape {self.kind!r}")
        if len(self.params) != _ARITY[self.kind]:
            raise ValueError(
                f"{self.kind} takes {_ARITY[self.kind]} parameters, "
                f"got {len(self.params)} for MF {self.name!r}"
            )

    @property
    def n_params(self) -> int:
        return _ARITY[self.kind]

    def __call__(self, x) -> np.ndarray:
        return _SHAPES[self.kind](x, self.params)


@dataclass(frozen=True)
class Variable:
    """A linguistic variable: a physical range carrying named fuzzy sets.

    `n_points` sets the discretisation of the universe. It only matters for
    output variables, where the aggregated fuzzy set is integrated numerically
    during centroid defuzzification; inputs are evaluated at the crisp reading
    directly and never touch the grid.
    """

    name: str
    unit: str
    lo: float
    hi: float
    mfs: tuple
    n_points: int = 501

    @property
    def universe(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, self.n_points)

    @property
    def mf_names(self) -> tuple:
        return tuple(mf.name for mf in self.mfs)

    def index_of(self, mf_name: str) -> int:
        try:
            return self.mf_names.index(mf_name)
        except ValueError:
            raise KeyError(
                f"variable {self.name!r} has no membership function {mf_name!r}; "
                f"available: {list(self.mf_names)}"
            ) from None

    @property
    def n_params(self) -> int:
        """Total tunable parameters on this variable -- the Part 2 gene count."""
        return sum(mf.n_params for mf in self.mfs)

    def fuzzify(self, x) -> np.ndarray:
        """Degrees of membership of x in every MF.

        Accepts a scalar or an array of N crisp readings and returns an array
        shaped (n_mfs,) or (N, n_mfs) respectively. The vectorised form is what
        makes the Part 2 fitness evaluation affordable: a whole dataset is
        fuzzified in one call.
        """
        x = np.asarray(x, dtype=float)
        cols = [mf(np.atleast_1d(x)) for mf in self.mfs]
        out = np.stack(cols, axis=-1)
        return out[0] if x.ndim == 0 else out

    def mf_matrix(self) -> np.ndarray:
        """Every MF sampled on the universe grid, shaped (n_mfs, n_points)."""
        return np.stack([mf(self.universe) for mf in self.mfs])

    # -- parameter vector interface, used by the Part 2 genetic algorithm ----

    def get_params(self) -> np.ndarray:
        """Flatten this variable's MF parameters into one vector."""
        return np.concatenate([np.asarray(mf.params, dtype=float) for mf in self.mfs])

    def with_params(self, vec) -> "Variable":
        """Rebuild this variable from a flat parameter vector.

        Each MF's slice is sorted ascending before use. The GA searches raw
        real numbers and has no notion of shape validity, so a decoded
        chromosome can easily propose `trimf(24, 19, 21)`. Sorting is a repair
        operator: it maps every point in the search space onto a valid fuzzy
        set instead of discarding the individual, which keeps selection
        pressure on fitness rather than on feasibility.
        """
        vec = np.asarray(vec, dtype=float)
        if vec.size != self.n_params:
            raise ValueError(
                f"variable {self.name!r} expects {self.n_params} parameters, got {vec.size}"
            )
        out, i = [], 0
        for mf in self.mfs:
            chunk = np.sort(vec[i : i + mf.n_params])
            out.append(replace(mf, params=tuple(chunk)))
            i += mf.n_params
        return replace(self, mfs=tuple(out))
