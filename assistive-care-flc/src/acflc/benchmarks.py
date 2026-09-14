"""Part 3: two functions from the CEC'2005 real-parameter benchmark suite.

Functions
---------
**F6 - Shifted Rosenbrock's Function.** Unimodal in two dimensions but with a
long, narrow, curved valley; multi-modal from D=3 upwards. Non-separable, so a
coordinate-wise search cannot decompose it.

    z    = x - o + 1
    F6(x) = sum_{i=1}^{D-1} [ 100 (z_i^2 - z_{i+1})^2 + (z_i - 1)^2 ]  + 390

    range [-100, 100]^D,  optimum at x = o with F6 = 390

**F9 - Shifted Rastrigin's Function.** Separable but massively multi-modal:
roughly 10^D local minima on a regular lattice, each a shallow basin on a
global parabolic bowl. The difficulty is not the geometry of any one basin but
escaping the one you land in.

    z    = x - o
    F9(x) = sum_{i=1}^{D} [ z_i^2 - 10 cos(2 pi z_i) + 10 ]  - 330

    range [-5, 5]^D,  optimum at x = o with F9 = -330

The pair is chosen to be complementary: one non-separable with a hard-to-follow
valley, one separable with an overwhelming number of local optima. An algorithm
that suits both is genuinely general; in practice they tend not to.

On the shift vectors -- read this before comparing with published results
--------------------------------------------------------------------------
The official CEC'2005 distribution ships the shift vectors `o` as data files.
Those files are not available on this machine, so the vectors used here are
**generated from a fixed seed and written out in full** to
`results/part3_shift_vectors.json`, and the generator is below.

That has one consequence and it matters: the numbers here are **not directly
comparable to published CEC'2005 results**, because a different shift moves
the optimum to a different place and changes how the boundary interacts with
the basin. Everything the report claims is a *within-study* comparison between
three algorithms on identical instances, which is what Part 3 actually asks
for. Any comparison against the literature would be unsound and is not made.

Both functions keep their official bias terms (+390, -330) so that the reported
optima match the suite's documented values, and both are shifted away from the
origin, which is the point of the shift: an algorithm that initialises at or
searches symmetrically about the centre of the domain gets no free advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Seed for the shift vectors. Fixed so every run of Part 3 uses identical
#: problem instances and results are reproducible.
SHIFT_SEED = 20260912


@dataclass(frozen=True)
class Benchmark:
    """One benchmark instance: the function, its domain and its known optimum."""

    key: str
    name: str
    lo: float
    hi: float
    bias: float
    dim: int
    shift: np.ndarray
    properties: tuple

    @property
    def optimum(self) -> float:
        """The global minimum value, which is the bias term by construction."""
        return self.bias

    def bounds(self):
        return (np.full(self.dim, self.lo), np.full(self.dim, self.hi))

    def __call__(self, X) -> np.ndarray:
        raise NotImplementedError


def _make_shift(dim, lo, hi, index, fraction=0.8):
    """A reproducible shift vector inside the search domain.

    Drawn uniformly from the middle `fraction` of the range so the optimum is
    never pinned against a boundary, which would let an algorithm that clamps
    to the bounds find it by accident. `index` decorrelates the two functions.
    """
    rng = np.random.default_rng(SHIFT_SEED + 1000 * index)
    half = fraction * (hi - lo) / 2.0
    centre = (hi + lo) / 2.0
    return rng.uniform(centre - half, centre + half, size=dim)


class ShiftedRosenbrock(Benchmark):
    """CEC'2005 F6."""

    def __init__(self, dim):
        super().__init__(
            key="F6", name="Shifted Rosenbrock", lo=-100.0, hi=100.0,
            bias=390.0, dim=dim, shift=_make_shift(dim, -100.0, 100.0, 0),
            properties=("multi-modal (D>2)", "shifted", "non-separable",
                        "scalable", "narrow curved valley"),
        )

    def __call__(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        z = X - self.shift + 1.0
        a = z[:, :-1] ** 2 - z[:, 1:]
        b = z[:, :-1] - 1.0
        return 100.0 * (a ** 2).sum(axis=1) + (b ** 2).sum(axis=1) + self.bias


class ShiftedRastrigin(Benchmark):
    """CEC'2005 F9."""

    def __init__(self, dim):
        super().__init__(
            key="F9", name="Shifted Rastrigin", lo=-5.0, hi=5.0,
            bias=-330.0, dim=dim, shift=_make_shift(dim, -5.0, 5.0, 1),
            properties=("multi-modal", "shifted", "separable", "scalable",
                        "very many local optima"),
        )

    def __call__(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        z = X - self.shift
        return (z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0).sum(axis=1) + self.bias


FUNCTIONS = {"F6": ShiftedRosenbrock, "F9": ShiftedRastrigin}


def make(key: str, dim: int) -> Benchmark:
    return FUNCTIONS[key](dim)


class BudgetedObjective:
    """Wraps a benchmark to count evaluations and record the best-so-far.

    Every algorithm in Part 3 is given the same number of *function
    evaluations*, not the same number of generations or iterations. That is the
    only comparison that means anything across a population method and a
    single-point method: a generation of 50 individuals costs 50 evaluations
    while an annealing step costs one, so matching "iterations" would hand
    simulated annealing a budget tens of times larger without saying so.

    The wrapper enforces the budget rather than trusting each algorithm to
    stop, and records the convergence trace used for the graphs.
    """

    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = int(budget)
        self.used = 0
        self.best = np.inf
        self.best_x = None
        self.trace = []          # (evaluations used, best so far)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def __call__(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.exhausted:
            # Return the true values but stop counting; algorithms are expected
            # to check `exhausted`, and this keeps a sloppy one from crashing.
            return self.fn(X)

        remaining = self.budget - self.used
        if len(X) > remaining:
            X = X[:remaining]
        y = self.fn(X)
        self.used += len(X)

        i = int(np.argmin(y))
        if y[i] < self.best:
            self.best = float(y[i])
            self.best_x = X[i].copy()
        self.trace.append((self.used, self.best))
        return y

    def convergence(self, n_points=200):
        """Best-so-far resampled onto a regular evaluation grid.

        Runs record traces at different points, so averaging across runs needs
        a common x-axis. Step interpolation is correct here because best-so-far
        is piecewise constant between improvements.
        """
        if not self.trace:
            return np.array([]), np.array([])
        ev = np.array([t[0] for t in self.trace], dtype=float)
        bs = np.array([t[1] for t in self.trace], dtype=float)
        grid = np.linspace(ev[0], self.budget, n_points)
        idx = np.searchsorted(ev, grid, side="right") - 1
        return grid, bs[np.clip(idx, 0, len(bs) - 1)]
