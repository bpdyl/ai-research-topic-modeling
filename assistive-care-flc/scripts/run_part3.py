"""Part 3 driver: GA vs PSO vs SA on two CEC'2005 functions.

    python scripts/run_part3.py [--runs 15]

Protocol
--------
Two functions (F6 Shifted Rosenbrock, F9 Shifted Rastrigin), three algorithms,
D = 2 and D = 10, 15 independent runs each: 180 runs in total. Every run gets
the same budget of 10,000 x D function evaluations, which is the CEC'2005
suite's own termination criterion.

Reported per cell: mean, standard deviation, best and worst of the final
objective value, plus the error against the known optimum, plus the mean
best-so-far convergence curve.

Seeding is `run index + 1000 x algorithm index`, fixed, so the whole table is
reproducible and every algorithm sees the same 15 starting conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acflc import benchmarks as bm  # noqa: E402
from acflc import optimisers as op  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

FUNCS = ["F6", "F9"]
DIMS = [2, 10]
ALGOS = ["GA", "PSO", "SA"]
EVALS_PER_DIM = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--evals-per-dim", type=int, default=EVALS_PER_DIM)
    args = ap.parse_args()

    print("=" * 78)
    print("PART 3 - OPTIMISER COMPARISON ON CEC'2005 FUNCTIONS")
    print("=" * 78)
    print(f"runs per cell : {args.runs}")
    print(f"budget        : {args.evals_per_dim:,} x D function evaluations "
          f"(CEC'2005 termination criterion)")
    print(f"total runs    : {len(FUNCS) * len(DIMS) * len(ALGOS) * args.runs}")

    shifts, out = {}, {"protocol": {
        "runs": args.runs, "evals_per_dim": args.evals_per_dim,
        "functions": FUNCS, "dims": DIMS, "algorithms": ALGOS,
        "budget_rule": "10000 x D function evaluations, identical for all algorithms",
        "shift_note": "shift vectors generated locally (see acflc/benchmarks.py); "
                      "results are NOT comparable with published CEC'2005 numbers",
    }, "cells": [], "algorithm_parameters": {
        k: v for k, (_, v) in op.ALGORITHMS.items()}}

    convergence = {}
    t_start = time.perf_counter()

    for fkey in FUNCS:
        for dim in DIMS:
            fn = bm.make(fkey, dim)
            shifts[f"{fkey}_D{dim}"] = [float(v) for v in fn.shift]
            lo, hi = fn.bounds()
            budget = args.evals_per_dim * dim

            print(f"\n{'=' * 78}")
            print(f"{fn.key}  {fn.name}   D={dim}   "
                  f"domain [{fn.lo:g}, {fn.hi:g}]   optimum {fn.optimum:g}")
            print(f"  properties: {', '.join(fn.properties)}")
            print(f"  budget: {budget:,} evaluations per run")
            print(f"{'-' * 78}")
            print(f"{'algorithm':<10}{'mean':>15}{'std':>14}{'best':>15}"
                  f"{'worst':>15}{'mean error':>14}{'median':>14}")

            for ai, akey in enumerate(ALGOS):
                solver, _params = op.ALGORITHMS[akey]
                values, curves = [], []
                t0 = time.perf_counter()

                for run in range(args.runs):
                    rng = np.random.default_rng(run + 1000 * ai
                                                + 7919 * dim + 104729 * FUNCS.index(fkey))
                    obj = bm.BudgetedObjective(fn, budget)
                    solver(obj, lo, hi, rng)
                    values.append(obj.best)
                    grid, best = obj.convergence()
                    curves.append(best)

                v = np.array(values)
                err = v - fn.optimum
                secs = time.perf_counter() - t0

                # Values on these functions span ten orders of magnitude once an
                # algorithm diverges, so fixed-point formatting runs the columns
                # together. Switch to scientific notation per cell when it must.
                def _f(x, w=15):
                    return (f"{x:>{w}.6f}" if abs(x) < 1e6 else f"{x:>{w}.4e}")

                print(f"{akey:<10}{_f(v.mean())}{_f(v.std(ddof=1), 14)}"
                      f"{_f(v.min())}{_f(v.max())}{err.mean():>14.4e}"
                      f"  {np.median(v):>12.4f}   [{secs:.1f}s]")

                out["cells"].append({
                    "function": fkey, "name": fn.name, "dim": dim,
                    "algorithm": akey, "runs": args.runs, "budget": budget,
                    "optimum": fn.optimum,
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "best": float(v.min()), "worst": float(v.max()),
                    "median": float(np.median(v)),
                    "mean_error": float(err.mean()),
                    "best_error": float(err.min()),
                    "values": [float(x) for x in v],
                    "seconds": secs,
                })
                convergence[(fkey, dim, akey)] = (grid, np.mean(curves, axis=0))

    out["shift_vectors"] = shifts
    (RES / "part3_results.json").write_text(json.dumps(out, indent=2))
    (RES / "part3_shift_vectors.json").write_text(json.dumps(shifts, indent=2))

    print(f"\n{'=' * 78}")
    print(f"completed in {time.perf_counter() - t_start:.0f}s")
    winners(out)
    make_figures(convergence)
    print(f"\nwrote {(RES / 'part3_results.json').relative_to(ROOT)}")
    print(f"wrote {(RES / 'part3_shift_vectors.json').relative_to(ROOT)}")


def winners(out):
    """Which algorithm won each cell, on mean final value."""
    print("\nWINNER PER CELL (lowest mean over the runs)")
    print(f"{'function':<10}{'D':>4}   {'winner':<8}{'mean':>16}"
          f"   {'runner-up':<8}{'mean':>16}")
    for fkey in FUNCS:
        for dim in DIMS:
            cells = [c for c in out["cells"]
                     if c["function"] == fkey and c["dim"] == dim]
            cells.sort(key=lambda c: c["mean"])
            a, b = cells[0], cells[1]
            print(f"{fkey:<10}{dim:>4}   {a['algorithm']:<8}{a['mean']:>16.6f}"
                  f"   {b['algorithm']:<8}{b['mean']:>16.6f}")


def make_figures(convergence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from acflc.viz import PALETTE

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    for r, fkey in enumerate(FUNCS):
        for c, dim in enumerate(DIMS):
            ax = axes[r, c]
            fn = bm.make(fkey, dim)
            for i, akey in enumerate(ALGOS):
                grid, curve = convergence[(fkey, dim, akey)]
                ax.plot(grid, curve - fn.optimum, color=PALETTE[i], lw=1.6,
                        label=akey)
            ax.set_yscale("log")
            ax.set_xlabel("function evaluations")
            ax.set_ylabel("error above optimum  $f - f^*$")
            ax.set_title(f"{fkey} {fn.name},  D={dim}", fontsize=9, loc="left")
            ax.legend(fontsize=8)
    fig.suptitle("Mean best-so-far convergence over 15 runs "
                 "(equal evaluation budget)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig14_convergence_cec2005.png")
    plt.close(fig)
    print("  wrote figures/fig14_convergence_cec2005.png")


if __name__ == "__main__":
    main()
