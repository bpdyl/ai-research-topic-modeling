"""Supplementary report figures (fig15-fig23), derived from existing results.

    python scripts/make_report_figures.py

This script runs **no new experiments**. Every figure it draws is either a new
view of a results file already written by `run_part1.py`, `run_part2.py` or
`run_part3.py`, or a cheap deterministic re-evaluation of the controller (which
is milliseconds, not minutes). Nothing here can disagree with the numbers in
the report, because nothing here computes a number the report quotes from
anywhere else.

Why these nine exist
--------------------
The three part drivers produce the figures each part *needs*. The report also
has to carry an appendix, and a reader auditing a claim wants the view that
makes that particular claim checkable. These are those views:

    fig15  rule base as FAM heatmaps        - the 54 rules, auditable at a glance
    fig16  the three defuzzifiers compared  - justifies centroid from measurement
    fig17  GA vs random search vs baseline  - the Part 2 control, as a distribution
    fig18  benchmark run distributions      - the 15 runs behind every Part 3 mean
    fig19  rule-activation coverage         - no dead rules, no uncovered inputs
    fig20  controller-minus-reference error - where the tuned controller still misses
    fig21  the two benchmark landscapes     - what F6 and F9 actually look like
    fig22  linguistic ordering after tuning - the 3-of-5 interpretability failure
    fig23  8-bit encoding cost              - what the chromosome quantisation costs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from acflc import build_controller  # noqa: E402
from acflc import defuzz, viz  # noqa: E402
from acflc.dataset import make_dataset, reference_policy, normalised_rmse  # noqa: E402
from acflc.flat import fam_tables  # noqa: E402
from acflc.ga import GAConfig, MembershipGA, linguistic_ordering_intact  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
PALETTE = viz.PALETTE


def _load(name):
    return json.loads((RES / name).read_text())


# --------------------------------------------------------------------- fig15

def fig15_fam_heatmaps(flc):
    """The rule base as four FAM matrices.

    The report claims the 54 rules are auditable because they come from small
    additive tables rather than being asserted one at a time. This is that
    claim as a picture: three temperature x activity slices, one per preference
    setting, plus the lighting table. The colour is the signed demand step, so
    the monotonicity the FAM guarantees is visible as a gradient that never
    reverses along either axis.
    """
    tables = fam_tables()
    hvac_order = ["CoolHigh", "CoolLow", "Off", "HeatLow", "HeatHigh"]
    dim_order = ["Off", "Low", "Medium", "High"]

    fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.9))
    keys = [k for k in tables if k.startswith("HVAC")] + ["Dimmer"]

    for ax, key in zip(axes, keys):
        rows, cols, cells = tables[key]
        order = hvac_order if key.startswith("HVAC") else dim_order
        Z = np.array([[order.index(c) for c in row] for row in cells], dtype=float)
        # centre the HVAC scale on Off so cooling and heating read as opposite
        if key.startswith("HVAC"):
            Z = Z - order.index("Off")
            im = ax.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        else:
            im = ax.imshow(Z, cmap="YlOrBr", vmin=0, vmax=3, aspect="auto")

        for i, row in enumerate(cells):
            for j, c in enumerate(row):
                # white text on the saturated cells, black on the pale ones,
                # so every label stays legible in print and in greyscale
                strong = (abs(Z[i, j]) >= 2) if key.startswith("HVAC") else (Z[i, j] >= 2)
                ax.text(j, i, c, ha="center", va="center", fontsize=6.6,
                        color=("white" if strong else "black"))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=7.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=7.5)
        title = (key.replace("HVAC | preference = ", "hvac  ·  preference = ")
                 if key.startswith("HVAC") else "dimmer  ·  daylight x activity")
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(False)
        ax.set_xlabel("activity", fontsize=7.5)

    axes[0].set_ylabel("room_temp", fontsize=7.5)
    axes[3].set_ylabel("daylight", fontsize=7.5)
    fig.suptitle("Rule base as fuzzy associative memory tables: 45 thermal + 9 lighting = 54 rules",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig15_fam_heatmaps.png")
    plt.close(fig)


# --------------------------------------------------------------------- fig16

def fig16_defuzzifier_comparison(flc):
    """Centroid, bisector and mean-of-maximum on one aggregated set.

    Part 1 marks the justification of the defuzzifier. Asserting that centroid
    is conventional is not a justification, so the three are drawn on the same
    aggregated set from the worked evening scenario, and the reachable-range
    measurement that motivates the choice is drawn beside them.
    """
    tr = flc.trace(room_temp=21.0, activity=7.5, daylight=25.0, preference=0.0)
    part1 = _load("part1_analysis.json")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))

    ax = axes[0]
    var = flc.outputs["hvac"]
    u = var.universe
    agg = tr.aggregated["hvac"]
    ax.fill_between(u, 0, agg, color=PALETTE[0], alpha=0.22,
                    label="aggregated set (max)")
    ax.plot(u, agg, color=PALETTE[0], lw=1.4)
    styles = {"centroid": ("-", PALETTE[3]), "bisector": ("--", PALETTE[2]),
              "mean_of_maximum": (":", PALETTE[1])}
    for method, fn in defuzz.METHODS.items():
        v = fn(agg, u)
        ls, c = styles[method]
        ax.axvline(v, color=c, ls=ls, lw=1.6,
                   label=f"{method.replace('_', ' ')} = {v:.2f}")
    ax.set_xlim(var.lo, var.hi)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("hvac  [% capacity (-cool/+heat)]")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("Three defuzzifiers on the same aggregated set\n"
                 "(room_temp 21, activity 7.5, daylight 25, preference 0)",
                 fontsize=8.5, loc="left")
    ax.legend(fontsize=7)

    # reachable range per method, measured over the input grid in run_part1
    ax = axes[1]
    rows = [d for d in part1["defuzzification"] if d["output"] == "hvac"]
    names = [r["method"].replace("_", " ") for r in rows]
    ypos = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.barh(i, r["max"] - r["min"], left=r["min"], height=0.5,
                color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax.text(0, i + 0.32, f"{r['pct_of_universe']:.1f}% of universe",
                ha="center", fontsize=7)
    ax.axvline(-100, color="0.3", ls="--", lw=1.0)
    ax.axvline(100, color="0.3", ls="--", lw=1.0)
    ax.text(100, len(rows) - 0.35, " actuator limit", fontsize=7, va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(-112, 132)
    ax.set_xlabel("reachable hvac command  [% capacity]")
    ax.set_title("Actuator authority actually reachable\n"
                 "(measured over a 70,875-point input grid)",
                 fontsize=8.5, loc="left")
    ax.grid(axis="y", alpha=0)

    fig.tight_layout()
    fig.savefig(FIG / "fig16_defuzzifier_comparison.png")
    plt.close(fig)


# --------------------------------------------------------------------- fig17

def fig17_part2_control():
    """The Part 2 control, as a distribution rather than a pair of means.

    Left: held-out test nRMSE for the hand design, the GA over five seeds and
    budget-matched random search over five seeds. Right: the same five GA runs
    as convergence curves against the level random search reached, which is the
    figure that shows *why* the control lost -- its best draw in 9,060 was the
    expert chromosome it was seeded with, so its curve is flat by construction.
    """
    d = _load("part2_ga.json")
    base = d["baseline_test_nrmse"]
    ga = [r["ga"]["test_nrmse"] for r in d["runs"]]
    rs = [r["random_search"]["test_nrmse"] for r in d["runs"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))

    ax = axes[0]
    groups = [("hand-designed\n(untuned)", [base], PALETTE[4]),
              ("GA\n(5 seeds)", ga, PALETTE[3]),
              ("random search\n(5 seeds, matched)", rs, PALETTE[0])]
    for i, (label, vals, colour) in enumerate(groups):
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.06, 0.06, len(vals)),
                   vals, s=34, color=colour, zorder=3, edgecolors="white",
                   linewidths=0.6)
        m = float(np.mean(vals))
        ax.hlines(m, i - 0.24, i + 0.24, color=colour, lw=2.2, zorder=2)
        ax.text(i + 0.3, m, f"{m:.5f}", fontsize=7.5, va="center", color=colour)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("held-out test nRMSE  (lower is better)")
    ax.set_title("Part 2: the GA against its budget-matched control",
                 fontsize=9, loc="left")
    ax.set_xlim(-0.45, 2.75)

    ax = axes[1]
    for i, r in enumerate(d["runs"]):
        curve = [1.0 / f - 1.0 for f in r["ga"]["history_best"]]
        ax.plot(curve, color=PALETTE[i % len(PALETTE)], lw=1.2, alpha=0.9,
                label=f"GA seed {i}")
    rs_train = d["runs"][0]["random_search"]["train_nrmse"]
    ax.axhline(rs_train, color="0.25", ls="-", lw=1.6,
               label="random search, all 5 seeds")
    ax.axhline(d["baseline"]["train_nrmse"], color="0.55", ls="--", lw=1.2,
               label="hand-designed baseline")
    ax.set_xlabel("generation")
    ax.set_ylabel("best training nRMSE")
    ax.set_title("Random search never left its seed", fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=2, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG / "fig17_part2_control.png")
    plt.close(fig)


# --------------------------------------------------------------------- fig18

def fig18_benchmark_distributions():
    """All 180 Part 3 runs, as per-cell distributions.

    The results table reports mean, std, best and worst per cell. Those four
    numbers hide the shape, and on F6 at D=10 the shape is the entire finding:
    PSO's mean is six orders of magnitude off because a minority of runs
    diverge, not because it is typically bad -- its *median* is competitive.
    A log axis and the individual runs make that visible.
    """
    d = _load("part3_results.json")
    cells = d["cells"]
    algos = ["GA", "PSO", "SA"]
    colours = {"GA": PALETTE[3], "PSO": PALETTE[0], "SA": PALETTE[1]}
    combos = [("F6", 2), ("F6", 10), ("F9", 2), ("F9", 10)]

    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.2))
    for ax, (fn, dim) in zip(axes, combos):
        sub = {c["algorithm"]: c for c in cells
               if c["function"] == fn and c["dim"] == dim}
        opt = sub["GA"]["optimum"]
        for i, a in enumerate(algos):
            vals = np.array(sub[a]["values"], dtype=float)
            err = np.maximum(vals - opt, 1e-12)      # error above the optimum
            ax.scatter(np.full(len(err), i) + np.random.default_rng(0)
                       .uniform(-0.13, 0.13, len(err)),
                       err, s=16, color=colours[a], alpha=0.75,
                       edgecolors="none", zorder=3)
            ax.hlines(np.median(err), i - 0.26, i + 0.26, color=colours[a],
                      lw=2.2, zorder=4)
        ax.set_yscale("log")
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, fontsize=8.5)
        ax.set_title(f"{fn}  {sub['GA']['name'].replace('Shifted ', '')}   D={dim}",
                     fontsize=8.5, loc="left")
        if ax is axes[0]:
            ax.set_ylabel("error above known optimum  (log)")
    fig.suptitle("Part 3: every one of the 180 runs, with the per-cell median marked",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig18_benchmark_distributions.png")
    plt.close(fig)


# --------------------------------------------------------------------- fig19

def fig19_rule_coverage(flc):
    """How many rules fire, and how often each one fires, across the space.

    Two properties the report asserts are checked here. No input state falls
    through the rule base (the left histogram has no zero bin), and no rule is
    dead (every bar on the right is non-zero). Both follow from the overlap
    chosen in Part 1, and both are the kind of claim that is cheap to state and
    easy to get wrong.
    """
    rng = np.random.default_rng(7)
    lo = np.array([flc.inputs[n].lo for n in flc.input_names])
    hi = np.array([flc.inputs[n].hi for n in flc.input_names])
    X = rng.uniform(lo, hi, size=(20000, len(lo)))
    S = flc._firing(X)
    n_active = (S > 0).sum(axis=1)
    per_rule = (S > 0).mean(axis=0) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2),
                             gridspec_kw={"width_ratios": [1, 2.1]})

    ax = axes[0]
    bins = np.arange(n_active.min() - 0.5, n_active.max() + 1.5)
    ax.hist(n_active, bins=bins, color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("rules firing simultaneously")
    ax.set_ylabel("input states (of 20,000 random)")
    ax.set_title(f"Simultaneous activations\nmean {n_active.mean():.2f}, "
                 f"min {n_active.min()}, max {n_active.max()}",
                 fontsize=8.5, loc="left")

    ax = axes[1]
    idx = np.arange(len(per_rule))
    is_light = np.array([("daylight" in flc.rules[i].antecedents)
                         for i in idx])
    ax.bar(idx[~is_light], per_rule[~is_light], color=PALETTE[3], width=0.85,
           label="thermal rules (45)")
    ax.bar(idx[is_light], per_rule[is_light], color=PALETTE[2], width=0.85,
           label="lighting rules (9)")
    ax.set_xlabel("rule index")
    ax.set_ylabel("% of input states\nthe rule fires on")
    ax.set_title(f"Per-rule activation frequency — dead rules: "
                 f"{int((per_rule == 0).sum())}", fontsize=8.5, loc="left")
    ax.legend(fontsize=7.5)
    ax.set_xlim(-1, len(per_rule))

    fig.tight_layout()
    fig.savefig(FIG / "fig19_rule_coverage.png")
    plt.close(fig)


# --------------------------------------------------------------------- fig20

def fig20_error_maps(flc):
    """Where the controller disagrees with the reference policy, and where it
    stops answering at all.

    Part 2 reports one scalar per model. A scalar cannot say *where* the error
    lives, and the answer turns out to be structural rather than uniform: the
    hand design is systematically wrong in the corners of the temperature x
    preference plane, because the FAM saturates the demand there while the
    reference policy keeps ramping. Tuning shrinks that band rather than
    removing it, which is expected given the target is a different functional
    form (see acflc/dataset.py).

    The third panel is the finding this figure exists for. Tuning the
    membership functions freely moves them apart, and where two adjacent sets
    stop overlapping the rule base has a hole: no rule fires, the aggregated
    set is empty, the centroid is undefined and the actuator receives no
    command. The hand design has no such state. Four of the five tuned
    controllers do. `normalised_rmse` charges an undefined output the full
    output span, so the fitness function *discourages* holes without
    forbidding them -- which is exactly how a safety property gets traded away
    for average-case accuracy without anyone deciding to.
    """
    d = _load("part2_ga.json")
    best = min(d["runs"], key=lambda r: r["ga"]["test_nrmse"])
    tuned = flc.with_params(np.array(best["ga"]["best_params"]))

    n = 61
    temps = np.linspace(flc.inputs["room_temp"].lo, flc.inputs["room_temp"].hi, n)
    prefs = np.linspace(flc.inputs["preference"].lo, flc.inputs["preference"].hi, n)
    TT, PP = np.meshgrid(temps, prefs)
    X = np.column_stack([TT.ravel(), np.full(TT.size, 3.0),
                         np.full(TT.size, 50.0), PP.ravel()])
    target = reference_policy(X)[:, 0].reshape(TT.shape)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.3),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    vmax = 60.0
    for ax, (src, label) in zip(axes[:2], ((flc, "hand-designed"),
                                          (tuned, f"GA-tuned (seed {best['seed']})"))):
        pred = src.evaluate(X)[:, 0].reshape(TT.shape)
        err = pred - target
        im = ax.pcolormesh(TT, PP, err, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        # undefined outputs are not "zero error" -- paint them as absent data
        holes = ~np.isfinite(err)
        if holes.any():
            ax.pcolormesh(TT, PP, np.where(holes, 1.0, np.nan),
                          cmap=matplotlib.colors.ListedColormap(["#3a3a3a"]),
                          vmin=0, vmax=1, shading="auto", zorder=4)
            ax.scatter([], [], marker="s", s=28, color="#3a3a3a",
                       label="no rule fires")
            ax.legend(fontsize=7, loc="lower left", framealpha=0.85,
                      frameon=True)
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        ax.set_xlabel("room_temp  [degC]")
        ax.set_title(f"{label}\nRMSE {rmse:.1f} % capacity"
                     + (f", {holes.mean() * 100:.1f}% undefined" if holes.any() else ""),
                     fontsize=8.5, loc="left")
        ax.grid(False)
    axes[0].set_ylabel("preference  [scale]")
    cb = fig.colorbar(im, ax=axes[1], label="controller $-$ reference\n[% capacity]",
                      fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)

    # coverage audit over the whole 4-D input space, per seed
    ax = axes[2]
    rng = np.random.default_rng(11)
    lo = np.array([flc.inputs[k].lo for k in flc.input_names])
    hi = np.array([flc.inputs[k].hi for k in flc.input_names])
    Xs = rng.uniform(lo, hi, size=(50000, len(lo)))
    labels, rates, cols = ["hand\ndesign"], [0.0], [PALETTE[4]]
    base_rate = float(np.isnan(flc.evaluate(Xs)).any(axis=1).mean() * 100)
    rates[0] = base_rate
    for r in d["runs"]:
        t = flc.with_params(np.array(r["ga"]["best_params"]))
        rate = float(np.isnan(t.evaluate(Xs)).any(axis=1).mean() * 100)
        labels.append(f"seed\n{r['seed']}")
        rates.append(rate)
        cols.append(PALETTE[3] if rate > 0 else PALETTE[2])
    ax.bar(range(len(rates)), rates, color=cols, width=0.65)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, ("0" if v == 0 else f"{v:.2f}"), ha="center",
                fontsize=7.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("% of input states with\nNO defined output")
    ax.set_ylim(0, max(rates) * 1.35 + 0.05)
    ax.set_title("Tuning punches holes in the rule base\n"
                 "(50,000 uniform states over the full 4-D input space)",
                 fontsize=8.5, loc="left")

    fig.tight_layout()
    fig.savefig(FIG / "fig20_error_maps.png")
    plt.close(fig)

    return {"n_states": 50000,
            "hand_designed_undefined_pct": base_rate,
            "per_seed_undefined_pct": {str(r["seed"]): rates[i + 1]
                                       for i, r in enumerate(d["runs"])}}


# --------------------------------------------------------------------- fig21

def fig21_benchmark_landscapes():
    """What F6 and F9 look like at D=2, with the shifted optimum marked.

    Part 3 argues the two functions were chosen to be complementary -- one a
    narrow curved non-separable valley, the other a separable lattice of
    roughly 10^D basins. That argument is much easier to accept from the
    contours than from the formulae, and the marked optimum shows the shift
    doing its job: neither optimum sits at the centre of the domain, so an
    algorithm that searches symmetrically about the origin gets nothing free.
    """
    from acflc.benchmarks import make as make_benchmark

    fig = plt.figure(figsize=(9.6, 4.2))
    for k, key in enumerate(("F6", "F9")):
        b = make_benchmark(key, 2)
        n = 260
        xs = np.linspace(b.lo, b.hi, n)
        ys = np.linspace(b.lo, b.hi, n)
        XX, YY = np.meshgrid(xs, ys)
        Z = b(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

        ax = fig.add_subplot(1, 2, k + 1)
        # log-scale the height above the optimum so structure is visible on both
        H = np.log10(np.maximum(Z - b.optimum, 1e-8))
        im = ax.pcolormesh(XX, YY, H, cmap="viridis", shading="auto")
        ax.contour(XX, YY, H, levels=12, colors="white", linewidths=0.35,
                   alpha=0.55)
        ax.plot(b.shift[0], b.shift[1], marker="*", ms=15, color="#D55E00",
                mec="white", mew=0.8, zorder=5)
        ax.annotate(f"optimum {b.optimum:g}\nat ({b.shift[0]:.2f}, {b.shift[1]:.2f})",
                    xy=(b.shift[0], b.shift[1]), xytext=(8, 8),
                    textcoords="offset points", fontsize=7.5, color="white")
        ax.set_title(f"{key}  {b.name}   D=2", fontsize=9, loc="left")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label=r"$\log_{10}(f - f^*)$", fraction=0.046,
                     pad=0.03)

    fig.suptitle("The two CEC'2005 instances actually optimised, with locally "
                 "generated shift vectors", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig21_benchmark_landscapes.png")
    plt.close(fig)


# --------------------------------------------------------------------- fig22

def fig22_linguistic_ordering(flc):
    """Does the tuned controller still mean what its rules say?

    The Part 1 justification for Mamdani is that the rule base stays legible to
    a carer. That survives Part 2 only if the fuzzy sets are still in their
    original left-to-right order afterwards, because a rule reading "IF
    room_temp is Cold" is false advertising once Cold has migrated to the right
    of Cool. Set centres of mass are plotted before and after for every seed;
    a crossing line is a broken ordering, and three of the five runs have one.
    """
    d = _load("part2_ga.json")
    runs = d["runs"]
    variables = flc.input_names + flc.output_names

    fig, axes = plt.subplots(1, len(runs), figsize=(12.0, 3.4), sharey=True)
    for ax, r in zip(axes, runs):
        params = np.array(r["ga"]["best_params"])
        rep = linguistic_ordering_intact(flc, params)
        broken = r["ga"]["ordering_broken_on"]
        for vi, name in enumerate(variables):
            info = rep[name]
            var = flc.inputs.get(name) or flc.outputs.get(name)
            span = var.hi - var.lo
            b = [(c - var.lo) / span for c in info["centres_before"]]
            a = [(c - var.lo) / span for c in info["centres_after"]]
            ok = info["order_preserved"]
            for j in range(len(b)):
                ax.plot([vi - 0.22, vi + 0.22], [b[j], a[j]],
                        color=(PALETTE[2] if ok else PALETTE[3]),
                        lw=1.3, alpha=0.9, zorder=2)
            ax.scatter(np.full(len(b), vi - 0.22), b, s=13, color="0.35",
                       zorder=3, edgecolors="none")
            ax.scatter(np.full(len(a), vi + 0.22), a, s=13,
                       color=(PALETTE[2] if ok else PALETTE[3]), zorder=3,
                       edgecolors="none")
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(variables, rotation=90, fontsize=7)
        ax.set_title(f"seed {r['seed']}  " +
                     ("ordering intact" if not broken
                      else "BROKEN: " + ", ".join(broken)),
                     fontsize=8, loc="left",
                     color=("black" if not broken else PALETTE[3]))
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("set centre of mass\n(normalised within variable)")
    handles = [Patch(color=PALETTE[2], label="order preserved"),
               Patch(color=PALETTE[3], label="order broken")]
    fig.legend(handles=handles, fontsize=7.5, ncol=2, loc="upper right",
               bbox_to_anchor=(0.995, 0.97), frameon=False)
    fig.suptitle("Left tick = hand-designed, right tick = GA-tuned. "
                 "A crossing means the rule base no longer means what it says.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig22_linguistic_ordering.png")
    plt.close(fig)


# --------------------------------------------------------------------- fig23

def fig23_encoding_cost(flc):
    """What the 8-bit-per-gene binary encoding costs before the search starts.

    The chromosome quantises every membership-function breakpoint to one of 256
    levels on its variable's physical range. That is a design choice with a
    price, and the price is measurable: encode the hand-designed controller and
    decode it back, and the error it can no longer represent is the floor the
    GA is working above. Left: the round-trip parameter displacement. Right:
    the nRMSE the encoding alone costs, against the improvement the GA then won.
    """
    data = make_dataset(flc, n=400)
    spans = np.array([flc.outputs[n].hi - flc.outputs[n].lo
                      for n in flc.output_names])
    ga = MembershipGA(flc, data, GAConfig())

    p0 = flc.get_params()
    p1 = ga.decode(ga.encode(p0))
    direct, _ = normalised_rmse(data.y_train, flc.evaluate(data.X_train), spans)
    roundtrip, _ = ga.evaluate_params(p1, data.X_train, data.y_train)

    d = _load("part2_ga.json")
    ga_train = float(np.mean([r["ga"]["train_nrmse"] for r in d["runs"]]))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))

    ax = axes[0]
    layout = flc.param_layout()
    bounds, pos, names = [], 0, []
    for name, role, k in layout:
        bounds.append((pos, pos + k, name))
        pos += k
    disp = np.abs(p1 - p0)
    for i, (a, b, name) in enumerate(bounds):
        ax.bar(np.arange(a, b), disp[a:b], color=PALETTE[i % len(PALETTE)],
               width=0.9, label=name)
        ax.axvline(b - 0.5, color="0.8", lw=0.6)
    ax.set_xlabel("gene index (81 membership-function parameters)")
    ax.set_ylabel("|round-trip displacement|\n[variable's own units]")
    ax.set_title(f"8-bit encode/decode error — max {disp.max():.3f}",
                 fontsize=8.5, loc="left")
    ax.legend(fontsize=6.5, ncol=3)
    ax.set_xlim(-1, len(disp))

    ax = axes[1]
    labels = ["hand design\nas built", "hand design\nafter 8-bit\nround trip",
              "GA best\n(mean of 5)"]
    vals = [direct, roundtrip, ga_train]
    cols = [PALETTE[4], PALETTE[1], PALETTE[3]]
    ax.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0015, f"{v:.5f}", ha="center", fontsize=7.5)
    ax.annotate("", xy=(1, roundtrip + 0.006), xytext=(0, roundtrip + 0.006),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.0))
    ax.text(0.5, roundtrip + 0.0085,
            f"encoding cost\n+{roundtrip - direct:.5f}", ha="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("training nRMSE")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("The encoding costs error before the GA gains any",
                 fontsize=8.5, loc="left")

    fig.tight_layout()
    fig.savefig(FIG / "fig23_encoding_cost.png")
    plt.close(fig)

    return {"direct": direct, "roundtrip": roundtrip,
            "encoding_cost": roundtrip - direct,
            "max_gene_displacement": float(disp.max()),
            "ga_train_mean": ga_train}


# ------------------------------------------------------------------- part 3 stats

def part3_pairwise_tests():
    """Rank tests on the Part 3 runs, because four means can mislead.

    The results table ranks algorithms by mean, and on F6 at D=10 that ranking
    is carried entirely by three PSO runs that diverged. A Mann-Whitney U test
    on the 15 raw values per cell asks the different question -- is the
    *typical* run better -- and on that cell the answer is that GA and PSO are
    indistinguishable (p = 0.62) and PSO's median is in fact the lower of the
    two. Reporting the mean ranking without this would overstate the GA.

    Mann-Whitney rather than a t-test: the run distributions are heavily
    skewed and, on the cells where an algorithm solves the problem, nearly
    degenerate, so normality is not available.
    """
    from itertools import combinations
    from scipy.stats import mannwhitneyu

    d = _load("part3_results.json")
    cells = {(c["function"], c["dim"], c["algorithm"]): np.asarray(c["values"])
             for c in d["cells"]}
    out = []
    for fn in ("F6", "F9"):
        for dim in (2, 10):
            for a, b in combinations(("GA", "PSO", "SA"), 2):
                x, y = cells[(fn, dim, a)], cells[(fn, dim, b)]
                u, p = mannwhitneyu(x, y, alternative="two-sided")
                out.append({
                    "function": fn, "dim": dim, "a": a, "b": b,
                    "U": float(u), "p": float(p),
                    "significant_at_0.05": bool(p < 0.05),
                    "median_a": float(np.median(x)),
                    "median_b": float(np.median(y)),
                    "lower_median": a if np.median(x) < np.median(y) else b,
                })
    return {"test": "Mann-Whitney U, two-sided, n=15 per group",
            "note": "within-study only; shift vectors are locally generated",
            "comparisons": out}


# ------------------------------------------------------------------------ main

def main():
    flc = build_controller()
    print("supplementary report figures")

    fig15_fam_heatmaps(flc);            print("  fig15_fam_heatmaps.png")
    fig16_defuzzifier_comparison(flc);  print("  fig16_defuzzifier_comparison.png")
    fig17_part2_control();              print("  fig17_part2_control.png")
    fig18_benchmark_distributions();    print("  fig18_benchmark_distributions.png")
    fig19_rule_coverage(flc);           print("  fig19_rule_coverage.png")
    cover = fig20_error_maps(flc);      print("  fig20_error_maps.png")
    fig21_benchmark_landscapes();       print("  fig21_benchmark_landscapes.png")
    fig22_linguistic_ordering(flc);     print("  fig22_linguistic_ordering.png")
    stats = fig23_encoding_cost(flc);   print("  fig23_encoding_cost.png")

    # The two audits the report quotes that no part driver produces.
    d = _load("part2_ga.json")
    audit = {
        "encoding": stats,
        "coverage": cover,
        "per_seed": [
            {"seed": r["seed"],
             "undefined_pct": cover["per_seed_undefined_pct"][str(r["seed"])],
             "ordering_broken_on": r["ga"]["ordering_broken_on"],
             "fully_covering": cover["per_seed_undefined_pct"][str(r["seed"])] == 0.0,
             "ordering_intact": not r["ga"]["ordering_broken_on"]}
            for r in d["runs"]],
    }
    audit["seeds_passing_both"] = sum(
        1 for s in audit["per_seed"] if s["fully_covering"] and s["ordering_intact"])
    (RES / "tuning_audit.json").write_text(json.dumps(audit, indent=2))

    stats3 = part3_pairwise_tests()
    (RES / "part3_stats.json").write_text(json.dumps(stats3, indent=2))
    print("\nPart 3 pairwise rank tests (Mann-Whitney U, n=15):")
    for c in stats3["comparisons"]:
        flag = "*" if c["significant_at_0.05"] else "ns"
        print(f"  {c['function']} D={c['dim']:<2} {c['a']:>3} vs {c['b']:<3} "
              f"p={c['p']:.2e} {flag:2s}  lower median: {c['lower_median']}")

    print(f"\nencoding cost: nRMSE {stats['direct']:.6f} -> "
          f"{stats['roundtrip']:.6f} (+{stats['encoding_cost']:.6f}), "
          f"max gene displacement {stats['max_gene_displacement']:.4f}")
    print(f"coverage: hand design {cover['hand_designed_undefined_pct']:.2f}% "
          f"undefined; tuned seeds "
          + ", ".join(f"{k}={v:.2f}%" for k, v
                      in cover["per_seed_undefined_pct"].items()))
    print(f"seeds that are BOTH fully covering and linguistically ordered: "
          f"{audit['seeds_passing_both']} of {len(audit['per_seed'])}")
    print("wrote results/tuning_audit.json")


if __name__ == "__main__":
    main()
