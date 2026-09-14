"""Figures for the Part 1 write-up.

Part 1 awards 16 marks for evidence of implementation -- "diagrams and
screenshots of the different components" -- and 7 more for analysis of output
behaviour via rule activation, controller output and control surfaces. With no
MATLAB Fuzzy Logic Toolbox on the machine there are no FIS Editor / Rule Viewer
/ Surface Viewer screenshots to paste, so the equivalent views are drawn here
instead, from the engine's own intermediate quantities.

Drawing them rather than screenshotting them is not purely a workaround. Every
figure below is generated from the same `InferenceTrace` the controller
actually produced, so the numbers in the report and the numbers in the plots
cannot drift apart, and any reader can regenerate all of it from the repository.

House style is kept deliberately plain: one idea per figure, axes labelled with
units, no colour used as the only carrier of meaning.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# A colourblind-safe qualitative sequence (Okabe-Ito), used consistently so a
# given linguistic term keeps its colour across every figure in the report.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _label(var):
    return f"{var.name}  [{var.unit}]"


def plot_variable(var, ax=None, mark=None):
    """One variable's membership functions, optionally with a reading marked.

    `mark` is a crisp value; where given, a dashed rule is dropped at that
    value and each set's degree of membership is dotted onto it. That is the
    fuzzification step made visible -- the equivalent of the Lab 3 barometer
    figure, and the first component the marking scheme asks for evidence of.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 2.3))
    u = var.universe
    for i, mf in enumerate(var.mfs):
        ax.plot(u, mf(u), color=PALETTE[i % len(PALETTE)], lw=1.6, label=mf.name)
    if mark is not None:
        mus = var.fuzzify(mark)
        ax.axvline(mark, color="0.35", ls="--", lw=1.0)
        for i, m in enumerate(mus):
            if m > 1e-9:
                ax.plot([mark], [m], "o", color=PALETTE[i % len(PALETTE)], ms=5)
                ax.annotate(f"{m:.2f}", (mark, m), textcoords="offset points",
                            xytext=(6, 2), fontsize=7.5)
    ax.set_xlim(var.lo, var.hi)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel(_label(var))
    ax.set_ylabel(r"$\mu$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=len(var.mfs),
              fontsize=7.5, handlelength=1.2, columnspacing=1.0)
    return ax


def figure_all_variables(flc, path, marks=None):
    """Every input and output variable on one sheet -- the FIS at a glance."""
    marks = marks or {}
    names = flc.input_names + flc.output_names
    variables = [flc.inputs.get(n) or flc.outputs.get(n) for n in names]
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(5.6, 1.85 * n))
    for ax, var in zip(np.atleast_1d(axes), variables):
        plot_variable(var, ax=ax, mark=marks.get(var.name))
        role = "input" if var.name in flc.inputs else "output"
        ax.text(0.0, 1.34, role, transform=ax.transAxes, fontsize=7,
                color="0.45", ha="left")
    fig.tight_layout(h_pad=1.9)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_rule_activation(flc, trace, path, top=12):
    """Which rules fired, and how strongly -- horizontal bars, strongest first.

    Rules are coloured by the output they drive, because the two subsystems
    are interleaved in rule order and the eye needs the grouping.
    """
    rows = trace.activation_table(flc.rules, top=top)
    if not rows:
        raise ValueError("no rules fired for this input; nothing to plot")

    labels, vals, colours = [], [], []
    out_order = flc.output_names
    for r, _text, s in rows:
        rule = flc.rules[r]
        ants = ", ".join(f"{v}={m}" for v, m in rule.antecedents.items())
        labels.append(f"R{r}  {ants}\n      -> {rule.consequent[1]}")
        vals.append(s)
        colours.append(PALETTE[out_order.index(rule.consequent[0])])

    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colours, height=0.66)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, min(1.0, max(vals) * 1.28))
    ax.set_xlabel("firing strength  " + r"$\alpha_r$" + "   (AND = min)")
    scen = ",  ".join(f"{k}={v:g}" for k, v in trace.crisp_inputs.items())
    ax.set_title(f"Rule activation\n{scen}", fontsize=8.5, loc="left")
    ax.legend(handles=[Line2D([], [], color=PALETTE[i], lw=6, label=n)
                       for i, n in enumerate(out_order)],
              loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_inference_stages(flc, trace, out_name, path):
    """Implication, aggregation and defuzzification for one output.

    Three stacked panels reproduce the Mamdani pipeline exactly as computed:
    the consequent sets at full height with their activation levels marked;
    the same sets clipped at those levels; and the aggregated set with the
    centroid dropped onto it. This is the figure that carries the worked
    example in the report.
    """
    var = flc.outputs[out_name]
    u = var.universe
    alphas = trace.clipped[out_name]
    agg = trace.aggregated[out_name]
    crisp = trace.outputs[out_name]

    fig, axes = plt.subplots(3, 1, figsize=(5.8, 6.2), sharex=True)

    ax = axes[0]
    for i, mf in enumerate(var.mfs):
        a = alphas.get(mf.name, 0.0)
        ax.plot(u, mf(u), color=PALETTE[i % len(PALETTE)],
                lw=1.7 if a > 0 else 0.9, alpha=1.0 if a > 0 else 0.35,
                label=f"{mf.name}" + (rf"  ($\alpha$={a:.3f})" if a > 0 else ""))
        if a > 0:
            ax.axhline(a, color=PALETTE[i % len(PALETTE)], ls=":", lw=0.9)
    ax.set_ylabel(r"$\mu$")
    ax.set_title("1. Consequent sets and their activation levels",
                 fontsize=8.5, loc="left")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    for i, mf in enumerate(var.mfs):
        a = alphas.get(mf.name, 0.0)
        if a > 0:
            ax.fill_between(u, np.minimum(a, mf(u)),
                            color=PALETTE[i % len(PALETTE)], alpha=0.35)
            ax.plot(u, np.minimum(a, mf(u)),
                    color=PALETTE[i % len(PALETTE)], lw=1.3)
    ax.set_ylabel(r"$\mu$")
    ax.set_title("2. Implication: each set clipped at its activation (min)",
                 fontsize=8.5, loc="left")

    ax = axes[2]
    ax.fill_between(u, agg, color="0.55", alpha=0.5)
    ax.plot(u, agg, color="0.2", lw=1.4)
    ax.axvline(crisp, color="#D55E00", lw=1.8)
    ax.annotate(f"centroid = {crisp:.2f}", (crisp, 0.55),
                textcoords="offset points", xytext=(8, 0),
                color="#D55E00", fontsize=8.5, fontweight="bold")
    ax.set_ylabel(r"$\mu$")
    ax.set_xlabel(_label(var))
    ax.set_title("3. Aggregation (max) and centroid defuzzification",
                 fontsize=8.5, loc="left")

    for ax in axes:
        ax.set_ylim(-0.04, 1.10)
        ax.set_xlim(var.lo, var.hi)
    fig.tight_layout(h_pad=1.1)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_control_surface(flc, x_var, y_var, out_var, fixed, path, n=61):
    """A 3D control surface plus its filled contour, side by side.

    The contour is not decoration: reading a gradient off a 3D surface is
    unreliable, and the flat regions where the controller saturates are far
    easier to see from above.
    """
    xs, ys, Z = flc.control_surface(x_var, y_var, out_var, fixed, n=n)
    XX, YY = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(9.8, 4.1))
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    # Subsample the mesh for the 3D panel. matplotlib draws plot_surface as
    # individual quads with no depth buffer, so a dense mesh renders as visible
    # striping where facets are sorted inconsistently. Around 40 facets per
    # axis is past the point where the eye can see the difference in shape, and
    # well before the point where the artefact appears. The contour panel keeps
    # the full resolution, and it is the one to read gradients off anyway.
    step = max(1, n // 30)
    ax.plot_surface(XX[::step, ::step], YY[::step, ::step], Z[::step, ::step],
                    cmap="viridis", linewidth=0, edgecolor="none",
                    antialiased=True, shade=True)
    ax.set_xlabel(_label(flc.inputs[x_var]), fontsize=7.5, labelpad=6)
    ax.set_ylabel(_label(flc.inputs[y_var]), fontsize=7.5, labelpad=6)
    ax.set_zlabel(_label(flc.outputs[out_var]), fontsize=7.5, labelpad=4)
    ax.tick_params(labelsize=6.5, pad=1)
    ax.view_init(elev=26, azim=-131)

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(XX, YY, Z, levels=18, cmap="viridis")
    # linestyles is forced solid: matplotlib dashes negative contours by
    # default, which on a signed axis like the HVAC command reads as a
    # meaningful distinction between heating and cooling when it is not.
    ax2.contour(XX, YY, Z, levels=9, colors="white", linewidths=0.5,
                alpha=0.7, linestyles="solid")
    ax2.set_xlabel(_label(flc.inputs[x_var]))
    ax2.set_ylabel(_label(flc.inputs[y_var]))
    fig.colorbar(cs, ax=ax2, label=_label(flc.outputs[out_var]))
    ax2.grid(visible=False)

    held = ",  ".join(f"{k}={v:g}" for k, v in fixed.items())
    fig.suptitle(f"Control surface: {out_var}   (held fixed: {held})",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_response_curves(flc, sweep_var, out_var, series, fixed, path):
    """Output against one input, one line per setting of a second input.

    Slices like this make the *shape* of the controller's response legible in
    a way a surface does not: the reader can see directly that the curve is
    monotone, and read off where it saturates.
    """
    var = flc.inputs[sweep_var]
    xs = np.linspace(var.lo, var.hi, 241)
    series_var, values = series

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, val in enumerate(values):
        X = np.empty((len(xs), len(flc.input_names)))
        for j, name in enumerate(flc.input_names):
            if name == sweep_var:
                X[:, j] = xs
            elif name == series_var:
                X[:, j] = val
            else:
                X[:, j] = fixed[name]
        y = flc.evaluate(X)[:, flc.output_names.index(out_var)]
        ax.plot(xs, y, color=PALETTE[i % len(PALETTE)], lw=1.7,
                label=f"{series_var} = {val:g}")

    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel(_label(var))
    ax.set_ylabel(_label(flc.outputs[out_var]))
    ax.set_xlim(var.lo, var.hi)
    ax.legend(fontsize=8)
    held = ",  ".join(f"{k}={v:g}" for k, v in fixed.items())
    ax.set_title(f"{out_var} response   (held fixed: {held})",
                 fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
