"""Part 2 driver: GA tuning of the membership functions, with a control.

    python scripts/run_part2.py [--seeds 5] [--generations 150]

Reports the hand-designed baseline, the GA, and a budget-matched random
search, each over several seeds, on a held-out test split.

Why the random-search control is here
-------------------------------------
A GA that improves on its starting point has not thereby shown that *evolution*
did the work: a search that merely samples the space finely can do the same.
The only way to separate the two is to spend an identical number of fitness
evaluations sampling at random and compare. If the GA cannot beat that, the
honest conclusion is that the encoding and the budget are doing the work and
the genetic operators are not, which is worth knowing and worth reporting.

Both searches are seeded with the expert controller and both are charged for
that evaluation, so neither gets a free head start.
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

from acflc import build_controller  # noqa: E402
from acflc.dataset import make_dataset, normalised_rmse  # noqa: E402
from acflc.ga import GAConfig, MembershipGA, linguistic_ordering_intact  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def random_search(ga, budget, seed):
    """Uniform random chromosomes, same budget, same seeding, same fitness.

    Deliberately the crudest possible baseline: no memory, no structure, no
    exploitation. It is the floor the GA has to clear to justify itself.
    """
    rng = np.random.default_rng(seed)
    best_fit, best_chrom = -np.inf, None
    history, spent = [], 0
    batch = 50

    seeded = ga.encode(ga.flc.get_params())[None, :]
    while spent < budget:
        k = min(batch, budget - spent)
        pop = rng.integers(0, 2, size=(k, ga.n_bits), dtype=np.uint8)
        if spent == 0:
            pop[0] = seeded[0]
        fits = ga.fitness(pop)
        spent += k
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best_fit, best_chrom = float(fits[i]), pop[i].copy()
        history.append(best_fit)

    params = ga.flc.with_params(ga.decode(best_chrom)).get_params()
    return params, best_fit, history


def summarise(name, values):
    v = np.asarray(values, dtype=float)
    return {"method": name, "n_runs": len(v), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "best": float(v.min()), "worst": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--population", type=int, default=60)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    flc = build_controller()
    data = make_dataset(flc, n=args.n)
    spans = np.array([flc.outputs[n].hi - flc.outputs[n].lo
                      for n in flc.output_names])

    print("=" * 74)
    print("PART 2 - GA TUNING OF MEMBERSHIP FUNCTIONS")
    print("=" * 74)
    print(f"dataset      : {data.n_train} train / {data.n_test} test "
          f"(SYNTHETIC - see acflc/dataset.py)")
    print(f"chromosome   : {flc.n_params} genes x 8 bits = {flc.n_params * 8} bits")
    print(f"budget       : {args.population} x ({args.generations} + 1) = "
          f"{args.population * (args.generations + 1):,} evaluations per run")
    print(f"seeds        : {args.seeds}")

    base_tr, base_tr_per = normalised_rmse(
        data.y_train, flc.evaluate(data.X_train), spans)
    base_te, base_te_per = normalised_rmse(
        data.y_test, flc.evaluate(data.X_test), spans)
    print(f"\nbaseline (hand-designed, untuned)")
    print(f"  train nRMSE {base_tr:.5f}   test nRMSE {base_te:.5f}")
    print(f"  per-output test: " + ", ".join(
        f"{n}={v:.5f}" for n, v in zip(flc.output_names, base_te_per)))

    results = {
        "dataset": data.describe(),
        "baseline": {"train_nrmse": base_tr, "test_nrmse": base_te,
                     "per_output_test": [float(v) for v in base_te_per]},
        "runs": [],
    }

    ga_test, rs_test, ga_hist = [], [], []
    budget = args.population * (args.generations + 1)

    for seed in range(args.seeds):
        print(f"\n--- seed {seed} " + "-" * 58)
        cfg = GAConfig(population=args.population, generations=args.generations,
                       seed=seed)
        ga = MembershipGA(flc, data, cfg)

        t0 = time.perf_counter()
        res = ga.run(seed_with_expert=True, verbose=True)
        ga_secs = time.perf_counter() - t0
        ga_te, ga_te_per = ga.evaluate_params(res.best_params,
                                              data.X_test, data.y_test)
        ga_test.append(ga_te)
        ga_hist.append(res.history_best)

        t0 = time.perf_counter()
        rs_params, rs_fit, rs_history = random_search(ga, budget, seed=1000 + seed)
        rs_secs = time.perf_counter() - t0
        rs_te, _ = ga.evaluate_params(rs_params, data.X_test, data.y_test)
        rs_test.append(rs_te)

        print(f"  GA            train nRMSE {res.best_nrmse:.5f}   "
              f"test {ga_te:.5f}   ({ga_secs:.0f}s, {res.evaluations:,} evals)")
        print(f"  random search train nRMSE {1 / rs_fit - 1:.5f}   "
              f"test {rs_te:.5f}   ({rs_secs:.0f}s, {budget:,} evals)")

        ordering = linguistic_ordering_intact(flc, res.best_params)
        broken = [k for k, v in ordering.items() if not v["order_preserved"]]
        print(f"  linguistic ordering preserved: "
              f"{'ALL variables' if not broken else 'BROKEN on ' + ', '.join(broken)}")

        results["runs"].append({
            "seed": seed,
            "ga": {"train_nrmse": res.best_nrmse, "test_nrmse": ga_te,
                   "per_output_test": [float(v) for v in ga_te_per],
                   "evaluations": res.evaluations, "seconds": ga_secs,
                   "history_best": res.history_best,
                   "best_params": [float(v) for v in res.best_params],
                   "ordering_broken_on": broken},
            "random_search": {"train_nrmse": 1 / rs_fit - 1, "test_nrmse": rs_te,
                              "evaluations": budget, "seconds": rs_secs},
        })

    print("\n" + "=" * 74)
    print("SUMMARY - held-out test nRMSE over "
          f"{args.seeds} seeds (lower is better)")
    print("=" * 74)
    print(f"{'method':<24}{'mean':>10}{'std':>10}{'best':>10}{'worst':>10}"
          f"{'vs baseline':>14}")
    rows = [summarise("GA", ga_test), summarise("random search", rs_test)]
    print(f"{'hand-designed baseline':<24}{base_te:>10.5f}{'-':>10}"
          f"{'-':>10}{'-':>10}{'-':>14}")
    for r in rows:
        gain = 100.0 * (base_te - r["mean"]) / base_te
        print(f"{r['method']:<24}{r['mean']:>10.5f}{r['std']:>10.5f}"
              f"{r['best']:>10.5f}{r['worst']:>10.5f}{gain:>13.1f}%")
    results["summary"] = rows
    results["baseline_test_nrmse"] = base_te

    # the comparison the control exists to make
    ga_m, rs_m = np.mean(ga_test), np.mean(rs_test)
    verdict = ("GA beats budget-matched random search"
               if ga_m < rs_m else
               "random search MATCHES OR BEATS the GA at equal budget")
    print(f"\nVERDICT: {verdict} "
          f"(GA {ga_m:.5f} vs RS {rs_m:.5f}, {args.seeds} seeds)")
    results["verdict"] = verdict

    (RES / "part2_ga.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {(RES / 'part2_ga.json').relative_to(ROOT)}")

    make_figures(flc, data, results, ga_hist, spans)


def make_figures(flc, data, results, ga_hist, spans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from acflc import viz

    best_run = min(results["runs"], key=lambda r: r["ga"]["test_nrmse"])
    tuned = flc.with_params(np.array(best_run["ga"]["best_params"]))

    # convergence
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for i, h in enumerate(ga_hist):
        ax.plot([1 / f - 1 for f in h], color=viz.PALETTE[i % len(viz.PALETTE)],
                lw=1.2, alpha=0.85, label=f"seed {i}")
    ax.axhline(results["baseline"]["train_nrmse"], color="0.35", ls="--", lw=1.2,
               label="hand-designed baseline")
    ax.set_xlabel("generation")
    ax.set_ylabel("best training nRMSE")
    ax.set_title("GA convergence", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_ga_convergence.png")
    plt.close(fig)

    # membership functions before / after, inputs only
    names = flc.input_names + flc.output_names
    fig, axes = plt.subplots(len(names), 2, figsize=(9.0, 1.75 * len(names)))
    for row, name in enumerate(names):
        for col, (src, title) in enumerate(((flc, "hand-designed"),
                                            (tuned, "GA-tuned"))):
            var = src.inputs.get(name) or src.outputs.get(name)
            ax = axes[row, col]
            u = var.universe
            for i, mf in enumerate(var.mfs):
                ax.plot(u, mf(u), color=viz.PALETTE[i % len(viz.PALETTE)],
                        lw=1.4, label=mf.name)
            ax.set_xlim(var.lo, var.hi)
            ax.set_ylim(-0.05, 1.1)
            ax.set_ylabel(r"$\mu$", fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.text(-0.22, 0.5, name, transform=ax.transAxes, rotation=90,
                        va="center", fontsize=8.5)
            if row == len(names) - 1:
                ax.set_xlabel(var.unit, fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 1:
                ax.legend(fontsize=6.5, ncol=3, loc="upper center",
                          bbox_to_anchor=(0.5, 1.65))
    fig.tight_layout()
    fig.savefig(FIG / "fig12_mf_before_after.png")
    plt.close(fig)

    # predicted vs target on the held-out split
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for j, out_name in enumerate(flc.output_names):
        ax = axes[j]
        for src, lab, colour in ((flc, "hand-designed", viz.PALETTE[0]),
                                 (tuned, "GA-tuned", viz.PALETTE[3])):
            pred = src.evaluate(data.X_test)[:, j]
            ax.scatter(data.y_test[:, j], pred, s=11, alpha=0.55,
                       color=colour, label=lab, edgecolors="none")
        var = flc.outputs[out_name]
        ax.plot([var.lo, var.hi], [var.lo, var.hi], color="0.4", ls="--", lw=1.0)
        ax.set_xlim(var.lo, var.hi)
        ax.set_ylim(var.lo, var.hi)
        ax.set_xlabel(f"target {out_name} [{var.unit}]")
        ax.set_ylabel(f"controller {out_name}")
        ax.legend(fontsize=7.5)
        ax.set_title(out_name, fontsize=9, loc="left")
    fig.suptitle("Held-out test split: controller output against reference policy",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig13_pred_vs_target.png")
    plt.close(fig)

    for p in ("fig11_ga_convergence.png", "fig12_mf_before_after.png",
              "fig13_pred_vs_target.png"):
        print(f"  wrote figures/{p}")


if __name__ == "__main__":
    main()
