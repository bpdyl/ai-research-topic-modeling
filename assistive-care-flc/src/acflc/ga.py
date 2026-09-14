"""Part 2: a genetic algorithm that tunes the controller's membership functions.

The four things Part 2 marks explicitly
---------------------------------------

**1. Problem encoding.** Binary. Every tunable membership-function parameter
is one gene of 8 bits, decoded linearly onto the physical range of the
variable it belongs to:

    value = lo + (bits / 255) * (hi - lo)

so a room-temperature breakpoint decodes somewhere in [14, 34] degrees and an
HVAC breakpoint somewhere in [-100, +100] percent. Binary encoding with a
fixed word length is used rather than a real-valued vector because it is the
representation the taught material and the comparison literature use, and
because it makes the chromosome length a stated, checkable quantity rather
than an implementation detail.

The resolution that buys is (hi - lo)/255: about 0.078 degrees on room
temperature and 0.78 percentage points on the HVAC command -- finer than
either sensor or actuator resolves, so the quantisation is not what limits the
result.

**2. Chromosome length.** 81 parameters x 8 bits = **648 bits**, laid out in
the order given by `FuzzyController.param_layout()`:

    room_temp 17, activity 11, daylight 11, preference 11   (inputs,  50)
    hvac      17, dimmer   14                               (outputs, 31)

**3. Genetic operators.** Roulette-wheel (fitness-proportionate) selection
with elitism; one-point crossover at p_c = 0.7; per-bit mutation at
p_m = 0.01. Those rates are the conventional ranges from the module material
(crossover 0.4-0.8, mutation 0.001-0.01) and match the values the comparison
submission used, which keeps the results comparable.

**4. Fitness function.** Span-normalised RMSE against the reference dataset,
turned into something to maximise:

    fitness = 1 / (1 + nRMSE)

Selection needs a positive quantity that is larger when the individual is
better, and roulette-wheel selection in particular needs it non-negative.
1/(1+e) satisfies both, is bounded in (0, 1], and is monotone in the error, so
it changes which individuals are picked without changing which is best.

Repair rather than rejection
----------------------------
A decoded chromosome routinely proposes a triangle whose parameters are out of
order. Rather than discard the individual -- which wastes the evaluation and
biases the search towards whatever region happens to be feasible -- each
membership function's parameters are sorted ascending on decode. Every point
in the 648-bit space therefore maps to a valid controller, and selection
pressure stays on fitness where it belongs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import normalised_rmse

BITS_PER_GENE = 8
_LEVELS = (1 << BITS_PER_GENE) - 1          # 255


@dataclass
class GAConfig:
    """Every knob, in one place, so the report can tabulate it."""

    population: int = 60
    generations: int = 120
    p_crossover: float = 0.70
    p_mutation: float = 0.01                 # per bit
    elitism: int = 2
    seed: int = 0
    bits_per_gene: int = BITS_PER_GENE

    def as_dict(self):
        return {
            "population": self.population,
            "generations": self.generations,
            "p_crossover": self.p_crossover,
            "p_mutation_per_bit": self.p_mutation,
            "elitism": self.elitism,
            "bits_per_gene": self.bits_per_gene,
            "seed": self.seed,
        }


@dataclass
class GAResult:
    best_params: np.ndarray
    best_fitness: float
    best_nrmse: float
    history_best: list = field(default_factory=list)
    history_mean: list = field(default_factory=list)
    evaluations: int = 0
    config: dict = field(default_factory=dict)


class MembershipGA:
    """Tunes a controller's membership functions against labelled examples."""

    def __init__(self, flc, dataset, config: GAConfig | None = None):
        self.flc = flc
        self.data = dataset
        self.cfg = config or GAConfig()

        # Per-parameter decode bounds: every gene inherits the physical range
        # of the variable whose membership function it belongs to.
        lo, hi = [], []
        for name, role, k in flc.param_layout():
            var = flc.inputs.get(name) or flc.outputs.get(name)
            lo.extend([var.lo] * k)
            hi.extend([var.hi] * k)
        self.lo = np.array(lo)
        self.hi = np.array(hi)

        self.n_genes = flc.n_params
        self.n_bits = self.n_genes * self.cfg.bits_per_gene
        self.spans = np.array([flc.outputs[n].hi - flc.outputs[n].lo
                               for n in flc.output_names])

        # Powers of two for vectorised binary -> integer decoding.
        self._weights = (1 << np.arange(self.cfg.bits_per_gene - 1, -1, -1))

    # ------------------------------------------------------------- encoding

    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """648 bits -> 81 physical parameters.

        Accepts a single chromosome (n_bits,) or a population (P, n_bits) and
        returns (n_genes,) or (P, n_genes) respectively.
        """
        c = np.atleast_2d(chromosome)
        genes = c.reshape(c.shape[0], self.n_genes, self.cfg.bits_per_gene)
        ints = (genes * self._weights).sum(axis=2)
        vals = self.lo + (ints / _LEVELS) * (self.hi - self.lo)
        return vals[0] if chromosome.ndim == 1 else vals

    def encode(self, params: np.ndarray) -> np.ndarray:
        """81 physical parameters -> 648 bits. The inverse of `decode`.

        Used to seed the population with the hand-designed controller, so the
        search starts from expert knowledge rather than from noise.
        """
        params = np.clip(np.asarray(params, dtype=float), self.lo, self.hi)
        ints = np.rint((params - self.lo) / (self.hi - self.lo) * _LEVELS).astype(int)
        bits = ((ints[:, None] >> np.arange(self.cfg.bits_per_gene - 1, -1, -1)) & 1)
        return bits.reshape(-1).astype(np.uint8)

    # -------------------------------------------------------------- fitness

    def evaluate_params(self, params, X, y):
        """Span-normalised RMSE of one parameter vector on one split."""
        candidate = self.flc.with_params(params)     # sorting repairs order
        pred = candidate.evaluate(X)
        nrmse, per_output = normalised_rmse(y, pred, self.spans)
        return nrmse, per_output

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """1 / (1 + nRMSE) for every individual, on the training split."""
        params = self.decode(population)
        out = np.empty(len(population))
        for i, p in enumerate(params):
            nrmse, _ = self.evaluate_params(p, self.data.X_train, self.data.y_train)
            out[i] = 1.0 / (1.0 + nrmse)
        return out

    # ------------------------------------------------------------ operators

    def _select(self, population, fits, rng):
        """Roulette wheel: P(i) proportional to fitness.

        Fitness here is already positive and bounded, so no shifting is needed.
        The wheel is spun once per required parent.
        """
        total = fits.sum()
        probs = fits / total if total > 0 else np.full(len(fits), 1 / len(fits))
        idx = rng.choice(len(population), size=len(population), p=probs)
        return population[idx]

    def _crossover(self, parents, rng):
        """One-point crossover applied to consecutive pairs at p_c.

        The cut may fall anywhere in the chromosome, including inside a gene,
        which is deliberate: cutting only on gene boundaries would make the
        8-bit words atomic and prevent the search from refining a single
        breakpoint by a small amount.
        """
        children = parents.copy()
        for i in range(0, len(parents) - 1, 2):
            if rng.random() < self.cfg.p_crossover:
                pt = rng.integers(1, self.n_bits)
                children[i, pt:] = parents[i + 1, pt:]
                children[i + 1, pt:] = parents[i, pt:]
        return children

    def _mutate(self, population, rng):
        """Independent bit flips at p_m."""
        mask = rng.random(population.shape) < self.cfg.p_mutation
        population[mask] ^= 1
        return population

    # ------------------------------------------------------------------ run

    def run(self, seed_with_expert=True, verbose=False) -> GAResult:
        """Evolve, returning the best parameters found on the training split."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        population = rng.integers(0, 2, size=(cfg.population, self.n_bits),
                                  dtype=np.uint8)
        if seed_with_expert:
            # One individual carries the hand-designed controller. The GA
            # should never do worse than the expert design, and seeding makes
            # that a property of the search rather than a hope.
            population[0] = self.encode(self.flc.get_params())

        fits = self.fitness(population)
        evaluations = len(population)
        best_i = int(np.argmax(fits))
        best_chrom, best_fit = population[best_i].copy(), float(fits[best_i])
        hist_best, hist_mean = [best_fit], [float(fits.mean())]

        for gen in range(cfg.generations):
            parents = self._select(population, fits, rng)
            children = self._mutate(self._crossover(parents, rng), rng)

            # Elitism: the best individuals survive untouched, so the best
            # fitness can never fall between generations.
            if cfg.elitism > 0:
                elite = population[np.argsort(fits)[-cfg.elitism:]]
                children[:cfg.elitism] = elite

            population = children
            fits = self.fitness(population)
            evaluations += len(population)

            i = int(np.argmax(fits))
            if fits[i] > best_fit:
                best_fit = float(fits[i])
                best_chrom = population[i].copy()
            hist_best.append(best_fit)
            hist_mean.append(float(fits.mean()))

            if verbose and (gen % 20 == 0 or gen == cfg.generations - 1):
                print(f"    gen {gen:>4}  best fitness {best_fit:.6f}  "
                      f"(nRMSE {1 / best_fit - 1:.6f})  mean {fits.mean():.6f}")

        best_params = self.decode(best_chrom)
        # store the *repaired* parameters, which is what the controller uses
        best_params = self.flc.with_params(best_params).get_params()

        return GAResult(
            best_params=best_params,
            best_fitness=best_fit,
            best_nrmse=1.0 / best_fit - 1.0,
            history_best=hist_best,
            history_mean=hist_mean,
            evaluations=evaluations,
            config=cfg.as_dict(),
        )


def linguistic_ordering_intact(flc, params) -> dict:
    """Do the tuned sets still sit in their original left-to-right order?

    Mamdani is chosen in Part 1 because the rule base stays readable. That
    argument survives optimisation only if "Cold" is still to the left of
    "Cool" afterwards -- a GA that reorders the sets produces a controller
    whose rules no longer mean what they say, and the interpretability
    justification collapses with it.

    Ordering is compared on each set's centre of mass, per variable. This is
    reported rather than enforced: whether unconstrained tuning preserves
    interpretability is a finding, not something to assume.
    """
    tuned = flc.with_params(params)
    report = {}
    for name in list(flc.inputs) + list(flc.outputs):
        before = flc.inputs.get(name) or flc.outputs.get(name)
        after = tuned.inputs.get(name) or tuned.outputs.get(name)
        c0 = [float(np.mean(mf.params)) for mf in before.mfs]
        c1 = [float(np.mean(mf.params)) for mf in after.mfs]
        report[name] = {
            "names": list(before.mf_names),
            "centres_before": c0,
            "centres_after": c1,
            "order_preserved": list(np.argsort(c0)) == list(np.argsort(c1)),
        }
    return report
