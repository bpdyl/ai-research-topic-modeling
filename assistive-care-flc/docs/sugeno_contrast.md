# How the Part 2 GA would change under a Sugeno controller

Part 2 requires a description of how the solution would change if the other inference model were
used. Our Part 1 controller is **Mamdani**; this is the Sugeno (TSK) counterpart.

Written to be reused directly as a report section. Numbers referenced here come from
`results/part1_analysis.json` and `results/part2_ga.json`.

---

## 1. What actually changes

Only the **consequent** side of the system changes. A Sugeno rule replaces the output fuzzy set
with a function of the inputs:

| | Mamdani (what we built) | Sugeno (TSK) |
|---|---|---|
| Rule consequent | `THEN hvac is HeatLow` — a fuzzy set | `THEN hvac = p₀ + p₁·T + p₂·A + p₃·D + p₄·P` |
| Implication | clip the set at α (min) | evaluate the function |
| Aggregation | union of clipped sets (max) | none needed |
| Defuzzification | centroid of the aggregate | weighted average `Σαᵢyᵢ / Σαᵢ` |

The antecedents, the input membership functions and all 54 rules stay exactly as they are. That
matters for the GA: **the input half of the chromosome is unchanged.**

---

## 2. The effect on the chromosome

Our chromosome is 81 parameters × 8 bits = **648 bits**, split:

```
inputs   room_temp 17 · activity 11 · daylight 11 · preference 11  =  50 parameters
outputs  hvac      17 · dimmer   14                                =  31 parameters
```

Under Sugeno the 31 **output** parameters are deleted — there are no output membership functions
to tune — and replaced by the coefficients of the consequent functions. What replaces them
depends entirely on the order chosen, and the range is large:

### Zero-order Sugeno (constant consequents)

Each output fuzzy set becomes a single singleton value. Our 5 HVAC sets and 4 dimmer sets become
**9 constants**:

```
50 input parameters + 9 consequent constants = 59 parameters
59 × 8 bits = 472 bits          (a 27 % shorter chromosome than Mamdani's 648)
```

This is the closest analogue to what we have, and the cheapest search.

### First-order Sugeno (linear consequents)

Each *rule* — not each output set — carries its own linear function of the four inputs, so five
coefficients per rule:

```
50 input parameters + 54 rules × 5 coefficients = 320 parameters
320 × 8 bits = 2,560 bits       (nearly 4× Mamdani's chromosome)
```

**This is the decisive practical point.** The search space grows from 2⁶⁴⁸ to 2²⁵⁶⁰. Our GA
already needs 9,060 evaluations per run against a 648-bit chromosome and reaches only a modest
improvement over the hand design; a four-times-longer chromosome with the same population and
budget would almost certainly do worse, not better, and the honest options would be to raise the
budget substantially or to restrict linear consequents to the thermal rules only.

---

## 3. What gets easier

**Least-squares becomes available, and it is a large win.** With the antecedents fixed, a Sugeno
system is *linear in its consequent parameters*. The firing strengths α depend only on the input
membership functions, so for a fixed set of input MFs the optimal consequent coefficients can be
solved directly by least squares rather than searched.

That enables the standard **hybrid ANFIS-style scheme**, and it is the approach we would actually
take:

- **GA (or gradient descent) tunes only the 50 input MF parameters** — 400 bits.
- **Least squares solves the consequent coefficients exactly** at each fitness evaluation.

The GA's search space shrinks by more than a third against our current 648 bits, *and* every
individual is scored using optimal consequents rather than whatever the chromosome happened to
encode. Both effects push in the same direction. This is why ANFIS is built on Sugeno and not on
Mamdani — Mamdani's centroid defuzzification is not linear in the output MF parameters, so no
equivalent closed-form step exists for the system we built.

**Evaluation also gets cheaper.** Our Mamdani engine integrates a 501-point aggregated set per
output per sample. Sugeno replaces that with a weighted average over active rules — no grid, no
integration. Given that Part 2's GA spends essentially all of its time inside the controller
(≈18 ms per fitness evaluation over 280 training samples), this would be a large constant-factor
speed-up, plausibly an order of magnitude.

**The repair operator becomes partly unnecessary.** Our decoder sorts each membership function's
parameters ascending, because a GA freely proposes `trimf(24, 19, 21)`. Consequent *coefficients*
have no ordering constraint at all, so under Sugeno the repair applies only to the input half.

---

## 4. What gets harder or worse

**Interpretability, which is the reason we chose Mamdani in the first place.** A Mamdani rule
reads `IF room_temp is Cold AND activity is Resting THEN hvac is HeatHigh` — a carer or
occupational therapist can check that sentence against their own judgement. The first-order
Sugeno equivalent reads `THEN hvac = 71.2 − 3.4·T + 1.1·A + 0.02·D + 4.6·P`, which is not
something a domain expert can audit, and after GA tuning the coefficients would carry no
particular meaning at all. For an assistive-care system whose behaviour someone has to sign off
on, that is a real loss, not a cosmetic one.

**The linguistic-ordering check we run (`ga.linguistic_ordering_intact`) would only half apply.**
It verifies that GA tuning has not reordered the fuzzy sets and destroyed the meaning of the
rules. Under Sugeno it would still cover the input sets, but there would be nothing equivalent to
check on the output side — the consequents would simply be numbers, sound or not.

**The safety property would need re-establishing.** Our clamped FAM guarantees the controller can
never heat a hot room or cool a cold one, and that follows from the *structure* of the rule base
(DEC-030). With free linear consequents a GA could produce coefficients that violate it while
still scoring well on average error. It would have to be reimposed either as a constraint on the
coefficients or as a penalty in the fitness function — and a penalty only discourages violations
rather than preventing them, which is weaker than what we have now.

---

## 5. What we would keep unchanged

- **The fitness function.** Span-normalised RMSE, `fitness = 1/(1 + nRMSE)`, still applies without
  modification — it is defined on the controller's crisp outputs and does not care how they were
  produced.
- **The dataset and the train/test split**, including its synthetic provenance and the declaration
  that goes with it.
- **The genetic operators** — roulette-wheel selection, one-point crossover at 0.7, per-bit
  mutation at 0.01, elitism — all act on a bit string and are indifferent to what it encodes.
- **The budget-matched random-search control.** If anything it becomes *more* important: with
  least squares solving the consequents, a large part of the fitness improvement would come from
  the closed-form step rather than from evolution, and the control is the only way to show how
  much of the gain the GA is actually responsible for.

---

## 6. Summary

| | Mamdani (built) | Zero-order Sugeno | First-order Sugeno | First-order + least squares |
|---|---|---|---|---|
| Chromosome | **648 bits** | 472 bits | 2,560 bits | **400 bits** |
| Tuned by GA | 81 parameters | 59 | 320 | 50 (inputs only) |
| Consequents | fuzzy sets | constants | linear functions | solved exactly |
| Defuzzification | centroid over a grid | weighted average | weighted average | weighted average |
| Interpretable to a carer | **yes** | partly | no | no |
| Safety by construction | **yes** | yes | no — needs a penalty | no — needs a penalty |

The short answer: **Sugeno would make the optimisation problem easier and the controller harder to
trust.** For a benchmark or an inner loop that is a good trade. For an assistive-care flat, where
the behaviour has to be explicable to the resident and signed off by a carer, it is the wrong way
round, which is why Part 1 uses Mamdani.
