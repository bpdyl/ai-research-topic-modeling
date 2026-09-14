# Draft email — two questions for the module leader

**To:** Shrawan Thakur <stw0049@softwarica.edu.np>
**Subject:** ST7085CEM CW — two clarifications (Task 1 / Task 2 overlap; Task 2 Part 1 tooling)

Closes [OQ-016](../../../.brain/OPEN_QUESTIONS.md) and [OQ-020](../../../.brain/OPEN_QUESTIONS.md).
Both have been flagged internally and neither has been raised. Send as one message.

---

Dear Shrawan,

We are a group of three (Bibek Paudyal 250288, Siddhartha Bhatta 250620, Sajan Mahat 250289)
and have two short questions on the coursework before we submit.

**1. A genetic algorithm appears in both tasks. Is that a problem?**

Our Task 1 applies LDA topic modelling to AI research with Nepal-affiliated authors, and one of
the four models we compare uses a GA to search the LDA hyperparameters (K, alpha, beta). Task 2
Part 2 separately requires a GA to tune the membership functions of our fuzzy controller.

The two optimise entirely different things and the tasks are independent, but we wanted to check
you are content with a GA appearing in both rather than it reading as the same content submitted
twice. If you would prefer, we can foreground a different model as Task 1's headline method — we
also ran a budget-matched random search control, which in fact outperformed the GA, and that
negative result is already reported in the paper.

**2. Task 2 Part 1 is implemented in Python rather than MATLAB. Is that acceptable?**

The brief specifies the fuzzy logic controller should be demonstrable and simulated using "MATLAB
Fuzzy Logic Toolbox, FuzzyLite, or Juzzy". We do not have access to a MATLAB licence, so we have
implemented the Mamdani controller in Python.

We would rather ask than assume, because Part 1 is 30 marks. For what it is worth, we have taken
some care to make the implementation verifiable rather than just asserting it works:

- The inference engine is written from first principles — fuzzification, min t-norm, min
  implication, max aggregation, centroid defuzzification — rather than taken from a library.
- It is cross-checked against `scikit-fuzzy` as an independent oracle: the two agree on every
  consequent activation to 1e-12, and converge on the defuzzified output as the output universe
  is refined.
- All the component views the brief asks for evidence of (membership functions, rule activation,
  implication and aggregation stages, control surfaces) are produced, and every figure is
  generated from the same inference the controller actually ran.

If you would prefer to see Fuzzy Logic Toolbox output specifically, we can reproduce the same
system in MATLAB Online if a campus licence is available to us — could you confirm whether one is?

Thank you,

Sajan Mahat (250289), on behalf of the group

---

## Why each question is being asked

**Q1 (OQ-016)** was flagged twice in our own planning notes and never closed. The risk is low on
the merits — the GA is a tool inside Task 1's method, not the method itself — but it is cheap
insurance, and the honest framing (offering to reorder the emphasis) costs nothing because the
random-search control genuinely did beat the GA.

**Q2 (OQ-020)** matters more. The brief's tool list is not hedged with "e.g.", and Part 1 is 30
marks with 16 of them for implementation evidence. The email is worth sending on its own for this
question alone; asking early also leaves time to capture MATLAB Online screenshots if the answer
is that the toolbox is expected.
