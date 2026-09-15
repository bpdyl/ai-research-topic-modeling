# Task 2 completeness check - 15 September 2026

**Assessment: the required implementation and report components for all three
parts are present.** This confirms coverage against the supplied PDF, not a
guarantee of marks, live deployment suitability or complete re-execution in
every supported environment.

## Part 1 - controller design and implementation (30 marks)

The report describes an assistive-care room, four crisp inputs, two outputs,
fuzzy sets, 54 Mamdani rules, inference and centroid defuzzification. It includes
membership-function diagrams, MATLAB visualisations, worked rule activation,
output traces and control surfaces. The code is in `assistive-care-flc/src/acflc/`
and `assistive-care-flc/matlab/`.

A wording error was corrected: the current MATLAB implementation is a custom
base-MATLAB engine, not a Fuzzy Logic Toolbox system. The assignment says a
simulated system may be programmed in Matlab, FuzzyLite or Juzzy, then gives
Toolbox reference links. It does not explicitly restrict MATLAB to the Toolbox.
The older requirements summary overstated that restriction and has been corrected.
The report now identifies the implementation accurately.

The historical decision record reports the MATLAB-compatible engine checked in
GNU Octave, with worked-scenario differences around 3.6e-7 against Python. The
Python engine is also cross-checked against scikit-fuzzy in passing tests. Those
are distinct checks; they should not be described as a fresh Toolbox execution.

## Part 2 - membership-function optimisation (10 marks)

The implementation preserves the rule structure and tunes input/output membership
functions. The report specifies the synthetic dataset, training/test split,
81 parameters, 648-bit chromosome, operators, fitness and evaluation budget.
It compares five GA trials with a matched random-search control and explains
how the encoding and optimisation change under Sugeno inference.

The lower error is not concealed behind an unqualified improvement claim:
none of the five tuned controllers retains both input coverage and linguistic
ordering. Constrained tuning would be a useful extension, not a missing stated
assignment requirement. Real-resident validation is outside this simulation.

## Part 3 - CEC2005 comparison (10 marks)

Official F6 and F9 shifts are vendored with provenance. GA, PSO and SA were run
at D=2 and D=10, with 15 repetitions each: 180 runs. The report contains function
descriptions, MATLAB function code, parameters, mean/standard deviation/best/worst
results, convergence plots and distribution-aware discussion. D=100 and literature
comparisons are optional in the brief.

The 180 trials were rerun in Python using the official shifts. The added MATLAB
benchmark listing implements the same equations and official data files but has
not been executed in MATLAB during this revision. That is a verification limit,
not a claim that MATLAB benchmark trials were run. The prior locally shifted
results remain explicitly archived.

## Packaging and verification

The separate Task 2 PDF and shared sources are included in the output package.
The Python tests pass, references/figures are checked and the rendered report is
reviewed. Updated local files have not been pushed to the GitHub repository;
the linked repository must be synchronised with these files before submission
if the GitHub link is used to supply the code.

Before an assessed live demonstration, execute the supplied MATLAB driver in
the intended environment and retain its output. The new benchmark listing can
also be checked at the official optima. This would strengthen the runtime evidence
without changing the already reported Python experiment.
