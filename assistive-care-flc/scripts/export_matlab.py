"""Emit the controller as MATLAB Fuzzy Logic Toolbox code.

    python scripts/export_matlab.py

Writes `matlab/buildAssistiveCareFIS.m` and `matlab/runAssistiveCareFIS.m`.

Why generate rather than hand-write
-----------------------------------
The brief expects Part 1 to be demonstrable in the MATLAB Fuzzy Logic Toolbox
(or FuzzyLite/Juzzy), and the toolbox viewers -- FIS Editor, Membership
Function Editor, Rule Editor, Rule Viewer, Surface Viewer -- are exactly the
five component screenshots the 16-mark implementation-evidence criterion asks
for. MATLAB is not installed here, but MATLAB Online can run the emitted code.

Writing the FIS out twice by hand would mean maintaining two definitions of the
same controller and hoping they stay in step. Emitting the .m file from
`acflc.flat` makes drift impossible: the MATLAB system is the Python system, by
construction. If a membership function changes, re-running this script updates
MATLAB too.

The emitted script also prints the firing strengths and defuzzified outputs for
the worked scenario, so the MATLAB run can be checked against
`results/part1_analysis.json` rather than merely looking plausible.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acflc import build_controller  # noqa: E402

OUT = ROOT / "matlab"
OUT.mkdir(exist_ok=True)

# The scenario traced by hand in the report; kept in step with run_part1.py.
WORKED = {"room_temp": 21.0, "activity": 7.5, "daylight": 25.0, "preference": 0.0}


def _mf_call(mf):
    params = ", ".join(f"{v:g}" for v in mf.params)
    return f"'{mf.kind}', [{params}]"


def emit_builder(flc) -> str:
    L = []
    a = L.append
    a("function fis = buildAssistiveCareFIS()")
    a("%BUILDASSISTIVECAREFIS  Mamdani FLC for one room of an assistive-care flat.")
    a("%")
    a("%   fis = BUILDASSISTIVECAREFIS() returns the fuzzy inference system used in")
    a("%   Task 2 Part 1: four sensor inputs, two actuator outputs, 54 rules.")
    a("%")
    a("%   GENERATED FILE - do not edit by hand.")
    a("%   Emitted from the Python definition in src/acflc/flat.py by")
    a("%   scripts/export_matlab.py, so the MATLAB and Python systems cannot")
    a("%   drift apart. Re-run that script after any change to the controller.")
    a("%")
    a("%   Requires the Fuzzy Logic Toolbox, R2018b or later.")
    a("")
    a("fis = mamfis('Name', 'AssistiveCareRoomFLC', ...")
    a("             'AndMethod', 'min', ...")
    a("             'OrMethod', 'max', ...")
    a("             'ImplicationMethod', 'min', ...")
    a("             'AggregationMethod', 'max', ...")
    a("             'DefuzzificationMethod', 'centroid');")
    a("")

    for role, names in (("Input", flc.input_names), ("Output", flc.output_names)):
        for name in names:
            var = flc.inputs.get(name) or flc.outputs.get(name)
            a(f"%% {role}: {var.name}  [{var.unit}]")
            a(f"fis = add{role}(fis, [{var.lo:g} {var.hi:g}], 'Name', '{var.name}');")
            for mf in var.mfs:
                a(f"fis = addMF(fis, '{var.name}', {_mf_call(mf)}, "
                  f"'Name', '{mf.name}');")
            a("")

    # Rule matrix. Columns are the four inputs, then the two outputs, then the
    # rule weight and the antecedent connective (1 = AND). A zero means the
    # variable is not referenced by that rule.
    a("%% Rules")
    a("% Columns: [room_temp activity daylight preference | hvac dimmer | weight connective]")
    a("% 0 means the variable is not used by that rule; connective 1 is AND.")
    a("% Thermal rules constrain temperature, activity and preference; lighting")
    a("% rules constrain daylight and activity. Generated from the additive FAM")
    a("% tables described in src/acflc/flat.py.")
    a("ruleList = [")
    for i, rule in enumerate(flc.rules):
        cols = []
        for name in flc.input_names:
            mf_name = rule.antecedents.get(name)
            cols.append(0 if mf_name is None
                        else flc.inputs[name].index_of(mf_name) + 1)
        ov, om = rule.consequent
        for name in flc.output_names:
            cols.append(flc.outputs[name].index_of(om) + 1 if name == ov else 0)
        cols += [rule.weight, 1]
        body = " ".join(f"{int(c):>2}" if float(c).is_integer() else f"{c:g}"
                        for c in cols)
        note = f"  % R{i}: {rule.text()}"
        a(f"    {body}{note}")
    a("];")
    a("fis = addRule(fis, ruleList);")
    a("")
    a("end")
    return "\n".join(L) + "\n"


def emit_runner(flc) -> str:
    scen = ", ".join(f"{k} = {v:g}" for k, v in WORKED.items())
    order = ", ".join(flc.input_names)
    vals = " ".join(f"{WORKED[n]:g}" for n in flc.input_names)

    L = []
    a = L.append
    a("%% Assistive-care flat FLC - Part 1 evidence script")
    a("%")
    a("%  Builds the controller, evaluates the worked scenario, and opens the")
    a("%  five Fuzzy Logic Toolbox views that Part 1 asks for evidence of.")
    a("%")
    a("%  GENERATED FILE - emitted by scripts/export_matlab.py.")
    a("%")
    a("%  Cross-check: the printed values below must match")
    a("%  results/part1_analysis.json from the Python implementation. They are")
    a("%  two independent evaluations of the same system, so agreement is")
    a("%  evidence and disagreement is a defect.")
    a("")
    a("clear; clc; close all;")
    a("")
    a("fis = buildAssistiveCareFIS();")
    a("")
    a(f"%% Worked scenario: {scen}")
    a(f"%  Input order: [{order}]")
    a(f"x = [{vals}];")
    a("")
    a("[y, ~, ~, ~, ruleFiring] = evalfis(fis, x);")
    a("")
    a("fprintf('Worked scenario\\n');")
    for i, n in enumerate(flc.input_names):
        a(f"fprintf('  {n:<11} = %g\\n', x({i + 1}));")
    a("fprintf('\\nDefuzzified outputs (centroid)\\n');")
    for i, n in enumerate(flc.output_names):
        a(f"fprintf('  {n:<7} = %.4f\\n', y({i + 1}));")
    a("")
    a("fprintf('\\nActive rules (firing strength > 0)\\n');")
    a("active = find(ruleFiring > 0);")
    a("for k = 1:numel(active)")
    a("    r = active(k);")
    a("    fprintf('  R%-3d  alpha = %.4f\\n', r, ruleFiring(r));")
    a("end")
    a("")
    a("%% Component views - screenshot each of these")
    a("% 1. Membership functions for every variable")
    a("figure('Name', 'Input membership functions', 'Position', [80 80 900 620]);")
    for i, n in enumerate(flc.input_names):
        a(f"subplot({len(flc.input_names)}, 1, {i + 1});")
        a(f"plotmf(fis, 'input', {i + 1}); title('{n}'); ylabel('\\mu');")
    a("")
    a("figure('Name', 'Output membership functions', 'Position', [120 100 900 400]);")
    for i, n in enumerate(flc.output_names):
        a(f"subplot({len(flc.output_names)}, 1, {i + 1});")
        a(f"plotmf(fis, 'output', {i + 1}); title('{n}'); ylabel('\\mu');")
    a("")
    a("% 2. Control surfaces")
    a("figure('Name', 'Surface: hvac vs room_temp and activity');")
    a("gensurf(fis, [1 2], 1);")
    a("xlabel('room temp [degC]'); ylabel('activity [index]'); "
      "zlabel('hvac [% capacity]');")
    a("")
    a("figure('Name', 'Surface: dimmer vs daylight and activity');")
    a("gensurf(fis, [3 2], 2);")
    a("xlabel('daylight [%]'); ylabel('activity [index]'); zlabel('dimmer [%]');")
    a("")
    a("% 3. Interactive views - screenshot these from the GUI")
    a("fuzzy(fis)      % FIS Editor, MF Editor and Rule Editor")
    a("ruleview(fis)   % Rule Viewer - set the inputs to the scenario above")
    a("surfview(fis)   % Surface Viewer")
    a("")
    a("% Optional: save the system so it can be attached to the report appendix")
    a("% writeFIS(fis, 'assistive_care_room');")
    return "\n".join(L) + "\n"


def main():
    flc = build_controller()
    b = OUT / "buildAssistiveCareFIS.m"
    r = OUT / "runAssistiveCareFIS.m"
    b.write_text(emit_builder(flc), encoding="utf-8")
    r.write_text(emit_runner(flc), encoding="utf-8")

    print(f"wrote {b.relative_to(ROOT)}  ({len(flc.rules)} rules, "
          f"{len(flc.input_names)} inputs, {len(flc.output_names)} outputs)")
    print(f"wrote {r.relative_to(ROOT)}")
    print("\nIn MATLAB Online: upload both files, then run `runAssistiveCareFIS`.")
    print("Expected outputs for the worked scenario, from the Python engine:")
    y = flc(**WORKED)
    for k, v in y.items():
        print(f"  {k:<7} = {v:.4f}")


if __name__ == "__main__":
    main()
