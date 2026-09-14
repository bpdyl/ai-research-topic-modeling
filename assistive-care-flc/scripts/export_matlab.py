"""Emit the controller as base-MATLAB code. No Fuzzy Logic Toolbox required.

    python scripts/export_matlab.py

Writes `matlab/buildAssistiveCareFIS.m`, `matlab/runAssistiveCareFIS.m` and
`matlab/acExportFig.m`.

Why generate rather than hand-write
-----------------------------------
The brief expects Part 1 to be demonstrable in MATLAB. Writing the FIS out
twice by hand would mean maintaining two definitions of one controller and
hoping they stay in step. Emitting the .m files from `acflc.flat` makes drift
impossible: the MATLAB system *is* the Python system, by construction. Change a
membership function in `src/acflc/flat.py` and re-run this script.

Why there is no toolbox call anywhere in the output
---------------------------------------------------
The first version of this exporter emitted `mamfis`/`addInput`/`addMF`/
`addRule`, and `runAssistiveCareFIS.m` called `evalfis`, `plotmf` and
`gensurf`. All of those belong to the Fuzzy Logic Toolbox, which is *not*
included in the MATLAB Online licence available for this project:

    >> license('test','Fuzzy_Toolbox')
    ans = 0
    >> which mamfis
    'mamfis' not found.

The toolbox cannot be installed, so the dependency had to be removed rather
than worked around. The emitted code now describes the controller as a plain
struct and hands it to a Mamdani engine written in base MATLAB:

    matlab/acEvalMF.m              trimf / trapmf          (hand-written)
    matlab/evalAssistiveCareFIS.m  the five-stage engine   (hand-written)
    matlab/buildAssistiveCareFIS.m the controller itself   (generated here)
    matlab/runAssistiveCareFIS.m   driver, checks, figures (generated here)
    matlab/acExportFig.m           portable PNG export     (generated here)

The two engine files are hand-written and checked in rather than generated,
because they are *generic*: they contain no knowledge of this controller's
variables or rules, so they cannot drift from `flat.py` and there is nothing
for a generator to specialise. Only the definition-derived files are emitted.

The emitted runner cross-checks itself against the Python engine on the worked
scenario, using constants baked in below from the very same controller object,
so they cannot be quietly adjusted by hand to make a failing check pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acflc import build_controller  # noqa: E402

OUT = ROOT / "matlab"
OUT.mkdir(exist_ok=True)

#: Hand-written engine files the emitted code depends on. Checked, not written.
ENGINE_FILES = ("acEvalMF.m", "evalAssistiveCareFIS.m")

# The scenario traced by hand in the report; kept in step with run_part1.py.
WORKED = {"room_temp": 21.0, "activity": 7.5, "daylight": 25.0, "preference": 0.0}

# Representative states swept by the emitted runner as a smoke test. Each is a
# corner of the design space the report makes a claim about, so a regression in
# any of them is visible without having to read a figure.
EDGE_CASES = [
    ("heating failure, resting, night",  14.0,  0.0,   0.0,  0.0),
    ("very cold, active, warmer pref",   14.5,  9.5,  10.0,  4.0),
    ("comfortable, light activity",      22.5,  5.0,  50.0,  0.0),
    ("heatwave, resting, cooler pref",   34.0,  0.0, 100.0, -5.0),
    ("hot, active, neutral",             31.0,  9.0,  80.0,  0.0),
    ("dark room, active occupant",       22.0,  9.0,   0.0,  0.0),
    ("bright room, resting occupant",    22.0,  0.0, 100.0,  0.0),
    ("universe corner, all low",         14.0,  0.0,   0.0, -5.0),
    ("universe corner, all high",        34.0, 10.0, 100.0,  5.0),
]


def _num(v) -> str:
    """MATLAB literal for a number, without a trailing '.0'."""
    return f"{v:g}"


# ---------------------------------------------------------------- builder

def emit_builder(flc) -> str:
    """`buildAssistiveCareFIS.m`: the controller as a plain MATLAB struct."""
    L = []
    a = L.append

    a("function fis = buildAssistiveCareFIS()")
    a("%BUILDASSISTIVECAREFIS  Mamdani FLC for one room of an assistive-care flat.")
    a("%")
    a("%   fis = BUILDASSISTIVECAREFIS() returns the fuzzy inference system used")
    a(f"%   in Task 2 Part 1: {len(flc.input_names)} sensor inputs, "
      f"{len(flc.output_names)} actuator outputs, {len(flc.rules)} rules.")
    a("%")
    a("%   NO FUZZY LOGIC TOOLBOX REQUIRED. The system is a plain struct, not a")
    a("%   mamfis object, and is evaluated by EVALASSISTIVECAREFIS, which")
    a("%   implements Mamdani inference in base MATLAB. This is a deliberate")
    a("%   substitution: the MATLAB Online licence available for this project")
    a("%   reports license('test','Fuzzy_Toolbox') == 0, so mamfis, addInput,")
    a("%   addMF, addOutput, addRule, evalfis, plotmf and gensurf are all")
    a("%   unavailable. The controller itself is unchanged.")
    a("%")
    a("%   GENERATED FILE - do not edit by hand. Emitted from the Python")
    a("%   definition in src/acflc/flat.py by scripts/export_matlab.py, so the")
    a("%   MATLAB and Python systems cannot drift apart. Re-run that script")
    a("%   after any change to the controller.")
    a("%")
    a("%   Struct layout")
    a("%     fis.inputs(i).name  .unit  .range  .mfs(k).name .kind .params")
    a("%     fis.outputs(o).name .unit  .range  .mfs(k).name .kind .params")
    a("%                         .universe  .mfMatrix         (cached below)")
    a("%     fis.ruleAnt         nRules x nInputs,  0 = variable not used")
    a("%     fis.ruleCons        nRules x nOutputs, 0 = output not driven")
    a("%     fis.ruleWeight      nRules x 1")
    a("%     fis.ruleText        nRules x 1 cell, the readable rule")
    a("%")
    a("%   See also EVALASSISTIVECAREFIS, ACEVALMF, RUNASSISTIVECAREFIS.")
    a("")
    a("fis = struct();")
    a(f"fis.name = '{flc.name}';")
    a("")
    a("%% Inference configuration")
    a("%  Unchanged from the Toolbox version, and from src/acflc/controller.py.")
    a("fis.andMethod             = 'min';    % AND across a rule's antecedents")
    a("fis.orMethod              = 'max';    % unused: every rule here is pure AND")
    a("fis.implicationMethod     = 'min';    % clip the consequent at alpha")
    a("fis.aggregationMethod     = 'max';    % union the clipped sets")
    a("fis.defuzzificationMethod = 'centroid';")
    a("")
    a("%  Output universe resolution. This is the authoritative value, set by")
    a("%  Variable.n_points in src/acflc/membership.py. The Toolbox version had")
    a("%  to override MATLAB's default of 101 to match it; here it is simply the")
    a("%  definition. Changing it shifts the defuzzified outputs slightly and")
    a("%  breaks comparability with the Python reference.")
    a("fis.nPoints = 501;")
    a("")
    a("fis.inputNames  = {"
      + ", ".join(f"'{n}'" for n in flc.input_names) + "};")
    a("fis.outputNames = {"
      + ", ".join(f"'{n}'" for n in flc.output_names) + "};")
    a("")

    for role, names, field in (("Input", flc.input_names, "inputs"),
                               ("Output", flc.output_names, "outputs")):
        for vi, name in enumerate(names, start=1):
            var = flc.inputs.get(name) or flc.outputs.get(name)
            a(f"%% {role}: {var.name}  [{var.unit}]")
            a(f"fis.{field}({vi}).name  = '{var.name}';")
            a(f"fis.{field}({vi}).unit  = '{var.unit}';")
            a(f"fis.{field}({vi}).range = [{_num(var.lo)} {_num(var.hi)}];")
            for mi, mf in enumerate(var.mfs, start=1):
                params = " ".join(_num(p) for p in mf.params)
                a(f"fis.{field}({vi}).mfs({mi}).name   = '{mf.name}';")
                a(f"fis.{field}({vi}).mfs({mi}).kind   = '{mf.kind}';")
                a(f"fis.{field}({vi}).mfs({mi}).params = [{params}];")
            a("")

    # ---- rules -----------------------------------------------------------
    n_in, n_out = len(flc.input_names), len(flc.output_names)
    a("%% Rules")
    a("%  Columns: [" + " ".join(flc.input_names) + " | "
      + " ".join(flc.output_names) + " | weight]")
    a("%")
    a("%  A zero means the rule does not reference that variable: the lighting")
    a("%  rules constrain daylight and activity and say nothing about")
    a("%  temperature or preference. Silence means 'do not care', not 'zero'.")
    a("%")
    a("%  The Toolbox rule matrix carried an extra trailing connective column")
    a("%  (1 = AND, 2 = OR). It is dropped here because it was constant: every")
    a("%  rule in this controller is a pure conjunction, and the operator is")
    a("%  stated once in fis.andMethod above. No rule semantics change.")
    a("%")
    a("%  Generated from the additive FAM tables described in src/acflc/flat.py.")
    a("ruleList = [")

    rule_texts = []
    for i, rule in enumerate(flc.rules):
        cols = []
        for name in flc.input_names:
            mf_name = rule.antecedents.get(name)
            cols.append(0 if mf_name is None
                        else flc.inputs[name].index_of(mf_name) + 1)
        ov, om = rule.consequent
        for name in flc.output_names:
            cols.append(flc.outputs[name].index_of(om) + 1 if name == ov else 0)
        cols.append(rule.weight)
        body = " ".join(f"{int(c):>2}" if float(c).is_integer() else f"{c:g}"
                        for c in cols)
        a(f"    {body}   % R{i + 1}: {rule.text()}")
        rule_texts.append(rule.text())
    a("];")
    a("")
    a(f"fis.ruleAnt    = ruleList(:, 1:{n_in});")
    a(f"fis.ruleCons   = ruleList(:, {n_in + 1}:{n_in + n_out});")
    a(f"fis.ruleWeight = ruleList(:, {n_in + n_out + 1});")
    a("")
    a("fis.ruleText = { ...")
    for t in rule_texts:
        a(f"    '{t}'; ...")
    a("};")
    a("")
    a("%% Cache each output's universe and its sampled membership functions.")
    a("%  These are fixed for a given parameter set and are touched on every")
    a("%  inference, so they are computed once here rather than per call. This")
    a("%  mirrors FuzzyController._compile() in src/acflc/controller.py.")
    a("for k = 1:numel(fis.outputs)")
    a("    u = linspace(fis.outputs(k).range(1), fis.outputs(k).range(2), ...")
    a("                 fis.nPoints);")
    a("    M = zeros(numel(fis.outputs(k).mfs), fis.nPoints);")
    a("    for m = 1:numel(fis.outputs(k).mfs)")
    a("        M(m, :) = acEvalMF(fis.outputs(k).mfs(m).kind, ...")
    a("                           fis.outputs(k).mfs(m).params, u);")
    a("    end")
    a("    fis.outputs(k).universe = u;")
    a("    fis.outputs(k).mfMatrix = M;")
    a("end")
    a("")
    a("end")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------- runner

def emit_runner(flc, expected) -> str:
    """`runAssistiveCareFIS.m`: validation, evidence figures, edge-case sweep."""
    scen = ", ".join(f"{k} = {v:g}" for k, v in WORKED.items())
    order = ", ".join(flc.input_names)
    vals = " ".join(f"{WORKED[n]:g}" for n in flc.input_names)
    n_in, n_out = len(flc.input_names), len(flc.output_names)
    n_vars = n_in + n_out

    L = []
    a = L.append
    a("%% Assistive-care flat FLC - Part 1 evidence script")
    a("%")
    a("%  Builds the controller, verifies it against the Python reference")
    a("%  implementation, sweeps a set of edge cases, checks the rule base for")
    a("%  holes and dead rules, and exports the three evidence figures.")
    a("%")
    a("%  NO FUZZY LOGIC TOOLBOX REQUIRED. Nothing here calls mamfis, addMF,")
    a("%  addRule, evalfis, plotmf or gensurf. Inference is done by")
    a("%  evalAssistiveCareFIS and the figures are drawn with base MATLAB")
    a("%  plotting, because license('test','Fuzzy_Toolbox') returns 0 on the")
    a("%  MATLAB Online licence available for this project.")
    a("%")
    a("%  GENERATED FILE - emitted by scripts/export_matlab.py. Do not edit by")
    a("%  hand; edit src/acflc/flat.py and re-run the exporter.")
    a("%")
    a("%  HOW TO RUN (MATLAB Online, https://matlab.mathworks.com, or Octave):")
    a("%    1. Upload the whole matlab/ folder.")
    a("%    2. In the Command Window type:  runAssistiveCareFIS")
    a("%    3. Screenshot the Command Window output, and collect the three PNGs")
    a("%       written to the figures/ folder beside this script.")
    a("")
    a("clear; clc; close all;")
    a("")
    a("fis = buildAssistiveCareFIS();")
    a("")
    a("outDir = fullfile(fileparts(mfilename('fullpath')), 'figures');")
    a("if ~exist(outDir, 'dir'); mkdir(outDir); end")
    a("")
    a("nIn    = numel(fis.inputs);")
    a("nOut   = numel(fis.outputs);")
    a("nRules = size(fis.ruleAnt, 1);")
    a("")
    a("fprintf('==============================================================\\n');")
    a("fprintf(' ASSISTIVE-CARE FLAT FLC - base MATLAB, no Fuzzy Logic Toolbox\\n');")
    a("fprintf('==============================================================\\n');")
    a("fprintf('interpreter      : %s\\n', version());")
    a("fprintf('mamfis available : %d  (not required either way)\\n', ...")
    a("        ~isempty(which('mamfis')));")
    a("fprintf('inputs           : %d\\n', nIn);")
    a("fprintf('outputs          : %d\\n', nOut);")
    a("fprintf('rules            : %d\\n', nRules);")
    a("fprintf('and / implication: %s / %s\\n', fis.andMethod, ...")
    a("        fis.implicationMethod);")
    a("fprintf('aggregation      : %s\\n', fis.aggregationMethod);")
    a("fprintf('defuzzification  : %s\\n', fis.defuzzificationMethod);")
    a("fprintf('output samples   : %d\\n', fis.nPoints);")
    a("")
    a(f"%% Worked scenario: {scen}")
    a(f"%  Input order: [{order}]")
    a(f"x = [{vals}];")
    a("")
    a("[y, d] = evalAssistiveCareFIS(fis, x);")
    a("")
    a("fprintf('\\nWorked scenario\\n');")
    for i, n in enumerate(flc.input_names):
        a(f"fprintf('  {n:<11} = %g\\n', x({i + 1}));")
    a("")
    a("fprintf('\\nStage 1 - fuzzified inputs (non-zero memberships)\\n');")
    a("for i = 1:nIn")
    a("    for k = 1:numel(fis.inputs(i).mfs)")
    a("        if d.mu{i}(k) > 0")
    a("            fprintf('  %-11s %-12s %.4f\\n', fis.inputs(i).name, ...")
    a("                    fis.inputs(i).mfs(k).name, d.mu{i}(k));")
    a("        end")
    a("    end")
    a("end")
    a("")
    a("fprintf('\\nStage 2 - active rules (firing strength > 0)\\n');")
    a("for k = 1:numel(d.active)")
    a("    r = d.active(k);")
    a("    fprintf('  R%-3d  alpha = %.4f   %s\\n', r, d.firing(r), ...")
    a("            fis.ruleText{r});")
    a("end")
    a("fprintf('  %d of %d rules active\\n', numel(d.active), nRules);")
    a("")
    a("fprintf('\\nStages 3-5 - defuzzified outputs (centroid)\\n');")
    for i, n in enumerate(flc.output_names):
        a(f"fprintf('  {n:<7} = %.6f\\n', y({i + 1}));")
    a("")
    a("%% Cross-check against the Python reference implementation")
    a("%  These constants are emitted by scripts/export_matlab.py from the very")
    a("%  same controller object, so they are not transcribed by hand and cannot")
    a("%  be quietly adjusted to make a failing check pass.")
    py = ", ".join(f"{expected[n]:.6f}" for n in flc.output_names)
    a(f"pythonOutputs = [{py}];")
    a("names = {" + ", ".join(f"'{n}'" for n in flc.output_names) + "};")
    a("tol = 1e-3;")
    a("fprintf('\\nCross-check against the Python engine (tolerance %.0e)\\n', tol);")
    a("allPass = true;")
    a("for k = 1:numel(names)")
    a("    diffK = abs(y(k) - pythonOutputs(k));")
    a("    ok = diffK < tol;")
    a("    allPass = allPass && ok;")
    a("    verdict = 'FAIL';")
    a("    if ok; verdict = 'PASS'; end")
    a("    fprintf('  %-7s MATLAB %10.6f   Python %10.6f   diff %.2e   %s\\n', ...")
    a("            names{k}, y(k), pythonOutputs(k), diffK, verdict);")
    a("end")
    a("if allPass")
    a("    fprintf('  ==> AGREE: two independent engines, same controller.\\n');")
    a("else")
    a("    fprintf('  ==> DISAGREE: investigate before quoting either.\\n');")
    a("end")
    a("")
    a("%% Edge-case sweep")
    a("%  Representative corners of the design space. The check is not that any")
    a("%  particular number is right -- the control surfaces say that -- but that")
    a("%  every state produces a defined command. A NaN here would mean some")
    a("%  input activates no rule at all, i.e. a hole in the rule base.")
    a("edgeNames = { ...")
    for label, *_ in EDGE_CASES:
        a(f"    '{label}'; ...")
    a("};")
    a("edgeInputs = [ ...")
    for _, t, act, day, pref in EDGE_CASES:
        a(f"    {t:>6g} {act:>5g} {day:>6g} {pref:>5g}; ...")
    a("];")
    a("fprintf('\\nEdge-case sweep\\n');")
    a("fprintf('  %-32s %8s %8s %6s   %s\\n', 'scenario', 'hvac', 'dimmer', ...")
    a("        'rules', 'status');")
    a("nBad = 0;")
    a("for k = 1:size(edgeInputs, 1)")
    a("    [yk, dk] = evalAssistiveCareFIS(fis, edgeInputs(k, :));")
    a("    if any(isnan(yk))")
    a("        status = 'UNDEFINED OUTPUT';")
    a("        nBad = nBad + 1;")
    a("    else")
    a("        status = 'ok';")
    a("    end")
    a("    fprintf('  %-32s %8.2f %8.2f %6d   %s\\n', edgeNames{k}, ...")
    a("            yk(1), yk(2), numel(dk.active), status);")
    a("end")
    a("fprintf('  %d of %d edge cases undefined\\n', nBad, size(edgeInputs, 1));")
    a("")
    a("%% Rule-base coverage check")
    a("%  A coarse sweep of the whole input space, looking for the two defects")
    a("%  the design claims are absent: an input state that activates no rule,")
    a("%  and a rule that never fires anywhere.")
    a("nGrid = [11 7 7 7];")
    a("g = cell(1, nIn);")
    a("for i = 1:nIn")
    a("    g{i} = linspace(fis.inputs(i).range(1), fis.inputs(i).range(2), ...")
    a("                    nGrid(i));")
    a("end")
    a("everFired    = false(nRules, 1);")
    a("nUndefined   = 0;")
    a("nStates      = 0;")
    a("nActiveTotal = 0;")
    a("maxActive    = 0;")
    a("for i1 = 1:nGrid(1)")
    a("  for i2 = 1:nGrid(2)")
    a("    for i3 = 1:nGrid(3)")
    a("      for i4 = 1:nGrid(4)")
    a("        [yy, dd] = evalAssistiveCareFIS(fis, ...")
    a("            [g{1}(i1) g{2}(i2) g{3}(i3) g{4}(i4)]);")
    a("        nStates      = nStates + 1;")
    a("        nActiveTotal = nActiveTotal + numel(dd.active);")
    a("        maxActive    = max(maxActive, numel(dd.active));")
    a("        if any(isnan(yy)); nUndefined = nUndefined + 1; end")
    a("        everFired(dd.active) = true;")
    a("      end")
    a("    end")
    a("  end")
    a("end")
    a("fprintf('\\nRule-base coverage over %d grid states\\n', nStates);")
    a("fprintf('  states with no defined output : %d\\n', nUndefined);")
    a("fprintf('  rules that never fire         : %d\\n', sum(~everFired));")
    a("fprintf('  rules active per state, mean  : %.2f\\n', ...")
    a("        nActiveTotal / nStates);")
    a("fprintf('  rules active per state, max   : %d\\n', maxActive);")
    a("")
    a("%% ---------------------------------------------------------------")
    a("%  FIGURE 1 - membership functions for all six variables")
    a("%  The Membership Function Editor view, drawn with base MATLAB. Each")
    a("%  set is sampled on its variable's own physical axis via acEvalMF.")
    a("%% ---------------------------------------------------------------")
    a("f1 = figure('Name', 'Membership functions', 'Color', 'w', ...")
    a("            'Position', [60 60 900 950]);")
    a("%  fis.inputs and fis.outputs are NOT concatenated into one struct array:")
    a("%  the outputs carry the cached .universe and .mfMatrix fields that the")
    a("%  inputs do not, and struct arrays require matching field sets. They are")
    a("%  therefore walked separately into a shared subplot grid.")
    a("panel = 0;")
    a("for role = 1:2")
    a("    if role == 1")
    a("        nVarsInRole = nIn;  roleName = 'input';")
    a("    else")
    a("        nVarsInRole = nOut; roleName = 'output';")
    a("    end")
    a("    for v = 1:nVarsInRole")
    a("        if role == 1")
    a("            vr = fis.inputs(v);")
    a("        else")
    a("            vr = fis.outputs(v);")
    a("        end")
    a("        panel = panel + 1;")
    a(f"        subplot({n_vars}, 1, panel);")
    a("        hold on;")
    a("        u = linspace(vr.range(1), vr.range(2), 501);")
    a("        labels = cell(1, numel(vr.mfs));")
    a("        for m = 1:numel(vr.mfs)")
    a("            plot(u, acEvalMF(vr.mfs(m).kind, vr.mfs(m).params, u), ...")
    a("                 'LineWidth', 1.4);")
    a("            labels{m} = vr.mfs(m).name;")
    a("        end")
    a("        xlim(vr.range); ylim([-0.05 1.15]); grid on; box on;")
    a("        ylabel('\\mu');")
    a("        xlabel(sprintf('%s  [%s]', strrep(vr.name, '_', '\\_'), vr.unit));")
    a("        title(sprintf('%s %d:  %s', roleName, v, ...")
    a("              strrep(vr.name, '_', '\\_')), 'FontWeight', 'normal');")
    a("        legend(labels, 'Location', 'eastoutside');")
    a("        hold off;")
    a("    end")
    a("end")
    a("acExportFig(f1, fullfile(outDir, 'matlab_fig1_membership.png'));")
    a("")
    a("%% ---------------------------------------------------------------")
    a("%  FIGURE 2 - rule inference for the worked scenario")
    a("%  The Rule Viewer view, rebuilt from the engine's own intermediate")
    a("%  quantities: which rules fired, and the aggregated set that each")
    a("%  output was defuzzified from.")
    a("%% ---------------------------------------------------------------")
    a("f2 = figure('Name', 'Rule inference', 'Color', 'w', ...")
    a("            'Position', [80 80 1000 420]);")
    a("")
    a(f"subplot(1, {n_out + 1}, 1);")
    a("bar(d.active, d.firing(d.active), 0.6);")
    a("hold on;")
    a("for k = 1:numel(d.active)")
    a("    text(d.active(k), d.firing(d.active(k)) + 0.04, ...")
    a("         sprintf('R%d', d.active(k)), 'HorizontalAlignment', 'center', ...")
    a("         'FontSize', 8);")
    a("end")
    a("ylim([0 1.2]); grid on; box on;")
    a("xlabel('rule index'); ylabel('firing strength \\alpha');")
    a("title(sprintf('%d of %d rules fire', numel(d.active), nRules), ...")
    a("      'FontWeight', 'normal');")
    a("hold off;")
    a("")
    a("for o = 1:nOut")
    a(f"    subplot(1, {n_out + 1}, o + 1);")
    a("    u   = d.universe{o};")
    a("    agg = d.aggregated{o};")
    a("    fill([u(1) u u(end)], [0 agg 0], [0.00 0.45 0.70], ...")
    a("         'FaceAlpha', 0.25, 'EdgeColor', [0.00 0.45 0.70], ...")
    a("         'LineWidth', 1.3);")
    a("    hold on;")
    a("    plot([y(o) y(o)], [0 1.1], '-', 'Color', [0.84 0.37 0.00], ...")
    a("         'LineWidth', 1.8);")
    a("    text(y(o), 1.14, sprintf('centroid = %.2f', y(o)), ...")
    a("         'HorizontalAlignment', 'center', 'FontSize', 8, ...")
    a("         'Color', [0.84 0.37 0.00]);")
    a("    xlim(fis.outputs(o).range); ylim([0 1.3]); grid on; box on;")
    a("    xlabel(sprintf('%s  [%s]', strrep(fis.outputs(o).name, '_', '\\_'), ...")
    a("           fis.outputs(o).unit));")
    a("    ylabel('\\mu');")
    a("    title(sprintf('aggregated set -> %s', ...")
    a("          strrep(fis.outputs(o).name, '_', '\\_')), 'FontWeight', 'normal');")
    a("    hold off;")
    a("end")
    a("acExportFig(f2, fullfile(outDir, 'matlab_fig2_ruleinference.png'));")
    a("")
    a("%% ---------------------------------------------------------------")
    a("%  FIGURE 3 - control surfaces")
    a("%  The Surface Viewer view. gensurf is a Toolbox function, so the input")
    a("%  grid is swept explicitly and handed to base MATLAB's surf.")
    a("%% ---------------------------------------------------------------")
    a("f3 = figure('Name', 'Control surfaces', 'Color', 'w', ...")
    a("            'Position', [100 100 1000 420]);")
    a("nSurf = 41;")
    a("")
    a("%  hvac against room_temp x activity, at mid daylight and neutral")
    a("%  preference (the two variables held fixed do not drive hvac and")
    a("%  daylight, respectively).")
    a("subplot(1, 2, 1);")
    a("tGrid = linspace(fis.inputs(1).range(1), fis.inputs(1).range(2), nSurf);")
    a("aGrid = linspace(fis.inputs(2).range(1), fis.inputs(2).range(2), nSurf);")
    a("Zh = zeros(nSurf, nSurf);")
    a("for ii = 1:nSurf")
    a("    for jj = 1:nSurf")
    a("        yy = evalAssistiveCareFIS(fis, [tGrid(jj) aGrid(ii) 50 0]);")
    a("        Zh(ii, jj) = yy(1);")
    a("    end")
    a("end")
    a("surf(tGrid, aGrid, Zh);")
    a("xlabel('room temp [degC]'); ylabel('activity [index]');")
    a("zlabel('hvac [% capacity]');")
    a("title('hvac  vs  room\\_temp x activity', 'FontWeight', 'normal');")
    a("view(-37.5, 30); shading interp; grid on; box on; colorbar;")
    a("")
    a("%  dimmer against daylight x activity")
    a("subplot(1, 2, 2);")
    a("dGrid = linspace(fis.inputs(3).range(1), fis.inputs(3).range(2), nSurf);")
    a("Zd = zeros(nSurf, nSurf);")
    a("for ii = 1:nSurf")
    a("    for jj = 1:nSurf")
    a("        yy = evalAssistiveCareFIS(fis, [22 aGrid(ii) dGrid(jj) 0]);")
    a("        Zd(ii, jj) = yy(2);")
    a("    end")
    a("end")
    a("surf(dGrid, aGrid, Zd);")
    a("xlabel('daylight [%]'); ylabel('activity [index]');")
    a("zlabel('dimmer [%]');")
    a("title('dimmer  vs  daylight x activity', 'FontWeight', 'normal');")
    a("view(-37.5, 30); shading interp; grid on; box on; colorbar;")
    a("acExportFig(f3, fullfile(outDir, 'matlab_fig3_surfaces.png'));")
    a("")
    a("fprintf('\\nWrote three figures to %s\\n', outDir);")
    a("fprintf('  matlab_fig1_membership.png\\n');")
    a("fprintf('  matlab_fig2_ruleinference.png\\n');")
    a("fprintf('  matlab_fig3_surfaces.png\\n');")
    a("")
    a("if allPass && nBad == 0 && nUndefined == 0 && sum(~everFired) == 0")
    a("    fprintf('\\nALL CHECKS PASSED\\n');")
    a("else")
    a("    fprintf('\\nSOME CHECKS FAILED - see above\\n');")
    a("end")
    return "\n".join(L) + "\n"


def emit_export_helper() -> str:
    """`acExportFig.m`: save a figure across MATLAB and Octave versions."""
    return "\n".join([
        "function acExportFig(figHandle, outPath)",
        "%ACEXPORTFIG  Save a figure to PNG, portably.",
        "%",
        "%   exportgraphics arrived in MATLAB R2020a and does not exist in",
        "%   Octave, so print is used as the fallback. Both are base",
        "%   functionality; no toolbox is involved either way.",
        "%",
        "%   GENERATED FILE - emitted by scripts/export_matlab.py.",
        "",
        "if exist('exportgraphics', 'file')",
        "    exportgraphics(figHandle, outPath, 'Resolution', 200);",
        "else",
        "    print(figHandle, outPath, '-dpng', '-r200');",
        "end",
        "end",
    ]) + "\n"


def main():
    flc = build_controller()
    expected = flc(**WORKED)

    missing = [f for f in ENGINE_FILES if not (OUT / f).exists()]
    if missing:
        raise SystemExit(
            f"missing hand-written engine file(s) in {OUT}: {missing}\n"
            "These are not generated; they are checked in alongside this script."
        )

    files = {
        "buildAssistiveCareFIS.m": emit_builder(flc),
        "runAssistiveCareFIS.m": emit_runner(flc, expected),
        "acExportFig.m": emit_export_helper(),
    }
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8")

    print(f"wrote matlab/buildAssistiveCareFIS.m  ({len(flc.rules)} rules, "
          f"{len(flc.input_names)} inputs, {len(flc.output_names)} outputs)")
    print("wrote matlab/runAssistiveCareFIS.m")
    print("wrote matlab/acExportFig.m")
    print("hand-written engine (not generated): " + ", ".join(ENGINE_FILES))
    print("\nNo Fuzzy Logic Toolbox function is emitted anywhere.")
    print("\nIn MATLAB Online or Octave: upload matlab/, then run")
    print("  runAssistiveCareFIS")
    print("\nThe script cross-checks itself against these Python values:")
    for k, v in expected.items():
        print(f"  {k:<7} = {v:.6f}")


if __name__ == "__main__":
    main()
