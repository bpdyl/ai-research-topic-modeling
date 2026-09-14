%% Assistive-care flat FLC - Part 1 evidence script
%
%  Builds the controller, verifies it against the Python reference
%  implementation, sweeps a set of edge cases, checks the rule base for
%  holes and dead rules, and exports the three evidence figures.
%
%  NO FUZZY LOGIC TOOLBOX REQUIRED. Nothing here calls mamfis, addMF,
%  addRule, evalfis, plotmf or gensurf. Inference is done by
%  evalAssistiveCareFIS and the figures are drawn with base MATLAB
%  plotting, because license('test','Fuzzy_Toolbox') returns 0 on the
%  MATLAB Online licence available for this project.
%
%  GENERATED FILE - emitted by scripts/export_matlab.py. Do not edit by
%  hand; edit src/acflc/flat.py and re-run the exporter.
%
%  HOW TO RUN (MATLAB Online, https://matlab.mathworks.com, or Octave):
%    1. Upload the whole matlab/ folder.
%    2. In the Command Window type:  runAssistiveCareFIS
%    3. Screenshot the Command Window output, and collect the three PNGs
%       written to the figures/ folder beside this script.

clear; clc; close all;

fis = buildAssistiveCareFIS();

outDir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end

nIn    = numel(fis.inputs);
nOut   = numel(fis.outputs);
nRules = size(fis.ruleAnt, 1);

fprintf('==============================================================\n');
fprintf(' ASSISTIVE-CARE FLAT FLC - base MATLAB, no Fuzzy Logic Toolbox\n');
fprintf('==============================================================\n');
fprintf('interpreter      : %s\n', version());
fprintf('mamfis available : %d  (not required either way)\n', ...
        ~isempty(which('mamfis')));
fprintf('inputs           : %d\n', nIn);
fprintf('outputs          : %d\n', nOut);
fprintf('rules            : %d\n', nRules);
fprintf('and / implication: %s / %s\n', fis.andMethod, ...
        fis.implicationMethod);
fprintf('aggregation      : %s\n', fis.aggregationMethod);
fprintf('defuzzification  : %s\n', fis.defuzzificationMethod);
fprintf('output samples   : %d\n', fis.nPoints);

%% Worked scenario: room_temp = 21, activity = 7.5, daylight = 25, preference = 0
%  Input order: [room_temp, activity, daylight, preference]
x = [21 7.5 25 0];

[y, d] = evalAssistiveCareFIS(fis, x);

fprintf('\nWorked scenario\n');
fprintf('  room_temp   = %g\n', x(1));
fprintf('  activity    = %g\n', x(2));
fprintf('  daylight    = %g\n', x(3));
fprintf('  preference  = %g\n', x(4));

fprintf('\nStage 1 - fuzzified inputs (non-zero memberships)\n');
for i = 1:nIn
    for k = 1:numel(fis.inputs(i).mfs)
        if d.mu{i}(k) > 0
            fprintf('  %-11s %-12s %.4f\n', fis.inputs(i).name, ...
                    fis.inputs(i).mfs(k).name, d.mu{i}(k));
        end
    end
end

fprintf('\nStage 2 - active rules (firing strength > 0)\n');
for k = 1:numel(d.active)
    r = d.active(k);
    fprintf('  R%-3d  alpha = %.4f   %s\n', r, d.firing(r), ...
            fis.ruleText{r});
end
fprintf('  %d of %d rules active\n', numel(d.active), nRules);

fprintf('\nStages 3-5 - defuzzified outputs (centroid)\n');
fprintf('  hvac    = %.6f\n', y(1));
fprintf('  dimmer  = %.6f\n', y(2));

%% Cross-check against the Python reference implementation
%  These constants are emitted by scripts/export_matlab.py from the very
%  same controller object, so they are not transcribed by hand and cannot
%  be quietly adjusted to make a failing check pass.
pythonOutputs = [-22.769724, 72.601620];
names = {'hvac', 'dimmer'};
tol = 1e-3;
fprintf('\nCross-check against the Python engine (tolerance %.0e)\n', tol);
allPass = true;
for k = 1:numel(names)
    diffK = abs(y(k) - pythonOutputs(k));
    ok = diffK < tol;
    allPass = allPass && ok;
    verdict = 'FAIL';
    if ok; verdict = 'PASS'; end
    fprintf('  %-7s MATLAB %10.6f   Python %10.6f   diff %.2e   %s\n', ...
            names{k}, y(k), pythonOutputs(k), diffK, verdict);
end
if allPass
    fprintf('  ==> AGREE: two independent engines, same controller.\n');
else
    fprintf('  ==> DISAGREE: investigate before quoting either.\n');
end

%% Edge-case sweep
%  Representative corners of the design space. The check is not that any
%  particular number is right -- the control surfaces say that -- but that
%  every state produces a defined command. A NaN here would mean some
%  input activates no rule at all, i.e. a hole in the rule base.
edgeNames = { ...
    'heating failure, resting, night'; ...
    'very cold, active, warmer pref'; ...
    'comfortable, light activity'; ...
    'heatwave, resting, cooler pref'; ...
    'hot, active, neutral'; ...
    'dark room, active occupant'; ...
    'bright room, resting occupant'; ...
    'universe corner, all low'; ...
    'universe corner, all high'; ...
};
edgeInputs = [ ...
        14     0      0     0; ...
      14.5   9.5     10     4; ...
      22.5     5     50     0; ...
        34     0    100    -5; ...
        31     9     80     0; ...
        22     9      0     0; ...
        22     0    100     0; ...
        14     0      0    -5; ...
        34    10    100     5; ...
];
fprintf('\nEdge-case sweep\n');
fprintf('  %-32s %8s %8s %6s   %s\n', 'scenario', 'hvac', 'dimmer', ...
        'rules', 'status');
nBad = 0;
for k = 1:size(edgeInputs, 1)
    [yk, dk] = evalAssistiveCareFIS(fis, edgeInputs(k, :));
    if any(isnan(yk))
        status = 'UNDEFINED OUTPUT';
        nBad = nBad + 1;
    else
        status = 'ok';
    end
    fprintf('  %-32s %8.2f %8.2f %6d   %s\n', edgeNames{k}, ...
            yk(1), yk(2), numel(dk.active), status);
end
fprintf('  %d of %d edge cases undefined\n', nBad, size(edgeInputs, 1));

%% Rule-base coverage check
%  A coarse sweep of the whole input space, looking for the two defects
%  the design claims are absent: an input state that activates no rule,
%  and a rule that never fires anywhere.
nGrid = [11 7 7 7];
g = cell(1, nIn);
for i = 1:nIn
    g{i} = linspace(fis.inputs(i).range(1), fis.inputs(i).range(2), ...
                    nGrid(i));
end
everFired    = false(nRules, 1);
nUndefined   = 0;
nStates      = 0;
nActiveTotal = 0;
maxActive    = 0;
for i1 = 1:nGrid(1)
  for i2 = 1:nGrid(2)
    for i3 = 1:nGrid(3)
      for i4 = 1:nGrid(4)
        [yy, dd] = evalAssistiveCareFIS(fis, ...
            [g{1}(i1) g{2}(i2) g{3}(i3) g{4}(i4)]);
        nStates      = nStates + 1;
        nActiveTotal = nActiveTotal + numel(dd.active);
        maxActive    = max(maxActive, numel(dd.active));
        if any(isnan(yy)); nUndefined = nUndefined + 1; end
        everFired(dd.active) = true;
      end
    end
  end
end
fprintf('\nRule-base coverage over %d grid states\n', nStates);
fprintf('  states with no defined output : %d\n', nUndefined);
fprintf('  rules that never fire         : %d\n', sum(~everFired));
fprintf('  rules active per state, mean  : %.2f\n', ...
        nActiveTotal / nStates);
fprintf('  rules active per state, max   : %d\n', maxActive);

%% ---------------------------------------------------------------
%  FIGURE 1 - membership functions for all six variables
%  The Membership Function Editor view, drawn with base MATLAB. Each
%  set is sampled on its variable's own physical axis via acEvalMF.
%% ---------------------------------------------------------------
f1 = figure('Name', 'Membership functions', 'Color', 'w', ...
            'Position', [60 60 900 950]);
%  fis.inputs and fis.outputs are NOT concatenated into one struct array:
%  the outputs carry the cached .universe and .mfMatrix fields that the
%  inputs do not, and struct arrays require matching field sets. They are
%  therefore walked separately into a shared subplot grid.
panel = 0;
for role = 1:2
    if role == 1
        nVarsInRole = nIn;  roleName = 'input';
    else
        nVarsInRole = nOut; roleName = 'output';
    end
    for v = 1:nVarsInRole
        if role == 1
            vr = fis.inputs(v);
        else
            vr = fis.outputs(v);
        end
        panel = panel + 1;
        subplot(6, 1, panel);
        hold on;
        u = linspace(vr.range(1), vr.range(2), 501);
        labels = cell(1, numel(vr.mfs));
        for m = 1:numel(vr.mfs)
            plot(u, acEvalMF(vr.mfs(m).kind, vr.mfs(m).params, u), ...
                 'LineWidth', 1.4);
            labels{m} = vr.mfs(m).name;
        end
        xlim(vr.range); ylim([-0.05 1.15]); grid on; box on;
        ylabel('\mu');
        xlabel(sprintf('%s  [%s]', strrep(vr.name, '_', '\_'), vr.unit));
        title(sprintf('%s %d:  %s', roleName, v, ...
              strrep(vr.name, '_', '\_')), 'FontWeight', 'normal');
        legend(labels, 'Location', 'eastoutside');
        hold off;
    end
end
acExportFig(f1, fullfile(outDir, 'matlab_fig1_membership.png'));

%% ---------------------------------------------------------------
%  FIGURE 2 - rule inference for the worked scenario
%  The Rule Viewer view, rebuilt from the engine's own intermediate
%  quantities: which rules fired, and the aggregated set that each
%  output was defuzzified from.
%% ---------------------------------------------------------------
f2 = figure('Name', 'Rule inference', 'Color', 'w', ...
            'Position', [80 80 1000 420]);

subplot(1, 3, 1);
bar(d.active, d.firing(d.active), 0.6);
hold on;
for k = 1:numel(d.active)
    text(d.active(k), d.firing(d.active(k)) + 0.04, ...
         sprintf('R%d', d.active(k)), 'HorizontalAlignment', 'center', ...
         'FontSize', 8);
end
ylim([0 1.2]); grid on; box on;
xlabel('rule index'); ylabel('firing strength \alpha');
title(sprintf('%d of %d rules fire', numel(d.active), nRules), ...
      'FontWeight', 'normal');
hold off;

for o = 1:nOut
    subplot(1, 3, o + 1);
    u   = d.universe{o};
    agg = d.aggregated{o};
    fill([u(1) u u(end)], [0 agg 0], [0.00 0.45 0.70], ...
         'FaceAlpha', 0.25, 'EdgeColor', [0.00 0.45 0.70], ...
         'LineWidth', 1.3);
    hold on;
    plot([y(o) y(o)], [0 1.1], '-', 'Color', [0.84 0.37 0.00], ...
         'LineWidth', 1.8);
    text(y(o), 1.14, sprintf('centroid = %.2f', y(o)), ...
         'HorizontalAlignment', 'center', 'FontSize', 8, ...
         'Color', [0.84 0.37 0.00]);
    xlim(fis.outputs(o).range); ylim([0 1.3]); grid on; box on;
    xlabel(sprintf('%s  [%s]', strrep(fis.outputs(o).name, '_', '\_'), ...
           fis.outputs(o).unit));
    ylabel('\mu');
    title(sprintf('aggregated set -> %s', ...
          strrep(fis.outputs(o).name, '_', '\_')), 'FontWeight', 'normal');
    hold off;
end
acExportFig(f2, fullfile(outDir, 'matlab_fig2_ruleinference.png'));

%% ---------------------------------------------------------------
%  FIGURE 3 - control surfaces
%  The Surface Viewer view. gensurf is a Toolbox function, so the input
%  grid is swept explicitly and handed to base MATLAB's surf.
%% ---------------------------------------------------------------
f3 = figure('Name', 'Control surfaces', 'Color', 'w', ...
            'Position', [100 100 1000 420]);
nSurf = 41;

%  hvac against room_temp x activity, at mid daylight and neutral
%  preference (the two variables held fixed do not drive hvac and
%  daylight, respectively).
subplot(1, 2, 1);
tGrid = linspace(fis.inputs(1).range(1), fis.inputs(1).range(2), nSurf);
aGrid = linspace(fis.inputs(2).range(1), fis.inputs(2).range(2), nSurf);
Zh = zeros(nSurf, nSurf);
for ii = 1:nSurf
    for jj = 1:nSurf
        yy = evalAssistiveCareFIS(fis, [tGrid(jj) aGrid(ii) 50 0]);
        Zh(ii, jj) = yy(1);
    end
end
surf(tGrid, aGrid, Zh);
xlabel('room temp [degC]'); ylabel('activity [index]');
zlabel('hvac [% capacity]');
title('hvac  vs  room\_temp x activity', 'FontWeight', 'normal');
view(-37.5, 30); shading interp; grid on; box on; colorbar;

%  dimmer against daylight x activity
subplot(1, 2, 2);
dGrid = linspace(fis.inputs(3).range(1), fis.inputs(3).range(2), nSurf);
Zd = zeros(nSurf, nSurf);
for ii = 1:nSurf
    for jj = 1:nSurf
        yy = evalAssistiveCareFIS(fis, [22 aGrid(ii) dGrid(jj) 0]);
        Zd(ii, jj) = yy(2);
    end
end
surf(dGrid, aGrid, Zd);
xlabel('daylight [%]'); ylabel('activity [index]');
zlabel('dimmer [%]');
title('dimmer  vs  daylight x activity', 'FontWeight', 'normal');
view(-37.5, 30); shading interp; grid on; box on; colorbar;
acExportFig(f3, fullfile(outDir, 'matlab_fig3_surfaces.png'));

fprintf('\nWrote three figures to %s\n', outDir);
fprintf('  matlab_fig1_membership.png\n');
fprintf('  matlab_fig2_ruleinference.png\n');
fprintf('  matlab_fig3_surfaces.png\n');

if allPass && nBad == 0 && nUndefined == 0 && sum(~everFired) == 0
    fprintf('\nALL CHECKS PASSED\n');
else
    fprintf('\nSOME CHECKS FAILED - see above\n');
end
