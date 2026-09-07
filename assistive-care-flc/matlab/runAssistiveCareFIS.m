%% Assistive-care flat FLC - Part 1 evidence script
%
%  Builds the controller, evaluates the worked scenario, and opens the
%  five Fuzzy Logic Toolbox views that Part 1 asks for evidence of.
%
%  GENERATED FILE - emitted by scripts/export_matlab.py.
%
%  Cross-check: the printed values below must match
%  results/part1_analysis.json from the Python implementation. They are
%  two independent evaluations of the same system, so agreement is
%  evidence and disagreement is a defect.

clear; clc; close all;

fis = buildAssistiveCareFIS();

%% Worked scenario: room_temp = 21, activity = 7.5, daylight = 25, preference = 0
%  Input order: [room_temp, activity, daylight, preference]
x = [21 7.5 25 0];

[y, ~, ~, ~, ruleFiring] = evalfis(fis, x);

fprintf('Worked scenario\n');
fprintf('  room_temp   = %g\n', x(1));
fprintf('  activity    = %g\n', x(2));
fprintf('  daylight    = %g\n', x(3));
fprintf('  preference  = %g\n', x(4));
fprintf('\nDefuzzified outputs (centroid)\n');
fprintf('  hvac    = %.4f\n', y(1));
fprintf('  dimmer  = %.4f\n', y(2));

fprintf('\nActive rules (firing strength > 0)\n');
active = find(ruleFiring > 0);
for k = 1:numel(active)
    r = active(k);
    fprintf('  R%-3d  alpha = %.4f\n', r, ruleFiring(r));
end

%% Component views - screenshot each of these
% 1. Membership functions for every variable
figure('Name', 'Input membership functions', 'Position', [80 80 900 620]);
subplot(4, 1, 1);
plotmf(fis, 'input', 1); title('room_temp'); ylabel('\mu');
subplot(4, 1, 2);
plotmf(fis, 'input', 2); title('activity'); ylabel('\mu');
subplot(4, 1, 3);
plotmf(fis, 'input', 3); title('daylight'); ylabel('\mu');
subplot(4, 1, 4);
plotmf(fis, 'input', 4); title('preference'); ylabel('\mu');

figure('Name', 'Output membership functions', 'Position', [120 100 900 400]);
subplot(2, 1, 1);
plotmf(fis, 'output', 1); title('hvac'); ylabel('\mu');
subplot(2, 1, 2);
plotmf(fis, 'output', 2); title('dimmer'); ylabel('\mu');

% 2. Control surfaces
figure('Name', 'Surface: hvac vs room_temp and activity');
gensurf(fis, [1 2], 1);
xlabel('room temp [degC]'); ylabel('activity [index]'); zlabel('hvac [% capacity]');

figure('Name', 'Surface: dimmer vs daylight and activity');
gensurf(fis, [3 2], 2);
xlabel('daylight [%]'); ylabel('activity [index]'); zlabel('dimmer [%]');

% 3. Interactive views - screenshot these from the GUI
fuzzy(fis)      % FIS Editor, MF Editor and Rule Editor
ruleview(fis)   % Rule Viewer - set the inputs to the scenario above
surfview(fis)   % Surface Viewer

% Optional: save the system so it can be attached to the report appendix
% writeFIS(fis, 'assistive_care_room');
