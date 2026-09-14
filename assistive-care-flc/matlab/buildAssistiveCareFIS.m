function fis = buildAssistiveCareFIS()
%BUILDASSISTIVECAREFIS  Mamdani FLC for one room of an assistive-care flat.
%
%   fis = BUILDASSISTIVECAREFIS() returns the fuzzy inference system used
%   in Task 2 Part 1: 4 sensor inputs, 2 actuator outputs, 54 rules.
%
%   NO FUZZY LOGIC TOOLBOX REQUIRED. The system is a plain struct, not a
%   mamfis object, and is evaluated by EVALASSISTIVECAREFIS, which
%   implements Mamdani inference in base MATLAB. This is a deliberate
%   substitution: the MATLAB Online licence available for this project
%   reports license('test','Fuzzy_Toolbox') == 0, so mamfis, addInput,
%   addMF, addOutput, addRule, evalfis, plotmf and gensurf are all
%   unavailable. The controller itself is unchanged.
%
%   GENERATED FILE - do not edit by hand. Emitted from the Python
%   definition in src/acflc/flat.py by scripts/export_matlab.py, so the
%   MATLAB and Python systems cannot drift apart. Re-run that script
%   after any change to the controller.
%
%   Struct layout
%     fis.inputs(i).name  .unit  .range  .mfs(k).name .kind .params
%     fis.outputs(o).name .unit  .range  .mfs(k).name .kind .params
%                         .universe  .mfMatrix         (cached below)
%     fis.ruleAnt         nRules x nInputs,  0 = variable not used
%     fis.ruleCons        nRules x nOutputs, 0 = output not driven
%     fis.ruleWeight      nRules x 1
%     fis.ruleText        nRules x 1 cell, the readable rule
%
%   See also EVALASSISTIVECAREFIS, ACEVALMF, RUNASSISTIVECAREFIS.

fis = struct();
fis.name = 'AssistiveCareRoomFLC';

%% Inference configuration
%  Unchanged from the Toolbox version, and from src/acflc/controller.py.
fis.andMethod             = 'min';    % AND across a rule's antecedents
fis.orMethod              = 'max';    % unused: every rule here is pure AND
fis.implicationMethod     = 'min';    % clip the consequent at alpha
fis.aggregationMethod     = 'max';    % union the clipped sets
fis.defuzzificationMethod = 'centroid';

%  Output universe resolution. This is the authoritative value, set by
%  Variable.n_points in src/acflc/membership.py. The Toolbox version had
%  to override MATLAB's default of 101 to match it; here it is simply the
%  definition. Changing it shifts the defuzzified outputs slightly and
%  breaks comparability with the Python reference.
fis.nPoints = 501;

fis.inputNames  = {'room_temp', 'activity', 'daylight', 'preference'};
fis.outputNames = {'hvac', 'dimmer'};

%% Input: room_temp  [degC]
fis.inputs(1).name  = 'room_temp';
fis.inputs(1).unit  = 'degC';
fis.inputs(1).range = [14 34];
fis.inputs(1).mfs(1).name   = 'Cold';
fis.inputs(1).mfs(1).kind   = 'trapmf';
fis.inputs(1).mfs(1).params = [14 14 16 19];
fis.inputs(1).mfs(2).name   = 'Cool';
fis.inputs(1).mfs(2).kind   = 'trimf';
fis.inputs(1).mfs(2).params = [17 19.5 22];
fis.inputs(1).mfs(3).name   = 'Comfortable';
fis.inputs(1).mfs(3).kind   = 'trimf';
fis.inputs(1).mfs(3).params = [20.5 22.5 24.5];
fis.inputs(1).mfs(4).name   = 'Warm';
fis.inputs(1).mfs(4).kind   = 'trimf';
fis.inputs(1).mfs(4).params = [23 25.5 28];
fis.inputs(1).mfs(5).name   = 'Hot';
fis.inputs(1).mfs(5).kind   = 'trapmf';
fis.inputs(1).mfs(5).params = [26 29 34 34];

%% Input: activity  [index]
fis.inputs(2).name  = 'activity';
fis.inputs(2).unit  = 'index';
fis.inputs(2).range = [0 10];
fis.inputs(2).mfs(1).name   = 'Resting';
fis.inputs(2).mfs(1).kind   = 'trapmf';
fis.inputs(2).mfs(1).params = [0 0 1.5 3.5];
fis.inputs(2).mfs(2).name   = 'Light';
fis.inputs(2).mfs(2).kind   = 'trimf';
fis.inputs(2).mfs(2).params = [2.5 5 7.5];
fis.inputs(2).mfs(3).name   = 'Active';
fis.inputs(2).mfs(3).kind   = 'trapmf';
fis.inputs(2).mfs(3).params = [6.5 8.5 10 10];

%% Input: daylight  [% of design lux]
fis.inputs(3).name  = 'daylight';
fis.inputs(3).unit  = '% of design lux';
fis.inputs(3).range = [0 100];
fis.inputs(3).mfs(1).name   = 'Dark';
fis.inputs(3).mfs(1).kind   = 'trapmf';
fis.inputs(3).mfs(1).params = [0 0 10 30];
fis.inputs(3).mfs(2).name   = 'Dim';
fis.inputs(3).mfs(2).kind   = 'trimf';
fis.inputs(3).mfs(2).params = [20 45 70];
fis.inputs(3).mfs(3).name   = 'Bright';
fis.inputs(3).mfs(3).kind   = 'trapmf';
fis.inputs(3).mfs(3).params = [60 80 100 100];

%% Input: preference  [scale]
fis.inputs(4).name  = 'preference';
fis.inputs(4).unit  = 'scale';
fis.inputs(4).range = [-5 5];
fis.inputs(4).mfs(1).name   = 'Cooler';
fis.inputs(4).mfs(1).kind   = 'trapmf';
fis.inputs(4).mfs(1).params = [-5 -5 -3 -1];
fis.inputs(4).mfs(2).name   = 'Neutral';
fis.inputs(4).mfs(2).kind   = 'trimf';
fis.inputs(4).mfs(2).params = [-2 0 2];
fis.inputs(4).mfs(3).name   = 'Warmer';
fis.inputs(4).mfs(3).kind   = 'trapmf';
fis.inputs(4).mfs(3).params = [1 3 5 5];

%% Output: hvac  [% capacity (-cool/+heat)]
fis.outputs(1).name  = 'hvac';
fis.outputs(1).unit  = '% capacity (-cool/+heat)';
fis.outputs(1).range = [-100 100];
fis.outputs(1).mfs(1).name   = 'CoolHigh';
fis.outputs(1).mfs(1).kind   = 'trapmf';
fis.outputs(1).mfs(1).params = [-100 -100 -80 -45];
fis.outputs(1).mfs(2).name   = 'CoolLow';
fis.outputs(1).mfs(2).kind   = 'trimf';
fis.outputs(1).mfs(2).params = [-70 -35 0];
fis.outputs(1).mfs(3).name   = 'Off';
fis.outputs(1).mfs(3).kind   = 'trimf';
fis.outputs(1).mfs(3).params = [-15 0 15];
fis.outputs(1).mfs(4).name   = 'HeatLow';
fis.outputs(1).mfs(4).kind   = 'trimf';
fis.outputs(1).mfs(4).params = [0 35 70];
fis.outputs(1).mfs(5).name   = 'HeatHigh';
fis.outputs(1).mfs(5).kind   = 'trapmf';
fis.outputs(1).mfs(5).params = [45 80 100 100];

%% Output: dimmer  [%]
fis.outputs(2).name  = 'dimmer';
fis.outputs(2).unit  = '%';
fis.outputs(2).range = [0 100];
fis.outputs(2).mfs(1).name   = 'Off';
fis.outputs(2).mfs(1).kind   = 'trapmf';
fis.outputs(2).mfs(1).params = [0 0 5 20];
fis.outputs(2).mfs(2).name   = 'Low';
fis.outputs(2).mfs(2).kind   = 'trimf';
fis.outputs(2).mfs(2).params = [10 30 50];
fis.outputs(2).mfs(3).name   = 'Medium';
fis.outputs(2).mfs(3).kind   = 'trimf';
fis.outputs(2).mfs(3).params = [40 60 80];
fis.outputs(2).mfs(4).name   = 'High';
fis.outputs(2).mfs(4).kind   = 'trapmf';
fis.outputs(2).mfs(4).params = [70 88 100 100];

%% Rules
%  Columns: [room_temp activity daylight preference | hvac dimmer | weight]
%
%  A zero means the rule does not reference that variable: the lighting
%  rules constrain daylight and activity and say nothing about
%  temperature or preference. Silence means 'do not care', not 'zero'.
%
%  The Toolbox rule matrix carried an extra trailing connective column
%  (1 = AND, 2 = OR). It is dropped here because it was constant: every
%  rule in this controller is a pure conjunction, and the operator is
%  stated once in fis.andMethod above. No rule semantics change.
%
%  Generated from the additive FAM tables described in src/acflc/flat.py.
ruleList = [
     1  1  0  1  5  0  1   % R1: IF room_temp is Cold AND activity is Resting AND preference is Cooler THEN hvac is HeatHigh
     1  1  0  2  5  0  1   % R2: IF room_temp is Cold AND activity is Resting AND preference is Neutral THEN hvac is HeatHigh
     1  1  0  3  5  0  1   % R3: IF room_temp is Cold AND activity is Resting AND preference is Warmer THEN hvac is HeatHigh
     1  2  0  1  4  0  1   % R4: IF room_temp is Cold AND activity is Light AND preference is Cooler THEN hvac is HeatLow
     1  2  0  2  5  0  1   % R5: IF room_temp is Cold AND activity is Light AND preference is Neutral THEN hvac is HeatHigh
     1  2  0  3  5  0  1   % R6: IF room_temp is Cold AND activity is Light AND preference is Warmer THEN hvac is HeatHigh
     1  3  0  1  4  0  1   % R7: IF room_temp is Cold AND activity is Active AND preference is Cooler THEN hvac is HeatLow
     1  3  0  2  4  0  1   % R8: IF room_temp is Cold AND activity is Active AND preference is Neutral THEN hvac is HeatLow
     1  3  0  3  5  0  1   % R9: IF room_temp is Cold AND activity is Active AND preference is Warmer THEN hvac is HeatHigh
     2  1  0  1  4  0  1   % R10: IF room_temp is Cool AND activity is Resting AND preference is Cooler THEN hvac is HeatLow
     2  1  0  2  5  0  1   % R11: IF room_temp is Cool AND activity is Resting AND preference is Neutral THEN hvac is HeatHigh
     2  1  0  3  5  0  1   % R12: IF room_temp is Cool AND activity is Resting AND preference is Warmer THEN hvac is HeatHigh
     2  2  0  1  3  0  1   % R13: IF room_temp is Cool AND activity is Light AND preference is Cooler THEN hvac is Off
     2  2  0  2  4  0  1   % R14: IF room_temp is Cool AND activity is Light AND preference is Neutral THEN hvac is HeatLow
     2  2  0  3  5  0  1   % R15: IF room_temp is Cool AND activity is Light AND preference is Warmer THEN hvac is HeatHigh
     2  3  0  1  3  0  1   % R16: IF room_temp is Cool AND activity is Active AND preference is Cooler THEN hvac is Off
     2  3  0  2  3  0  1   % R17: IF room_temp is Cool AND activity is Active AND preference is Neutral THEN hvac is Off
     2  3  0  3  4  0  1   % R18: IF room_temp is Cool AND activity is Active AND preference is Warmer THEN hvac is HeatLow
     3  1  0  1  3  0  1   % R19: IF room_temp is Comfortable AND activity is Resting AND preference is Cooler THEN hvac is Off
     3  1  0  2  4  0  1   % R20: IF room_temp is Comfortable AND activity is Resting AND preference is Neutral THEN hvac is HeatLow
     3  1  0  3  4  0  1   % R21: IF room_temp is Comfortable AND activity is Resting AND preference is Warmer THEN hvac is HeatLow
     3  2  0  1  2  0  1   % R22: IF room_temp is Comfortable AND activity is Light AND preference is Cooler THEN hvac is CoolLow
     3  2  0  2  3  0  1   % R23: IF room_temp is Comfortable AND activity is Light AND preference is Neutral THEN hvac is Off
     3  2  0  3  4  0  1   % R24: IF room_temp is Comfortable AND activity is Light AND preference is Warmer THEN hvac is HeatLow
     3  3  0  1  2  0  1   % R25: IF room_temp is Comfortable AND activity is Active AND preference is Cooler THEN hvac is CoolLow
     3  3  0  2  2  0  1   % R26: IF room_temp is Comfortable AND activity is Active AND preference is Neutral THEN hvac is CoolLow
     3  3  0  3  3  0  1   % R27: IF room_temp is Comfortable AND activity is Active AND preference is Warmer THEN hvac is Off
     4  1  0  1  2  0  1   % R28: IF room_temp is Warm AND activity is Resting AND preference is Cooler THEN hvac is CoolLow
     4  1  0  2  3  0  1   % R29: IF room_temp is Warm AND activity is Resting AND preference is Neutral THEN hvac is Off
     4  1  0  3  3  0  1   % R30: IF room_temp is Warm AND activity is Resting AND preference is Warmer THEN hvac is Off
     4  2  0  1  1  0  1   % R31: IF room_temp is Warm AND activity is Light AND preference is Cooler THEN hvac is CoolHigh
     4  2  0  2  2  0  1   % R32: IF room_temp is Warm AND activity is Light AND preference is Neutral THEN hvac is CoolLow
     4  2  0  3  3  0  1   % R33: IF room_temp is Warm AND activity is Light AND preference is Warmer THEN hvac is Off
     4  3  0  1  1  0  1   % R34: IF room_temp is Warm AND activity is Active AND preference is Cooler THEN hvac is CoolHigh
     4  3  0  2  1  0  1   % R35: IF room_temp is Warm AND activity is Active AND preference is Neutral THEN hvac is CoolHigh
     4  3  0  3  2  0  1   % R36: IF room_temp is Warm AND activity is Active AND preference is Warmer THEN hvac is CoolLow
     5  1  0  1  1  0  1   % R37: IF room_temp is Hot AND activity is Resting AND preference is Cooler THEN hvac is CoolHigh
     5  1  0  2  2  0  1   % R38: IF room_temp is Hot AND activity is Resting AND preference is Neutral THEN hvac is CoolLow
     5  1  0  3  2  0  1   % R39: IF room_temp is Hot AND activity is Resting AND preference is Warmer THEN hvac is CoolLow
     5  2  0  1  1  0  1   % R40: IF room_temp is Hot AND activity is Light AND preference is Cooler THEN hvac is CoolHigh
     5  2  0  2  1  0  1   % R41: IF room_temp is Hot AND activity is Light AND preference is Neutral THEN hvac is CoolHigh
     5  2  0  3  2  0  1   % R42: IF room_temp is Hot AND activity is Light AND preference is Warmer THEN hvac is CoolLow
     5  3  0  1  1  0  1   % R43: IF room_temp is Hot AND activity is Active AND preference is Cooler THEN hvac is CoolHigh
     5  3  0  2  1  0  1   % R44: IF room_temp is Hot AND activity is Active AND preference is Neutral THEN hvac is CoolHigh
     5  3  0  3  1  0  1   % R45: IF room_temp is Hot AND activity is Active AND preference is Warmer THEN hvac is CoolHigh
     0  1  1  0  0  2  1   % R46: IF daylight is Dark AND activity is Resting THEN dimmer is Low
     0  2  1  0  0  3  1   % R47: IF daylight is Dark AND activity is Light THEN dimmer is Medium
     0  3  1  0  0  4  1   % R48: IF daylight is Dark AND activity is Active THEN dimmer is High
     0  1  2  0  0  2  1   % R49: IF daylight is Dim AND activity is Resting THEN dimmer is Low
     0  2  2  0  0  2  1   % R50: IF daylight is Dim AND activity is Light THEN dimmer is Low
     0  3  2  0  0  3  1   % R51: IF daylight is Dim AND activity is Active THEN dimmer is Medium
     0  1  3  0  0  1  1   % R52: IF daylight is Bright AND activity is Resting THEN dimmer is Off
     0  2  3  0  0  1  1   % R53: IF daylight is Bright AND activity is Light THEN dimmer is Off
     0  3  3  0  0  2  1   % R54: IF daylight is Bright AND activity is Active THEN dimmer is Low
];

fis.ruleAnt    = ruleList(:, 1:4);
fis.ruleCons   = ruleList(:, 5:6);
fis.ruleWeight = ruleList(:, 7);

fis.ruleText = { ...
    'IF room_temp is Cold AND activity is Resting AND preference is Cooler THEN hvac is HeatHigh'; ...
    'IF room_temp is Cold AND activity is Resting AND preference is Neutral THEN hvac is HeatHigh'; ...
    'IF room_temp is Cold AND activity is Resting AND preference is Warmer THEN hvac is HeatHigh'; ...
    'IF room_temp is Cold AND activity is Light AND preference is Cooler THEN hvac is HeatLow'; ...
    'IF room_temp is Cold AND activity is Light AND preference is Neutral THEN hvac is HeatHigh'; ...
    'IF room_temp is Cold AND activity is Light AND preference is Warmer THEN hvac is HeatHigh'; ...
    'IF room_temp is Cold AND activity is Active AND preference is Cooler THEN hvac is HeatLow'; ...
    'IF room_temp is Cold AND activity is Active AND preference is Neutral THEN hvac is HeatLow'; ...
    'IF room_temp is Cold AND activity is Active AND preference is Warmer THEN hvac is HeatHigh'; ...
    'IF room_temp is Cool AND activity is Resting AND preference is Cooler THEN hvac is HeatLow'; ...
    'IF room_temp is Cool AND activity is Resting AND preference is Neutral THEN hvac is HeatHigh'; ...
    'IF room_temp is Cool AND activity is Resting AND preference is Warmer THEN hvac is HeatHigh'; ...
    'IF room_temp is Cool AND activity is Light AND preference is Cooler THEN hvac is Off'; ...
    'IF room_temp is Cool AND activity is Light AND preference is Neutral THEN hvac is HeatLow'; ...
    'IF room_temp is Cool AND activity is Light AND preference is Warmer THEN hvac is HeatHigh'; ...
    'IF room_temp is Cool AND activity is Active AND preference is Cooler THEN hvac is Off'; ...
    'IF room_temp is Cool AND activity is Active AND preference is Neutral THEN hvac is Off'; ...
    'IF room_temp is Cool AND activity is Active AND preference is Warmer THEN hvac is HeatLow'; ...
    'IF room_temp is Comfortable AND activity is Resting AND preference is Cooler THEN hvac is Off'; ...
    'IF room_temp is Comfortable AND activity is Resting AND preference is Neutral THEN hvac is HeatLow'; ...
    'IF room_temp is Comfortable AND activity is Resting AND preference is Warmer THEN hvac is HeatLow'; ...
    'IF room_temp is Comfortable AND activity is Light AND preference is Cooler THEN hvac is CoolLow'; ...
    'IF room_temp is Comfortable AND activity is Light AND preference is Neutral THEN hvac is Off'; ...
    'IF room_temp is Comfortable AND activity is Light AND preference is Warmer THEN hvac is HeatLow'; ...
    'IF room_temp is Comfortable AND activity is Active AND preference is Cooler THEN hvac is CoolLow'; ...
    'IF room_temp is Comfortable AND activity is Active AND preference is Neutral THEN hvac is CoolLow'; ...
    'IF room_temp is Comfortable AND activity is Active AND preference is Warmer THEN hvac is Off'; ...
    'IF room_temp is Warm AND activity is Resting AND preference is Cooler THEN hvac is CoolLow'; ...
    'IF room_temp is Warm AND activity is Resting AND preference is Neutral THEN hvac is Off'; ...
    'IF room_temp is Warm AND activity is Resting AND preference is Warmer THEN hvac is Off'; ...
    'IF room_temp is Warm AND activity is Light AND preference is Cooler THEN hvac is CoolHigh'; ...
    'IF room_temp is Warm AND activity is Light AND preference is Neutral THEN hvac is CoolLow'; ...
    'IF room_temp is Warm AND activity is Light AND preference is Warmer THEN hvac is Off'; ...
    'IF room_temp is Warm AND activity is Active AND preference is Cooler THEN hvac is CoolHigh'; ...
    'IF room_temp is Warm AND activity is Active AND preference is Neutral THEN hvac is CoolHigh'; ...
    'IF room_temp is Warm AND activity is Active AND preference is Warmer THEN hvac is CoolLow'; ...
    'IF room_temp is Hot AND activity is Resting AND preference is Cooler THEN hvac is CoolHigh'; ...
    'IF room_temp is Hot AND activity is Resting AND preference is Neutral THEN hvac is CoolLow'; ...
    'IF room_temp is Hot AND activity is Resting AND preference is Warmer THEN hvac is CoolLow'; ...
    'IF room_temp is Hot AND activity is Light AND preference is Cooler THEN hvac is CoolHigh'; ...
    'IF room_temp is Hot AND activity is Light AND preference is Neutral THEN hvac is CoolHigh'; ...
    'IF room_temp is Hot AND activity is Light AND preference is Warmer THEN hvac is CoolLow'; ...
    'IF room_temp is Hot AND activity is Active AND preference is Cooler THEN hvac is CoolHigh'; ...
    'IF room_temp is Hot AND activity is Active AND preference is Neutral THEN hvac is CoolHigh'; ...
    'IF room_temp is Hot AND activity is Active AND preference is Warmer THEN hvac is CoolHigh'; ...
    'IF daylight is Dark AND activity is Resting THEN dimmer is Low'; ...
    'IF daylight is Dark AND activity is Light THEN dimmer is Medium'; ...
    'IF daylight is Dark AND activity is Active THEN dimmer is High'; ...
    'IF daylight is Dim AND activity is Resting THEN dimmer is Low'; ...
    'IF daylight is Dim AND activity is Light THEN dimmer is Low'; ...
    'IF daylight is Dim AND activity is Active THEN dimmer is Medium'; ...
    'IF daylight is Bright AND activity is Resting THEN dimmer is Off'; ...
    'IF daylight is Bright AND activity is Light THEN dimmer is Off'; ...
    'IF daylight is Bright AND activity is Active THEN dimmer is Low'; ...
};

%% Cache each output's universe and its sampled membership functions.
%  These are fixed for a given parameter set and are touched on every
%  inference, so they are computed once here rather than per call. This
%  mirrors FuzzyController._compile() in src/acflc/controller.py.
for k = 1:numel(fis.outputs)
    u = linspace(fis.outputs(k).range(1), fis.outputs(k).range(2), ...
                 fis.nPoints);
    M = zeros(numel(fis.outputs(k).mfs), fis.nPoints);
    for m = 1:numel(fis.outputs(k).mfs)
        M(m, :) = acEvalMF(fis.outputs(k).mfs(m).kind, ...
                           fis.outputs(k).mfs(m).params, u);
    end
    fis.outputs(k).universe = u;
    fis.outputs(k).mfMatrix = M;
end

end
