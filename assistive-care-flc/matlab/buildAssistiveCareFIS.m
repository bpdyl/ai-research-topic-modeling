function fis = buildAssistiveCareFIS()
%BUILDASSISTIVECAREFIS  Mamdani FLC for one room of an assistive-care flat.
%
%   fis = BUILDASSISTIVECAREFIS() returns the fuzzy inference system used in
%   Task 2 Part 1: four sensor inputs, two actuator outputs, 54 rules.
%
%   GENERATED FILE - do not edit by hand.
%   Emitted from the Python definition in src/acflc/flat.py by
%   scripts/export_matlab.py, so the MATLAB and Python systems cannot
%   drift apart. Re-run that script after any change to the controller.
%
%   Requires the Fuzzy Logic Toolbox, R2018b or later.

fis = mamfis('Name', 'AssistiveCareRoomFLC', ...
             'AndMethod', 'min', ...
             'OrMethod', 'max', ...
             'ImplicationMethod', 'min', ...
             'AggregationMethod', 'max', ...
             'DefuzzificationMethod', 'centroid');

%% Input: room_temp  [degC]
fis = addInput(fis, [14 34], 'Name', 'room_temp');
fis = addMF(fis, 'room_temp', 'trapmf', [14, 14, 16, 19], 'Name', 'Cold');
fis = addMF(fis, 'room_temp', 'trimf', [17, 19.5, 22], 'Name', 'Cool');
fis = addMF(fis, 'room_temp', 'trimf', [20.5, 22.5, 24.5], 'Name', 'Comfortable');
fis = addMF(fis, 'room_temp', 'trimf', [23, 25.5, 28], 'Name', 'Warm');
fis = addMF(fis, 'room_temp', 'trapmf', [26, 29, 34, 34], 'Name', 'Hot');

%% Input: activity  [index]
fis = addInput(fis, [0 10], 'Name', 'activity');
fis = addMF(fis, 'activity', 'trapmf', [0, 0, 1.5, 3.5], 'Name', 'Resting');
fis = addMF(fis, 'activity', 'trimf', [2.5, 5, 7.5], 'Name', 'Light');
fis = addMF(fis, 'activity', 'trapmf', [6.5, 8.5, 10, 10], 'Name', 'Active');

%% Input: daylight  [% of design lux]
fis = addInput(fis, [0 100], 'Name', 'daylight');
fis = addMF(fis, 'daylight', 'trapmf', [0, 0, 10, 30], 'Name', 'Dark');
fis = addMF(fis, 'daylight', 'trimf', [20, 45, 70], 'Name', 'Dim');
fis = addMF(fis, 'daylight', 'trapmf', [60, 80, 100, 100], 'Name', 'Bright');

%% Input: preference  [scale]
fis = addInput(fis, [-5 5], 'Name', 'preference');
fis = addMF(fis, 'preference', 'trapmf', [-5, -5, -3, -1], 'Name', 'Cooler');
fis = addMF(fis, 'preference', 'trimf', [-2, 0, 2], 'Name', 'Neutral');
fis = addMF(fis, 'preference', 'trapmf', [1, 3, 5, 5], 'Name', 'Warmer');

%% Output: hvac  [% capacity (-cool/+heat)]
fis = addOutput(fis, [-100 100], 'Name', 'hvac');
fis = addMF(fis, 'hvac', 'trapmf', [-100, -100, -80, -45], 'Name', 'CoolHigh');
fis = addMF(fis, 'hvac', 'trimf', [-70, -35, 0], 'Name', 'CoolLow');
fis = addMF(fis, 'hvac', 'trimf', [-15, 0, 15], 'Name', 'Off');
fis = addMF(fis, 'hvac', 'trimf', [0, 35, 70], 'Name', 'HeatLow');
fis = addMF(fis, 'hvac', 'trapmf', [45, 80, 100, 100], 'Name', 'HeatHigh');

%% Output: dimmer  [%]
fis = addOutput(fis, [0 100], 'Name', 'dimmer');
fis = addMF(fis, 'dimmer', 'trapmf', [0, 0, 5, 20], 'Name', 'Off');
fis = addMF(fis, 'dimmer', 'trimf', [10, 30, 50], 'Name', 'Low');
fis = addMF(fis, 'dimmer', 'trimf', [40, 60, 80], 'Name', 'Medium');
fis = addMF(fis, 'dimmer', 'trapmf', [70, 88, 100, 100], 'Name', 'High');

%% Rules
% Columns: [room_temp activity daylight preference | hvac dimmer | weight connective]
% 0 means the variable is not used by that rule; connective 1 is AND.
% Thermal rules constrain temperature, activity and preference; lighting
% rules constrain daylight and activity. Generated from the additive FAM
% tables described in src/acflc/flat.py.
ruleList = [
     1  1  0  1  5  0  1  1  % R0: IF room_temp is Cold AND activity is Resting AND preference is Cooler THEN hvac is HeatHigh
     1  1  0  2  5  0  1  1  % R1: IF room_temp is Cold AND activity is Resting AND preference is Neutral THEN hvac is HeatHigh
     1  1  0  3  5  0  1  1  % R2: IF room_temp is Cold AND activity is Resting AND preference is Warmer THEN hvac is HeatHigh
     1  2  0  1  4  0  1  1  % R3: IF room_temp is Cold AND activity is Light AND preference is Cooler THEN hvac is HeatLow
     1  2  0  2  5  0  1  1  % R4: IF room_temp is Cold AND activity is Light AND preference is Neutral THEN hvac is HeatHigh
     1  2  0  3  5  0  1  1  % R5: IF room_temp is Cold AND activity is Light AND preference is Warmer THEN hvac is HeatHigh
     1  3  0  1  4  0  1  1  % R6: IF room_temp is Cold AND activity is Active AND preference is Cooler THEN hvac is HeatLow
     1  3  0  2  4  0  1  1  % R7: IF room_temp is Cold AND activity is Active AND preference is Neutral THEN hvac is HeatLow
     1  3  0  3  5  0  1  1  % R8: IF room_temp is Cold AND activity is Active AND preference is Warmer THEN hvac is HeatHigh
     2  1  0  1  4  0  1  1  % R9: IF room_temp is Cool AND activity is Resting AND preference is Cooler THEN hvac is HeatLow
     2  1  0  2  5  0  1  1  % R10: IF room_temp is Cool AND activity is Resting AND preference is Neutral THEN hvac is HeatHigh
     2  1  0  3  5  0  1  1  % R11: IF room_temp is Cool AND activity is Resting AND preference is Warmer THEN hvac is HeatHigh
     2  2  0  1  3  0  1  1  % R12: IF room_temp is Cool AND activity is Light AND preference is Cooler THEN hvac is Off
     2  2  0  2  4  0  1  1  % R13: IF room_temp is Cool AND activity is Light AND preference is Neutral THEN hvac is HeatLow
     2  2  0  3  5  0  1  1  % R14: IF room_temp is Cool AND activity is Light AND preference is Warmer THEN hvac is HeatHigh
     2  3  0  1  3  0  1  1  % R15: IF room_temp is Cool AND activity is Active AND preference is Cooler THEN hvac is Off
     2  3  0  2  3  0  1  1  % R16: IF room_temp is Cool AND activity is Active AND preference is Neutral THEN hvac is Off
     2  3  0  3  4  0  1  1  % R17: IF room_temp is Cool AND activity is Active AND preference is Warmer THEN hvac is HeatLow
     3  1  0  1  3  0  1  1  % R18: IF room_temp is Comfortable AND activity is Resting AND preference is Cooler THEN hvac is Off
     3  1  0  2  4  0  1  1  % R19: IF room_temp is Comfortable AND activity is Resting AND preference is Neutral THEN hvac is HeatLow
     3  1  0  3  4  0  1  1  % R20: IF room_temp is Comfortable AND activity is Resting AND preference is Warmer THEN hvac is HeatLow
     3  2  0  1  2  0  1  1  % R21: IF room_temp is Comfortable AND activity is Light AND preference is Cooler THEN hvac is CoolLow
     3  2  0  2  3  0  1  1  % R22: IF room_temp is Comfortable AND activity is Light AND preference is Neutral THEN hvac is Off
     3  2  0  3  4  0  1  1  % R23: IF room_temp is Comfortable AND activity is Light AND preference is Warmer THEN hvac is HeatLow
     3  3  0  1  2  0  1  1  % R24: IF room_temp is Comfortable AND activity is Active AND preference is Cooler THEN hvac is CoolLow
     3  3  0  2  2  0  1  1  % R25: IF room_temp is Comfortable AND activity is Active AND preference is Neutral THEN hvac is CoolLow
     3  3  0  3  3  0  1  1  % R26: IF room_temp is Comfortable AND activity is Active AND preference is Warmer THEN hvac is Off
     4  1  0  1  2  0  1  1  % R27: IF room_temp is Warm AND activity is Resting AND preference is Cooler THEN hvac is CoolLow
     4  1  0  2  3  0  1  1  % R28: IF room_temp is Warm AND activity is Resting AND preference is Neutral THEN hvac is Off
     4  1  0  3  3  0  1  1  % R29: IF room_temp is Warm AND activity is Resting AND preference is Warmer THEN hvac is Off
     4  2  0  1  1  0  1  1  % R30: IF room_temp is Warm AND activity is Light AND preference is Cooler THEN hvac is CoolHigh
     4  2  0  2  2  0  1  1  % R31: IF room_temp is Warm AND activity is Light AND preference is Neutral THEN hvac is CoolLow
     4  2  0  3  3  0  1  1  % R32: IF room_temp is Warm AND activity is Light AND preference is Warmer THEN hvac is Off
     4  3  0  1  1  0  1  1  % R33: IF room_temp is Warm AND activity is Active AND preference is Cooler THEN hvac is CoolHigh
     4  3  0  2  1  0  1  1  % R34: IF room_temp is Warm AND activity is Active AND preference is Neutral THEN hvac is CoolHigh
     4  3  0  3  2  0  1  1  % R35: IF room_temp is Warm AND activity is Active AND preference is Warmer THEN hvac is CoolLow
     5  1  0  1  1  0  1  1  % R36: IF room_temp is Hot AND activity is Resting AND preference is Cooler THEN hvac is CoolHigh
     5  1  0  2  2  0  1  1  % R37: IF room_temp is Hot AND activity is Resting AND preference is Neutral THEN hvac is CoolLow
     5  1  0  3  2  0  1  1  % R38: IF room_temp is Hot AND activity is Resting AND preference is Warmer THEN hvac is CoolLow
     5  2  0  1  1  0  1  1  % R39: IF room_temp is Hot AND activity is Light AND preference is Cooler THEN hvac is CoolHigh
     5  2  0  2  1  0  1  1  % R40: IF room_temp is Hot AND activity is Light AND preference is Neutral THEN hvac is CoolHigh
     5  2  0  3  2  0  1  1  % R41: IF room_temp is Hot AND activity is Light AND preference is Warmer THEN hvac is CoolLow
     5  3  0  1  1  0  1  1  % R42: IF room_temp is Hot AND activity is Active AND preference is Cooler THEN hvac is CoolHigh
     5  3  0  2  1  0  1  1  % R43: IF room_temp is Hot AND activity is Active AND preference is Neutral THEN hvac is CoolHigh
     5  3  0  3  1  0  1  1  % R44: IF room_temp is Hot AND activity is Active AND preference is Warmer THEN hvac is CoolHigh
     0  1  1  0  0  2  1  1  % R45: IF daylight is Dark AND activity is Resting THEN dimmer is Low
     0  2  1  0  0  3  1  1  % R46: IF daylight is Dark AND activity is Light THEN dimmer is Medium
     0  3  1  0  0  4  1  1  % R47: IF daylight is Dark AND activity is Active THEN dimmer is High
     0  1  2  0  0  2  1  1  % R48: IF daylight is Dim AND activity is Resting THEN dimmer is Low
     0  2  2  0  0  2  1  1  % R49: IF daylight is Dim AND activity is Light THEN dimmer is Low
     0  3  2  0  0  3  1  1  % R50: IF daylight is Dim AND activity is Active THEN dimmer is Medium
     0  1  3  0  0  1  1  1  % R51: IF daylight is Bright AND activity is Resting THEN dimmer is Off
     0  2  3  0  0  1  1  1  % R52: IF daylight is Bright AND activity is Light THEN dimmer is Off
     0  3  3  0  0  2  1  1  % R53: IF daylight is Bright AND activity is Active THEN dimmer is Low
];
fis = addRule(fis, ruleList);

end
