function [y, details] = evalAssistiveCareFIS(fis, x)
%EVALASSISTIVECAREFIS  Mamdani inference in base MATLAB. No toolbox required.
%
%   y = EVALASSISTIVECAREFIS(FIS, X) evaluates the controller built by
%   BUILDASSISTIVECAREFIS for one crisp input row X (1 x nInputs, in
%   fis.inputNames order) and returns the crisp output row Y (1 x nOutputs,
%   in fis.outputNames order).
%
%   [Y, DETAILS] = EVALASSISTIVECAREFIS(...) also returns every intermediate
%   quantity of the inference, so that each stage can be tabulated or plotted
%   exactly as the engine computed it:
%
%       details.mu           1 x nInputs cell, degree of membership per set
%       details.firing       nRules x 1, firing strength alpha of every rule
%       details.active       indices of the rules with alpha > 0
%       details.clipped      1 x nOutputs cell, implication level per output set
%       details.aggregated   1 x nOutputs cell, the aggregated set on the grid
%       details.universe     1 x nOutputs cell, the output grids
%
%   This replaces the Fuzzy Logic Toolbox's evalfis. The five Mamdani stages
%   are implemented explicitly below, in the order the report describes them:
%
%       1. fuzzification     crisp reading  -> degree of membership per set
%       2. rule firing       AND = min across a rule's antecedents
%       3. implication       min, i.e. clip each consequent set at its alpha
%       4. aggregation       max, i.e. union the clipped sets per output
%       5. defuzzification   centroid of the aggregated set
%
%   NUMERICAL NOTE. Defuzzification integrates over fis.nPoints = 501 samples
%   of each output universe. That value is not arbitrary: it is the
%   authoritative resolution set by Variable.n_points in
%   src/acflc/membership.py, and matching it is what allows this
%   implementation and the Python reference to be compared directly. The
%   Toolbox version of this script had to raise MATLAB's default of 101 to
%   501 for the same reason.
%
%   Where a reading activates no rule at all, the aggregated set is
%   identically zero, the centroid is undefined, and NaN is returned. That is
%   a real condition of the rule base rather than a numerical artefact, so it
%   is reported rather than silently defaulted to the middle of the universe.
%
%   See also BUILDASSISTIVECAREFIS, ACEVALMF, RUNASSISTIVECAREFIS.

x = double(x(:)).';                     % accept a row or a column
nIn  = numel(fis.inputs);
nOut = numel(fis.outputs);
nRules = size(fis.ruleAnt, 1);

if numel(x) ~= nIn
    error('evalAssistiveCareFIS:inputs', ...
          'expected %d inputs, got %d', nIn, numel(x));
end

%% ---------------------------------------------------------------------
%  STAGE 1 - FUZZIFICATION
%  Each crisp reading is mapped to a degree of membership in every fuzzy set
%  of its own variable. Inputs are evaluated at the reading itself; only the
%  outputs ever touch a sampled grid.
%% ---------------------------------------------------------------------
mu = cell(1, nIn);
for i = 1:nIn
    v = fis.inputs(i);
    m = zeros(1, numel(v.mfs));
    for k = 1:numel(v.mfs)
        m(k) = acEvalMF(v.mfs(k).kind, v.mfs(k).params, x(i));
    end
    mu{i} = m;
end

%% ---------------------------------------------------------------------
%  STAGE 2 - RULE FIRING STRENGTHS
%  The antecedents of a rule are combined with the AND operator, which is the
%  minimum t-norm. A zero in fis.ruleAnt means the rule does not constrain
%  that variable at all -- the lighting rules say nothing about temperature --
%  so those columns are skipped rather than treated as a membership of zero.
%% ---------------------------------------------------------------------
firing = zeros(nRules, 1);
for r = 1:nRules
    alpha = Inf;
    for i = 1:nIn
        setIdx = fis.ruleAnt(r, i);
        if setIdx > 0
            alpha = min(alpha, mu{i}(setIdx));      % AND = min
        end
    end
    if isinf(alpha)
        error('evalAssistiveCareFIS:emptyRule', ...
              'rule %d has no antecedents', r);
    end
    firing(r) = alpha * fis.ruleWeight(r);
end

%% ---------------------------------------------------------------------
%  STAGES 3 and 4 - IMPLICATION AND AGGREGATION
%  Rules sharing a consequent set are collapsed first: the set is clipped at
%  the strongest alpha among them. That is algebraically identical to
%  clipping each rule separately and then taking the union, because
%  max_r min(alpha_r, mf) == min(max_r alpha_r, mf) for a shared mf, and it
%  is how the Python reference does it.
%
%  The clipped sets of one output are then unioned with max, giving the
%  aggregated fuzzy set that defuzzification consumes.
%% ---------------------------------------------------------------------
clipped    = cell(1, nOut);
aggregated = cell(1, nOut);
universe   = cell(1, nOut);
y = zeros(1, nOut);

for o = 1:nOut
    v = fis.outputs(o);
    nSets = numel(v.mfs);

    % Implication level per consequent set: max over the rules that name it.
    alphaSet = zeros(1, nSets);
    for r = 1:nRules
        setIdx = fis.ruleCons(r, o);
        if setIdx > 0
            alphaSet(setIdx) = max(alphaSet(setIdx), firing(r));
        end
    end

    % Clip each set at its level (min-implication), then union them (max).
    % v.mfMatrix is the sets already sampled on the universe grid, cached at
    % build time because it is fixed for a given parameter set and is touched
    % on every single inference.
    clippedSets = min(repmat(alphaSet(:), 1, fis.nPoints), v.mfMatrix);
    agg = max(clippedSets, [], 1);

    clipped{o}    = alphaSet;
    aggregated{o} = agg;
    universe{o}   = v.universe;

    %% -----------------------------------------------------------------
    %  STAGE 5 - DEFUZZIFICATION (centroid)
    %% -----------------------------------------------------------------
    y(o) = acCentroid(agg, v.universe);
end

if nargout > 1
    details = struct( ...
        'mu',         {mu}, ...
        'firing',     firing, ...
        'active',     find(firing > 0), ...
        'clipped',    {clipped}, ...
        'aggregated', {aggregated}, ...
        'universe',   {universe});
end
end


% ------------------------------------------------------------------ local

function c = acCentroid(agg, universe)
%ACCENTROID  Centre of gravity by exact trapezoidal integration.
%
%   The aggregated set is piecewise linear between grid points, so both its
%   area and its first moment can be integrated exactly rather than
%   approximated. Over one segment [x1, x2] with heights y1, y2:
%
%       area          = (dx/2) (y1 + y2)
%       moment * area = (dx/2) [ x1 (y1 + y2) + dx (y1 + 2 y2) / 3 ]
%
%   and the centroid is the ratio of the sums. Writing the moment
%   pre-multiplied by the area avoids a 0/0 division on the many flat-zero
%   segments of a typical aggregated set.
%
%   This is the same quantity the Toolbox's defuzz(..., 'centroid') returns,
%   and the same one src/acflc/controller.py computes, which is what makes
%   the cross-check in runAssistiveCareFIS a real test rather than a
%   comparison of two different formulas.

x1 = universe(1:end-1);
x2 = universe(2:end);
dx = x2 - x1;

y1 = agg(1:end-1);
y2 = agg(2:end);

area       = 0.5 .* dx .* (y1 + y2);
momentArea = 0.5 .* dx .* (x1 .* (y1 + y2) + dx .* (y1 + 2.0 .* y2) ./ 3.0);

total = sum(area);
if total > 0
    c = sum(momentArea) / total;
else
    % No rule fired: the aggregated set is empty and the centroid is
    % undefined. Reported rather than defaulted -- a controller that cannot
    % decide should say so.
    c = NaN;
end
end
