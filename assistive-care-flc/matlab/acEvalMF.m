function mu = acEvalMF(kind, params, x)
%ACEVALMF  Degree of membership of x in one fuzzy set. No toolbox required.
%
%   mu = ACEVALMF(KIND, PARAMS, X) evaluates the membership function named
%   KIND ('trimf' or 'trapmf') with parameter vector PARAMS at every element
%   of X, returning an array the same size as X.
%
%   This is the base-MATLAB replacement for the Fuzzy Logic Toolbox's trimf
%   and trapmf. It is written to reproduce src/acflc/membership.py exactly,
%   including its treatment of the degenerate cases, because the Python
%   engine is the reference implementation this file is validated against.
%
%   trimf(a, b, c)      feet at a and c, apex at b.
%   trapmf(a, b, c, d)  feet at a and d, plateau over [b, c].
%
%   Degenerate forms are deliberately permitted and are load-bearing in this
%   controller: a == b gives a vertical left edge, which is how the outermost
%   sets are made to saturate at the end of a universe. Cold is
%   trapmf(14, 14, 16, 19), so 14 degrees is *fully* Cold rather than
%   half Cold, which is the physically correct reading.
%
%   See also EVALASSISTIVECAREFIS, BUILDASSISTIVECAREFIS.

x = double(x);

switch lower(kind)
    case 'trimf'
        if numel(params) ~= 3
            error('acEvalMF:params', 'trimf takes 3 parameters, got %d', ...
                  numel(params));
        end
        mu = acTrimf(x, params(1), params(2), params(3));

    case 'trapmf'
        if numel(params) ~= 4
            error('acEvalMF:params', 'trapmf takes 4 parameters, got %d', ...
                  numel(params));
        end
        mu = acTrapmf(x, params(1), params(2), params(3), params(4));

    otherwise
        error('acEvalMF:kind', 'unknown membership function shape ''%s''', kind);
end
end


% ------------------------------------------------------------------ local

function y = acTrimf(x, a, b, c)
%ACTRIMF  Triangular membership function, feet at a and c, apex at b.
if ~(a <= b && b <= c)
    error('acEvalMF:order', ...
          'trimf parameters must be ascending, got [%g %g %g]', a, b, c);
end

y = zeros(size(x));

% Rising edge. Skipped when a == b, which leaves a vertical left side.
if a ~= b
    idx = (x > a) & (x < b);
    y(idx) = (x(idx) - a) ./ (b - a);
end

% Falling edge. Skipped when b == c, which leaves a vertical right side.
if b ~= c
    idx = (x > b) & (x < c);
    y(idx) = (c - x(idx)) ./ (c - b);
end

% The apex is set last so it wins over both edges, and so that the fully
% degenerate case a == b == c still returns 1 at the single point.
y(x == b) = 1;
end


function y = acTrapmf(x, a, b, c, d)
%ACTRAPMF  Trapezoidal membership function, plateau over [b, c].
if ~(a <= b && b <= c && c <= d)
    error('acEvalMF:order', ...
          'trapmf parameters must be ascending, got [%g %g %g %g]', a, b, c, d);
end

% Start from the plateau and carve the two shoulders out of it, which is how
% membership.py does it and keeps the degenerate cases identical.
y = ones(size(x));

idx = (x <= b);
y(idx) = acTrimf(x(idx), a, b, b);

idx = (x >= c);
y(idx) = acTrimf(x(idx), c, c, d);

y(x < a) = 0;
y(x > d) = 0;
end
