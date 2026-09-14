function acExportFig(figHandle, outPath)
%ACEXPORTFIG  Save a figure to PNG, portably.
%
%   exportgraphics arrived in MATLAB R2020a and does not exist in
%   Octave, so print is used as the fallback. Both are base
%   functionality; no toolbox is involved either way.
%
%   GENERATED FILE - emitted by scripts/export_matlab.py.

if exist('exportgraphics', 'file')
    exportgraphics(figHandle, outPath, 'Resolution', 200);
else
    print(figHandle, outPath, '-dpng', '-r200');
end
end
