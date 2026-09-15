function values = cec2005_f6_f9(X, function_number)
% CEC2005 F6/F9 with official shift data, first D coordinates.
% Equations follow Suganthan et al. (2005); data provenance is in
% ../data/cec2005/provenance.json. Rows of X are candidate solutions.
    data_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'cec2005');
    D = size(X, 2);
    assert(D >= 2 && D <= 100, 'Supported dimensions: 2 through 100');
    if function_number == 6
        shift = readmatrix(fullfile(data_dir, 'rosenbrock_func_data.txt'));
        Z = X - reshape(shift(1:D), 1, D) + 1;
        values = sum(100 .* (Z(:,1:end-1).^2 - Z(:,2:end)).^2 ...
                   + (Z(:,1:end-1) - 1).^2, 2) + 390;
    elseif function_number == 9
        shift = readmatrix(fullfile(data_dir, 'rastrigin_func_data.txt'));
        Z = X - reshape(shift(1:D), 1, D);
        values = sum(Z.^2 - 10 .* cos(2*pi*Z) + 10, 2) - 330;
    else
        error('Only CEC2005 functions 6 and 9 are implemented here');
    end
end
