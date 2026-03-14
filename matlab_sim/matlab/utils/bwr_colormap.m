function cmap = bwr_colormap(n)
% bwr_colormap  Returns an n-colour blue-white-red diverging colormap.
%   cmap = bwr_colormap(n)   — n defaults to 256

    if nargin < 1 || isempty(n), n = 256; end
    half = floor(n / 2);
    rest = n - half;

    % Blue → White
    blue_to_white = [linspace(0, 1, half)', linspace(0, 1, half)', ones(half, 1)];
    % White → Red
    white_to_red  = [ones(rest, 1), linspace(1, 0, rest)', linspace(1, 0, rest)'];

    cmap = [blue_to_white; white_to_red];
end
