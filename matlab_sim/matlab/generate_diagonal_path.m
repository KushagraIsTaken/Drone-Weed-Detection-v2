function [waypoints_diag, n_strips, path_len_m] = generate_diagonal_path(altitude, fov_deg, field_w, field_h)
% generate_diagonal_path  Generates a 45-degree diagonal boustrophedon path.
%
%   waypoints_diag = Nx3 matrix [x, y, z] of flight waypoints
%   n_strips       = number of diagonal strips actually generated
%   path_len_m     = total path length in metres

    footprint   = 2 * altitude * tand(fov_deg / 2);
    spacing     = footprint * 0.9;   % perpendicular distance between strips

    % Diagonal strips: y = x - d  →  offsets d span entire field diagonal
    offsets = -field_h : spacing : field_w + field_h;

    waypoints_diag = [];
    n_strips = 0;

    for i = 1:length(offsets)
        d = offsets(i);

        % Clip the infinite line y = x - d to the field rectangle [0,field_w] x [0,field_h]
        % Intersection with x=0:      (0,    -d)
        % Intersection with x=field_w: (field_w, field_w-d)
        % Intersection with y=0:      (d,    0)
        % Intersection with y=field_h:(d+field_h, field_h)
        pts = [];
        if -d >= 0 && -d <= field_h,          pts = [pts; 0,       -d      ]; end
        if field_w-d >= 0 && field_w-d <= field_h, pts = [pts; field_w, field_w-d]; end
        if d >= 0 && d <= field_w,            pts = [pts; d,       0       ]; end
        if d+field_h >= 0 && d+field_h <= field_w, pts = [pts; d+field_h, field_h ]; end

        % Remove duplicate points
        if size(pts, 1) >= 2
            pts = unique(pts, 'rows');
        end

        if size(pts, 1) < 2
            continue;   % line does not intersect field
        end

        % Take the two extreme endpoints along the diagonal
        [~, order] = sort(pts(:, 1));
        pts = pts(order, :);
        p1 = pts(1, :);
        p2 = pts(end, :);

        n_strips = n_strips + 1;
        if mod(n_strips, 2) == 1
            waypoints_diag = [waypoints_diag; p1, altitude; p2, altitude]; %#ok<AGROW>
        else
            waypoints_diag = [waypoints_diag; p2, altitude; p1, altitude]; %#ok<AGROW>
        end
    end

    % Landing
    waypoints_diag = [waypoints_diag; 0, 0, 0];

    % Total path length
    path_len_m = sum(sqrt(sum(diff(waypoints_diag).^2, 2)));
end
