function [waypoints_full, n_rows, path_len_m] = generate_full_coverage_path(altitude, fov_deg, field_w, field_h)
% generate_full_coverage_path  Generates a lawnmower path that covers the ENTIRE field.
%
%   waypoints_full = Nx3 matrix [x, y, z] of flight waypoints
%   n_rows         = number of strips
%   path_len_m     = total path length in metres

    footprint  = 2 * altitude * tand(fov_deg / 2);
    row_spacing = footprint * 0.9;   % 10% overlap between adjacent strips

    x_positions = 0 : row_spacing : field_w;
    n_rows = length(x_positions);

    waypoints_full = [];
    for i = 1:n_rows
        x = x_positions(i);
        if mod(i, 2) == 1     % odd strips: south → north
            waypoints_full = [waypoints_full; x, 0,       altitude; ...
                                              x, field_h, altitude];
        else                  % even strips: north → south (boustrophedon)
            waypoints_full = [waypoints_full; x, field_h, altitude; ...
                                              x, 0,       altitude];
        end
    end

    % Add landing at origin
    waypoints_full = [waypoints_full; 0, 0, 0];

    % Compute total path length
    path_len_m = sum(sqrt(sum(diff(waypoints_full(:, 1:3)).^2, 2)));
end
