function overlap_map = compute_overlap_map(waypoints, altitude, fov_deg, field_w, field_h, grid_res)
% compute_overlap_map  Counts how many times each field cell is imaged by the camera footprint.
%
%   overlap_map(gy, gx) = number of distinct waypoints whose footprint covers that cell

    if nargin < 6, grid_res = 1.0; end

    gridW = ceil(field_w / grid_res);
    gridH = ceil(field_h / grid_res);
    overlap_map = zeros(gridH, gridW);

    footprint = 2 * altitude * tand(fov_deg / 2);
    half_fp   = footprint / 2;

    % Skip takeoff (row 1) and landing (last row)
    for i = 2:(size(waypoints, 1) - 1)
        wx = waypoints(i, 1);
        wy = waypoints(i, 2);

        x_min = max(0, wx - half_fp);
        x_max = min(field_w, wx + half_fp);
        y_min = max(0, wy - half_fp);
        y_max = min(field_h, wy + half_fp);

        gx_min = max(1, floor(x_min / grid_res) + 1);
        gx_max = min(gridW, ceil(x_max / grid_res));
        gy_min = max(1, floor(y_min / grid_res) + 1);
        gy_max = min(gridH, ceil(y_max / grid_res));

        if gx_min <= gx_max && gy_min <= gy_max
            overlap_map(gy_min:gy_max, gx_min:gx_max) = ...
                overlap_map(gy_min:gy_max, gx_min:gx_max) + 1;
        end
    end
end
