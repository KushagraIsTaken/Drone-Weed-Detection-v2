function [battery_cycles, coverage_per_cycle] = battery_model(waypoints, altitude, drone_speed_ms, battery_range_m)
% battery_model Calculates battery cycles needed to fly the waypoints
    if nargin < 3
        drone_speed_ms = 3.0;
    end
    if nargin < 4
        battery_range_m = 3000.0;
    end
    
    total_path_length = 0;
    for i = 1:(size(waypoints, 1) - 1)
        p1 = waypoints(i, 1:2);
        p2 = waypoints(i+1, 1:2);
        dist = norm(p2 - p1);
        total_path_length = total_path_length + dist;
    end
    
    battery_cycles = ceil(total_path_length / battery_range_m);
    
    % Field area is 100x100 = 10000 m2
    field_area = 10000;
    coverage_per_cycle = field_area / battery_cycles;
end
