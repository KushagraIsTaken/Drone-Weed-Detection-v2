function [coverageMap, detectionMap, missedMap] = build_coverage_map(detectionLog, gtLog, waypoints, altitude, fov_deg, fieldW, fieldH, gridRes)
% build_coverage_map Generates grid maps of coverage and detections

    if nargin < 8
        gridRes = 1.0;
    end
    
    gridW = ceil(fieldW / gridRes);
    gridH = ceil(fieldH / gridRes);
    
    coverageMap = zeros(gridH, gridW);
    detectionMap = zeros(gridH, gridW);
    missedMap = zeros(gridH, gridW);
    
    % 1. Coverage Map — mark cells under each waypoint footprint
    footprint = 2 * altitude * tand(fov_deg / 2);
    half_fp = footprint / 2;
    
    for i = 2:(size(waypoints, 1) - 1)  % skip takeoff/landing
        wx = waypoints(i, 1);
        wy = waypoints(i, 2);
        
        x_min = max(0, wx - half_fp);
        x_max = min(fieldW, wx + half_fp);
        y_min = max(0, wy - half_fp);
        y_max = min(fieldH, wy + half_fp);
        
        gx_min = max(1, floor(x_min / gridRes) + 1);
        gx_max = min(gridW, ceil(x_max / gridRes));
        gy_min = max(1, floor(y_min / gridRes) + 1);
        gy_max = min(gridH, ceil(y_max / gridRes));
        
        if gx_min <= gx_max && gy_min <= gy_max
            coverageMap(gy_min:gy_max, gx_min:gx_max) = 1;
        end
    end
    
    % 2. Detection Map
    all_det_x = [];
    all_det_y = [];
    
    if ~isempty(detectionLog)
        for i = 1:length(detectionLog)
            boxes = detectionLog(i).boxes_field;
            scores = detectionLog(i).scores;
            if isempty(boxes) || size(boxes, 2) < 4
                continue;
            end
            for j = 1:size(boxes, 1)
                cx = (boxes(j, 1) + boxes(j, 3)) / 2;
                cy = (boxes(j, 2) + boxes(j, 4)) / 2;
                
                gx = min(gridW, max(1, floor(cx / gridRes) + 1));
                gy = min(gridH, max(1, floor(cy / gridRes) + 1));
                
                sc = 1.0;
                if ~isempty(scores) && j <= length(scores)
                    sc = scores(j);
                end
                detectionMap(gy, gx) = detectionMap(gy, gx) + sc;
                
                all_det_x(end+1) = cx; %#ok<AGROW>
                all_det_y(end+1) = cy; %#ok<AGROW>
            end
        end
    end
    
    % 3. Missed Map — GT locations with no nearby detection
    if ~isempty(gtLog)
        for i = 1:length(gtLog)
            gx_pos = gtLog(i).field_x;
            gy_pos = gtLog(i).field_y;
            
            if isempty(all_det_x)
                is_missed = true;
            else
                dists = sqrt((all_det_x - gx_pos).^2 + (all_det_y - gy_pos).^2);
                is_missed = min(dists) > 2.0;
            end
            
            if is_missed
                gx = min(gridW, max(1, floor(gx_pos / gridRes) + 1));
                gy = min(gridH, max(1, floor(gy_pos / gridRes) + 1));
                missedMap(gy, gx) = 1;
            end
        end
    end
end
