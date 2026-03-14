function [scenario, waypoints, imageAssignments] = setup_scenario(resultsJson, altitudeM)
% setup_scenario Creates uavScenario and waypoint trajectory

    % Create UAV scenario (requires UAV Toolbox)
    try
        scenario = uavScenario('UpdateRate', 10, 'ReferenceLocation', [0 0 0]);
        
        % Add terrain mesh (100x100m at z=0)
        color = [0.2 0.8 0.2]; % green
        addMesh(scenario, 'Polygon', {[0 0; 100 0; 100 100; 0 100], [-0.1 0]}, color);
        
        % Add red markers for unique GT weed locations
        if isfield(resultsJson, 'gt_boxes')
            gts = resultsJson.gt_boxes;
            if iscell(gts)
                numBoxes = length(gts);
                positions = zeros(numBoxes, 2);
                for i = 1:numBoxes
                    positions(i, :) = [gts{i}.field_x, gts{i}.field_y];
                end
            else
                numBoxes = length(gts);
                fx = vertcat(gts.field_x);
                fy = vertcat(gts.field_y);
                positions = [fx, fy];
            end
            
            if numBoxes > 0
                % Simple clustering within 3m
                keepIdx = true(size(positions, 1), 1);
                for i = 1:size(positions, 1)
                    if ~keepIdx(i), continue; end
                    for j = (i+1):size(positions, 1)
                        if norm(positions(i, :) - positions(j, :)) <= 3.0
                            keepIdx(j) = false;
                        end
                    end
                end
                uniquePos = positions(keepIdx, :);
                
                % addMesh Cylinder: {[cx cy cz], radius, height}, color
                for i = 1:size(uniquePos, 1)
                    try
                        addMesh(scenario, 'Cylinder', ...
                            {[uniquePos(i,1), uniquePos(i,2), 0], 0.25, 0.5}, [1 0 0]);
                    catch
                        % Skip if cylinder API differs between toolbox versions
                    end
                end
            end
        end
    catch ME
        warning('uavScenario creation failed: %s. Using empty scenario placeholder.', ME.message);
        scenario = struct('type', 'placeholder');
    end
    
    % Generate lawnmower waypoints
    fov = resultsJson.metadata.camera_fov_deg;
    footprint = 2 * altitudeM * tand(fov / 2);
    row_spacing = footprint * 0.9; % 10% overlap
    
    waypoints = [0, 0, altitudeM]; % takeoff
    
    y = row_spacing / 2;
    direction = 1;
    while y <= resultsJson.metadata.field_height_m
        if direction == 1
            waypoints = [waypoints; 0, y, altitudeM];
            waypoints = [waypoints; resultsJson.metadata.field_width_m, y, altitudeM];
        else
            waypoints = [waypoints; resultsJson.metadata.field_width_m, y, altitudeM];
            waypoints = [waypoints; 0, y, altitudeM];
        end
        y = y + row_spacing;
        direction = direction * -1;
    end
    waypoints = [waypoints; 0, 0, 0]; % landing
    
    % Assign nearest image to each waypoint (excluding takeoff/landing)
    imageAssignments = struct();
    if isfield(resultsJson, 'field_positions')
        posList = resultsJson.field_positions;
        if iscell(posList)
            numPos = length(posList);
            posMat = zeros(numPos, 2);
            for k = 1:numPos
                posMat(k,:) = [posList{k}.field_x, posList{k}.field_y];
            end
        else
            numPos = length(posList);
            fx = vertcat(posList.field_x);
            fy = vertcat(posList.field_y);
            posMat = [fx, fy];
        end
        
        for i = 2:(size(waypoints, 1)-1)
            wp = waypoints(i, 1:2);
            dists = sum((posMat - wp).^2, 2);
            [~, minIdx] = min(dists);
            if iscell(posList)
                imageAssignments(i).image_id = posList{minIdx}.image_id;
            else
                imageAssignments(i).image_id = posList(minIdx).image_id;
            end
        end
    end
end
