function altitudeSweepResults = altitude_sweep(resultsJson, modelNames, altitudes)
% altitude_sweep Computes metrics across all models and altitudes

    % Add utils to path (battery_model, compute_metrics live here)
    scriptDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(scriptDir, 'utils'));
    
    numModels = length(modelNames);
    numAltitudes = length(altitudes);
    
    mAP50_matrix = zeros(numModels, numAltitudes);
    coverage_matrix = zeros(numModels, numAltitudes);
    missed_patches_matrix = zeros(numModels, numAltitudes);
    
    fieldW = resultsJson.metadata.field_width_m;
    fieldH = resultsJson.metadata.field_height_m;
    
    for i = 1:numModels
        mName = modelNames{i};
        for j = 1:numAltitudes
            alt = altitudes(j);
            altStr = num2str(alt);
            
            % Get mAP50 directly from JSON
            if isfield(resultsJson.mAP50_by_model_altitude, mName)
                modelScores = resultsJson.mAP50_by_model_altitude.(mName);
                % MATLAB jsondecode prepends 'x' to numeric struct field names
                if isfield(modelScores, ['x', altStr])
                    mAP50_matrix(i, j) = modelScores.(['x', altStr]);
                elseif isfield(modelScores, altStr)
                    mAP50_matrix(i, j) = modelScores.(altStr);
                else
                    mAP50_matrix(i, j) = 0;
                end
            end
            
            % Load detections
            [detectionLog, gtLog] = load_detections(resultsJson, mName, alt);
            
            % Compute missed-patches metric
            metrics = compute_metrics(detectionLog, gtLog, fieldW, fieldH);
            missed_patches_matrix(i, j) = metrics.missed_patches_per_hectare;
            
            % Compute field coverage %
            [~, waypoints, ~] = setup_scenario(resultsJson, alt);
            [coverageMap, ~, ~] = build_coverage_map(detectionLog, gtLog, waypoints, alt, ...
                resultsJson.metadata.camera_fov_deg, fieldW, fieldH);
            coverage_pct = (sum(coverageMap(:)) / numel(coverageMap)) * 100;
            coverage_matrix(i, j) = coverage_pct;
            
            fprintf('  [%s @ %dm] mAP50=%.3f  coverage=%.1f%%  missed/ha=%.1f\n', ...
                mName, alt, mAP50_matrix(i,j), coverage_pct, missed_patches_matrix(i,j));
        end
    end
    
    altitudeSweepResults = struct();
    altitudeSweepResults.model_names = modelNames;
    altitudeSweepResults.altitudes = altitudes;
    altitudeSweepResults.mAP50_matrix = mAP50_matrix;
    altitudeSweepResults.coverage_matrix = coverage_matrix;
    altitudeSweepResults.missed_patches_matrix = missed_patches_matrix;
end
