%% WEED DETECTION UAV SIMULATION — MASTER SCRIPT
% Run this after running python/run_inference.py
% Usage: cd to the matlab/ directory, then run: main_simulation

% 0. Configuration
ALTITUDES = [10, 15, 20, 25, 30];
MODEL_NAMES = {'teacher','small_baseline','nano_baseline',...
               'nano_feature_kd','small_feature_kd','nano_pseudo_kd','small_pseudo_kd'};
FOV_DEG  = 60.0;
FIELD_W  = 100.0;
FIELD_H  = 100.0;
GRID_RES = 1.0;

% Resolve paths relative to THIS script's location so it works from any cwd
scriptDir   = fileparts(mfilename('fullpath'));
RESULTS_JSON = fullfile(scriptDir, '..', 'python', 'results', 'results.json');

addpath(scriptDir);
addpath(fullfile(scriptDir, 'utils'));

% 1. Verify results.json exists and read it
if ~exist(RESULTS_JSON, 'file')
    error(['Cannot find results.json at: %s\n' ...
           'Please run: python python/run_inference.py'], RESULTS_JSON);
end

fprintf('Loading results.json...\n');
fid = fopen(RESULTS_JSON, 'r');
if fid == -1
    error('Cannot open results.json. Check read permissions.');
end
raw = fread(fid, '*char')';
fclose(fid);

% 2. Decode JSON
fprintf('Decoding JSON (this may take a moment for large files)...\n');
resultsJson = jsondecode(raw);
fprintf('Done. %d detections loaded.\n', length(resultsJson.detections));

% 3. Run altitude sweep for all models
fprintf('\nRunning altitude sweep metrics...\n');
sweepResults = altitude_sweep(resultsJson, MODEL_NAMES, ALTITUDES);

% 4. Print full metrics table to console
fprintf('\n===== UAV SIMULATION METRICS SUMMARY =====\n');
fprintf('%-20s | %-3s | %-6s | %-10s | %-11s | %s\n', ...
    'Model', 'Alt', 'mAP50', 'Coverage%', 'Missed/ha', 'Battery cycles');
fprintf('%s\n', repmat('-', 1, 75));

% 5. Write CSV summary
outDir = fullfile(scriptDir, 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

csvFile = fullfile(outDir, 'metrics_summary.csv');
fid = fopen(csvFile, 'w');
fprintf(fid, 'Model,Altitude,mAP50,CoveragePercent,MissedPerHa,BatteryCycles\n');

for i = 1:length(MODEL_NAMES)
    mName = MODEL_NAMES{i};
    for j = 1:length(ALTITUDES)
        alt    = ALTITUDES(j);
        mAP50  = sweepResults.mAP50_matrix(i, j);
        cov    = sweepResults.coverage_matrix(i, j);
        missed = sweepResults.missed_patches_matrix(i, j);
        
        [~, wp, ~] = setup_scenario(resultsJson, alt);
        [bat_cyc, ~] = battery_model(wp, alt);
        
        fprintf('%-20s | %3d | %6.3f | %9.1f%% | %11.1f | %d\n', ...
            mName, alt, mAP50, cov, missed, bat_cyc);
        fprintf(fid, '%s,%d,%.4f,%.2f,%.2f,%d\n', ...
            mName, alt, mAP50, cov, missed, bat_cyc);
    end
end
fclose(fid);
fprintf('%s\n', repmat('=', 1, 75));
fprintf('CSV saved: %s\n', csvFile);

% 6. Generate all figures
fprintf('\nGenerating and saving figures...\n');
generate_figures(resultsJson, sweepResults);

fprintf('\nDone! Figures saved in: %s\n', outDir);
