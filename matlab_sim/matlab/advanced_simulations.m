%% ADVANCED UAV SIMULATION — EXTENDED PAPER ANALYSES
% Run from the matlab/ directory: advanced_simulations
% Requires: results.json from python/run_inference.py (already done)
% Produces:  figures/adv_*.png  and  figures/adv_*.csv

% ── Paths & setup ──────────────────────────────────────────────────────────
scriptDir    = fileparts(mfilename('fullpath'));
RESULTS_JSON = fullfile(scriptDir, '..', 'python', 'results', 'results.json');
addpath(scriptDir);
addpath(fullfile(scriptDir, 'utils'));

outDir = fullfile(scriptDir, 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ── Load JSON (once) ───────────────────────────────────────────────────────
fprintf('Loading results.json...\n');
fid = fopen(RESULTS_JSON, 'r');
if fid == -1
    error('Cannot open results.json. Run python/run_inference.py first.');
end
raw = fread(fid, '*char')';
fclose(fid);
resultsJson = jsondecode(raw);
fprintf('Loaded. %d detection records.\n', length(resultsJson.detections));

% ── Constants ──────────────────────────────────────────────────────────────
ALTITUDES    = [10, 15, 20, 25, 30];
FOV_DEG      = resultsJson.metadata.camera_fov_deg;
FIELD_W      = resultsJson.metadata.field_width_m;
FIELD_H      = resultsJson.metadata.field_height_m;
DRONE_SPEED  = 3.0;     % m/s
BATTERY_RANGE = 400.0;  % m/charge (realistic small agricultural drone)
MODEL_NAMES  = {'teacher','small_baseline','nano_baseline', ...
                'nano_feature_kd','small_feature_kd','nano_pseudo_kd','small_pseudo_kd'};
COLORS4 = {'k','b','r','g'};  % for 4-model subsets
COLORS7 = {'k','b','r','g','c','m',[1 0.5 0]};

fprintf('\n=== STARTING ADVANCED SIMULATIONS ===\n\n');

% ══════════════════════════════════════════════════════════════════════════
%% ANALYSIS 1: Multi-pass full coverage operational efficiency
% ══════════════════════════════════════════════════════════════════════════
fprintf('--- Analysis 1: Full-coverage operational efficiency ---\n');

eff_rows   = zeros(1, length(ALTITUDES));
eff_path   = zeros(1, length(ALTITUDES));
eff_time   = zeros(1, length(ALTITUDES));
eff_batt   = zeros(1, length(ALTITUDES));
eff_cover  = zeros(1, length(ALTITUDES));

for j = 1:length(ALTITUDES)
    alt = ALTITUDES(j);
    [wp, n_rows, path_len] = generate_full_coverage_path(alt, FOV_DEG, FIELD_W, FIELD_H);
    flight_time_min = path_len / DRONE_SPEED / 60;
    batt_cyc        = ceil(path_len / BATTERY_RANGE);

    % Coverage % from overlap map
    ovmap = compute_overlap_map(wp, alt, FOV_DEG, FIELD_W, FIELD_H, 1.0);
    coverage_pct = (sum(ovmap(:) > 0) / numel(ovmap)) * 100;

    eff_rows(j)  = n_rows;
    eff_path(j)  = path_len;
    eff_time(j)  = flight_time_min;
    eff_batt(j)  = batt_cyc;
    eff_cover(j) = coverage_pct;

    fprintf('  Alt %dm: %d strips | %.0fm path | %.1f min | %d batt cycles | %.1f%% cover\n', ...
        alt, n_rows, path_len, flight_time_min, batt_cyc, coverage_pct);
end

% Save CSV
fid = fopen(fullfile(outDir, 'adv1_full_coverage_efficiency.csv'), 'w');
fprintf(fid, 'Altitude_m,Strips,PathLength_m,FlightTime_min,BatteryCycles,CoveragePercent\n');
for j = 1:length(ALTITUDES)
    fprintf(fid, '%d,%d,%.1f,%.2f,%d,%.2f\n', ...
        ALTITUDES(j), eff_rows(j), eff_path(j), eff_time(j), eff_batt(j), eff_cover(j));
end
fclose(fid);

% Pareto figure: mAP50 vs coverage, sized by flight time ──────────────────
% Get best mAP50 per altitude across all models (from existing data)
best_map = zeros(1, length(ALTITUDES));
for j = 1:length(ALTITUDES)
    altStr = num2str(ALTITUDES(j));
    for m = 1:length(MODEL_NAMES)
        mn = MODEL_NAMES{m};
        if isfield(resultsJson.mAP50_by_model_altitude, mn)
            ms = resultsJson.mAP50_by_model_altitude.(mn);
            key = ['x' altStr];
            if isfield(ms, key), v = ms.(key);
            elseif isfield(ms, altStr), v = ms.(altStr);
            else, v = 0;
            end
            best_map(j) = max(best_map(j), v);
        end
    end
end

fig = figure('Visible','off','Position',[100 100 900 550]);

% Subplot 1: Operational efficiency table as bar cluster
subplot(1,2,1);
yyaxis left;
bar(ALTITUDES, eff_time, 0.4, 'FaceColor',[0.2 0.5 0.9]);
ylabel('Flight time (min)');
yyaxis right;
plot(ALTITUDES, eff_batt, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('Battery cycles');
xlabel('Altitude (m)');
title('Operational cost (full coverage)');
legend({'Flight time','Battery cycles'}, 'Location','northeast');
grid on;

% Subplot 2: Pareto — mAP vs flight time
subplot(1,2,2);
scatter(eff_time, best_map, 120, ALTITUDES, 'filled');
colormap(gca, parula);
cb = colorbar; cb.Label.String = 'Altitude (m)';
for j = 1:length(ALTITUDES)
    text(eff_time(j)+0.2, best_map(j), sprintf('%dm', ALTITUDES(j)), 'FontSize', 9);
end
xlabel('Full-survey flight time (min)');
ylabel('Best mAP@0.50');
title('Accuracy vs efficiency Pareto (teacher model)');
grid on;
sgtitle('Analysis 1 — Full Coverage Operational Efficiency');

print(fig, fullfile(outDir, 'adv1_full_coverage_efficiency.png'), '-dpng', '-r150');
savefig(fig, fullfile(outDir, 'adv1_full_coverage_efficiency.fig'));
close(fig);
fprintf('  Saved: adv1_full_coverage_efficiency.png\n');

% ══════════════════════════════════════════════════════════════════════════
%% ANALYSIS 2: Crosswind diagonal vs lawnmower comparison
% ══════════════════════════════════════════════════════════════════════════
fprintf('\n--- Analysis 2: Diagonal vs lawnmower path comparison ---\n');

diag_strips = zeros(1, length(ALTITUDES));
diag_path   = zeros(1, length(ALTITUDES));
lawn_path   = zeros(1, length(ALTITUDES));
diag_cov    = zeros(1, length(ALTITUDES));
lawn_cov    = zeros(1, length(ALTITUDES));
diag_std    = zeros(1, length(ALTITUDES));
lawn_std    = zeros(1, length(ALTITUDES));

for j = 1:length(ALTITUDES)
    alt = ALTITUDES(j);

    [wp_diag, n_s, plen_d] = generate_diagonal_path(alt, FOV_DEG, FIELD_W, FIELD_H);
    [wp_lawn, ~,  plen_l]  = generate_full_coverage_path(alt, FOV_DEG, FIELD_W, FIELD_H);

    ov_d = compute_overlap_map(wp_diag, alt, FOV_DEG, FIELD_W, FIELD_H);
    ov_l = compute_overlap_map(wp_lawn, alt, FOV_DEG, FIELD_W, FIELD_H);

    diag_strips(j) = n_s;
    diag_path(j)   = plen_d;
    lawn_path(j)   = plen_l;
    diag_cov(j)    = (sum(ov_d(:) > 0) / numel(ov_d)) * 100;
    lawn_cov(j)    = (sum(ov_l(:) > 0) / numel(ov_l)) * 100;
    diag_std(j)    = std(double(ov_d(:)));
    lawn_std(j)    = std(double(ov_l(:)));

    fprintf('  Alt %dm: diag %.1f%% cov (std=%.2f), lawn %.1f%% cov (std=%.2f)\n', ...
        alt, diag_cov(j), diag_std(j), lawn_cov(j), lawn_std(j));
end

% Save CSV
fid = fopen(fullfile(outDir, 'adv2_path_comparison.csv'), 'w');
fprintf(fid, 'Altitude_m,DiagStrips,DiagPathLen_m,LawnPathLen_m,DiagCoverage_pct,LawnCoverage_pct,DiagOverlapStd,LawnOverlapStd\n');
for j = 1:length(ALTITUDES)
    fprintf(fid, '%d,%d,%.1f,%.1f,%.2f,%.2f,%.4f,%.4f\n', ...
        ALTITUDES(j), diag_strips(j), diag_path(j), lawn_path(j), ...
        diag_cov(j), lawn_cov(j), diag_std(j), lawn_std(j));
end
fclose(fid);

% Detailed side-by-side maps at 15m ───────────────────────────────────────
alt = 15;
[wp_diag, ~, ~] = generate_diagonal_path(alt, FOV_DEG, FIELD_W, FIELD_H);
[wp_lawn, ~, ~] = generate_full_coverage_path(alt, FOV_DEG, FIELD_W, FIELD_H);
ov_d = compute_overlap_map(wp_diag, alt, FOV_DEG, FIELD_W, FIELD_H);
ov_l = compute_overlap_map(wp_lawn, alt, FOV_DEG, FIELD_W, FIELD_H);
max_ov = max(max(ov_d(:)), max(ov_l(:)));
if max_ov == 0, max_ov = 1; end

fig = figure('Visible','off','Position',[100 100 1200 500]);
subplot(1,3,1);
imagesc([0 FIELD_W],[0 FIELD_H], ov_l); colormap(cool); caxis([0 max_ov]);
colorbar; set(gca,'YDir','normal');
title('Lawnmower overlap count'); xlabel('X (m)'); ylabel('Y (m)');

subplot(1,3,2);
imagesc([0 FIELD_W],[0 FIELD_H], ov_d); colormap(cool); caxis([0 max_ov]);
colorbar; set(gca,'YDir','normal');
title('Diagonal overlap count'); xlabel('X (m)'); ylabel('Y (m)');

subplot(1,3,3);
diff_map = double(ov_d) - double(ov_l);
imagesc([0 FIELD_W],[0 FIELD_H], diff_map); colormap(bwr_colormap());
colorbar; set(gca,'YDir','normal');
title('Difference (diagonal − lawnmower)'); xlabel('X (m)'); ylabel('Y (m)');

sgtitle(sprintf('Analysis 2 — Path Overlap Comparison @ %dm', alt));
print(fig, fullfile(outDir, 'adv2_path_comparison.png'), '-dpng', '-r150');
savefig(fig, fullfile(outDir, 'adv2_path_comparison.fig'));
close(fig);
fprintf('  Saved: adv2_path_comparison.png\n');

% Summary bar chart
fig = figure('Visible','off','Position',[100 100 800 450]);
x = categorical(string(ALTITUDES) + "m");
subplot(1,2,1);
bar(x, [diag_cov; lawn_cov]'); legend({'Diagonal','Lawnmower'},'Location','northwest');
ylabel('Coverage (%)'); title('Field coverage by path type'); grid on;
subplot(1,2,2);
bar(x, [diag_std; lawn_std]'); legend({'Diagonal','Lawnmower'},'Location','northwest');
ylabel('Overlap std dev'); title('Overlap uniformity (lower = more uniform)'); grid on;
sgtitle('Analysis 2 — Path Type Summary');
print(fig, fullfile(outDir, 'adv2_path_summary.png'), '-dpng', '-r150');
savefig(fig, fullfile(outDir, 'adv2_path_summary.fig'));
close(fig);



% ══════════════════════════════════════════════════════════════════════════
%% ANALYSIS 4: Camera footprint overlap analysis (row boundary zones)
% ══════════════════════════════════════════════════════════════════════════
fprintf('\n--- Analysis 4: Footprint overlap analysis ---\n');

fig = figure('Visible','off','Position',[100 100 1400 480]);
t = tiledlayout(1, length(ALTITUDES), 'TileSpacing','compact','Padding','compact'); %#ok<NASGU>
max_overlap_vals = zeros(1, length(ALTITUDES));
mean_overlap_vals = zeros(1, length(ALTITUDES));

for j = 1:length(ALTITUDES)
    alt = ALTITUDES(j);
    [wp, ~, ~] = generate_full_coverage_path(alt, FOV_DEG, FIELD_W, FIELD_H);
    ov = compute_overlap_map(wp, alt, FOV_DEG, FIELD_W, FIELD_H);
    max_overlap_vals(j)  = max(ov(:));
    mean_overlap_vals(j) = mean(ov(ov > 0));

    nexttile;
    imagesc([0 FIELD_W],[0 FIELD_H], ov);
    colormap(hot); colorbar;
    set(gca,'YDir','normal');
    title(sprintf('%dm (max=%d)', alt, max(ov(:))));
    xlabel('X (m)'); ylabel('Y (m)');
end
sgtitle('Analysis 4 — Camera Footprint Overlap Count per Field Cell');
print(fig, fullfile(outDir, 'adv4_overlap_analysis.png'), '-dpng', '-r150');
savefig(fig, fullfile(outDir, 'adv4_overlap_analysis.fig'));
close(fig);
fprintf('  Saved: adv4_overlap_analysis.png\n');

% Save overlap stats CSV
fid = fopen(fullfile(outDir, 'adv4_overlap_stats.csv'), 'w');
fprintf(fid, 'Altitude_m,MaxOverlapCount,MeanOverlapCount\n');
for j = 1:length(ALTITUDES)
    fprintf(fid, '%d,%d,%.2f\n', ALTITUDES(j), max_overlap_vals(j), mean_overlap_vals(j));
end
fclose(fid);

% ══════════════════════════════════════════════════════════════════════════
%% ANALYSIS 5: Drone speed vs effective detection quality
% ══════════════════════════════════════════════════════════════════════════
fprintf('\n--- Analysis 5: Speed vs detection quality ---\n');

speeds_ms    = [2.0, 3.0, 5.0, 7.0, 10.0];
% Blur factor: faster = more motion blur = effective confidence penalty
% Model: blur_factor = 1 - 0.022*(speed-2)  (empirical, saturates at 0)
blur_factors = max(0.5, 1.0 - 0.022 * (speeds_ms - 2.0));

% Apply blur as a multiplicative confidence penalty to existing detections
% At each speed, recompute "effective mAP50" proxy using teacher @ 15m
[det15, gt15] = load_detections(resultsJson, 'teacher', 15);

speed_eff_map = zeros(length(speeds_ms), 1);
speed_path    = zeros(length(speeds_ms), 1);
speed_time    = zeros(length(speeds_ms), 1);

[~, wp15, ~] = setup_scenario(resultsJson, 15);
base_path = sum(sqrt(sum(diff(wp15).^2, 2)));

for si = 1:length(speeds_ms)
    bf = blur_factors(si);
    % Apply blur factor to scores
    det_blurred = det15;
    for i = 1:length(det_blurred)
        det_blurred(i).scores = det_blurred(i).scores * bf;
    end

    % Count detections still above 0.25 threshold after blur
    total_kept = 0;
    for i = 1:length(det_blurred)
        total_kept = total_kept + sum(det_blurred(i).scores >= 0.25);
    end

    % Estimate mAP proxy: scale linearly with kept detections vs base
    base_total = 0;
    for i = 1:length(det15)
        base_total = base_total + length(det15(i).scores);
    end
    if base_total > 0
        speed_eff_map(si) = 0.786 * (total_kept / max(base_total,1)) * bf;
    else
        speed_eff_map(si) = 0;
    end

    speed_path(si) = base_path;
    speed_time(si) = base_path / speeds_ms(si) / 60;  % mins for this partial pass
end

% Save CSV
fid = fopen(fullfile(outDir, 'adv5_speed_quality.csv'), 'w');
fprintf(fid, 'Speed_ms,BlurFactor,EffectivemAP50Proxy,PathLength_m,FlightTime_min\n');
for si = 1:length(speeds_ms)
    fprintf(fid, '%.1f,%.3f,%.4f,%.1f,%.2f\n', ...
        speeds_ms(si), blur_factors(si), speed_eff_map(si), speed_path(si), speed_time(si));
    fprintf('  %.1f m/s: blur=%.3f  eff_mAP~%.3f  time=%.1f min\n', ...
        speeds_ms(si), blur_factors(si), speed_eff_map(si), speed_time(si));
end
fclose(fid);

fig = figure('Visible','off','Position',[100 100 900 420]);
subplot(1,2,1);
yyaxis left;
plot(speeds_ms, speed_eff_map, 'bo-', 'LineWidth', 2, 'MarkerSize', 7);
ylabel('Effective mAP@0.50 (proxy)');
yyaxis right;
plot(speeds_ms, speed_time, 'rs--', 'LineWidth', 2, 'MarkerSize', 7);
ylabel('Flight time (min)');
xlabel('Drone speed (m/s)');
title('Detection quality vs speed @ 15 m');
legend({'Effective mAP50','Flight time'},'Location','east');
grid on;
xline(3.0, 'k--', '3 m/s baseline', 'LabelVerticalAlignment','bottom');

subplot(1,2,2);
bar(speeds_ms, blur_factors*100, 'FaceColor',[0.3 0.7 0.4]);
xlabel('Drone speed (m/s)');
ylabel('Effective image quality (%)');
title('Motion blur degradation model');
grid on;
sgtitle('Analysis 5 — Speed vs Detection Quality (teacher @ 15 m)');
print(fig, fullfile(outDir, 'adv5_speed_quality.png'), '-dpng', '-r150');
savefig(fig, fullfile(outDir, 'adv5_speed_quality.fig'));
close(fig);
fprintf('  Saved: adv5_speed_quality.png\n');



% ══════════════════════════════════════════════════════════════════════════
%% ANALYSIS 7: Battery model — corrected to 500x500m field / 400m range
% ══════════════════════════════════════════════════════════════════════════
fprintf('\n--- Analysis 7: Battery model (500x500m field, 400m battery) ---\n');

BIG_FIELD_W = 500; BIG_FIELD_H = 500;
batt_cycles_big = zeros(1, length(ALTITUDES));
batt_cov_big    = zeros(1, length(ALTITUDES));
batt_time_big   = zeros(1, length(ALTITUDES));
batt_path_big   = zeros(1, length(ALTITUDES));

for j = 1:length(ALTITUDES)
    alt = ALTITUDES(j);
    [wp, ~, plen] = generate_full_coverage_path(alt, FOV_DEG, BIG_FIELD_W, BIG_FIELD_H);
    cyc = ceil(plen / BATTERY_RANGE);
    field_area = BIG_FIELD_W * BIG_FIELD_H;
    cov_per_cyc = field_area / cyc;
    batt_cycles_big(j) = cyc;
    batt_cov_big(j)    = cov_per_cyc;
    batt_time_big(j)   = plen / DRONE_SPEED / 60;
    batt_path_big(j)   = plen;
    fprintf('  Alt %dm: %d battery cycles | %.0f m path | %.1f min | %.0f m²/cycle\n', ...
        alt, cyc, plen, batt_time_big(j), cov_per_cyc);
end

% Save CSV
fid = fopen(fullfile(outDir, 'adv7_battery_model_500m.csv'), 'w');
fprintf(fid, 'Altitude_m,BatteryCycles,PathLength_m,TotalFlightTime_min,AreaPerCycle_m2\n');
for j = 1:length(ALTITUDES)
    fprintf(fid, '%d,%d,%.1f,%.2f,%.0f\n', ...
        ALTITUDES(j), batt_cycles_big(j), batt_path_big(j), ...
        batt_time_big(j), batt_cov_big(j));
end
fclose(fid);

fig = figure('Visible','off','Position',[100 100 850 450]);
subplot(1,2,1);
b = bar(ALTITUDES, batt_cycles_big, 'FaceColor',[0.2 0.5 0.9]);
xlabel('Altitude (m)'); ylabel('Battery cycles required');
title(sprintf('Battery cycles — %dx%dm field, %dm range', ...
    BIG_FIELD_W, BIG_FIELD_H, round(BATTERY_RANGE)));
grid on;
xtips = b.XEndPoints; ytips = b.YEndPoints;
text(xtips, ytips, string(batt_cycles_big), ...
    'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',11,'FontWeight','bold');

subplot(1,2,2);
yyaxis left;
plot(ALTITUDES, batt_time_big, 'bo-','LineWidth',2,'MarkerSize',8);
ylabel('Total flight time (min)');
yyaxis right;
plot(ALTITUDES, batt_cov_big/10000, 'rs--','LineWidth',2,'MarkerSize',8);
ylabel('Area per battery cycle (ha)');
xlabel('Altitude (m)');
title('Flight time & area efficiency');
legend({'Flight time','Area/cycle'},'Location','northwest');
grid on;
sgtitle('Analysis 7 — Battery Model (500×500 m Field)');
print(fig, fullfile(outDir, 'adv7_battery_model_500m.png'), '-dpng', '-r150');
savefig(fig, fullfile(outDir, 'adv7_battery_model_500m.fig'));
close(fig);
fprintf('  Saved: adv7_battery_model_500m.png\n');

% ══════════════════════════════════════════════════════════════════════════
%% ANALYSIS 8: Combined Pareto — all models, mAP vs full-survey time
%% (The key paper figure — accuracy–efficiency trade-off)
% ══════════════════════════════════════════════════════════════════════════
fprintf('\n--- Analysis 8: Full Pareto — all models x altitudes ---\n');

pareto_rows = {};
fig = figure('Visible','off','Position',[100 100 900 580]);
hold on;
for mi = 1:length(MODEL_NAMES)
    mn = MODEL_NAMES{mi};
    mAP_vals = zeros(1, length(ALTITUDES));
    time_vals = zeros(1, length(ALTITUDES));
    cov_vals  = zeros(1, length(ALTITUDES));
    for j = 1:length(ALTITUDES)
        alt = ALTITUDES(j);
        altStr = num2str(alt);
        if isfield(resultsJson.mAP50_by_model_altitude, mn)
            ms = resultsJson.mAP50_by_model_altitude.(mn);
            key = ['x' altStr];
            if isfield(ms, key), mAP_vals(j) = ms.(key);
            elseif isfield(ms, altStr), mAP_vals(j) = ms.(altStr);
            end
        end
        [~, ~, plen] = generate_full_coverage_path(alt, FOV_DEG, FIELD_W, FIELD_H);
        time_vals(j) = plen / DRONE_SPEED / 60;
        cov_vals(j)  = eff_cover(j);

        pareto_rows{end+1} = {mn, alt, mAP_vals(j), time_vals(j), cov_vals(j)}; %#ok<AGROW>
    end
    clr = COLORS7{mi};
    if isnumeric(clr)
        plot(time_vals, mAP_vals, 'o-', 'Color', clr, 'LineWidth', 1.5, ...
            'MarkerSize', 6, 'DisplayName', mn);
    else
        plot(time_vals, mAP_vals, [clr 'o-'], 'LineWidth', 1.5, ...
            'MarkerSize', 6, 'DisplayName', mn);
    end
    % Label altitudes on teacher curve only
    if strcmp(mn, 'teacher')
        for j = 1:length(ALTITUDES)
            text(time_vals(j)+0.05, mAP_vals(j)+0.003, ...
                sprintf('%dm', ALTITUDES(j)), 'FontSize', 8);
        end
    end
end
hold off;
xlabel('Full-survey flight time at 3 m/s (min)');
ylabel('mAP@0.50');
title('Accuracy–Efficiency Pareto: all models × altitudes (100×100 m field)');
legend('Location','southeast','Interpreter','none','FontSize',8);
grid on;
sgtitle('Analysis 8 — Full Pareto Front');
print(fig, fullfile(outDir, 'adv8_pareto_all_models.png'), '-dpng', '-r150');
savefig(fig, fullfile(outDir, 'adv8_pareto_all_models.fig'));
close(fig);

% Save CSV
fid = fopen(fullfile(outDir, 'adv8_pareto_all_models.csv'), 'w');
fprintf(fid, 'Model,Altitude_m,mAP50,FlightTime_min,CoveragePercent\n');
for k = 1:length(pareto_rows)
    r = pareto_rows{k};
    fprintf(fid, '%s,%d,%.4f,%.2f,%.2f\n', r{1}, r{2}, r{3}, r{4}, r{5});
end
fclose(fid);
fprintf('  Saved: adv8_pareto_all_models.png\n');

% ══════════════════════════════════════════════════════════════════════════
%% COMPLETE — Print summary of all outputs
% ══════════════════════════════════════════════════════════════════════════
fprintf('\n=== ADVANCED SIMULATIONS COMPLETE ===\n');
fprintf('All outputs saved in: %s\n\n', outDir);
fprintf('  CSVs:\n');
fprintf('    adv1_full_coverage_efficiency.csv\n');
fprintf('    adv2_path_comparison.csv\n');
fprintf('    adv4_overlap_stats.csv\n');
fprintf('    adv5_speed_quality.csv\n');
fprintf('    adv7_battery_model_500m.csv\n');
fprintf('    adv8_pareto_all_models.csv\n');
fprintf('    metrics_summary.csv  (from main_simulation.m — previous run)\n');
fprintf('\n  Figures:\n');
fprintf('    adv1, adv2, adv4, adv5, adv7, adv8: *.png + *.fig\n');

% bwr_colormap is in utils/bwr_colormap.m (added via addpath above)
