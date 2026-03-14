function generate_figures(resultsJson, sweepResults)
% generate_figures Generates and saves all required plots

    scriptDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(scriptDir, 'utils'));
    
    outDir = fullfile(scriptDir, 'figures');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    models = sweepResults.model_names;
    alts = sweepResults.altitudes;
    
    colors  = {'k', 'b', 'r', 'g', 'c', 'm', [1 0.5 0]};
    markers = {'o', 's', 'd', '^', 'v', 'p', 'h'};
    
    %% Figure 1: Altitude vs mAP50 curve
    fig1 = figure('Name', 'Altitude vs mAP50', 'Visible', 'off', 'Position', [100, 100, 800, 600]);
    hold on;
    for i = 1:length(models)
        if isnumeric(colors{i})
            plot(alts, sweepResults.mAP50_matrix(i, :), '-', 'Color', colors{i}, ...
                'Marker', markers{i}, 'LineWidth', 2, 'DisplayName', models{i});
        else
            plot(alts, sweepResults.mAP50_matrix(i, :), [colors{i} markers{i} '-'], ...
                'LineWidth', 2, 'DisplayName', models{i});
        end
    end
    hold off; grid on;
    xlabel('Altitude (m)'); ylabel('mAP@0.50');
    title('Detection performance vs simulated drone altitude');
    legend('Location', 'southwest', 'Interpreter', 'none');
    print(fig1, fullfile(outDir, 'Figure1_mAP50.png'), '-dpng', '-r150');
    savefig(fig1, fullfile(outDir, 'Figure1_mAP50.fig'));
    close(fig1);
    fprintf('  Saved Figure 1: mAP50 vs Altitude\n');
    
    %% Figure 2: Coverage heatmap for best KD model at 15m
    kdIdx = min(4, length(models));   % 'nano_feature_kd' is index 4
    mName = models{kdIdx};
    alt = 15;
    
    [detectionLog, gtLog] = load_detections(resultsJson, mName, alt);
    [~, waypoints, ~] = setup_scenario(resultsJson, alt);
    [~, detectionMap, missedMap] = build_coverage_map(detectionLog, gtLog, waypoints, alt, ...
        resultsJson.metadata.camera_fov_deg, 100, 100);
    
    fig2 = figure('Name', 'Coverage Heatmap', 'Visible', 'off', 'Position', [100, 100, 800, 600]);
    imagesc([0 100], [0 100], detectionMap);
    colormap(hot); colorbar;
    set(gca, 'YDir', 'normal');
    hold on;
    for i = 1:length(gtLog)
        plot(gtLog(i).field_x, gtLog(i).field_y, 'g+', 'MarkerSize', 8, 'LineWidth', 2);
    end
    [my, mx] = find(missedMap > 0);
    if ~isempty(my)
        plot(mx, my, 'wx', 'MarkerSize', 8, 'LineWidth', 2);
    end
    xlabel('Field X (m)'); ylabel('Field Y (m)');
    title(sprintf('Coverage Heatmap: %s @ %dm', mName, alt), 'Interpreter', 'none');
    hold off;
    print(fig2, fullfile(outDir, 'Figure2_Coverage_Heatmap.png'), '-dpng', '-r150');
    savefig(fig2, fullfile(outDir, 'Figure2_Coverage_Heatmap.fig'));
    close(fig2);
    fprintf('  Saved Figure 2: Coverage Heatmap\n');
    
    %% Figure 3: Coverage heatmap comparison grid
    fig3 = figure('Name', 'Comparison Grid', 'Visible', 'off', 'Position', [100, 100, 1600, 800]);
    numCols = min(4, length(models));
    numRows = ceil(length(models) / numCols);
    t = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact'); %#ok<NASGU>
    
    alt = 15;
    maxDet = 0;
    allMaps = cell(length(models), 1);
    for i = 1:length(models)
        [detLog, gLog] = load_detections(resultsJson, models{i}, alt);
        [~, wp, ~]     = setup_scenario(resultsJson, alt);
        [~, dMap, ~]   = build_coverage_map(detLog, gLog, wp, alt, ...
            resultsJson.metadata.camera_fov_deg, 100, 100);
        allMaps{i} = dMap;
        maxDet = max(maxDet, max(dMap(:)));
    end
    if maxDet == 0, maxDet = 1; end  % avoid zero-range colorbar
    
    for i = 1:length(models)
        nexttile;
        imagesc([0 100], [0 100], allMaps{i});
        colormap(hot); caxis([0 maxDet]);
        set(gca, 'YDir', 'normal');
        title(models{i}, 'Interpreter', 'none');
        xlabel('X (m)'); ylabel('Y (m)');
    end
    cb = colorbar;
    cb.Layout.Tile = 'east';
    print(fig3, fullfile(outDir, 'Figure3_Comparison_Grid.png'), '-dpng', '-r150');
    savefig(fig3, fullfile(outDir, 'Figure3_Comparison_Grid.fig'));
    close(fig3);
    fprintf('  Saved Figure 3: Comparison Grid\n');
    
    %% Figure 4: Missed patches per hectare bar chart
    fig4 = figure('Name', 'Missed Patches', 'Visible', 'off', 'Position', [100, 100, 800, 600]);
    bar(sweepResults.missed_patches_matrix');
    set(gca, 'XTickLabel', string(alts));
    xlabel('Altitude (m)'); ylabel('Missed patches / ha');
    title('Missed patches per hectare across altitudes');
    legend(models, 'Location', 'northwest', 'Interpreter', 'none');
    grid on;
    print(fig4, fullfile(outDir, 'Figure4_Missed_Patches.png'), '-dpng', '-r150');
    savefig(fig4, fullfile(outDir, 'Figure4_Missed_Patches.fig'));
    close(fig4);
    fprintf('  Saved Figure 4: Missed Patches\n');
    
    %% Figure 5: Field coverage percentage vs altitude
    fig5 = figure('Name', 'Field Coverage', 'Visible', 'off', 'Position', [100, 100, 800, 600]);
    hold on;
    for i = 1:length(models)
        if isnumeric(colors{i})
            plot(alts, sweepResults.coverage_matrix(i, :), '-', 'Color', colors{i}, ...
                'Marker', markers{i}, 'LineWidth', 2, 'DisplayName', models{i});
        else
            plot(alts, sweepResults.coverage_matrix(i, :), [colors{i} markers{i} '-'], ...
                'LineWidth', 2, 'DisplayName', models{i});
        end
    end
    hold off; grid on;
    xlabel('Altitude (m)'); ylabel('Field Coverage %');
    title('Field coverage vs simulated drone altitude');
    legend('Location', 'southwest', 'Interpreter', 'none');
    print(fig5, fullfile(outDir, 'Figure5_Field_Coverage.png'), '-dpng', '-r150');
    savefig(fig5, fullfile(outDir, 'Figure5_Field_Coverage.fig'));
    close(fig5);
    fprintf('  Saved Figure 5: Field Coverage\n');
    
    %% Figure 6: UAV Scenario 3D visualisation (requires UAV Toolbox)
    [scenario, waypoints6, ~] = setup_scenario(resultsJson, 15);
    fig6 = figure('Name', '3D Scenario', 'Visible', 'off', 'Position', [100, 100, 800, 600]);
    try
        if isstruct(scenario) && isfield(scenario, 'type')
            % Placeholder — UAV Toolbox unavailable
            text(0.5, 0.5, 'UAV Toolbox not available', 'Units', 'normalized', ...
                'HorizontalAlignment', 'center', 'FontSize', 14);
            axis off;
        else
            show3D(scenario);
            hold on;
            plot3(waypoints6(:,1), waypoints6(:,2), waypoints6(:,3), 'b-', 'LineWidth', 2);
            plot3(waypoints6(:,1), waypoints6(:,2), waypoints6(:,3), 'r.', 'MarkerSize', 10);
            hold off;
        end
    catch
        text(0.5, 0.5, 'UAV Toolbox not available', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
    end
    title('UAV Scenario Visualization at 15m');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    print(fig6, fullfile(outDir, 'Figure6_3D_Scenario.png'), '-dpng', '-r150');
    savefig(fig6, fullfile(outDir, 'Figure6_3D_Scenario.fig'));
    close(fig6);
    fprintf('  Saved Figure 6: 3D Scenario\n');
    
    %% Figure 7: Battery cycles vs altitude bar chart
    cycles  = zeros(1, length(alts));
    cov_per = zeros(1, length(alts));
    for i = 1:length(alts)
        [~, wp, ~]         = setup_scenario(resultsJson, alts(i));
        [cyc, cov]         = battery_model(wp, alts(i), 3.0, 3000.0);
        cycles(i)          = cyc;
        cov_per(i)         = cov;
    end
    
    fig7 = figure('Name', 'Battery Cycles', 'Visible', 'off', 'Position', [100, 100, 800, 600]);
    b = bar(alts, cycles);
    xlabel('Altitude (m)'); ylabel('Battery Cycles Required');
    title('Battery usage vs altitude');
    xtips  = b.XEndPoints;
    ytips  = b.YEndPoints;
    labels = string(round(cov_per)) + " m^2/cycle";
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    print(fig7, fullfile(outDir, 'Figure7_Battery_Cycles.png'), '-dpng', '-r150');
    savefig(fig7, fullfile(outDir, 'Figure7_Battery_Cycles.fig'));
    close(fig7);
    fprintf('  Saved Figure 7: Battery Cycles\n');
end
