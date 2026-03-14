data = jsondecode(fileread('../python/results/results.json'));
f1 = data.mAP50_by_model_altitude.small_baseline;
f2 = data.mAP50_by_model_altitude.small_feature_kd;
fprintf('small_baseline: %.15f\n', f1.x10);
fprintf('small_feature_kd: %.15f\n', f2.x10);
