data = jsondecode(fileread('../python/results/results.json'));
f1 = data.mAP50_by_model_altitude.small_baseline;
f2 = data.mAP50_by_model_altitude.small_feature_kd;
disp('small_baseline:'); disp(f1);
disp('small_feature_kd:'); disp(f2);
