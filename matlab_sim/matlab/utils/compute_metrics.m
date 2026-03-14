function metrics = compute_metrics(detectionLog, gtLog, fieldWidthM, fieldHeightM, gridResM)
% compute_metrics Calculates coverage and detection statistics

    if nargin < 5
        gridResM = 1.0;
    end
    
    gt_patches_total = length(gtLog);
    gt_patches_detected = 0;
    total_detections = 0;
    
    % Gather all detection centres
    all_det_x = [];
    all_det_y = [];
    
    if ~isempty(detectionLog)
        for i = 1:length(detectionLog)
            boxes = detectionLog(i).boxes_field;
            if isempty(boxes) || size(boxes, 2) < 4
                continue;
            end
            % boxes are x1, y1, x2, y2 in field coordinates
            cx = (boxes(:, 1) + boxes(:, 3)) / 2;
            cy = (boxes(:, 2) + boxes(:, 4)) / 2;
            all_det_x = [all_det_x; cx]; %#ok<AGROW>
            all_det_y = [all_det_y; cy]; %#ok<AGROW>
            total_detections = total_detections + size(boxes, 1);
        end
    end
    
    % Check each GT patch: any detection within 2m is a "hit"
    for i = 1:length(gtLog)
        patch_x = gtLog(i).field_x;
        patch_y = gtLog(i).field_y;
        
        if isempty(all_det_x)
            continue;
        end
        
        dists = sqrt((all_det_x - patch_x).^2 + (all_det_y - patch_y).^2);
        if any(dists <= 2.0)
            gt_patches_detected = gt_patches_detected + 1;
        end
    end
    
    gt_patches_missed = gt_patches_total - gt_patches_detected;
    field_area_ha = (fieldWidthM * fieldHeightM) / 10000;
    if field_area_ha > 0
        missed_patches_per_hectare = gt_patches_missed / field_area_ha;
    else
        missed_patches_per_hectare = 0;
    end
    
    metrics = struct();
    metrics.gt_patches_total = gt_patches_total;
    metrics.gt_patches_detected = gt_patches_detected;
    metrics.gt_patches_missed = gt_patches_missed;
    metrics.missed_patches_per_hectare = missed_patches_per_hectare;
    metrics.total_detections = total_detections;
end
