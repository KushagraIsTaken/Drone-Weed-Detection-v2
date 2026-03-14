function [boxes, scores, classIds] = yolo_postprocess(rawOutput, confThresh, iouThresh, numClasses)
% yolo_postprocess Processes raw YOLOv11 ONNX output into bounding boxes
% rawOutput shape: [4+numClasses, numAnchors]
% Boxes returned are in normalized pixel coordinates [x1, y1, x2, y2]

    % Extract coordinates and class logits
    cx = rawOutput(1, :);
    cy = rawOutput(2, :);
    w = rawOutput(3, :);
    h = rawOutput(4, :);
    
    classLogits = rawOutput(5:end, :);
    classScores = 1 ./ (1 + exp(-classLogits)); % sigmoid
    
    [maxScores, classIdsRaw] = max(classScores, [], 1);
    
    % Filter by confidence
    validIdx = maxScores >= confThresh;
    
    cx = cx(validIdx);
    cy = cy(validIdx);
    w = w(validIdx);
    h = h(validIdx);
    scores = maxScores(validIdx)';
    classIds = classIdsRaw(validIdx)' - 1; % 0-indexed to match Python
    
    if isempty(scores)
        boxes = [];
        return;
    end
    
    % Convert cx, cy, w, h to x1, y1, x2, y2
    x1 = cx - w / 2;
    y1 = cy - h / 2;
    x2 = cx + w / 2;
    y2 = cy + h / 2;
    
    boxes = [x1', y1', x2', y2'];
    
    % Apply NMS per class
    keepIdx = [];
    uniqueClasses = unique(classIds);
    for i = 1:length(uniqueClasses)
        c = uniqueClasses(i);
        cIdx = find(classIds == c);
        
        cBoxes = boxes(cIdx, :);
        cScores = scores(cIdx);
        
        % Sort by score
        [cScores, sortIdx] = sort(cScores, 'descend');
        cBoxes = cBoxes(sortIdx, :);
        originalIdx = cIdx(sortIdx);
        
        cKeep = [];
        while ~isempty(cBoxes)
            cKeep = [cKeep; originalIdx(1)]; %#ok<AGROW>
            if size(cBoxes, 1) == 1
                break;
            end
            
            % Compute IoU of highest score box with rest
            b1 = cBoxes(1, :);
            bRest = cBoxes(2:end, :);
            
            xx1 = max(b1(1), bRest(:, 1));
            yy1 = max(b1(2), bRest(:, 2));
            xx2 = min(b1(3), bRest(:, 3));
            yy2 = min(b1(4), bRest(:, 4));
            
            w_int = max(0, xx2 - xx1);
            h_int = max(0, yy2 - yy1);
            interArea = w_int .* h_int;
            
            area1 = (b1(3) - b1(1)) * (b1(4) - b1(2));
            areaRest = (bRest(:, 3) - bRest(:, 1)) .* (bRest(:, 4) - bRest(:, 2));
            unionArea = area1 + areaRest - interArea;
            
            iou = interArea ./ unionArea;
            
            % Keep those with IoU < threshold
            keepMask = iou < iouThresh;
            cBoxes = cBoxes([false; keepMask], :);
            cScores = cScores([false; keepMask]);
            originalIdx = originalIdx([false; keepMask]);
        end
        keepIdx = [keepIdx; cKeep]; %#ok<AGROW>
    end
    
    boxes = boxes(keepIdx, :);
    scores = scores(keepIdx);
    classIds = classIds(keepIdx);
end
