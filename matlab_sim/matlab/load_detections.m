function [detectionLog, gtLog] = load_detections(resultsJson, modelName, altitudeM)
% load_detections Filters the full decoded JSON struct for specific model and altitude

    if ~isfield(resultsJson, 'detections')
        error('resultsJson does not contain ''detections'' field.');
    end
    
    dets = resultsJson.detections;
    
    % Robust filtering for struct arrays
    if isstruct(dets) && ~isempty(dets)
        % Build logical mask per-element (safe for non-uniform struct arrays)
        numDets = length(dets);
        modelMask = false(1, numDets);
        altMask   = false(1, numDets);
        for k = 1:numDets
            modelMask(k) = strcmp(dets(k).model, modelName);
            altMask(k)   = (dets(k).altitude == altitudeM);
        end
        
        detectionLog = dets(modelMask & altMask);
        
        % Ensure boxes_field, scores, class_ids are proper numeric arrays
        for i = 1:length(detectionLog)
            % boxes_field: should be Nx4 double
            bf = detectionLog(i).boxes_field;
            if iscell(bf)
                if isempty(bf)
                    bf = zeros(0, 4);
                else
                    bf = cell2mat(bf);
                end
            end
            if ~isempty(bf)
                bf = reshape(bf, [], 4);
            end
            detectionLog(i).boxes_field = bf;
            
            % scores: should be column vector
            sc = detectionLog(i).scores;
            if iscell(sc), sc = cell2mat(sc); end
            detectionLog(i).scores = sc(:);
            
            % class_ids: should be column vector
            ci = detectionLog(i).class_ids;
            if iscell(ci), ci = cell2mat(ci); end
            detectionLog(i).class_ids = ci(:);
        end
        
    elseif iscell(dets)
        % Fallback for cell arrays (if JSON is non-uniform)
        detectionLog = [];
        for i = 1:length(dets)
            d = dets{i};
            if strcmp(d.model, modelName) && d.altitude == altitudeM
                bf = d.boxes_field;
                if iscell(bf)
                    if isempty(bf), bf = zeros(0,4); else, bf = cell2mat(bf); end
                end
                if ~isempty(bf), bf = reshape(bf, [], 4); end
                d.boxes_field = bf;
                
                sc = d.scores;
                if iscell(sc), sc = cell2mat(sc); end
                d.scores = sc(:);
                
                ci = d.class_ids;
                if iscell(ci), ci = cell2mat(ci); end
                d.class_ids = ci(:);
                
                if isempty(detectionLog)
                    detectionLog = d;
                else
                    detectionLog(end+1) = d; %#ok<AGROW>
                end
            end
        end
    else
        detectionLog = [];
    end
    
    % Process GT (Ground Truth — no model/altitude filtering needed)
    if isfield(resultsJson, 'gt_boxes')
        gts = resultsJson.gt_boxes;
        if iscell(gts)
            gtLog = [gts{:}];
        else
            gtLog = gts;
        end
        
        % Clean up GT box and class_id fields
        for i = 1:length(gtLog)
            bx = gtLog(i).boxes;
            if iscell(bx)
                if isempty(bx), bx = zeros(0, 4); else, bx = cell2mat(bx); end
            end
            if ~isempty(bx), bx = reshape(bx, [], 4); end
            gtLog(i).boxes = bx;
            
            ci = gtLog(i).class_ids;
            if iscell(ci), ci = cell2mat(ci); end
            gtLog(i).class_ids = ci(:);
        end
    else
        gtLog = [];
    end
end
