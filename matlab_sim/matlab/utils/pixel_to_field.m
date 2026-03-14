function [field_boxes] = pixel_to_field(pixel_boxes, field_cx, field_cy, altitude, fov_deg, input_size)
% pixel_to_field Converts normalized pixel boxes to absolute field coordinates
% field_cx, field_cy are the drone's centre location on the field (m)
% pixel_boxes: Nx4 [x1, y1, x2, y2] normalized 0-1

    if isempty(pixel_boxes)
        field_boxes = [];
        return;
    end

    % Camera footprint in metres
    footprint = 2 * altitude * tand(fov_deg / 2);
    
    % Compute offsets from image centre (normalized -0.5 to 0.5)
    % then multiply by footprint to get metres
    % x1, y1, x2, y2
    
    offset_x1 = (pixel_boxes(:, 1) - 0.5) * footprint;
    offset_y1 = (pixel_boxes(:, 2) - 0.5) * footprint;
    offset_x2 = (pixel_boxes(:, 3) - 0.5) * footprint;
    offset_y2 = (pixel_boxes(:, 4) - 0.5) * footprint;
    
    field_x1 = field_cx + offset_x1;
    field_y1 = field_cy + offset_y1;
    field_x2 = field_cx + offset_x2;
    field_y2 = field_cy + offset_y2;
    
    field_boxes = [field_x1, field_y1, field_x2, field_y2];
end
