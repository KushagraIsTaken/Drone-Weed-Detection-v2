import os
import glob
import json
import math
import time
import datetime
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import onnxruntime as ort

# Configuration block
ALTITUDES = [10, 15, 20, 25, 30]  # metres
CONF_THRESHOLD = 0.001           # Use low threshold for accurate mAP
SAVE_CONF_THRESHOLD = 0.10       # Save detections >= 0.10 for MATLAB conf-sweep analysis
IOU_THRESHOLD = 0.65             # NMS IoU threshold (standard for evaluation)
INPUT_SIZE = 832
FIELD_WIDTH_M = 100.0
FIELD_HEIGHT_M = 100.0
CAMERA_FOV_DEG = 60.0
MODEL_NAMES = ['teacher', 'small_baseline', 'nano_baseline',
               'nano_feature_kd', 'small_feature_kd', 'nano_pseudo_kd', 'small_pseudo_kd']

def simulate_altitude(image, base_altitude=10, target_altitude=15, input_size=832):
    if target_altitude <= base_altitude:
        if image.shape[0] != input_size or image.shape[1] != input_size:
            return cv2.resize(image, (input_size, input_size))
        return image.copy()
        
    scale_factor = base_altitude / target_altitude
    new_h = int(input_size * scale_factor)
    new_w = int(input_size * scale_factor)
    
    scaled_img = cv2.resize(image, (new_w, new_h))
    
    pad_h = input_size - new_h
    pad_w = input_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Use gray padding (114) common in YOLO training
    padded_img = cv2.copyMakeBorder(scaled_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    return padded_img

def run_onnx_inference(session, image_np, conf_threshold, iou_threshold):
    img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    
    raw_out = outputs[0][0] # [4+num_classes, num_anchors]
    boxes_raw = raw_out[:4, :]
    scores_raw = raw_out[4:, :]
    
    # Check if sigmoid is needed (some exports already have it)
    if np.max(scores_raw) > 1.0 or np.min(scores_raw) < 0.0:
        scores_raw = 1.0 / (1.0 + np.exp(-scores_raw))
    
    max_scores = np.max(scores_raw, axis=0)
    class_ids = np.argmax(scores_raw, axis=0)
    
    mask = max_scores >= conf_threshold
    if not np.any(mask):
        return []
        
    cx = boxes_raw[0, mask]
    cy = boxes_raw[1, mask]
    w = boxes_raw[2, mask]
    h = boxes_raw[3, mask]
    f_scores = max_scores[mask]
    f_class_ids = class_ids[mask]
    
    # Normalize pixel coords to 0-1
    x1 = (cx - w / 2) / INPUT_SIZE
    y1 = (cy - h / 2) / INPUT_SIZE
    x2 = (cx + w / 2) / INPUT_SIZE
    y2 = (cy + h / 2) / INPUT_SIZE
    
    detections = []
    unique_classes = np.unique(f_class_ids)
    for c in unique_classes:
        c_mask = f_class_ids == c
        c_inds = np.where(c_mask)[0]
        
        c_x1, c_y1, c_x2, c_y2 = x1[c_mask], y1[c_mask], x2[c_mask], y2[c_mask]
        c_s = f_scores[c_mask]
        
        order = np.argsort(c_s)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(c_inds[i])
            if order.size == 1: break
            
            xx1 = np.maximum(c_x1[i], c_x1[order[1:]])
            yy1 = np.maximum(c_y1[i], c_y1[order[1:]])
            xx2 = np.minimum(c_x2[i], c_x2[order[1:]])
            yy2 = np.minimum(c_y2[i], c_y2[order[1:]])
            
            w_int = np.maximum(0.0, xx2 - xx1)
            h_int = np.maximum(0.0, yy2 - yy1)
            inter = w_int * h_int
            area_i = (c_x2[i] - c_x1[i]) * (c_y2[i] - c_y1[i])
            areas = (c_x2[order[1:]] - c_x1[order[1:]]) * (c_y2[order[1:]] - c_y1[order[1:]])
            union = area_i + areas - inter
            iou = inter / (union + 1e-6)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        for idx in keep:
            detections.append({
                "box": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                "score": float(f_scores[idx]),
                "class_id": int(f_class_ids[idx])
            })
    return detections

def pixel_to_field(pixel_boxes, field_x, field_y, altitude, fov_deg):
    footprint = 2 * altitude * math.tan(math.radians(fov_deg / 2))
    field_boxes = []
    for box in pixel_boxes:
        off_x1, off_y1 = (box[0] - 0.5) * footprint, (box[1] - 0.5) * footprint
        off_x2, off_y2 = (box[2] - 0.5) * footprint, (box[3] - 0.5) * footprint
        fx1 = min(max(field_x + off_x1, 0.0), FIELD_WIDTH_M)
        fy1 = min(max(field_y + off_y1, 0.0), FIELD_HEIGHT_M)
        fx2 = min(max(field_x + off_x2, 0.0), FIELD_WIDTH_M)
        fy2 = min(max(field_y + off_y2, 0.0), FIELD_HEIGHT_M)
        field_boxes.append([fx1, fy1, fx2, fy2])
    return field_boxes

def box_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_map50(all_detections, all_gt):
    all_classes = set([d['class_id'] for d in all_detections])
    for gts in all_gt.values():
        all_classes.update([g['class_id'] for g in gts])
    if not all_classes: return 0.0
    
    aps = []
    for c in all_classes:
        class_gt_count = 0
        img_gts = {}
        for img_id, gts in all_gt.items():
            img_gts[img_id] = [g.copy() for g in gts if g['class_id'] == c]
            for g in img_gts[img_id]: g['matched'] = False
            class_gt_count += len(img_gts[img_id])
        if class_gt_count == 0: continue
        
        class_dets = sorted([d for d in all_detections if d['class_id'] == c], key=lambda x: x['score'], reverse=True)
        tp, fp = np.zeros(len(class_dets)), np.zeros(len(class_dets))
        
        for i, d in enumerate(class_dets):
            best_iou, best_gt_idx = 0.5, -1
            for j, gt in enumerate(img_gts.get(d['image_id'], [])):
                iou = box_iou(d['box'], gt['box'])
                if iou > best_iou: best_iou, best_gt_idx = iou, j
            if best_gt_idx != -1 and not img_gts[d['image_id']][best_gt_idx]['matched']:
                tp[i], img_gts[d['image_id']][best_gt_idx]['matched'] = 1, True
            else: fp[i] = 1
            
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        recalls, precisions = tp_c / class_gt_count, tp_c / np.maximum(tp_c + fp_c, 1e-16)
        
        # All-point interpolation (COCO style)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        i = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
        
    return sum(aps) / len(aps) if aps else 0.0

def main():
    print("=== UAV Weed Detection Simulation ===")
    base_dir = Path(__file__).parent.parent
    data_dir, mod_dir, res_dir = base_dir / "data", base_dir / "models", base_dir / "python/results"
    os.makedirs(res_dir, exist_ok=True)
    
    with open(data_dir / "classes.txt", 'r') as f:
        classes = [l.strip() for l in f if l.strip()]
        
    img_files = list(Path(data_dir / "test_images").glob("*.jpg")) + list(Path(data_dir / "test_images").glob("*.JPG"))
    test_images = []
    for img_path in img_files:
        lbl_path = data_dir / "test_labels" / (img_path.stem + ".txt")
        if not lbl_path.exists(): continue
        gt_b, gt_c = [], []
        with open(lbl_path, 'r') as f:
            for l in f:
                p = l.strip().split()
                if len(p) >= 5:
                    c, cx, cy, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    gt_b.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
                    gt_c.append(c)
        test_images.append({'image_id': img_path.stem, 'filepath': str(img_path), 'gt_boxes': gt_b, 'gt_classes': gt_c})
    
    grid_size = math.ceil(math.sqrt(len(test_images)))
    sx, sy = FIELD_WIDTH_M / max(1, grid_size-1), FIELD_HEIGHT_M / max(1, grid_size-1)
    for i, r in enumerate(test_images):
        r['field_x'], r['field_y'] = (i % grid_size) * sx, (i // grid_size) * sy

    all_dets_json, map_res = [], {m: {} for m in MODEL_NAMES}
    gt_json = [{"image_id": r['image_id'], "field_x": r['field_x'], "field_y": r['field_y'], "boxes": r['gt_boxes'], "class_ids": r['gt_classes']} for r in test_images]
    
    # ── GPU / accelerator setup ────────────────────────────────────────────
    avail = ort.get_available_providers()
    print(f"Available providers: {avail}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    sess_options.log_severity_level = 3  # suppress verbose logs

    if 'CoreMLExecutionProvider' in avail:
        # MLComputeUnits=ALL lets CoreML choose ANE, GPU, or CPU per op
        providers = [
            ('CoreMLExecutionProvider', {
                'MLComputeUnits': 'ALL',
                'ModelFormat': 'MLProgram',
            }),
            'CPUExecutionProvider'
        ]
        print("Using CoreML (Apple Neural Engine + GPU) acceleration")
    elif 'CUDAExecutionProvider' in avail:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("Using CUDA GPU acceleration")
    else:
        providers = ['CPUExecutionProvider']
        print("Using CPU only")

    import onnx
    for m_name in MODEL_NAMES:
        m_path = mod_dir / (m_name + ".onnx")
        if not m_path.exists() or os.path.getsize(m_path) < 1000: continue
        print(f"\nLoading model: {m_name}")
        session = ort.InferenceSession(str(m_path), sess_options=sess_options, providers=providers)
        active = session.get_providers()
        print(f"  Active providers: {active}")
        
        for alt in ALTITUDES:
            print(f"  -> {alt}m")
            all_dets_eval, all_gt_eval = [], {}
            s_fact = 10.0 / alt
            pad = (INPUT_SIZE - (INPUT_SIZE * s_fact)) / 2
            
            for r in tqdm(test_images, desc=f"{m_name} @ {alt}m"):
                img = cv2.imread(r['filepath'])
                all_gt_eval[r['image_id']] = [{'box': [b[0]*s_fact+pad/INPUT_SIZE, b[1]*s_fact+pad/INPUT_SIZE, b[2]*s_fact+pad/INPUT_SIZE, b[3]*s_fact+pad/INPUT_SIZE], 'class_id': c} for b, c in zip(r['gt_boxes'], r['gt_classes'])]
                
                dets = run_onnx_inference(session, simulate_altitude(img, 10, alt, INPUT_SIZE), CONF_THRESHOLD, IOU_THRESHOLD)
                for d in dets:
                    all_dets_eval.append({"image_id": r['image_id'], "box": d['box'], "score": d['score'], "class_id": d['class_id']})
                    if d['score'] >= SAVE_CONF_THRESHOLD:
                        f_boxes = pixel_to_field([d['box']], r['field_x'], r['field_y'], alt, CAMERA_FOV_DEG)
                        all_dets_json.append({"image_id": r['image_id'], "model": m_name, "altitude": alt, "field_x": r['field_x'], "field_y": r['field_y'], "boxes_pixel": [d['box']], "boxes_field": f_boxes, "scores": [d['score']], "class_ids": [d['class_id']]})
            
            map_res[m_name][str(alt)] = compute_map50(all_dets_eval, all_gt_eval)
            print(f"     mAP@0.50: {map_res[m_name][str(alt)]:.3f}")
            
    with open(res_dir / "results.json", 'w') as f:
        json.dump({"metadata": {"altitudes": ALTITUDES, "models": MODEL_NAMES, "classes": classes, "input_size": INPUT_SIZE, "field_width_m": FIELD_WIDTH_M, "field_height_m": FIELD_HEIGHT_M, "camera_fov_deg": CAMERA_FOV_DEG, "generated_at": datetime.datetime.now().isoformat()}, "field_positions": [{"image_id": r['image_id'], "field_x": r['field_x'], "field_y": r['field_y']} for r in test_images], "detections": all_dets_json, "mAP50_by_model_altitude": map_res, "gt_boxes": gt_json}, f)
        
if __name__ == "__main__":
    main()
