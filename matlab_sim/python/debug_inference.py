import onnxruntime as ort
import numpy as np
import cv2
import json
from pathlib import Path
import os

# ── Config ───────────────────────────────────────────────────────────────
base_dir = Path(__file__).parent.parent
MODEL_PATH  = base_dir / "models/teacher.onnx"
# Support both lowercase and uppercase extensions
TEST_IMAGES = sorted(list((base_dir / "data/test_images").glob("*.jpg")) + list((base_dir / "data/test_images").glob("*.JPG")))
TEST_LABELS = base_dir / "data/test_labels"
INPUT_SIZE  = 832
CONF_THRESH = 0.25

if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}")
    exit(1)
if len(TEST_IMAGES) == 0:
    print(f"No test images found in {base_dir / 'data/test_images'}")
    exit(1)

# ── Step 1: Load model, print raw I/O shapes ─────────────────────────────
print("=" * 60)
print("STEP 1 — Model I/O shapes")
print("=" * 60)
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

for inp in session.get_inputs():
    print(f"  Input  name={inp.name}  shape={inp.shape}  dtype={inp.type}")
for out in session.get_outputs():
    print(f"  Output name={out.name}  shape={out.shape}  dtype={out.type}")

# ── Step 2: Run one image, print raw output stats ────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Raw output tensor inspection (first test image)")
print("=" * 60)

img_path = TEST_IMAGES[0]
print(f"  Processing: {img_path.name}")
img_bgr  = cv2.imread(str(img_path))
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_res  = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
tensor   = img_res.astype(np.float32) / 255.0
tensor   = np.transpose(tensor, (2, 0, 1))[np.newaxis]   # BCHW

input_name = session.get_inputs()[0].name
raw = session.run(None, {input_name: tensor})

print(f"  Number of output tensors: {len(raw)}")
for i, r in enumerate(raw):
    print(f"\n  Output[{i}]:")
    print(f"    shape : {r.shape}")
    print(f"    dtype : {r.dtype}")
    print(f"    min   : {r.min():.6f}")
    print(f"    max   : {r.max():.6f}")
    print(f"    mean  : {r.mean():.6f}")

# ── Step 3: Decode ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Decode with assumed YOLOv11 layout")
print("=" * 60)

out = raw[0]
print(f"  Raw shape: {out.shape}")

if out.ndim == 3:
    if out.shape[1] < out.shape[2]:
        print("  Layout detected: [batch, 4+nc, anchors]  ✅ standard")
        preds = out[0]
        boxes_raw  = preds[:4, :]
        class_logits = preds[4:, :]
    else:
        print("  Layout detected: [batch, anchors, 4+nc]  ⚠️  transposed")
        preds = out[0]
        boxes_raw    = preds[:, :4].T
        class_logits = preds[:, 4:].T
else:
    print(f"  ❌ Unexpected output dims: {out.ndim}")
    exit(1)

print(f"  Num classes detected: {class_logits.shape[0]}")

# ── Step 4: Activation Check ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Activation Check")
print("=" * 60)

raw_max = class_logits.max()
raw_min = class_logits.min()
if 0.0 <= raw_min and raw_max <= 1.0:
    print("  → Values already in [0,1] ✅")
    scores = class_logits
else:
    print("  → Applying sigmoid")
    scores = 1.0 / (1.0 + np.exp(-class_logits))

# ── Step 5: Confidence analysis ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Confidence threshold analysis")
print("=" * 60)

max_scores = scores.max(axis=0)
for thresh in [0.01, 0.05, 0.1, 0.15, 0.25]:
    n = (max_scores >= thresh).sum()
    print(f"  conf >= {thresh:.2f}  →  {n:5d} anchors pass")

# ── Step 6: Scale check ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Box coordinate scale check")
print("=" * 60)

if boxes_raw[0].max() > 1.5:
    print("  → Boxes are in PIXEL coords ⚠️")
    boxes_norm = boxes_raw / INPUT_SIZE
else:
    print("  → Boxes are NORMALIZED ✅")
    boxes_norm = boxes_raw

# ── Step 7: GT Comparison ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Ground truth comparison")
print("=" * 60)

label_file = TEST_LABELS / (img_path.stem + ".txt")
if not label_file.exists():
    print(f"  ⚠️  No label file: {label_file.name}")
else:
    gt_lines = label_file.read_text().strip().splitlines()
    print(f"  GT boxes: {len(gt_lines)}")
    passing = np.where(max_scores >= 0.1)[0]
    print(f"  Preds passing 0.1: {len(passing)}")
    if len(passing) > 0:
        top_idx = passing[max_scores[passing].argmax()]
        cls = scores[:, top_idx].argmax()
        s   = max_scores[top_idx]
        b   = boxes_norm[:, top_idx]
        print(f"  Top Pred: class={cls} score={s:.4f} box={b}")

print("\nDIAGNOSIS COMPLETE")
