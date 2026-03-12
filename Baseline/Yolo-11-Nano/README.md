# YOLOv11 Teacher Model — Evaluation Export
Generated : 2026-03-12 09:34

## Key Results (Test Set)
| Metric        | Value   |
|---------------|---------|
| mAP@0.50      | 0.7648 |
| mAP@0.50:0.95 | 0.4744 |
| Precision     | 0.7478 |
| Recall        | 0.7157 |
| F1            | 0.7314 |

## Plots
| File | Description |
|------|-------------|
| 01_loss_curves.png | Train/Val box, cls, dfl loss |
| 02_map_curves.png | mAP50 and mAP50-95 over epochs |
| 03_precision_recall_curves.png | P & R over epochs |
| 04_lr_schedule.png | Learning rate schedule |
| 05_summary_dashboard.png | 6-panel overview |
| 06_per_class_metrics.png | P/R/F1/mAP per class |
| 07_speed_metrics.png | Inference latency breakdown |
| 08_confidence_distribution.png | KD calibration histogram |
| 09_iou_sensitivity.png | mAP vs IoU threshold sweep |
| 10_conf_threshold_sweep.png | P-R-F1 vs confidence threshold |
| 11_boxsize_vs_confidence.png | Small-object detection analysis |
| 12_sample_predictions.png | Visual prediction grid |

## KD Usage
Load `teacher_kd_summary.json` in your student training script.
`recommended_kd_conf` = conf threshold that maximises teacher F1.

## Model
YOLOv11-m | 100 epochs | imgsz=640 | multi-GPU | AMP
