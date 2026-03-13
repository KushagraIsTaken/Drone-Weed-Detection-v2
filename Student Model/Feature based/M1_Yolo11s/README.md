# YOLOv11 Teacher Model — Evaluation Export
Generated : 2026-03-13 19:30

## Key Results (Test Set)
| Metric        | Value   |
|---------------|---------|
| mAP@0.50      | 0.8719 |
| mAP@0.50:0.95 | 0.5545 |
| Precision     | 0.8747 |
| Recall        | 0.8043 |
| F1            | 0.8380 |
| Min KD Loss    | 0.04918 @ epoch 99 |

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
| 08b_distillation_loss.png | KD loss per epoch + mAP50 convergence |
| 08c_kd_vs_training_combined.png | KD loss overlaid with training losses |
| 09_iou_sensitivity.png | mAP vs IoU threshold sweep |
| 10_conf_threshold_sweep.png | P-R-F1 vs confidence threshold |
| 11_boxsize_vs_confidence.png | Small-object detection analysis |
| 12_sample_predictions.png | Visual prediction grid |

## Distillation Loss Notes
- Method: epoch checkpoints (Mode A) if save_period was set during training,
  otherwise proxy from val_cls_loss x (1 - mAP50) (Mode B).
- Temperature scaling T=4.0 applied for soft-label computation.
- CSV: csv/distillation_loss_per_epoch.csv

## KD Usage
Load teacher_kd_summary.json in your student training script.
recommended_kd_conf = conf threshold that maximises teacher F1.
distillation_loss.min_kd_epoch = best epoch to use as soft-label source.

## Model
YOLOv11-m | 100 epochs | imgsz=640 | multi-GPU | AMP
