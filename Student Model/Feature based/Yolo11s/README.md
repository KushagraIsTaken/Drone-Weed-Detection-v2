# YOLOv11s Student (KD) — Evaluation Export
Generated : 2026-03-12 19:53

Teacher  : YOLOv11m  
Student  : YOLOv11s (Feature KD, λ=0.5, layers=[15, 18, 21])  
Training : 100 epochs, imgsz=832, AdamW, cosine LR

## Key Results (Test Set)
| Metric        | Value   |
|---------------|---------|
| mAP@0.50      | 0.8945 |
| mAP@0.50:0.95 | 0.6101 |
| Precision     | 0.8908 |
| Recall        | 0.8223 |
| F1            | 0.8551 |

## Distillation Loss (MSE, L2-normalised)
| Stat          | Value   |
|---------------|---------|
| First epoch   | N/A |
| Last epoch    | N/A |
| Minimum       | N/A |
| Reduction     | N/A |

## Output Files
| Folder     | Contents |
|------------|----------|
| models/    | student_best.pt, student_last.pt |
| plots/     | 15 PNG charts (00a–12) |
| csv/       | training_metrics, distill_loss_log, kd_loss_summary_stats, per_class, speed, iou, conf_sweep |
| predictions/ | Per-image prediction JPEGs |

## KD Loss Plots
| File | Description |
|------|-------------|
| 00a_kd_loss_curve.png    | Raw + weighted distillation loss |
| 00b_kd_loss_vs_map.png   | KD loss vs mAP50 dual-axis |
| 00c_kd_loss_smoothed.png | Smoothed trend + std band |
| 00d_kd_loss_stats.png    | Summary statistics bar chart |
| 00e_kd_loss_table.png    | Epoch milestone table |
