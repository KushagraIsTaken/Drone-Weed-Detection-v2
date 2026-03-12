# YOLO11n Feature Distillation Export
Generated: 2026-03-12 18:03

## Student
- Model: YOLO11n
- Teacher: YOLO11m best.pt
- KD method: FPN attention transfer + raw head logit matching

## Best Test Metrics
- mAP@0.50: 0.7617
- mAP@0.50:0.95: 0.4735
- Precision: 0.7630
- Recall: 0.6875
- F1: 0.7232

## Distillation
- Lowest combined KD loss: 0.2529 at epoch 98
- Feature KD curve: plots/02_distillation_losses.png
- Full training dashboard: plots/01_kd_training_dashboard.png

## Files
- csv/training_metrics.csv
- csv/distillation_loss_history.csv
- csv/test_overall_summary.csv
- csv/test_per_class_metrics.csv
- student_kd_summary.json
- models/best.pt
