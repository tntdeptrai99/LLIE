# Báo cáo benchmark hiện tại

## Dataset quality

| Dataset | N | PSNR mean | SSIM mean | MAE mean | PC inference ms |
|---|---:|---:|---:|---:|---:|
| LOL_v2_Real_val | 100 | 19.4780 | 0.7858 | 0.111396 | 1.9178 |
| LOL_val | 15 | 14.5200 | 0.7170 | 0.188023 | 2.2275 |

Artifact dataset:
- CSV: `D:\LLIE_Project\reports\benchmarks\current_pipeline_20260726\dataset_quality_metrics.csv`
- Ảnh mẫu/contact sheet/histogram: `D:\LLIE_Project\reports\figures\current_pipeline_20260726\dataset_quality`

## Architecture benchmark

- CSV: `D:\LLIE_Project\reports\benchmarks\current_pipeline_20260726\architecture_benchmark.csv`
- Chỉ model hiện tại có artifact ONNX để đo. Các kiến trúc Conv2D/Separable/GhostSep/Ghost-ESP+Distill được ghi là thiếu artifact nếu không có checkpoint/ONNX tương ứng.

## Loss function benchmark

- CSV: `D:\LLIE_Project\reports\benchmarks\current_pipeline_20260726\loss_function_benchmark_from_train_logs.csv`
- Báo cáo này tổng hợp từ các `train_log.csv` còn trong thư mục `experiments`.

## Quantization và PC-board equivalence

- Trạng thái: `ok`
- Log: `D:\LLIE_Project\reports\benchmarks\current_pipeline_20260726\pc_board_equivalence.log`

## Benchmark thị giác thực tế

- Artifact: `D:\LLIE_Project\reports\figures\current_pipeline_20260726\real_camera_visual`
- Dữ liệu lấy từ frame board đã dump: input preprocess, AI output, histogram và metric brightness/contrast/saturation/sharpness/clipping.