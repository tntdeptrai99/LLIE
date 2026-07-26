# Current Project Snapshot

## Active firmware

- `stm32/firmware/LLIE_E2E_Benchmark`
- Current board behavior: live camera -> preprocess -> Cube.AI inference -> LCD AI output.
- Current display mode in UART report: `m=3` for AI output.

## Current validation artifacts

- `board_dump_metrics_frame_COM3.log`
- `board_dump_current_real_input_COM3.log`
- `board_benchmark_full_pipeline_current_30s_COM3.log`
- `reports/figures/board_dump_metrics_frame/`
- `reports/figures/current_real_input_model_output_compare/`

## Current PC tools

- `scripts/analyze_board_dump_image_metrics.py`
- `scripts/compare_board_tensor_dump.py`

## Current ONNX models

- `stm32/onnx/ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx`
- `stm32/onnx/ghost_esp_dark_w12_m24_enhanced_rgb_u8out.onnx`

## Current measured baseline

- Full pipeline: about 191 ms/frame, about 5 FPS.
- AI inference: about 170 ms/frame.
- LCD transfer: about 16 ms/frame.
- Latest image metric report: `reports/figures/board_dump_metrics_frame/image_metrics.md`.
