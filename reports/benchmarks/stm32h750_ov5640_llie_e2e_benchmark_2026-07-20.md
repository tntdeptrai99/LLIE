# STM32H750 OV5640 LLIE End-to-End Benchmark

Date: 2026-07-20

## Setup

- MCU: STM32H750VBT6
- Camera: OV5640
- Display: ST7735 0.96 inch LCD
- Firmware: `LLIE_E2E_Benchmark`
- Pipeline: `OV5640 camera -> resize/preprocess -> LLIE model -> ST7735 LCD`
- Display mode: `display_mode = 3`
- Model output: QDQ MCU-friendly uint8 output, clipped before the problematic float output tail

## Measured Results

| Metric | Value |
|---|---:|
| `camera_display_ms` | 36 ms |
| `preprocess_ms` | 14 ms |
| `inference_ms` | 989 ms |
| `total_ms` | 1081 ms |
| `total_fps` | 1 FPS |

Additional runtime status:

| Variable | Value |
|---|---:|
| `display_mode` | 3 |
| `boot_stage` | 70 |
| `dcmi_error_count` | 0 |

## Artifacts

- Watch variables screenshot: `reports/figures/stm32_e2e_watch_2026-07-20.png`
- LCD output photo: `reports/figures/stm32_e2e_lcd_2026-07-20.png`

## Interpretation

The end-to-end pipeline is running without DCMI errors. Camera capture, preprocessing, model inference, and LCD display are all active.

The measured bottleneck is model inference:

- Inference takes about `989 ms`.
- Total frame time is about `1081 ms`.
- The resulting throughput is about `1 FPS`.

The LCD output captured during this run shows noisy colors, indicating that the model output path is active but the tensor layout/postprocessing still needs correction. After this measurement, firmware preprocessing and input preview were updated to use CHW layout to match the model input shape `1x3x96x96`.

## Next Step

The previous `display_mode = 3` path used the final ONNX output directly. That path is not reliable on the STM32 target: the final float/QDQ tail can produce all-zero output, and an output layout mismatch can make the LCD image look like heavy color noise.

Implemented next direction:

- Created MCU-friendly tail model:
  `stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_tail2_u8out.onnx`
- Script:
  `scripts/make_mcu_tail2_output_onnx.py`
- Public outputs:
  `_Add_1_output_0_QuantizeLinear_Output` = gain tensor, uint8
  `_Mul_1_output_0_QuantizeLinear_Output` = residual tensor, uint8
- PC verification:
  reconstructed final image differs from the full ONNX output by mean `0.45/255`, max `2/255`.
- Firmware was updated to use NCHW layout for model input/output and to compose:
  `output = input * gain + residual`
  in fixed-point integer arithmetic.

Retest after generating X-CUBE-AI code from the tail2 ONNX and flashing the firmware:

1. Set `display_mode = 2` and verify that the preprocessed input preview looks like the camera image.
2. Set `display_mode = 3` and capture the final model-output LCD image.
3. Record updated values for `camera_display_ms`, `preprocess_ms`, `inference_ms`, `total_ms`, and `total_fps`.

Optimization priority remains inference time, because it dominates the total end-to-end latency.
