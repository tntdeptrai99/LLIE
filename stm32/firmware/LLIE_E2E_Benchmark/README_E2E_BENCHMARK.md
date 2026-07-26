# LLIE_E2E_Benchmark

Clean end-to-end benchmark firmware for:

`OV5640 camera -> resize/preprocess -> LLIE model -> ST7735 LCD`

This project is intentionally based on the known-working `08-DCMI2LCD` camera/LCD baseline and imports X-CUBE-AI generated model code as a library layer.

Do not regenerate this project from CubeMX while validating camera/LCD timing. CubeMX regeneration can remove the baseline DCMI/SPI/LCD glue that made `08-DCMI2LCD` work.

Watch variables:

- `display_mode`
- `camera_display_ms`
- `preprocess_ms`
- `inference_ms`
- `total_ms`
- `total_fps`
- `camera_raw_min`, `camera_raw_max`
- `model_input_min`, `model_input_max`
- `model_output_min`, `model_output_max`
- `dcmi_error_count`

Display modes:

- `0`: LCD color test
- `1`: camera raw preview
- `2`: preprocessed model input preview
- `3`: full pipeline model output
- `4`: model output with constant input
- `5`: diagnostic candidate output buffer
- `6`: diagnostic external IO buffer
