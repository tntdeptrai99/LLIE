# Cube.AI vs ONNX Mismatch

## Current Status

- Firmware before this audit dump:
  `v7263`
- Firmware after audit changes:
  `v7265`
- PC ONNX:
  `ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_tail2_u8out.onnx`
- ONNX SHA-256:
  `3874C2F5621C0DA6495754943D9B3BF830C3BF866ABF2BB5943F572D416008D0`
- Cube.AI model hash:
  `0xd9fae00b20da69cb58a406c197e238c1`

## Existing Full-Pipeline Dump Result

Using `D:\LLIE_Project\board_dump` from firmware `v7263`:

- Board input checksum passed after parser was updated to tolerate the old UART ordering issue.
- Board input was fed directly to ONNX Runtime after dequantization with scale `0.0037334118969738483`, zero-point `0`.

Output0 gain:

- Best layout:
  CHW / NCHW
- Quantized MAE:
  `43.269`
- Quantized max abs:
  `127`
- Dequantized MAE:
  `0.339365`
- Quantized cosine:
  `0.979769`

Output1 residual:

- Best layout:
  CHW / NCHW
- Quantized MAE:
  `82.126`
- Quantized max abs:
  `255`
- Dequantized MAE:
  `0.064413`
- Quantized cosine:
  `0.905147`

## Signedness Check

Both public outputs are Cube.AI `uint8`, not `int8`.

Residual `max_abs=255` initially looked like a signedness problem, but an `int8` reinterpret comparison did not resolve it:

- Residual `uint8` MAE:
  `82.126`
- Residual `int8` reinterpret MAE:
  `61.709`
- Residual `int8` cosine:
  `0.348652`

Signedness is therefore not sufficient to explain the mismatch.

## Output Order Check

Cross-comparison did not show a simple output swap:

- Board output0 vs ONNX output1 raw uint8 MAE:
  `84.368`
- Board output1 vs ONNX output0 raw uint8 MAE:
  `52.175`

Output order alone is not the root cause.

## Next Required Test

Run the new `v7265` dump and verify the `tdump_begin` metadata:

- `in_off` should be `279960`
- `out0_off` should be `0`
- `out1_off` should be `138240`

Then run the fixed-input model-only test by setting:

```c
#define TEST_CUBEAI_FIXED_INPUT_ONLY 1U
```

Expected behavior:

- No camera frame is required.
- One fixed input inference is run.
- Firmware dumps `input_runtime`, `output0_public`, and `output1_public`.
- Program then waits forever.

Interpretation:

- If model-only matches ONNX but full pipeline does not, focus on activation overwrite, cache, DMA, or buffer race.
- If model-only still mismatches ONNX, focus on generated model conversion, output binding, or runtime/operator behavior.
