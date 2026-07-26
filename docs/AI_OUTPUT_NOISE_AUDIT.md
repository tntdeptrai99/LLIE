# AI Output Noise Audit

Date: 2026-07-22
Target firmware: `stm32/firmware/LLIE_E2E_Benchmark`
Current probe build: `FW_PROBE_VERSION 7233`

## Symptom

The full pipeline can show object structure on the ST7735, but AI display modes still show vertical or horizontal noise. Camera input display mode is clean, so the LCD path is not globally broken.

Recent board logs show:

- `display_mode=2`: camera input is visible.
- `display_mode=3`: AI output/postprocessed image shows structure with noise.
- `display_mode=7/8`: raw output visualizations show stripe/noise patterns depending on `ai_output_layout`.
- `postprocess_gain_q8=0` and `postprocess_residual_q8=0` can look clean only because the firmware falls back to camera input; that is not an AI-output fix.

## Changed Files From Git Status

The working tree contains many modified and untracked files. The active firmware under test is currently untracked as a whole:

- `stm32/firmware/LLIE_E2E_Benchmark/`

Other modified projects exist and were not touched for this audit:

- `stm32/firmware/LLIE/`
- `stm32/firmware/LLIE_Benchmark/`

For the active target, the source files relevant to this issue are:

- `stm32/firmware/LLIE_E2E_Benchmark/Core/Src/main.c`
- `stm32/firmware/LLIE_E2E_Benchmark/X-CUBE-AI/App/llieai.c`
- `stm32/firmware/LLIE_E2E_Benchmark/X-CUBE-AI/App/llieai.h`
- `stm32/firmware/LLIE_E2E_Benchmark/X-CUBE-AI/App/llieai_data_params.h`
- `stm32/firmware/LLIE_E2E_Benchmark/STM32H750VBTX_FLASH.ld`
- `stm32/firmware/LLIE_E2E_Benchmark/Drivers/BSP/ST7735/st7735.c`
- `stm32/firmware/LLIE_E2E_Benchmark/Drivers/BSP/ST7735/lcd.c`

## Evidence From Source

1. The model input and both outputs are `AI_BUFFER_FORMAT_U8`.
2. Each AI output is 27648 bytes, matching 96 x 96 x 3.
3. Generated public output tensors are transposed tensors with `AI_SHAPE_INIT(4, 1, 96, 96, 3)` and `AI_STRIDE_INIT(4, 1, 1, 96, 9216)`.
4. Internal non-public tensors use `AI_SHAPE_INIT(4, 1, 3, 96, 96)`.
5. Cube.AI binds public output 0 at activation offset `0`.
6. Cube.AI binds public output 1 at activation offset `138240`.
7. The application also allocates an external `ai_output_data` buffer in RAM_D1, immediately after `ai_activations`.
8. The current LCD driver sends rows through blocking `HAL_SPI_Transmit`, not `HAL_SPI_Transmit_DMA`.
9. The only explicit cache maintenance in the active code is camera-buffer invalidate before CPU reads DCMI output.
10. No source evidence currently supports LCD DMA reading a buffer while CPU writes the next frame.

## Most Likely Causes, In Order

1. AI public tensor layout is being interpreted incorrectly in postprocess. This matches "structure visible but striped/noisy", and the generated public tensor metadata is easy to misread because the header macros report `HEIGHT=96, WIDTH=3, CHANNEL=96` while generated tensor declarations show a public transposed 96 x 96 x 3 tensor.
2. Postprocess is treating gain/residual semantics incorrectly. Output 0 scale is about 1/127.5 and output 1 scale is about 1/1275, so manual blend values can mask the real issue.
3. AI output stability has not yet been proven with identical fixed input. If checksums differ run-to-run for fixed input, memory/cache/activation corruption becomes the lead cause.
4. Camera buffer race is possible in the full pipeline, but mode 2 camera display is clean and fixed-input tests are the required next separator.
5. LCD DMA race and missing LCD cache clean are low probability for the current source because the active LCD transmit path is blocking SPI, not DMA.
6. RGB565 byte order or RGB/BGR channel order can still affect color, but should not by itself create deterministic spatial stripes if test patterns are clean.

## Tests Before Any Root Fix

Patch 1 adds deterministic probes:

- `ENABLE_AI_OUTPUT_DEBUG`
- `PIPELINE_CORRECTNESS_MODE`
- `TEST_FIXED_INPUT_AI`
- `TEST_FIXED_INPUT_AI_TO_LCD`
- `TEST_CAMERA_TO_AI_NO_LCD`
- `TEST_FULL_PIPELINE`

The first required board test is:

```text
display_mode=4
ai_input_layout=0
ai_output_layout=2
```

Every 10 fixed-input runs the firmware prints:

```text
dbg,o0crc,o0min,o0max,o1crc,o1min,o1max,lcdcrc,o0mm,o1mm,lcdmm
```

Expected result for a deterministic fixed input:

```text
o0mm=0
o1mm=0
lcdmm=0
```

If any mismatch is nonzero, the first bad point is at or before the corresponding checksum stage. If all mismatches are zero while LCD is still noisy, the issue is deterministic tensor interpretation/postprocess/LCD conversion rather than a random cache or DMA race.

