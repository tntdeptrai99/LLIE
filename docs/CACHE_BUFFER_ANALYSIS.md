# Cache And Buffer Analysis

Target firmware: `stm32/firmware/LLIE_E2E_Benchmark`
Probe build: `FW_PROBE_VERSION 7233`

## Memory Map

From `Debug/LLIE_E2E_Benchmark.map`:

| Buffer/section | Address | Size | Region |
| --- | ---: | ---: | --- |
| `ai_activations` | `0x24004158` | `0x5ec80` = 388224 | RAM_D1 |
| `ai_output_data` | `0x24062dd8` | `0x6c00` = 27648 | RAM_D1 |
| `.ram_d2` | `0x30000000` | `0x13e00` = 81408 | RAM_D2 |

The `.ram_d2` section is exactly the combined size expected for:

```text
camera_frame        160 * 120 * 2 = 38400 bytes
model_display_rgb565 96 * 80 * 2 = 15360 bytes
model_input_shadow 96 * 96 * 3 = 27648 bytes
total = 81408 bytes
```

This separates camera/LCD/input-shadow buffers in RAM_D2 from AI activations/output in RAM_D1.

## Activation Size

From `llieai_data_params.h`:

```text
AI_LLIEAI_DATA_ACTIVATIONS_SIZE = 388224 bytes
```

The application map shows `ai_activations` is also 388224 bytes, so there is no evidence of an undersized activation arena in the current build.

## Cache Operations

Current source evidence:

- DCache is enabled in `main.c`.
- `SCB_InvalidateDCache_by_Addr` is used in `invalidate_camera_frame()`.
- The invalidate helper aligns the start address down and size up to 32-byte cache-line boundaries.
- No `SCB_InvalidateDCache_by_Addr` is applied to AI output after inference.
- No `SCB_CleanDCache_by_Addr` is applied to LCD output before transfer.

Current interpretation:

- Camera DMA writes then CPU reads: invalidate before CPU reads is correct.
- Cube.AI CPU writes then CPU reads: no invalidate is expected.
- LCD path is currently blocking CPU SPI transmit, so cache clean before SPI DMA is not required for the current source.

## Current Risk

The highest cache-related risk is not an explicit bad AI-output invalidate. It is whether a DMA peripheral is active against a buffer that CPU code is also reading or writing. Patch 1 therefore adds fixed-input modes that disable camera DMA and report stable checksums.

## Required Board Checks

1. Fixed input to AI, no camera DMA.
2. Fixed input to AI to LCD.
3. Camera to AI without LCD.
4. Full pipeline.

For identical fixed input, the checksums after AI and after postprocess should not change across repeated runs.

