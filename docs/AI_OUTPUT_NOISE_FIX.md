# AI Output Noise Fix Log

Target firmware: `stm32/firmware/LLIE_E2E_Benchmark`
Probe build: `FW_PROBE_VERSION 7245`

## Status

Patch 1 is implemented: checksum, tensor stats, and correctness modes.
Patch 2 is implemented: default tensor layout and full gain/residual composition.
Patch 3 is implemented: use scale-derived integer factors for gain/residual composition. The temporary mode-2 change that rendered `ai_runtime_input` directly was rolled back.
Patch 4 returns to the full AI pipeline after the temporary LCD-only rescue tests.
Patch 5 disables the residual term by default to isolate the observed noise overlay.
Patch 6 changes the AI input feed layout to CHW/NCHW while keeping residual disabled.
Patch 7 changes only the AI output read layout to the alternate CHW interpretation.

The current leading root cause is wrong default IO layout plus partial manual output blending. The old `g=64/0` setting was a debug blend, not a validated model output path.

## Patch 1 Contents

File changed:

- `Core/Src/main.c`

Added:

- `ENABLE_AI_OUTPUT_DEBUG`
- `PIPELINE_CORRECTNESS_MODE`
- `TEST_LCD_PATTERN`
- `TEST_FAKE_AI_OUTPUT_TO_LCD`
- `TEST_FIXED_INPUT_AI`
- `TEST_FIXED_INPUT_AI_TO_LCD`
- `TEST_CAMERA_TO_AI_NO_LCD`
- `TEST_FULL_PIPELINE`
- AI output min/max/checksum probes.
- LCD postprocess checksum probe.
- Fixed-input repeated-run comparator.
- Camera DMA guard for fixed-input correctness modes.

Compact debug line format:

```text
dbg,o0crc,o0min,o0max,o1crc,o1min,o1max,lcdcrc,o0mm,o1mm,lcdmm
```

The mismatch fields compare each run with the first fixed-input run in the current debug window.

## Build Result

Build passed for `LLIE_E2E_Benchmark.elf`:

```text
text=114764
data=16240
bss=504832
dec=635836
hex=9b3bc
```

The project is very close to the 128 KB internal FLASH limit, so Patch 1 uses compact checksum/min/max reporting instead of verbose per-frame tensor dumps.

## Patch 2 Contents

File changed:

- `Core/Src/main.c`

Changed defaults:

```text
FW_PROBE_VERSION: 7233 -> 7234
ai_input_layout: 0 -> 1
ai_output_layout: 2 -> 1
postprocess_gain_q8: 64 -> 256
postprocess_residual_q8: 0 -> 256
```

Reason:

- Cube.AI report exposes public input/output as `uint8(1x3x96x96)`.
- The firmware layout mode `1` matches CHW with `index = c * 96 * 96 + y * 96 + x`.
- Full model composition should use the full gain and residual terms. The previous `64/0` was only a manual debug blend.

## Patch 3 Contents

File changed:

- `Core/Src/main.c`

Changed:

```text
FW_PROBE_VERSION: 7234 -> 7235
gain pixel factor: 489 -> 514
residual pixel factor: 13107/65536 -> 51/256
```

Reason:

- Output 0 scale is `0.0078431377`, so pixel contribution is approximately `input_q * gain_q / 127.5`.
- Output 1 scale is `0.0007843137`, so pixel residual is approximately `residual_q * 0.2`.

## Patch 4 Contents

File changed:

- `Core/Src/main.c`

Changed:

```text
FW_PROBE_VERSION: 7240 -> 7241
TEST_LCD_PATTERN: 1 -> 0
```

Reason:

- The LCD-only loop was temporary recovery/debug code after a no-display report.
- The active target is back to camera -> preprocessing -> AI -> postprocess -> LCD.

## Patch 5 Contents

File changed:

- `Core/Src/main.c`

Changed:

```text
FW_PROBE_VERSION: 7242 -> 7243
postprocess_residual_q8: 256 -> 0
```

Reason:

- Board logs show output 1 as `u=0/255`, meaning the residual public tensor spans the full U8 range in live camera mode.
- The visible image in `display_mode=3` is covered by noise, which matches an overlaid residual/noise term.
- This is a diagnostic isolation patch: keep scale-correct gain output, remove residual contribution, and check whether the image becomes clean.

## Patch 6 Contents

File changed:

- `Core/Src/main.c`

Changed:

```text
FW_PROBE_VERSION: 7243 -> 7244
ai_input_layout: 0 -> 1
```

Reason:

- With residual disabled, the LCD still shows regular vertical stripes over a recognizable object.
- The generated Cube.AI report lists the public input as `uint8(1x3x96x96)`.
- Feeding HWC input to an NCHW model can preserve coarse structure while creating channel/column stripe artifacts.

## Patch 7 Contents

File changed:

- `Core/Src/main.c`

Changed:

```text
FW_PROBE_VERSION: 7244 -> 7245
ai_output_layout: 1 -> 2
```

Reason:

- v7244 still shows regular vertical stripe artifacts with residual disabled.
- Input layout change did not materially improve the image, so the next isolated variable is public output indexing.
- Residual remains disabled to keep the test focused on output0/gain layout.

## Tests Needed On Board

### Test A: fixed input to AI to LCD

Use:

```text
display_mode=4
ai_input_layout=1
ai_output_layout=1
```

Expected debug result:

```text
last three dbg fields = 0,0,0
```

If mismatch is nonzero:

- `o0mm != 0`: output 0 is not deterministic.
- `o1mm != 0`: output 1 is not deterministic.
- `lcdmm != 0`: postprocess/LCD buffer is not deterministic.

### Test B: normal full pipeline

Use:

```text
display_mode=3
ai_input_layout=1
ai_output_layout=1
```

Send both the normal `v=7233` lines and any `dbg` lines.

## Current Root-Cause Hypothesis

The current leading hypothesis is deterministic AI tensor layout or postprocess interpretation, not LCD DMA race.

Evidence:

- Camera-input display is clean.
- LCD path is blocking SPI.
- AI output is U8, not INT8/FLOAT.
- Public generated tensors are transposed 96 x 96 x 3 tensors, while header macros expose `HEIGHT=96, WIDTH=3, CHANNEL=96`.
- Noise direction changes with output layout selection.

## Rollback

To disable Patch 1 debug output without removing code:

```c
#define ENABLE_AI_OUTPUT_DEBUG 0U
```

To return to the normal live pipeline:

```c
#define PIPELINE_CORRECTNESS_MODE 0U
#define TEST_FIXED_INPUT_AI 0U
#define TEST_FIXED_INPUT_AI_TO_LCD 0U
#define TEST_CAMERA_TO_AI_NO_LCD 0U
#define TEST_FULL_PIPELINE 1U
```

## Pending

Patch 2 should only change output interpretation after fixed-input checksum results are available.

Patch 3 or Patch 4 should only be made if checksum evidence points to cache or DMA race.
