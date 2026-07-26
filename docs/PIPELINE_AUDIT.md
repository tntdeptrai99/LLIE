# LLIE E2E Pipeline Audit

Audit target: `stm32/firmware/LLIE_E2E_Benchmark`.

This document describes the current source-level pipeline before high-risk
optimization work. It intentionally records evidence from the current code so
the next patches can be measured instead of guessed.

## Relevant Source Tree

```text
stm32/firmware/LLIE_E2E_Benchmark/
  Core/Src/main.c                  main loop, preprocessing, postprocessing, AI run, DCMI callbacks, MPU, clock
  Core/Src/dcmi.c                  DCMI and DCMI DMA configuration
  Core/Src/spi.c                   SPI4 configuration for ST7735
  Core/Src/i2c.c                   I2C1 configuration for OV5640 SCCB/I2C control
  Core/Src/tim.c                   TIM1 PWM for LCD backlight and camera XCLK TIM mode
  Core/Src/stm32h7xx_it.c          IRQ handlers
  Drivers/BSP/Camera/camera.c      OV5640 control wrapper and XCLK selection
  Drivers/BSP/Camera/ov5640.c      OV5640 register setup and frame-size setup
  Drivers/BSP/ST7735/lcd.c         ST7735 IO bridge to HAL SPI
  Drivers/BSP/ST7735/st7735.c      ST7735 drawing/window routines
  X-CUBE-AI/App/llieai.c           generated Cube.AI network
  X-CUBE-AI/App/llieai.h           AI input/output sizes
  X-CUBE-AI/App/llieai_data*.c/h   AI weights/activation metadata
  STM32H750VBTX_FLASH.ld           memory regions and custom .ram_d2 section
```

## Current Pipeline

```text
OV5640 RGB565
-> DCMI/DMA writes camera_frame
-> CPU invalidates camera_frame D-cache
-> CPU scans camera stats
-> CPU preprocesses RGB565 into planar AI input
-> Cube.AI ai_llieai_forward() for allocated-IO activation model
-> CPU converts AI public output image into RGB565 LCD buffer
-> ST7735_FillRGBRect()
-> lcd_senddata()
-> blocking HAL_SPI_Transmit()
```

## Step-by-step Dataflow

| Step | File / function | Input buffer | Output buffer | Size | Memory section | Access | Blocking | Cache handling | Current risk |
|---|---|---|---|---:|---|---|---|---|---|
| Camera init | `Core/Src/main.c:769`, `Drivers/BSP/Camera/camera.c:230` `Camera_Init_Device` | OV5640 over I2C | `hcamera.device_id` | registers | n/a | CPU/I2C | blocking I2C | none | Board currently reaches red LCD when `hcamera.device_id != 0x5640`; XCLK/I2C/pin/clock remain suspects. |
| Camera capture start | `Core/Src/main.c:790` `HAL_DCMI_Start_DMA` | DCMI peripheral | `camera_frame` | 160 x 120 x 2 = 38400 bytes | `.ram_d2`, RAM_D2 | DMA writes | non-blocking start, continuous afterwards | invalidated later | Single buffer in circular DMA can be overwritten while CPU preprocesses. |
| DCMI config | `Core/Src/dcmi.c:31` `MX_DCMI_Init` | OV5640 parallel bus | DCMI FIFO | RGB565 bytes | peripheral | DMA | n/a | n/a | Polarity/byte order/frame timing not proven yet. |
| DMA config | `Core/Src/dcmi.c:131` `hdma_dcmi` | DCMI DR | RAM_D2 | word aligned transfer length | RAM_D2 | DMA1 Stream0 | circular | none here | `DMA_PRIORITY_LOW`, FIFO disabled; can lose data if bus contention occurs. |
| Frame callback | `Core/Src/main.c:680` `HAL_DCMI_FrameEventCallback`; `Core/Src/stm32h7xx_it.c` `DCMI_IRQHandler`/`DMA1_Stream0_IRQHandler` | interrupt | `dcmi_frame_ready` | flag | `.bss` | ISR/CPU | non-blocking | none | IRQ handlers were missing and have been added; only one flag remains, so dropped/overwritten frames are not counted separately. |
| Camera cache invalidate | `Core/Src/main.c:334` `invalidate_camera_frame` | `camera_frame` | same | 38400 bytes rounded to 32-byte lines | `.ram_d2` | CPU cache op | blocking cache op | `SCB_InvalidateDCache_by_Addr` | Correct direction for DMA-write/CPU-read, but no helper API yet and no protection against DMA still writing. |
| Camera stats | `Core/Src/main.c:350` `update_camera_raw_stats` | `camera_frame` | min/max globals | 19200 pixels | `.ram_d2` -> `.bss` | CPU | blocking scan | after invalidate | Extra full-frame memory pass every frame. |
| Preprocessing | `Core/Src/main.c:379` `preprocess_camera_to_ai_input` | `camera_frame` RGB565 | `ai_runtime_input` | 96 x 96 x 3 = 27648 bytes | input in Cube.AI activation arena | CPU | blocking | camera already invalidated | Nearest-neighbor resize and planar RGB write are direct, no extra input copy. |
| AI input | `Core/Src/main.c`, `llieai.h:51` | CPU-written tensor | Cube.AI input | 27648 bytes | `ai_activations + 279960` when `AI_LLIEAI_INPUTS_IN_ACTIVATIONS` | CPU/Cube.AI | n/a | cacheable by default | Must write into the generated public input activation offset, otherwise the model can run on zeros. |
| AI activations | `Core/Src/main.c:86`, `llieai_data_params.h:32` | Cube.AI | internal scratch | 388224 bytes | `.bss` RAM_D1 | CPU/Cube.AI | n/a | cacheable by default | Large RAM_D1 footprint; contention can occur if DMA also targets same bus/matrix. |
| AI run | `Core/Src/main.c:596` `run_llie_once` | `ai_runtime_input` | public outputs | 2 x 27648 bytes for current 2 outputs | generated model reports input/output in activation arena | CPU | blocking compute | none | This allocated-IO model must be run with `ai_llieai_forward()` or equivalent activation-backed IO handling; explicit output-buffer `ai_llieai_run()` caused zero outputs. |
| Output stats | `Core/Src/main.c:456` `update_ai_output_candidates_stats` | `ai_activations` slices | min/max globals | 5 x 27648 byte scans | RAM_D1 | CPU | blocking scans | none | This was a major debug overhead in model output mode before it was gated. |
| Postprocess | `Core/Src/main.c:496` `convert_ai_output_to_display` | AI public output #0 | `model_display_rgb565` | 96 x 80 x 2 = 15360 bytes | `.ram_d2`, RAM_D2 | CPU | blocking | none | Converts CHW U8 RGB output to RGB565; currently combined with RGB565 conversion. |
| LCD test/input conversion | `Core/Src/main.c:365`, `414`, `577` | camera/input/test | `model_display_rgb565` | 15360 bytes | `.ram_d2` | CPU | blocking | none | Useful for isolated tests, but currently display path still blocking. |
| LCD transfer | `Core/Src/main.c:823/837/898` `ST7735_FillRGBRect` | `model_display_rgb565` | ST7735 GRAM | 15360 bytes | `.ram_d2` | CPU/SPI polling | blocking | no clean | Uses blocking HAL SPI, not DMA. Cache clean is not needed for polling SPI, but will be needed for SPI DMA. |
| ST7735 row loop | `Drivers/BSP/ST7735/st7735.c:701` `ST7735_FillRGBRect` | RGB565 buffer | static row `pdata[640]` | 2 x Width each row | local static `.bss` | CPU | blocking per row | none | Calls `ST7735_SetCursor` once per row and sends one row at a time. |
| SPI IO bridge | `Drivers/BSP/ST7735/lcd.c:303` `lcd_senddata`, `lcd.c:313` `lcd_writereg` | command/data | SPI4 MOSI | variable | n/a | CPU/SPI | `HAL_SPI_Transmit(..., 100/500)` | none | Main LCD bottleneck candidate; no `HAL_SPI_Transmit_DMA` in current driver. |
| SPI config | `Core/Src/spi.c:30` `MX_SPI4_Init` | n/a | SPI4 | 8-bit 1-line TX | peripheral | SPI | n/a | n/a | Prescaler 8; actual pixel throughput depends on PCLK and current clock fallback. |

## Buffers and Placement

| Buffer | Declaration | Size | Region | DMA capable | Risk |
|---|---|---:|---|---|---|
| `camera_frame` | `Core/Src/main.c:101` | 38400 bytes | `.ram_d2` -> RAM_D2 | yes | Single buffer; DCMI circular DMA can overwrite while CPU reads. |
| `model_display_rgb565` | `Core/Src/main.c:102` | 15360 bytes | `.ram_d2` -> RAM_D2 | yes | OK for future SPI DMA, but needs clean D-cache before DMA reads. |
| `ai_activations` | `Core/Src/main.c:86` | 388224 bytes | `.bss` -> RAM_D1 | CPU only | Big cacheable CPU working set; Debug `-O0` makes AI slower. |
| `ai_runtime_input` | `Core/Src/main.c` | 27648 bytes | `ai_activations + 279960` when `AI_LLIEAI_INPUTS_IN_ACTIVATIONS` | CPU only | Public input #0, CHW U8 format. |
| `ai_output_data` | `Core/Src/main.c:88` | 27648 bytes | `.bss` -> RAM_D1 | CPU only | Used for single-output/external diagnostics. |
| `ai_runtime_output` | `Core/Src/main.c` | 27648 bytes | `ai_activations + 0` when `AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS` | CPU only | Public output #0, final RGB image in CHW U8 format. |
| `ai_runtime_gain` | `Core/Src/main.c` | 27648 bytes | `ai_activations + 138240` when `AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS` | CPU only | Public output #1 / auxiliary map; kept for debug/watch. |
| `ai_tail_gain_data` | `Core/Src/main.c` | 27648 bytes | `.ram_d2` -> RAM_D2 | CPU only | Compiled only if generated model allows external output buffers. |
| `ai_tail_residual_data` | `Core/Src/main.c` | 27648 bytes | `.ram_d2` -> RAM_D2 | CPU only | Compiled only if generated model allows external output buffers. |

Linker memory regions are defined in `STM32H750VBTX_FLASH.ld:47-50`:

```text
DTCMRAM 0x20000000 128K
RAM_D1  0x24000000 512K
RAM_D2  0x30000000 288K
RAM_D3  0x38000000 64K
```

`.bss` goes to RAM_D1 (`STM32H750VBTX_FLASH.ld:144-156`), `.ram_d2`
goes to RAM_D2 (`STM32H750VBTX_FLASH.ld:158-164`), and heap/stack go to
DTCMRAM (`STM32H750VBTX_FLASH.ld:167-175`).

## Blocking and Diagnostic Hotspots

| Pattern | Location | In frame loop? | Risk |
|---|---|---|---|
| `HAL_SPI_Transmit` | `Drivers/BSP/ST7735/lcd.c:303`, `313` | yes through `ST7735_FillRGBRect` | Strong LCD latency candidate. |
| Row-by-row cursor setup | `Drivers/BSP/ST7735/st7735.c:716` | yes | Repeats command overhead for every displayed row. |
| Full-frame raw stats | `Core/Src/main.c:350` | yes | Extra memory pass over camera frame. |
| Candidate activation stats | `Core/Src/main.c:456-461` | gated now | Multiple 27 KB scans; should stay disabled in performance runs. |
| UART `printf` | `Core/Src/main.c:658`, error/init logs | periodic | OK if once per second, but not per stage/frame for profiling. |
| `HAL_Delay` | `Core/Src/main.c:749`, `754`, `771`; `camera.c:148` | boot/init | Not current frame-loop latency. |
| Single `dcmi_frame_ready` flag | `Core/Src/main.c:680`, `811` | yes | Cannot distinguish dropped frames from overwritten frames. |

## Top Probable Causes of >1000 ms E2E Latency

1. LCD is blocking and row-based. Evidence: `ST7735_FillRGBRect` loops over
   each row and `lcd_senddata` uses `HAL_SPI_Transmit` with 500 ms timeout.
2. Integrated build is Debug `-O0`. Evidence: generated Debug `subdir.mk`
   compiles Core, BSP, HAL, and X-CUBE-AI with `-O0 -g3 -DDEBUG`.
3. Clock fallback can run at a different frequency than model-only. Evidence:
   current `SystemClock_Config` records `clock_config_source`; the board was
   observed using source `3` (HSI fallback) after HSE failed.
4. Camera uses one circular DMA buffer. Evidence: `camera_frame` is single
   buffer and DCMI DMA is circular; CPU reads it after a flag without stopping
   DMA or switching buffers.
5. Debug/stat scans add memory bandwidth pressure. Evidence:
   `update_camera_raw_stats` runs every frame; candidate stats can scan five
   output-sized activation regions when enabled.
6. RAM_D2 contains camera and LCD buffers. Evidence: both live in `.ram_d2`;
   this can create bus pressure once DMA and CPU work overlap.
7. Cache handling is partial. Evidence: camera invalidate exists, but there is
   no reusable aligned range helper and no LCD clean path for the future SPI
   DMA transfer.

## Confirmed Correctness Issue Fixed

After camera ID and AI init succeeded, UART stopped after:

```text
Starting camera -> preprocess -> model -> display
```

Source inspection found that `Core/Src/dcmi.c` enabled `DCMI_IRQn` and
`DMA1_Stream0_IRQn`, but `Core/Src/stm32h7xx_it.c` did not implement
`DCMI_IRQHandler` or `DMA1_Stream0_IRQHandler`. The startup file maps missing
handlers to `Default_Handler`, so DCMI/DMA interrupts could not reach
`HAL_DCMI_IRQHandler` / `HAL_DMA_IRQHandler`; therefore
`HAL_DCMI_FrameEventCallback` did not set `dcmi_frame_ready`.

Patch added:

```c
void DCMI_IRQHandler(void) { HAL_DCMI_IRQHandler(&hdcmi); }
void DMA1_Stream0_IRQHandler(void) { HAL_DMA_IRQHandler(&hdma_dcmi); }
```

## Confirmed Performance Baseline

Full pipeline runtime sample with `display_mode=3`:

```text
m=3,lcd=10..11,pre=2,ai=182,post=5,tot=200..201,fps=5..6,derr=0
```

This shows the previous >1000 ms/frame issue is not present after the IRQ and
boot fixes. Current E2E time is approximately:

```text
preprocess 2 ms + ai 182 ms + postprocess 5 ms + LCD 10-11 ms = about 200 ms
```

The pipeline is still sequential, but inference is not abnormally slower than
the model-only reference.

## Confirmed Output Mapping Issue Fixed

Runtime sample showed `out=0/0` in `display_mode=3`. The generated report says:

```text
input:      input_rgb_output, activation offset 279960
output 1/2: _Add_1...Transpose_0, QLinear(0.007843138,0,uint8)
output 2/2: _Mul_1...Transpose_1, QLinear(0.000784314,0,uint8)
AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS
AI_LLIEAI_INPUTS_IN_ACTIVATIONS
```

The normal firmware path now uses `ai_llieai_forward()` and reads public IO from
the activation arena offsets used by generated `llieai.c`:

```text
input0/public input       = ai_activations + 279960, public stride c*9216+x*96+y
output0/gain map          = ai_activations + 0, QLinear(0.007843138,0,uint8)
output1/residual map      = ai_activations + 138240, QLinear(0.000784314,0,uint8)
LCD/composed final image  = input * gain + residual, clipped to RGB888/RGB565
```

Confirmed runtime after the fix:

```text
v=7225,...,i=0/248,o=128/187,u=0/255,fb=1,io=279960/0,e=0
```

The key fixes were using activation-backed IO with the forward API for the
generated `allocate-inputs, allocate-outputs` model, writing/reading public IO
with the generated tensor stride, and composing the final LCD image from output
#0 gain plus output #1 residual instead of displaying either tensor directly.

## Profiling Patch Added

`Core/Src/main.c` now has `ENABLE_PIPELINE_PROFILING` and records 100 frames
of `PipelineProfile` in RAM. It reports min/mean/max/P95 only after the
sample window is full. Current LCD fields reflect the blocking LCD path:

```text
lcd_prepare_window = 0 until the LCD driver is split
lcd_cache_clean = 0 until SPI DMA is introduced
lcd_transfer_blocking = current ST7735_FillRGBRect blocking time
```

The next step is to build/flash, collect the DWT report, and use those numbers
to decide whether to fix camera correctness, LCD DMA, clock/build config, or
memory placement first.
