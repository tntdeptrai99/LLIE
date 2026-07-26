# LLIE E2E Performance Baseline

Status: profiling instrumentation added; board run is still required.

## How to Collect

Build and flash `stm32/firmware/LLIE_E2E_Benchmark` with:

```c
#define ENABLE_PIPELINE_PROFILING 1U
#define PIPELINE_PROFILE_FRAME_COUNT 100U
```

Open UART at 115200 baud. The firmware does not print every frame for the DWT
profile. It stores 100 frames in RAM, then prints one summary:

The UART report is intentionally compact to fit the STM32H750 128 KB FLASH:

```text
prof,n=100,sys=...,h=...,p1=...,p2=...,clk=...
p,<field>,<min_us>,<mean_us>,<max_us>,<p95_us>
```

Field mapping:

```text
0  camera_start_overhead
1  camera_wait_frame
2  camera_dma_transfer
3  camera_cache_invalidate
4  preprocessing
5  copy_to_ai_input
6  ai_run
7  postprocessing
8  rgb_to_rgb565
9  lcd_prepare_window
10 lcd_cache_clean
11 lcd_transfer_blocking
12 total_frame
```

Debugger watch variables:

```c
profile_frames_collected
profile_report_ready
pipeline_profile_last
clock_config_source
SystemCoreClock
```

`profile_report_ready` values:

```text
0 = collecting
1 = 100 frames collected, report pending
2 = report printed once
```

## Current Known Baseline Inputs

| Metric | Value | Source |
|---|---:|---|
| Preprocessing previously observed | 6-70 ms | user measurement |
| Model-only inference previously observed | about 189 ms | user measurement |
| Full E2E previously observed | often >1000 ms/frame | user measurement |
| Current build optimization | `-O0` Debug | `Debug/*/subdir.mk` |
| Camera frame | 160 x 120 RGB565, 38400 bytes | `Core/Src/main.c` |
| Model tensor | 96 x 96 x 3, 27648 bytes | `llieai.h` |
| LCD displayed block | 96 x 80 RGB565, 15360 bytes | `Core/Src/main.c` |

## Interpretation Rules

| Dominant field | Likely conclusion |
|---|---|
| `camera_wait_frame` or `camera_dma_transfer` | camera frame rate, XCLK, DCMI config, or callback cadence is limiting. |
| `camera_cache_invalidate` | cache range operation or memory placement needs review. |
| `preprocessing` | RGB565 resize/conversion or raw stats scan is expensive. |
| `ai_run` much greater than model-only | clock/build config/cache/DMA contention likely changed integrated inference. |
| `postprocessing` or `rgb_to_rgb565` | output combine/conversion path is expensive. |
| `lcd_transfer_blocking` | ST7735 blocking SPI path is the main bottleneck. |
| `total_frame` close to sum of all blocking fields | current system is purely sequential. |

## Pending Board Result

Paste the first 100-frame DWT profile here after flashing the profiling patch.

```text
TODO
```

## First Runtime Observation

UART sample with `display_mode=0`:

```text
m=0,lcd=10,pre=2..3,ai=0,post=0,tot=13..14,fps=26,raw=1/65469,in=0/248,out=0/0,derr=0
```

Interpretation:

| Field | Observation | Meaning |
|---|---:|---|
| `m=0` | LCD test pattern mode | AI is intentionally not running. |
| `lcd` | about 10 ms | Current 96 x 80 ST7735 blocking transfer is not the >1000 ms source by itself. |
| `pre` | 2-3 ms | Was measurement contamination: preprocessing ran before mode dispatch. Code has been adjusted so LCD-test and camera-raw modes do not preprocess. |
| `fps` | 26 | DCMI frame callback is now alive; effective camera callback cadence is about 26 Hz. |
| `derr` | 0 | No DCMI error callback reported in this run. |

Next required run: set `display_mode=3` for the full camera -> preprocess ->
AI -> postprocess -> LCD path and capture the DWT `prof`/`p,...` block.

UART sample with `display_mode=3` before output/API fix:

```text
m=3,lcd=10..11,pre=2,ai=182,post=5,tot=200..201,fps=5..6,raw=0..1/65469,in=0/248,out=0/0,derr=0
```

Interpretation:

| Field | Observation | Meaning |
|---|---:|---|
| `ai` | 182 ms | Integrated inference is close to the 189 ms model-only reference; no unexplained AI slowdown. |
| `tot` | 200-201 ms | The previous >1000 ms/frame behavior is no longer reproduced. |
| `lcd` | 10-11 ms | Current blocking LCD transfer is visible but not dominant. |
| `out` | 0/0 | Correctness issue: the model was generated with allocated IO in activations, but firmware called the normal run path with explicit output buffers. |

The normal path now uses activation-backed public IO and calls
`ai_llieai_forward()` for this allocated-IO model. Public input #0 is at
activation offset `279960` and must be written with the generated public stride
`c*9216+x*96+y`. Public output #0 is the gain map at offset `0`; public output
#1 is the residual map at offset `138240`. The LCD image is composed in
firmware as `input * gain + residual` and clipped before conversion to RGB565.

Confirmed fixed UART sample:

```text
v=7225,m=3,l=10..11,p=2..3,a=181..182,q=1..2,t=196..197,f=6,
r=1..2057/65469,i=0/248,o=128/187,u=0/255,fb=1,io=279960/0,e=0
```

Interpretation: output #0 and #1 are no longer zero. `fb=1` means the fallback
only fired during startup/probe, not continuously. Directly displaying output
#0 or #1 is expected to look like bands/texture because neither tensor is the
final RGB image.
