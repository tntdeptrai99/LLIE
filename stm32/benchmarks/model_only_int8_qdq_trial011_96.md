# Model-only benchmark: INT8 QDQ trial011 96x96

Target model:

```text
stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_qdq.onnx
```

Benchmark scope:

```text
input tensor ready
-> ai_network_run()
-> output tensor ready
```

Do not include camera capture, RGB565 conversion, resize, input quantization,
output dequantization, or LCD transfer in this number.

## 1. Import model with STM32Cube.AI

In STM32CubeMX or STM32CubeIDE:

1. Enable X-CUBE-AI.
2. Add the ONNX model above.
3. Analyze first, then generate code.
4. Keep the generated network name as `network` if possible.
5. Use Release build for benchmark.

Record the Analyze result in:

```text
stm32/benchmarks/model_only_int8_qdq_trial011_96_log.csv
```

The current project-side Analyze reference is:

| Resource | Value |
|---|---:|
| Total Flash | 55,464 B |
| Weights | 5,944 B |
| Total RAM | 441,820 B |
| Activations | 425,260 B |
| Input | 27,648 B |
| Output | 27,648 B |

Run Analyze again on the exact generated Cube.AI project before final reporting.

## 2. Add benchmark source

Copy these two files into the CubeIDE firmware project:

```text
stm32/benchmarks/stm32_model_only_benchmark.h
stm32/benchmarks/stm32_model_only_benchmark.c
```

Recommended location in CubeIDE:

```text
Core/Inc/stm32_model_only_benchmark.h
Core/Src/stm32_model_only_benchmark.c
```

If Cube.AI generated a name other than `network`, edit the include block and
macros at the top of `stm32_model_only_benchmark.c`.

## 3. Call from main.c

After HAL init, clock config, cache config, UART init, and Cube.AI network init:

```c
#include "stm32_model_only_benchmark.h"

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_USARTx_UART_Init();
  MX_X_CUBE_AI_Init();

  stm32_model_only_benchmark_run();

  while (1) {
  }
}
```

Use the UART terminal output to fill the CSV log.

## 4. Required run settings

Use:

```text
warmup_runs = 20
measured_runs = 200
cpu_clock_hz = 480000000
input_shape = 1x3x96x96
precision = INT8 QDQ
benchmark_mode = model-only
```

Report:

```text
mean_ms
median_ms
p95_ms
min_ms
max_ms
std_ms
fps_mean = 1000 / mean_ms
```

Pass/fail gate:

```text
mean_ms < 277
fps_mean > 3.6
```

