# STM32 Integration

STM32-related assets will live here.

Subdirectories:

- `cube_ai/`: STM32Cube.AI generated reports and model import notes.
- `firmware/`: firmware integration workspace or references.
- `benchmarks/`: model-only, processing pipeline and display pipeline benchmark logs.

Benchmark rules are defined in `../BENCHMARK_SPEC.md`.

Current first hardware benchmark target:

```text
benchmarks/model_only_int8_qdq_trial011_96.md
```

It measures the `ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_qdq.onnx`
model as model-only INT8 QDQ at `96x96`.
