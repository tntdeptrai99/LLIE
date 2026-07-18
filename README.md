# STM32H750 LLIE Project

This repository targets real-time or near-real-time Low-Light Image Enhancement on STM32H750VBT6 with external SDRAM.

Current project status:

- P0 documentation is locked enough to start the source tree.
- No training code has been implemented yet.
- No dependencies have been installed yet.
- `128x128` is the speed baseline.
- `256x256` is the quality variant.
- `96x96` is paper-reference/evaluation-only mode.

Primary references:

- `PROJECT_SCOPE.md`
- `PREPROCESSING_SPEC.md`
- `MODEL_SPEC.md`
- `METRIC_SPEC.md`
- `BENCHMARK_SPEC.md`
- `TASK_BREAKDOWN.md`

## Directory Overview

```text
configs/      Experiment configs.
data/         Local dataset mount/copy location. Raw datasets are not committed.
docs/         Extra implementation notes.
experiments/  Experiment outputs grouped by run.
reports/      Metrics, figures, tables and benchmark reports.
scripts/      Command-line utilities.
splits/       Reproducible split files generated with seed 42.
src/          Python source modules.
stm32/        STM32Cube.AI and firmware integration assets.
```

## Next Step

Implement the data pipeline from `PREPROCESSING_SPEC.md`, then run the dummy ONNX/Cube.AI feasibility check before long training.
