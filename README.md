# LLIE

Real-time low-light image enhancement research and embedded deployment pipeline for STM32H750-class edge devices.

This repository contains the training code, experiment artifacts, evaluation reports, ONNX export flow, and STM32Cube.AI firmware integration assets for a compact Low-Light Image Enhancement (LLIE) model designed for memory-constrained embedded vision.

## Highlights

- Target task: Low-Light Image Enhancement (LLIE)
- Main deployment target: STM32H750VBT6 with external SDRAM
- Best current model: `DG-GhostESP-96`
- Input resolution: RGB `96x96`
- Student architecture: Dark-guided Ghost/ESP blocks with gain-residual output
- Teacher signal: Retinexformer RGB distillation
- Export path: PyTorch checkpoint -> fused ONNX -> QDQ INT8 ONNX -> STM32Cube.AI
- Current best test result on `splits/lol_test.txt`, `count=15`:

| Model | PSNR | SSIM |
|---|---:|---:|
| Low-light input | 7.8353 | 0.144682 |
| Retinexformer teacher | 26.7819 | 0.953639 |
| DG-GhostESP-96, plateau best monitor | 19.6221 | 0.843353 |

## Best Model

Current best candidate:

```text
Dark-Guided GhostESP 96x96
W12/M24/B3, gain_max=3.0, residual_scale=0.35
plateau best_monitor checkpoint
```

Checkpoint:

```text
experiments/refine/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_from_long80_best_ssim/best_monitor.pt
```

Latest QDQ INT8 export target:

```text
stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq.onnx
```

STM32Cube.AI analyze summary for the current QDQ path:

| Resource | Value |
|---|---:|
| Total Flash | 55,464 B / 54.16 KiB |
| Weights | 5,944 B / 5.80 KiB |
| Library Flash | 49,520 B / 48.36 KiB |
| Total RAM | 441,820 B / 431.46 KiB |
| Activations | 425,260 B / 415.29 KiB |
| Library RAM | 16,560 B / 16.17 KiB |

See [`reports/tong_ket_model_tot_nhat.md`](reports/tong_ket_model_tot_nhat.md) for the detailed model summary.

## Repository Structure

```text
configs/       Experiment configuration files.
data/          Local dataset location. Raw datasets are not committed.
docs/          Implementation notes and model references.
experiments/   Checkpoints, Optuna databases, logs, and run artifacts.
reports/       Metrics, summaries, figures, and benchmark notes.
scripts/       Training, evaluation, split generation, and export utilities.
splits/        Local reproducible split files. Generated with seed 42.
src/           Python source modules for models, losses, metrics, and utilities.
stm32/         STM32Cube.AI, benchmark code, and firmware integration assets.
```

Important specification documents:

- [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md)
- [`PREPROCESSING_SPEC.md`](PREPROCESSING_SPEC.md)
- [`MODEL_SPEC.md`](MODEL_SPEC.md)
- [`METRIC_SPEC.md`](METRIC_SPEC.md)
- [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md)
- [`EXPERIMENTAL_PROTOCOL.md`](EXPERIMENTAL_PROTOCOL.md)
- [`SUCCESS_CRITERIA.md`](SUCCESS_CRITERIA.md)

## Dataset

Raw datasets are intentionally not stored in Git. Place local data under `data/` and generate reproducible splits under `splits/`.

The project expects paired low-light and normal-light images for supervised evaluation/training, plus optional precomputed Retinexformer teacher outputs for distillation.

Example workflow:

```powershell
python scripts\prepare_data.py
python scripts\generate_splits.py
python scripts\make_teacher_rgb_split.py
python scripts\make_distill_split.py
```

Adjust command arguments according to your local dataset paths.

## Training

Supervised baseline:

```powershell
python scripts\train_supervised.py --help
```

Distillation training:

```powershell
python scripts\train_distill.py --help
```

Optuna adaptive-loss tuning:

```powershell
python scripts\optuna_tune_adaptive_loss.py --help
```

The strongest current branch uses Retinexformer RGB distillation, adaptive dark-region weighting, SSIM/edge/color losses, and a dark-guided GhostESP student.

## Evaluation

Evaluate a checkpoint and save metrics/figures:

```powershell
python scripts\eval_model.py `
  --checkpoint experiments\refine\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_from_long80_best_ssim\best_monitor.pt `
  --split splits\lol_test.txt `
  --image-size 96 `
  --model-variant GHOST-ESP-DARK `
  --metric-name ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor `
  --name lol_test_ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor `
  --batch-size 4 `
  --save-images 8 `
  --figure-dir reports\figures\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor `
  --base-channels 12 `
  --mid-channels 24 `
  --e-blocks 3 `
  --gain-max 3.0 `
  --residual-scale 0.35
```

Generated outputs are stored in:

```text
reports/metrics/
reports/figures/
```

## ONNX and Quantization

Export fused ONNX:

```powershell
python scripts\export_dummy_onnx.py `
  --arch ghost-esp-dark `
  --blocks 3 `
  --base-channels 12 `
  --mid-channels 24 `
  --image-size 96 `
  --checkpoint experiments\refine\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_from_long80_best_ssim\best_monitor.pt `
  --out stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_fused.onnx
```

Export QDQ INT8 ONNX:

```powershell
python scripts\export_qdq_onnx.py `
  --input stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_fused.onnx `
  --output stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq.onnx `
  --calib-limit 64
```

The exported QDQ model is analyzed with STM32Cube.AI before firmware integration.

## STM32 Integration

STM32-related assets are stored under:

```text
stm32/
```

Key areas:

- `stm32/benchmarks/`: model-only benchmark code and timing logs.
- `stm32/firmware/LLIE/`: STM32 firmware integration project.
- `stm32/firmware/LLIE_Benchmark/`: benchmark-focused STM32CubeIDE project.

Current benchmark notes:

```text
stm32/benchmarks/model_only_int8_qdq_trial011_96.md
stm32/benchmarks/model_only_int8_qdq_trial011_96_log.csv
```

## Reproducibility Notes

- Default seed: `42`
- Main evaluation split: `splits/lol_test.txt`
- Main evaluation size: `96x96`
- Final comparisons should not mix different preprocessing protocols, teacher versions, or metric implementations.
- Large raw datasets and temporary outputs should remain outside Git.

## Roadmap

- Re-run STM32Cube.AI analyze for each final QDQ export.
- Compare FP32 vs QDQ outputs to quantify quantization drift.
- Measure end-to-end latency/FPS on the target STM32H750 board.
- Extend Optuna around the best plateau candidate if additional training time is available.

## License

No explicit license has been added yet. Treat this project as research code until a license is selected.
