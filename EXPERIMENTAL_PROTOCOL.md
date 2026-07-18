# Experimental Protocol for Real-time LLIE on STM32H750

Tai lieu nay dong bo protocol thuc nghiem theo 4 spec P0 da chot.

## 1. Uu tien nghien cuu

Thu tu uu tien:

1. Real-time / near-real-time tren STM32H750.
2. Chat luong anh cao hon Ghost-ESP paper trong dieu kien so sanh ro rang.
3. `128x128` la speed baseline.
4. `256x256` la quality variant co dieu kien.
5. `96x96` la paper-reference/evaluation-only mode.

## 2. Dataset va split

Dataset chinh:

- LOL.
- LOL-v2 Real.

Split:

- Giu official test split.
- Tu official train split: `90%` train, `10%` validation, seed `42`.
- Camera data chia theo scene: `70%/15%/15%`.
- Synthetic low-light sinh tu ground truth/normal-light, chi dung augmentation train.
- Calibration/representative set lay tu train, `300-500` anh, khong lay test.

Preprocessing chi tiet:

- Training `128x128`: reflection padding neu can, exact paired crop `128x128`, khong resize sau crop.
- Training `256x256`: reflection padding neu can, exact paired crop `256x256`.
- Validation/test: center-square-crop -> resize ve target resolution.
- Deployment: OV5640 RGB565 -> RGB888/RGB tensor -> quantized INT8.

## 3. Models

Student: shallow depthwise gain-residual network.

| Model | Dark Guidance | Loss/Training | Muc dich |
|---|---|---|---|
| A1 | OFF | Charbonnier | Sanity baseline |
| A2 | OFF | Charbonnier + SSIM | Baseline noi bo chinh |
| B | ON | A2 losses | Lightweight dark/illumination guidance |
| C | ON | Advanced losses + optional denoise refinement | Quality FP32 |
| D | ON | Retinexformer distillation + QAT INT8 | Deployable student |

Model B khong dung illumination decoder rieng.

Truoc training dai han bat buoc chay dummy ONNX -> STM32Cube.AI feasibility test cho operator chinh.

## 4. Resolutions

| Resolution | Vai tro |
|---:|---|
| `96x96` | Paper-reference/Ghost-ESP-like fairness |
| `128x128` | Speed baseline va deployment candidate chinh |
| `256x256` | Quality variant, phu thuoc external SDRAM |

Ket qua khong duoc gop chung giua cac resolution.

## 5. Metrics

Quality metrics:

- RGB PSNR.
- RGB SSIM = average cua SSIM_R, SSIM_G, SSIM_B.
- LPIPS-AlexNet.
- CIEDE2000.
- Visual grid co crop dark/highlight/texture/color.

Deployment metrics:

- Params.
- MACs.
- Internal SRAM.
- External SDRAM.
- Flash.
- Model-only latency/FPS.
- Processing pipeline FPS.
- Display pipeline FPS.

Metric implementation, library/version logging va confidence interval theo `METRIC_SPEC.md`.

## 6. Benchmark protocol

Benchmark chinh thuc:

- 20 warmups.
- 200 measured runs.
- Report mean, median, P95, min, max, std.
- DWT cycle counter, do tung inference rieng va xu ly overflow 32-bit.
- Bao cao `FPS_mean = 1000 / mean_latency_ms`.
- Bao cao `FPS_P95_equivalent = 1000 / P95_latency_ms`.

End-to-end phai tach:

- Model-only.
- Processing pipeline: camera -> preprocess -> inference -> postprocess.
- Display pipeline: camera -> preprocess -> inference -> postprocess -> LCD.
- Sequential vs pipelined neu co double buffering.

Cache/DMA coherency phai duoc log theo `BENCHMARK_SPEC.md`.

## 7. Success criteria

Model D INT8 dat muc toi thieu khi:

- Vuot A2 FP32 ve PSNR va SSIM.
- LPIPS thap hon A2 FP32.
- DeltaE00 thap hon A2 FP32.
- FP32 -> INT8 degradation trong nguong:
  - Delta PSNR <= `0.3 dB`.
  - Delta SSIM <= `0.01`.
  - LPIPS_INT8 <= `1.05 * LPIPS_FP32`.
  - DeltaE00_INT8 <= `1.05 * DeltaE00_FP32`.
- `128x128` model-only latency `<277 ms`.
- `128x128` model-only FPS `>3.6`, muc tieu thuc te `7-10`.

## 8. Ghost-ESP comparison

Ba muc so sanh:

| Muc | Dieu kien | Ket luan hop le |
|---|---|---|
| Direct reproduction | Cung dataset, split, preprocessing, resolution, metric va benchmark | Co the ket luan truc tiep |
| Ghost-ESP-like | Tai hien gan theo paper, cung protocol project | Reimplemented baseline |
| Paper reference | Chi dung so lieu paper | So sanh tham khao |

Paper-reference comparison phai ghi ro:

```text
Paper-reference comparison, not exact reproduction.
```

## 9. Reporting

Moi model can co:

- Per-image metric CSV.
- Summary JSON.
- Visual grid.
- Params/MACs.
- RAM/Flash.
- Latency/FPS.
- Environment/library versions.

Ket luan cuoi phai dua tren ca quality metrics, visual inspection va deployment feasibility.
