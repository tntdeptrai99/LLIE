# Plan Review After P0 Sync

Tai lieu nay thay the review cu. Review cu dua tren proposal goc va da co nhieu diem outdated sau khi project chot P0 specs.

## 1. Noi dung da ro

- Scope hien hanh nam trong `PROJECT_SCOPE.md`.
- Preprocessing nam trong `PREPROCESSING_SPEC.md`.
- Student architecture nam trong `MODEL_SPEC.md`.
- Metric nam trong `METRIC_SPEC.md`.
- Benchmark nam trong `BENCHMARK_SPEC.md`.
- Task trien khai nam trong `TASK_BREAKDOWN.md`.

Quyet dinh da chot:

- MCU: STM32H750VBT6 + external SDRAM.
- Camera: OV5640 RGB565.
- Display: ST7735 SPI.
- Training: PyTorch.
- Export: ONNX -> STM32Cube.AI.
- Teacher: Retinexformer.
- Student: shallow depthwise gain-residual network.
- Quantization: QAT INT8.
- Resolution: `128x128` speed baseline, `256x256` quality variant, `96x96` paper-reference.
- Dataset: LOL + LOL-v2 Real.
- Seed: `42`.
- Metrics: RGB PSNR, RGB SSIM, LPIPS-AlexNet, CIEDE2000.
- Benchmark: 20 warmups, 200 measured runs, mean/median/P95.

## 2. Noi dung da duoc giai quyet tu review cu

- Khong con dung U-Net Lite lam baseline hien hanh.
- Khong con dung `160x160` lam baseline.
- A1/A2 da chot tat Dark Guidance.
- Model B da chot la lightweight dark/illumination guidance, khong co illumination decoder rieng.
- Preprocessing `128x128` va `256x256` da thong nhat validation/test bang center-square-crop -> resize.
- Synthetic low-light da chot sinh tu ground truth/normal-light image.
- Benchmark da them `96x96` paper-reference mode.
- Benchmark da them cache/DMA coherency, DWT overflow, sequential/pipelined va display/processing split.

## 3. Quyet dinh con can theo doi

- Retinexformer license/checkpoint va cach tao teacher output.
- Ghost-ESP-like baseline co tai hien duoc den muc nao.
- `256x256` co dat latency/RAM khi activation nam trong SDRAM hay khong.
- ST7735 SPI co lam end-to-end FPS thap hon muc demo mong muon hay khong.
- Camera data that se thu thap du so luong va co ground truth nhu the nao.

## 4. Rui ro nghien cuu

- So sanh voi paper co the chi la paper-reference, khong phai reproduction.
- `256x256` activation lon, phu thuoc SDRAM va bandwidth.
- Cache/DMA coherency tren Cortex-M7 co the lam sai frame hoac sai latency neu khong log ro.
- Metric co the tot nhung anh van chay sang/le ch mau, can visual grid va crop.
- Calibration/representative set khong dai dien co the lam INT8 banding hoac lech mau.

## 5. Trang thai

Sau dong bo tai lieu, trang thai la:

```text
READY_FOR_SOURCE_TREE_AFTER_USER_APPROVAL
```

Buoc tiep theo hop ly:

1. User chap nhan P0 specs.
2. Tao source tree.
3. Implement data pipeline.
4. Export dummy ONNX -> STM32Cube.AI feasibility test.
