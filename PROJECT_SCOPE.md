# Project Scope: Real-time LLIE on STM32H750

Tai lieu nay la scope hien hanh cua project. `LLIE_PROPOSAL.md` duoc giu nhu proposal goc/historical; cac quyet dinh trien khai phai uu tien cac spec P0:

- `PREPROCESSING_SPEC.md`
- `MODEL_SPEC.md`
- `METRIC_SPEC.md`
- `BENCHMARK_SPEC.md`

## 1. Scope chinh thuc

Project la de tai Low-Light Image Enhancement chay truc tiep tren STM32H750VBT6 co external SDRAM.

Muc tieu chinh:

1. Real-time hoac near-real-time frame-by-frame LLIE tren STM32H750.
2. Anh dau ra tot hon Ghost-ESP paper trong dieu kien so sanh ro rang.
3. Model deploy bang QAT INT8 qua STM32Cube.AI.
4. Giu `128x128` lam speed baseline va `256x256` lam quality variant co dieu kien.

Project khong tuyen bo video enhancement day du neu chua danh gia temporal consistency.

## 2. Hardware va pipeline

- MCU: STM32H750VBT6.
- External SDRAM: dung cho framebuffer va activation lon neu can.
- Camera: OV5640 RGB565.
- Display: ST7735 SPI.
- Training: PyTorch.
- Export: ONNX -> STM32Cube.AI.
- Quantization: QAT INT8.

Pipeline runtime:

```text
OV5640 RGB565
-> DCMI/DMA
-> framebuffer
-> crop/resize
-> RGB565 to RGB tensor
-> INT8 model
-> RGB output
-> RGB565
-> ST7735 SPI
```

## 3. Input va output

### Input

- RGB low-light image.
- Logical range `[0,1]` trong training/evaluation.
- RGB565 tu OV5640 khi deployment.
- Resolutions:
  - `96x96`: paper-reference/evaluation-only mode.
  - `128x128`: speed baseline va deployment candidate chinh.
  - `256x256`: quality variant, phu thuoc external SDRAM va profiling.

### Output

Model du doan gain map va residual map:

```text
I_out = clip(I_in * G + Delta I, 0, 1)
```

Output can:

- Tang sang vung toi.
- Giu chi tiet va bien anh.
- Giam nhieu vua phai.
- Han che highlight clipping.
- Giu mau tu nhien.

## 4. Dataset

Dataset chinh:

- LOL.
- LOL-v2 Real.

Quy tac:

- Giu official test split.
- Official train split duoc chia `90%` train, `10%` validation bang seed `42`.
- Camera data chia theo scene: `70%` train, `15%` validation, `15%` test.
- Synthetic low-light chi dung nhu augmentation, sinh tu ground-truth/normal-light image, khong dung lam test chinh.
- Calibration/representative set lay tu train, `300-500` anh, khong lay tu test.

Chi tiet preprocessing nam trong `PREPROCESSING_SPEC.md`.

## 5. Model roadmap

Student la shallow depthwise gain-residual network, khong dung Transformer.

| Model | Dark Guidance | Noi dung | Vai tro |
|---|---|---|---|
| A1 | OFF | Student + Charbonnier | Pipeline sanity baseline |
| A2 | OFF | A1 + SSIM | Baseline noi bo chinh |
| B | ON | A2 + lightweight dark/illumination guidance | Kiem soat vung toi |
| C | ON | B + advanced losses + optional denoise refinement | Quality FP32 model |
| D | ON | C-compatible student + Retinexformer distillation + QAT INT8 | Deployable model |

Model B khong co illumination decoder rieng. Dark Guidance chinh la lightweight illumination guidance.

Teacher uu tien: Retinexformer, chi dung khi training/distillation.

## 6. Metric

Metric chat luong chinh:

- RGB PSNR.
- RGB SSIM, tinh theo tung channel roi lay trung binh.
- LPIPS-AlexNet.
- CIEDE2000.

Metric trien khai:

- Model-only latency.
- End-to-end latency.
- Processing pipeline FPS.
- Display pipeline FPS.
- Internal SRAM.
- External SDRAM.
- Flash.
- Params/MACs.

Chi tiet metric nam trong `METRIC_SPEC.md` va `BENCHMARK_SPEC.md`.

## 7. Ghost-ESP comparison

Paper reference:

- Ghost-ESP paper dung moc `96x96`.
- Latency paper: khoang `277 ms/frame`, tuong duong `3.6 FPS`.
- LOL paper reference: PSNR khoang `19.18 dB`, SSIM khoang `0.852`.

Project dung hai muc so sanh:

- Ghost-ESP-like baseline neu tai hien duoc.
- Paper-reference comparison neu chi dung so lieu paper.

Neu khong cung dataset, preprocessing, resolution, metric va phan cung, ket luan phai ghi ro la comparison tham khao.

## 8. Dieu kien thanh cong

Project dat muc toi thieu khi:

- Model D INT8 chay tren STM32H750.
- `128x128` model-only latency `<277 ms`.
- `128x128` model-only FPS `>3.6`, muc tieu thuc te `7-10 FPS`.
- Model D INT8 vuot A2 FP32 ve chat luong.
- Model D INT8 gan C FP32 theo nguong INT8 trong `METRIC_SPEC.md`.
- RAM/Flash khong vuot budget.
- Anh output it chay sang, it lech mau, it noise ro ret.

`256x256` chi la quality variant neu chua dat speed/RAM chinh thuc.

## 9. Ngoai scope hien tai

- Khong bat dau bang RAW Bayer hoac Neural ISP.
- Khong dung `160x160`/`192x192` lam pipeline chinh.
- Khong them illumination decoder nang.
- Khong ket luan vuot Ghost-ESP tuyet doi neu chi dung paper reference.
- Khong tao source tree cho den khi P0 spec duoc chap nhan.
