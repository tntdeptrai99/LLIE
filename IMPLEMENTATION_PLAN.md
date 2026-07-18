# Implementation Plan for LLIE on STM32H750

Tai lieu nay la ke hoach trien khai dong bo theo:

- `PROJECT_SCOPE.md`
- `PREPROCESSING_SPEC.md`
- `MODEL_SPEC.md`
- `METRIC_SPEC.md`
- `BENCHMARK_SPEC.md`
- `TASK_BREAKDOWN.md`

`LLIE_PROPOSAL.md` la proposal goc va co the chua phan anh cac quyet dinh moi nhat.

## 1. Muc tieu

Xay dung he thong LLIE frame-by-frame tren STM32H750VBT6 co external SDRAM, dung OV5640 RGB565 va ST7735 SPI, voi student QAT INT8 export bang ONNX -> STM32Cube.AI.

Muc tieu khong phai do phan giai toi da. Muc tieu la:

- Chat luong anh tot hon Ghost-ESP paper/reference trong dieu kien so sanh ro.
- `128x128` model-only latency `<277 ms`.
- `128x128` model-only FPS `>3.6`, muc tieu thuc te `7-10 FPS`.
- Model D INT8 vuot A2 FP32 va gan C FP32.

## 2. Dau vao va dau ra

Input:

- RGB low-light image.
- Training/evaluation logical range `[0,1]`.
- Deployment input tu OV5640 RGB565.

Output:

```text
I_out = clip(I_in * G + Delta I, 0, 1)
```

Model output la gain map va residual map, khong phai RGB truc tiep.

## 3. Phuong phap

Phuong phap chinh:

- Shallow depthwise gain-residual network.
- Block 5 voi Conv `1x1`, Depthwise Conv `3x3`, Conv `1x1`, residual add.
- Mot lan downsample va mot lan upsample.
- Dark Guidance nhe cho Model B tro len.
- Residual head dam nhiem sua mau, chi tiet va giam nhieu nhe.
- Retinexformer dung lam teacher khi distillation.
- QAT INT8 cho model deploy.

Khong dung:

- Transformer trong student.
- Illumination decoder rieng cho Model B.
- U-Net Lite lam baseline hien hanh.

## 4. Baseline

Baseline noi bo:

| Model | Dark Guidance | Noi dung |
|---|---|---|
| A1 | OFF | Student + Charbonnier |
| A2 | OFF | A1 + SSIM |

A2 la baseline noi bo chinh.

Baseline doi chieu:

- Ghost-ESP-like baseline neu co the tai hien.
- Paper reference neu chi co so lieu paper.
- `96x96` dung cho paper-reference fairness.

## 5. Du lieu

Can co:

- LOL.
- LOL-v2 Real.
- Camera data chia theo scene.
- Calibration/representative set tu train.

Khong dung LOL-v2 Synthetic trong phien ban dau tru khi mo rong sau.

Synthetic low-light:

- Sinh tu ground truth/normal-light image.
- Chi dung training augmentation.
- Khong dung test chinh.

## 6. Metrics

Quality:

- RGB PSNR.
- RGB SSIM.
- LPIPS-AlexNet.
- CIEDE2000.

Deployment:

- Params.
- MACs.
- Internal SRAM.
- External SDRAM.
- Flash.
- Model-only latency/FPS.
- Processing/display pipeline FPS.

## 7. Giai doan trien khai

### Giai doan 1: Khoa P0 specs

**Muc tieu**

Khoa preprocessing, model, metric va benchmark specs.

**Input can thiet**

- Proposal goc.
- Quyet dinh board/camera/display/training/export.
- 4 spec P0.

**Cong viec phai lam**

- Dong bo tai lieu.
- Kiem tra khong con mau thuan U-Net/illumination branch/160x160.
- Khoa `96x96`, `128x128`, `256x256` roles.

**Output du kien**

- Tai lieu Markdown dong bo.

**Tieu chi hoan thanh**

- P0 specs ro du de tao source tree.
- Model B duoc chot la lightweight dark/illumination guidance.

**Rui ro hoac thong tin con thieu**

- `256x256` chua chac deploy duoc neu SDRAM/latency khong dat.

### Giai doan 2: Tao source tree

**Muc tieu**

Tao cau truc source code toi thieu sau khi P0 specs duoc chap nhan.

**Input can thiet**

- 4 spec P0.
- `TASK_BREAKDOWN.md`.

**Cong viec phai lam**

- Tao folders cho configs, src, scripts, splits, reports, experiments.
- Tao config mau cho A1 `128x128`.
- Tao noi luu split artifacts.

**Output du kien**

- Source tree ban dau.

**Tieu chi hoan thanh**

- Co noi de implement dataset, model, loss, train, eval, export.

**Rui ro hoac thong tin con thieu**

- Chua co dataset paths that.

### Giai doan 3: Data pipeline

**Muc tieu**

Implement preprocessing va split dung spec.

**Input can thiet**

- LOL, LOL-v2 Real.
- `PREPROCESSING_SPEC.md`.

**Cong viec phai lam**

- Implement paired loader.
- Generate split files voi seed `42`.
- Implement exact crop train va center-square-crop test.
- Implement synthetic augmentation dung thu tu spec.
- Tao calibration list.

**Output du kien**

- Dataset loader.
- Split artifacts.
- Preprocessing validation report.

**Tieu chi hoan thanh**

- Khong leakage train/val/test.
- Same seed tao same split.

**Rui ro hoac thong tin con thieu**

- Dataset folder structure thuc te co the khac.

### Giai doan 4: Dummy export feasibility

**Muc tieu**

Kiem tra operator truoc training dai han.

**Input can thiet**

- `MODEL_SPEC.md`.
- STM32Cube.AI.

**Cong viec phai lam**

- Export dummy model co ReLU6, ResizeNearest, Concat, Clip, Add, Mul.
- Import vao STM32Cube.AI.
- Check FP32 va INT8/QAT graph support.

**Output du kien**

- Feasibility report.

**Tieu chi hoan thanh**

- Operator duoc Cube.AI chap nhan.

**Rui ro hoac thong tin con thieu**

- Clip/tanh co the can thay bang residual alternative.

### Giai doan 5: A1/A2 baseline

**Muc tieu**

Xay baseline noi bo.

**Input can thiet**

- Data pipeline.
- Student architecture.

**Cong viec phai lam**

- Train A1.
- Train A2.
- Report quality metrics.
- Chon A2 lam baseline.

**Output du kien**

- A1/A2 checkpoints.
- Baseline report.

**Tieu chi hoan thanh**

- A2 co RGB PSNR, RGB SSIM, LPIPS-AlexNet, CIEDE2000.

**Rui ro hoac thong tin con thieu**

- Baseline qua cham hoac qua lon can giam channel/block.

### Giai doan 6: B/C quality models

**Muc tieu**

Them guidance va quality losses theo ablation.

**Input can thiet**

- A2 baseline.

**Cong viec phai lam**

- B: bat Dark Guidance.
- C: advanced losses va optional denoise refinement.
- Kiem tra highlight clipping va saturated pixel percentage.
- Export dummy neu them operator moi.

**Output du kien**

- B/C reports.

**Tieu chi hoan thanh**

- C vuot A2 ve quality va khong tang artifact ro ret.

**Rui ro hoac thong tin con thieu**

- Loss phuc tap co the kho toi uu.

### Giai doan 7: Distillation va QAT

**Muc tieu**

Tao Model D deployable.

**Input can thiet**

- C-compatible student.
- Retinexformer teacher.
- Calibration set.

**Cong viec phai lam**

- Generate teacher outputs neu license/checkpoint hop le.
- Train distilled student.
- QAT INT8.
- Export ONNX/Cube.AI.

**Output du kien**

- D FP32.
- D INT8.
- FP32 vs INT8 report.

**Tieu chi hoan thanh**

- D INT8 vuot A2 FP32.
- INT8 degradation trong nguong metric.

**Rui ro hoac thong tin con thieu**

- Retinexformer license/checkpoint can xac minh.

### Giai doan 8: Ghost comparison

**Muc tieu**

So sanh voi Ghost-ESP-like va paper reference dung muc bang chung.

**Input can thiet**

- Paper Ghost-ESP.
- Dataset/test outputs.

**Cong viec phai lam**

- Neu co the, implement Ghost-ESP-like baseline.
- Chay `96x96` paper-reference mode.
- Report ro muc so sanh.

**Output du kien**

- Ghost comparison report.

**Tieu chi hoan thanh**

- Ket luan khong vuot qua bang chung.

**Rui ro hoac thong tin con thieu**

- Khong co code/checkpoint official cua Ghost-ESP.

### Giai doan 9: STM32 benchmark

**Muc tieu**

Do deploy thuc te tren STM32H750.

**Input can thiet**

- D INT8.
- Firmware STM32.

**Cong viec phai lam**

- Do model-only.
- Do processing pipeline.
- Do display pipeline.
- Log cache/DMA policy.
- Bao cao RAM/Flash tach internal SRAM va SDRAM.

**Output du kien**

- Hardware benchmark report.

**Tieu chi hoan thanh**

- `128x128` latency `<277 ms`.
- FPS `>3.6`, muc tieu `7-10`.

**Rui ro hoac thong tin con thieu**

- ST7735 SPI co the la bottleneck end-to-end.

### Giai doan 10: Final report

**Muc tieu**

Tong hop ablation, quality va deployment.

**Input can thiet**

- Tat ca reports.

**Cong viec phai lam**

- Tong hop bang metric.
- Tong hop visual grid.
- Bao cao CI neu tuyen bo vuot baseline/paper.
- Ghi ro direct/Ghost-like/paper-reference comparison.

**Output du kien**

- Final report.

**Tieu chi hoan thanh**

- Co ket luan dat/khong dat `SUCCESS_CRITERIA.md`.

**Rui ro hoac thong tin con thieu**

- Camera test it co the lam ket luan domain thuc yeu.

## 8. Cau truc thu muc de xuat

Chua tao o buoc nay.

```text
LLIE_Project/
  docs/
  configs/
  src/
    data/
    models/
    losses/
    train/
    eval/
    export/
  scripts/
  splits/
  experiments/
  reports/
  stm32/
```

## 9. Uu tien tiep theo

1. Xac nhan P0 specs da locked.
2. Tao source tree.
3. Implement data pipeline.
4. Export dummy model truoc khi training dai han.
5. Train A1/A2 `128x128`.
