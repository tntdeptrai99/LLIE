# Task Breakdown for STM32H750 LLIE

Tai lieu nay chia project thanh cac task trien khai nho, theo dung scope hien tai:

- MCU: STM32H750VBT6 + external SDRAM.
- Camera/display: OV5640 RGB565 -> ST7735 SPI.
- Training/export: PyTorch -> ONNX -> STM32Cube.AI.
- Teacher: Retinexformer.
- Student: shallow depthwise gain-residual network.
- Resolution: 96x96 paper-reference, 128x128 speed baseline, 256x256 quality variant.
- Quantization: QAT INT8.
- Metrics: RGB PSNR, RGB SSIM, LPIPS-AlexNet, CIEDE2000.
- Benchmark: model-only + processing/display pipeline, 20 warmups, 200 measured runs, mean/median/P95/min/max/std, DWT cycles, cache/DMA policy.

## 0. Task Status Legend

- `P0`: Bat buoc truoc khi code model chinh.
- `P1`: Can cho baseline va training dau tien.
- `P2`: Can cho export, QAT, benchmark hoac report.
- `Blocked`: Chua nen lam vi thieu quyet dinh/input.

## Phase 1: Specification Lock

### T1.1 - Preprocessing Protocol

**Priority:** P0

**Muc tieu**

Chot preprocessing tai lap duoc cho 96x96, 128x128 va 256x256.

**Input**

- LOL, LOL-v2 Real.
- Pipeline camera OV5640 RGB565.
- Fixed seed 42.

**Cong viec**

- Chot resize/crop cho 128x128 speed baseline.
- Chot resize/crop cho 256x256 quality variant.
- Chot 96x96 paper-reference mode.
- Chot normalize RGB: vi du `[0,1]`.
- Chot convert RGB565 -> RGB tensor.
- Chot augmentation ranges: exposure, gamma, white balance, shot noise, read noise.
- Chot synthetic low-light sinh tu ground truth/normal-light va thu tu augmentation.
- Chot split artifact format va seed 42.

**Output**

- Preprocessing spec.
- Augmentation spec.
- Split generation spec.

**Tieu chi hoan thanh**

- Cung input va seed 42 tao ra cung split/preprocessing.
- Khong co test leakage.

**Phu thuoc**

- Khong.

### T1.2 - Student Architecture Spec

**Priority:** P0

**Muc tieu**

Chot kien truc student du de viet code.

**Input**

- Scope: shallow depthwise gain-residual network.
- STM32Cube.AI operator constraints.
- Output formula: `I_out = clip(I_in * G + Delta I, 0, 1)`.

**Cong viec**

- Chot so stage/block/channel.
- Chot depthwise + pointwise block.
- Chot activation: ReLU/ReLU6.
- Chot A1/A2 tat lightweight dark/illumination guidance.
- Chot Model B bat lightweight dark/illumination guidance, khong dung illumination decoder rieng.
- Chot denoising module cho C.
- Chot output heads: gain map `G`, residual map `Delta I`.
- Chot range cua gain/residual va clamp output.

**Output**

- Architecture spec cho A1/A2/B/C/D.
- Operator list de check voi STM32Cube.AI.

**Tieu chi hoan thanh**

- Co the implement model ma khong phai hoi lai so layer/channel/head.
- Operator deu nam trong nhom STM32Cube.AI-friendly.

**Phu thuoc**

- T1.1.

### T1.3 - Benchmark Protocol Spec

**Priority:** P0

**Muc tieu**

Chot cach do performance chinh thuc.

**Input**

- SUCCESS_CRITERIA.md.
- STM32H750VBT6 firmware constraints.

**Cong viec**

- Chot model-only benchmark.
- Chot processing pipeline: camera -> preprocess -> inference -> postprocess.
- Chot display pipeline: camera -> preprocess -> inference -> postprocess -> LCD.
- Chot 20 warmups, 200 measured runs.
- Chot report mean, median, P95, min, max, std.
- Chot cach tinh FPS_mean va FPS_P95_equivalent tu latency.
- Chot DWT cycle counter va overflow handling.
- Chot cache/DMA coherency policy.
- Chot sequential vs pipelined measurement.
- Chot nguon RAM/Flash: STM32Cube.AI report, map file, runtime measurement neu co.

**Output**

- Benchmark protocol.
- Report table template.

**Tieu chi hoan thanh**

- Ket qua benchmark co the tai lap va so sanh giua model.

**Phu thuoc**

- Khong.

### T1.4 - Metric Implementation Spec

**Priority:** P0

**Muc tieu**

Chot cach tinh metric chat luong.

**Input**

- SUCCESS_CRITERIA.md.
- Dataset output/ground truth.

**Cong viec**

- Chot RGB PSNR tren mien gia tri nao.
- Chot RGB SSIM va window/settings.
- Chot LPIPS-AlexNet.
- Chot CIEDE2000 conversion RGB -> Lab.
- Chot cach tinh metric tren batch va aggregate.

**Output**

- Metric spec.
- Report table template.

**Tieu chi hoan thanh**

- Metric tinh ra tai lap duoc va khong phu thuoc setting ngam dinh.

**Phu thuoc**

- T1.1.

## Phase 2: Project Skeleton And Feasibility

### T2.1 - Create Source Structure

**Priority:** P1

**Muc tieu**

Tao cau truc source code toi thieu sau khi spec P0 da chot.

**Input**

- IMPLEMENTATION_PLAN.md.
- TASK_BREAKDOWN.md.

**Cong viec**

- Tao thu muc cho configs, src, scripts, experiments, reports.
- Tao README ngan cho cach chay.
- Tao file config mau cho A1 128x128.

**Output**

- Source tree ban dau.

**Tieu chi hoan thanh**

- Co noi de dat dataset loader, model, loss, train, eval, export.

**Phu thuoc**

- T1.1, T1.2, T1.4.

### T2.2 - Dummy ONNX/Cube.AI Feasibility

**Priority:** P1

**Muc tieu**

Kiem tra operator truoc khi train dai han.

**Input**

- `MODEL_SPEC.md`.
- STM32Cube.AI.

**Cong viec**

- Export dummy student co ReLU6, ResizeNearest, Concat, Clip, Add va Mul.
- Import vao STM32Cube.AI.
- Check FP32 graph.
- Check INT8/QAT graph support.
- Neu Clip/tanh khong on dinh, dung residual alternative trong `MODEL_SPEC.md`.

**Output**

- Feasibility report.

**Tieu chi hoan thanh**

- Operator duoc Cube.AI chap nhan truoc khi train A1/A2 dai han.

**Phu thuoc**

- T2.1.

## Phase 3: Data Pipeline

### T3.1 - Dataset Loader

**Priority:** P1

**Muc tieu**

Load LOL va LOL-v2 Real theo official split.

**Input**

- Dataset paths.
- Preprocessing spec.

**Cong viec**

- Implement paired image loader.
- Implement train/val split tu train split voi seed 42.
- Implement official test split loader.
- Implement scene-based camera split placeholder.

**Output**

- Dataset loader.
- Split files.

**Tieu chi hoan thanh**

- Train/val/test counts dung.
- Khong co test image trong train/val.

**Phu thuoc**

- T2.1.

### T3.2 - Augmentation Pipeline

**Priority:** P1

**Muc tieu**

Them augmentation low-light co kiem soat.

**Input**

- Augmentation spec.

**Cong viec**

- Implement exposure/gamma/noise/white-balance augmentation.
- Dam bao synthetic chi dung train augmentation, khong dung test.

**Output**

- Augmentation module.

**Tieu chi hoan thanh**

- Augmentation deterministic voi seed khi can.
- Khong lam thay doi ground truth/test.

**Phu thuoc**

- T3.1.

## Phase 4: Baseline Models

### T4.1 - Model A1

**Priority:** P1

**Muc tieu**

Implement shallow depthwise gain-residual network baseline voi Charbonnier.

**Input**

- Architecture spec.
- Dataset pipeline.

**Cong viec**

- Implement depthwise/pointwise blocks.
- Implement gain/residual heads.
- Implement output composition.
- Implement Charbonnier loss.

**Output**

- A1 model.
- A1 config 128x128.

**Tieu chi hoan thanh**

- Forward pass OK.
- Output shape dung.
- Output range duoc clamp.

**Phu thuoc**

- T3.1, T1.2, T2.2.

### T4.2 - Model A2

**Priority:** P1

**Muc tieu**

Them SSIM loss vao A1 de tao baseline noi bo chinh.

**Input**

- A1.
- Metric/loss spec.

**Cong viec**

- Implement SSIM loss.
- Train/eval A2 128x128.
- So sanh A1 vs A2.

**Output**

- A2 model/config.
- Baseline report.

**Tieu chi hoan thanh**

- A2 co metric report RGB PSNR, RGB SSIM, LPIPS-AlexNet, CIEDE2000.
- A2 duoc chon lam baseline noi bo.

**Phu thuoc**

- T4.1, T1.4.

## Phase 5: Proposed Models

### T5.1 - Model B

**Priority:** P1

**Muc tieu**

Them lightweight dark/illumination guidance vao A2.

**Input**

- A2.
- Architecture spec.

**Cong viec**

- Implement Dark Guidance theo `MODEL_SPEC.md`.
- Train/eval B.
- So sanh B vs A2.

**Output**

- Model B.
- Ablation report.

**Tieu chi hoan thanh**

- B cai thien vung toi ma khong lam chay sang ro ret.

**Phu thuoc**

- T4.2.

### T5.2 - Model C

**Priority:** P1

**Muc tieu**

Them dark-region attention, denoising va loss nang cao.

**Input**

- Model B.
- Loss spec.

**Cong viec**

- Them dark-region attention.
- Them denoising module.
- Them perceptual/color/edge/illumination loss theo ablation.
- Train/eval C.

**Output**

- Model C FP32.
- Ablation report.

**Tieu chi hoan thanh**

- C vuot A2 ve chat luong anh va giam artifact/noise.

**Phu thuoc**

- T5.1.

## Phase 6: Teacher and Distillation

### T6.1 - Retinexformer Teacher Setup

**Priority:** P2

**Muc tieu**

Dung Retinexformer lam teacher.

**Input**

- Retinexformer code/checkpoint.
- Dataset pipeline.

**Cong viec**

- Xac minh license/checkpoint.
- Sinh teacher outputs hoac feature targets.
- Chot distillation loss.

**Output**

- Teacher output cache hoac teacher inference pipeline.

**Tieu chi hoan thanh**

- Student co the train voi teacher ma inference khong phu thuoc teacher.

**Phu thuoc**

- T5.2.

### T6.2 - Model D FP32 Distilled Student

**Priority:** P2

**Muc tieu**

Train student distilled truoc QAT.

**Input**

- Model C/student.
- Teacher outputs.

**Cong viec**

- Implement distillation loss.
- Train/eval distilled student.
- So sanh D FP32 vs C FP32/A2.

**Output**

- D FP32.
- Distillation report.

**Tieu chi hoan thanh**

- D FP32 gan C FP32 va vuot A2.

**Phu thuoc**

- T6.1.

## Phase 7: Export and Quantization

### T7.1 - ONNX Export

**Priority:** P2

**Muc tieu**

Export student sang ONNX de dua vao STM32Cube.AI.

**Input**

- D FP32.
- Export spec.

**Cong viec**

- Export ONNX.
- Validate ONNX output gan PyTorch.
- Check operator compatibility.

**Output**

- ONNX model.
- Export validation report.

**Tieu chi hoan thanh**

- Sai khac PyTorch vs ONNX nam trong nguong chap nhan.

**Phu thuoc**

- T6.2.

### T7.2 - QAT INT8

**Priority:** P2

**Muc tieu**

Train/fine-tune QAT INT8.

**Input**

- D FP32.
- Calibration/representative train subset.

**Cong viec**

- Implement QAT flow.
- Export QAT model.
- Do suy giam FP32 -> INT8.

**Output**

- D INT8.
- FP32 vs INT8 report.

**Tieu chi hoan thanh**

- PSNR giam <= 0.3 dB.
- SSIM giam <= 0.01.
- LPIPS tang <= 5%.
- D INT8 vuot A2 FP32.

**Phu thuoc**

- T7.1.

## Phase 8: Ghost Comparison

### T8.1 - Ghost-ESP-like Baseline

**Priority:** P2

**Muc tieu**

Xay baseline Ghost-ESP-like de so sanh cong bang hon paper reference.

**Input**

- Paper PDF.
- Dataset pipeline.
- Metric spec.

**Cong viec**

- Implement Ghost-ESP-like model theo mo ta paper.
- Train/eval tren cung split/preprocessing neu co the.
- Bao cao la reimplemented baseline.

**Output**

- Ghost-ESP-like baseline report.

**Tieu chi hoan thanh**

- Co bang so sanh A2/C/D voi Ghost-ESP-like va paper reference.

**Phu thuoc**

- T3.1, T1.4.

## Phase 9: STM32 Deployment

### T9.1 - STM32Cube.AI Validation

**Priority:** P2

**Muc tieu**

Validate model voi STM32Cube.AI.

**Input**

- ONNX/QAT model.

**Cong viec**

- Import vao STM32Cube.AI.
- Lay report Flash/RAM/operator.
- Kiem tra warnings/errors.

**Output**

- Cube.AI report.

**Tieu chi hoan thanh**

- Model import duoc.
- RAM/Flash nam trong budget.

**Phu thuoc**

- T7.2.

### T9.2 - Firmware Benchmark

**Priority:** P2

**Muc tieu**

Do model-only va end-to-end tren STM32H750VBT6.

**Input**

- STM32 firmware.
- OV5640 RGB565.
- ST7735 SPI.
- D INT8.

**Cong viec**

- Tich hop inference.
- Do model-only.
- Do processing pipeline camera -> preprocess -> inference -> postprocess.
- Do display pipeline camera -> preprocess -> inference -> postprocess -> LCD.
- Chay 20 warmups, 200 measured runs.
- Bao cao mean, median, P95, min, max, std.
- Log DWT cycles, cache/DMA policy, sequential/pipelined mode.

**Output**

- Hardware benchmark report.

**Tieu chi hoan thanh**

- 128x128 dat >3.6 FPS toi thieu, muc tieu 5-10 FPS.
- Latency <277 ms.

**Phu thuoc**

- T9.1.

## Phase 10: Reporting

### T10.1 - Final Evaluation Report

**Priority:** P2

**Muc tieu**

Tong hop ket qua nghien cuu va trien khai.

**Input**

- Reports tu cac phase.

**Cong viec**

- Tong hop metric chat luong.
- Tong hop performance.
- Tao bang ablation.
- Tao bang so sanh Ghost-ESP-like + paper reference.
- Ghi ro dieu kien so sanh cong bang.

**Output**

- Final report.

**Tieu chi hoan thanh**

- Co ket luan ro dat/khong dat `SUCCESS_CRITERIA.md`.

**Phu thuoc**

- T8.1, T9.2.

## Immediate Next Tasks

Truoc khi tao source code, can lam 4 task P0:

1. T1.1 - Preprocessing Protocol.
2. T1.2 - Student Architecture Spec.
3. T1.3 - Benchmark Protocol Spec.
4. T1.4 - Metric Implementation Spec.

Sau khi 4 task nay duoc chot, moi nen tao source structure va bat dau code.
