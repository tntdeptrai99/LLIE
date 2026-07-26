# Ghi Chu Cho Phien Lam Viec Sau

Ngay tao: 2026-07-23

Project: `D:\LLIE_Project`

Firmware dang debug: `stm32/firmware/LLIE_E2E_Benchmark`

## Muc Tieu Cuoi Cung

Dat duoc pipeline chay tren STM32H750:

```text
OV5640 RGB565 -> preprocess -> Cube.AI LLIE model -> postprocess/display -> ST7735 LCD
```

Ket qua mong muon:

- Anh dau ra tren LCD la output AI that, khong phai fallback/input/manual gain.
- Tensor output tren board phai gan dung voi ONNX Runtime tren PC.
- Neu model tot tren PC thi board phai the hien cung xu huong tang sang.
- Hieu nang chap nhan duoc: AI khoang 180-190 ms, tong frame khoang 200-230 ms la baseline hien tai.

## Ket Luan Da Chung Minh

Khong nen quay lai debug LCD truoc. LCD khong phai loi chinh hien tai.

Da test fixed input `uint8=32`:

- Input vao Cube.AI tren board dung: `min=max=32`.
- ONNX Runtime voi cung input cho:
  - output0 gain: khoang `197..242`, mean khoang `230`.
  - output1 residual: toan `255`.
- Cube.AI tren board voi model `tail2_u8out` cho output sai:
  - output0/public: khoang `128..187`, mean khoang `143`.
  - output1/public: mean khoang `119`, khong phai `255`.
  - external output buffers khong dung duoc voi model nay vi model generated co `AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS`.

Ket luan: **model Cube.AI generated tu `tail2_u8out.onnx` khong tuong duong ONNX Runtime o phan tail gain/residual**.

## Nhung Viec Khong Nen Lam Lai

- Khong tiep tuc chinh `gain_q8` / `residual_q8` de lam dep LCD khi tensor AI output con sai.
- Khong danh gia bang mat tren LCD truoc khi tensor equivalence dung.
- Khong tiep tuc doan layout `ai_output_layout` cho `tail2_u8out`; cac layout da thu deu khong khop ONNX.
- Khong ket luan model train kem tu anh LCD hien tai. Loi dang nam o duong deploy Cube.AI/tail output, khong phai chat luong train.

## Huong Di Chinh Cho Phien Sau

Chuyen tu model 2 output gain/residual sang model 1 output anh RGB cuoi cung.

Dung ONNX co san:

```text
D:\LLIE_Project\stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_mcu_u8out.onnx
```

Ly do:

- `mcu_u8out.onnx` xuat thang `enhanced_rgb_QuantizeLinear_Output`.
- Firmware khong can tu ghep gain/residual nua.
- Duong output don gian hon, de validate hon, it kha nang Cube.AI loi o tail float/quantize/transpose.

## Ke Hoach Lam Viec Cu The

### Buoc 1: Regenerate Cube.AI Tu `mcu_u8out.onnx`

Trong STM32CubeMX/CubeIDE:

1. Mo project:

```text
D:\LLIE_Project\stm32\firmware\LLIE_E2E_Benchmark
```

2. Trong X-CUBE-AI, thay model hien tai bang:

```text
D:\LLIE_Project\stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_mcu_u8out.onnx
```

3. Generate lai code voi network name giu la:

```text
llieai
```

4. Kiem tra report moi:

```text
stm32/firmware/LLIE_E2E_Benchmark/X-CUBE-AI/App/llieai_generate_report.txt
```

Can thay:

- `AI_LLIEAI_OUT_NUM = 1`
- output la `uint8`
- shape public tuong duong `1x3x96x96`

### Buoc 2: Sua Firmware Sang Single Output

Trong `Core/Src/main.c`:

- Bo logic residual/gain tail cho test equivalence.
- Dump tensor:

```text
input_runtime
output0_public
```

- Khong can dump `output1_public`.
- Fixed input test van dung `32`.

### Buoc 3: So Sanh Tensor Board vs ONNX

Cap nhat `scripts/compare_board_tensor_dump.py` neu can de ho tro single-output ONNX.

Lenh test:

```powershell
python D:\LLIE_Project\scripts\compare_board_tensor_dump.py --log D:\LLIE_Project\board_dump_mcu_u8out_fixed32.log --fixed-input-u8 32 --onnx D:\LLIE_Project\stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_mcu_u8out.onnx
```

Tieu chi chap nhan:

- `board input u8`: `min=32 max=32`.
- `board output0` gan ONNX:
  - MAE quantized nho, ly tuong < 3-5.
  - Cosine > 0.99.
  - Min/max/mean cung xu huong.

Neu input dung nhung output van sai xa:

- Van de nam o Cube.AI conversion/runtime cho model nay.
- Can tao model ONNX don gian hon nua hoac dung Cube.AI validation CLI neu co.

### Buoc 4: Test Camera Frame That

Chi lam sau khi fixed input dung.

Quy trinh:

1. Bat camera/preprocess.
2. Dump `input_runtime` tu frame camera that.
3. Chay ONNX Runtime tren PC voi chinh input do.
4. So output board vs output ONNX.

Neu:

- Input board khac anh train: loi camera/preprocess/domain.
- Input giong, output board khac ONNX: loi Cube.AI.
- Output board giong ONNX nhung LCD xau: loi postprocess/display.

### Buoc 5: Dua Output AI Len LCD

Chi lam sau khi tensor output board vs ONNX da dung.

Voi `mcu_u8out`, postprocess se don gian:

```text
output_u8 CHW/RGB -> crop/resize neu can -> RGB565 -> LCD
```

Khong dung:

- manual gain blend
- residual blend
- fallback input
- layout guessing neu da co metadata ro

## Trang Thai Firmware Gan Nhat

Ban debug gan nhat:

```text
FW_PROBE_VERSION 7271
```

Da lam:

- Fixed input dump dung.
- Dump public output, activation offsets, external buffers.
- Loai tru:
  - LCD
  - camera
  - cache invalidate output
  - output external buffer
  - signedness int8/uint8
  - output order swap
  - layout HWC/CHW don gian

Can luu y: `main.c` hien dang o che do debug fixed input, khong phai ban firmware final.

## Cau Noi De Bat Dau Phien Sau

Neu tiep tuc voi Codex, hay noi:

```text
Tiep tuc tu docs/NEXT_SESSION_ACTION_PLAN.md. Hay regenerate/port firmware sang model mcu_u8out single-output va lam tensor equivalence fixed input truoc khi quay lai LCD.
```

## Ket Luan Ngan Gon

Duong `tail2_u8out` gain/residual hien tai khong dang tin tren Cube.AI board. Muon co ket qua cuoi cung tot, can chuyen sang `mcu_u8out` single-output, validate tensor equivalence truoc, roi moi hien thi LCD.

