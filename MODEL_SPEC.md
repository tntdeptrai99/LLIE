# Student Architecture Specification

## 1. Muc tieu

Student la CNN nhe chay tren STM32H750VBT6.

Muc tieu:

- Chat luong cao hon Ghost-ESP paper.
- Model-only latency nho hon `277 ms`.
- FPS lon hon `3.6`.
- Muc tieu thuc te `7-10 FPS`.
- Tuong thich ONNX va STM32Cube.AI.
- Ho tro QAT INT8.

Student khong chua Transformer.

Retinexformer chi duoc su dung lam teacher khi training.

## 2. Input shape

### Student-S

```text
Batch x Channel x Height x Width
1 x 3 x 128 x 128
```

### Student-HQ

```text
Batch x Channel x Height x Width
1 x 3 x 256 x 256
```

Hai phien ban dung cung kien truc, chi khac kich thuoc input.

Input logical range:

```text
RGB [0,1]
```

## 3. Kien truc tong the

So do duoi day la kien truc day du cho cac bien the co Dark Guidance. Voi A1/A2, Dark Guidance duoc tat theo bang variant o muc 17.

```text
Input RGB
   |
   v
Stem Conv
   |
   v
Stage 1: Block 5 x2
   |
   v
Downsample
   |
   v
Stage 2: Block 5 x3
   |
   v
Bottleneck: Block 5 x2
   |
   v
Dark Guidance
   |
   v
Upsample
   |
   v
Skip Add
   |
   v
Stage 3: Block 5 x1
   |
   v
Gain Head + Residual Head
   |
   v
Output = clip(Input * Gain + Residual, 0, 1)
```

## 4. Chi tiet tung stage

| Thanh phan | Output 128 | Output 256 | Cau hinh |
|---|---:|---:|---|
| Input | `3x128x128` | `3x256x256` | RGB |
| Stem | `8x128x128` | `8x256x256` | Conv `3x3`, `3->8` |
| Stage 1 | `8x128x128` | `8x256x256` | Block 5 x2 |
| Downsample | `16x64x64` | `16x128x128` | DWConv s2 + PWConv |
| Stage 2 | `16x64x64` | `16x128x128` | Block 5 x3 |
| Bottleneck | `16x64x64` | `16x128x128` | Block 5 x2 |
| Dark Guidance | `16x64x64` | `16x128x128` | concat + Conv `1x1` |
| Upsample | `8x128x128` | `8x256x256` | nearest x2 + Conv `1x1` |
| Skip Add | `8x128x128` | `8x256x256` | add Stage 1 |
| Stage 3 | `8x128x128` | `8x256x256` | Block 5 x1 |
| Gain Head | `3x128x128` | `3x256x256` | Conv `1x1` |
| Residual Head | `3x128x128` | `3x256x256` | Conv `1x1` |

Tong so Block 5:

```text
2 + 3 + 2 + 1 = 8
```

## 5. Stem

Conv2D:

- Kernel: `3x3`.
- Stride: `1`.
- Padding: `1`.
- Input channels: `3`.
- Output channels: `8`.

Activation:

- ReLU6.

BatchNorm co the su dung khi training nhung phai duoc fuse truoc export.

Tat ca convolution trong model dung zero padding. Reflection padding chi duoc dung trong dataset preprocessing, khong dung trong model runtime.

## 6. Block 5

Block 5 la don vi tinh toan chinh cua student.

```text
Input X
   |
   v
Conv 1x1, C->C
   |
   v
ReLU6
   |
   v
Depthwise Conv 3x3, stride 1, padding 1
   |
   v
ReLU6
   |
   v
Conv 1x1, C->C
   |
   v
Add voi X
   |
   v
ReLU6
```

Cong thuc:

```text
Y = ReLU6(X + P2(D(P1(X))))
```

Trong do:

- `P1`: pointwise Conv dau.
- `D`: depthwise Conv.
- `P2`: pointwise projection.

Khong su dung concat ben trong Block 5.

Khong mo rong channel trong phien ban V1 de giam RAM va latency.

## 7. Downsample

Depthwise Conv `3x3`:

- Stride: `2`.
- Padding: `1`.
- Channels: `8`.

Pointwise Conv `1x1`:

- `8 -> 16` channels.

Activation:

- ReLU6.

Chi downsample mot lan. Khong su dung pooling.

## 8. Stage 2 va bottleneck

Stage 2:

```text
Block 5 x3
C = 16
```

Bottleneck:

```text
Block 5 x2
C = 16
```

Khong tang len `32` channel trong phien ban dau. Muc tieu la bao ve FPS va activation RAM.

## 9. Dark Guidance

Dark map duoc tinh truc tiep tu input:

```text
M_dark = 1 - (R + G + B) / 3
```

Dark map co shape:

```text
1 x H x W
```

Sau do resize xuong kich thuoc bottleneck:

- `64x64` cho input `128x128`.
- `128x128` cho input `256x256`.

Vi tri:

```text
sau Bottleneck
truoc Upsample
```

Cach tich hop:

```text
Feature: 16 channels
Dark map: 1 channel
Concat: 17 channels
Conv 1x1: 17->16
ReLU6
```

Chi dung dark guidance mot lan. Khong dung self-attention hoac Transformer trong student.

## 10. Upsample

```text
Nearest-neighbor resize x2
-> Conv 1x1, 16->8
-> ReLU6
```

Khong su dung:

- Transposed convolution.
- Pixel shuffle.
- Dynamic resize.

## 11. Skip connection

Output Stage 1 duoc cong voi output upsample:

```text
F_skip = F_stage1 + F_upsample
```

Dung element-wise Add. Khong dung concatenation de tranh tang so channel va RAM.

## 12. Stage 3

```text
Block 5 x1
C = 8
```

Muc tieu la refine feature sau skip fusion.

## 13. Gain Head

```text
Conv 1x1, 8->3 channels
ReLU6
```

Gain duoc tinh:

```text
G = 1 + ReLU6(Z_G) / 6
```

Do do:

```text
G in [1, 2]
```

Gain theo tung pixel va tung kenh RGB.

Day la quyet dinh V1. Vi `G in [1,2]` khong truc tiep giam gain o highlight, can theo doi cac chi so sau trong ablation:

- Highlight clipping ratio.
- Saturated pixel percentage.
- Kha nang residual am sua vung sang.

Neu residual am khong du de bao ve highlight, bien the V2 co the thu `G in [0.75,2]`, nhung khong doi V1 truoc khi co ablation.

## 14. Residual Head

```text
Conv 1x1, 8->3 channels
```

Residual duoc gioi han:

```text
Delta I in [-0.2, 0.2]
```

Training reference:

```text
Delta I = 0.2 * tanh(Z_R)
```

Tuy nhien truoc khi export can kiem tra `tanh` voi Cube.AI.

Phuong an deploy uu tien:

```text
Delta I = 0.2 * clip(Z_R, -1, 1)
```

Neu `Clip` gay van de khi quantize, residual head co the su dung quantization scale de gioi han bien do ma khong can custom operator.

Phuong an thay the than thien hon voi quantization neu `Clip` hoac `tanh` khong import tot:

```text
Delta I = 0.2 * (ReLU6(Z_R + 3) / 3 - 1)
```

Phuong an nay chi can Add, ReLU6, Multiply va Subtract.

## 15. Output

```text
I_out = clip(I_in * G + Delta I, 0, 1)
```

Output shape bang input shape:

```text
1 x 3 x H x W
```

Output range:

```text
[0,1]
```

## 16. Denoising

Student V1 khong co denoising branch doc lap.

Residual head dam nhiem:

- Giam nhieu nhe.
- Sua mau.
- Phuc hoi chi tiet.
- Bu sai so tu gain map.

Model C co the thu them denoising module trong ablation, nhung module chi duoc giu neu:

- Chat luong tang dang ke.
- Latency van thap hon `277 ms`.
- RAM khong vuot budget.

## 17. Model variants

| Model | Dark Guidance | Gain/Residual | Ghi chu |
|---|---|---|---|
| A1 | OFF | ON | Baseline kien truc + Charbonnier |
| A2 | OFF | ON | A1 + SSIM loss |
| B | ON | ON | Lightweight dark/illumination guidance |
| C | ON | ON | B + advanced losses + optional denoise refinement |
| D | ON | ON | C-compatible student + Retinexformer distillation + QAT INT8 |

### A1

```text
Student architecture
+ Charbonnier loss
```

A1 tat Dark Guidance de tao baseline kien truc toi thieu.

### A2

```text
A1
+ SSIM loss
```

A2 tat Dark Guidance. A1 -> A2 chi khac loss, giup ablation co y nghia.

### B

```text
A2
+ lightweight dark/illumination guidance
```

Model B khong tao illumination decoder rieng. Dark Guidance chinh la lightweight illumination guidance cua Model B.

### C

```text
B
+ advanced losses
+ optional lightweight denoise refinement
```

### D

```text
C-compatible student
+ Retinexformer distillation
+ QAT INT8
```

Kien truc deploy cuoi cung phai giu operator tuong thich STM32Cube.AI.

## 18. 256x256 feasibility note

Student-HQ `256x256` la quality variant co dieu kien.

Activation dau vao lon hon dang ke, vi du:

```text
256 x 256 x 8 = 524288 phan tu
```

Voi INT8, rieng tensor nay da xap xi `512 KB`, chua tinh skip tensor, upsample output, input/output, dark map va intermediate activation.

Do do:

- `128x128` la deployment candidate chinh ban dau.
- `256x256` phu thuoc external-SDRAM activation placement va profiling.
- `256x256` khong duoc mac dinh xem la final deployment model neu chua do latency/RAM that.
- Neu internal SRAM budget chi khoang `90-100 KB`, budget nay chi ap dung cho internal SRAM; large activation tensors va framebuffer phai duoc dat trong external SDRAM.
- Can do latency rieng khi activation nam trong SDRAM vi memory access co the lam giam FPS.

## 19. Operator compatibility list

Operator duoc phep:

- Conv2D.
- DepthwiseConv2D.
- ReLU.
- ReLU6.
- Add.
- Multiply.
- Concat.
- ResizeNearest.
- Clip.
- Fixed Reshape/Transpose neu export yeu cau.

Operator can tranh:

- Transformer attention.
- LayerNorm.
- GELU.
- Grid sample.
- Dynamic shape.
- Dynamic interpolation.
- Custom PyTorch operator.
- Transposed convolution.
- Adaptive pooling phuc tap.
- Python control flow trong forward.

Truoc training A1 va truoc moi thay doi operator quan trong, phai export dummy model chua train sang ONNX va import thu bang STM32Cube.AI.

Blocking feasibility test:

1. Export dummy model co ReLU6, ResizeNearest, Concat, Clip, Add va Mul.
2. Import vao STM32Cube.AI.
3. Confirm FP32 graph import duoc.
4. Confirm INT8/QAT graph ho tro operator can thiet.
5. Neu `Clip` hoac `tanh` khong on dinh, dung residual alternative o muc 14.

Neu blocking test that bai, khong bat dau training dai han cho kien truc do.

## 20. Completion criteria

Spec nay duoc xem la chot khi:

- Input/output tensor shapes ro rang.
- Kien truc Student-S va Student-HQ ro rang.
- Block 5 duoc dinh nghia ro.
- Dark Guidance duoc dinh nghia ro.
- A1/A2 tat Dark Guidance, B/C/D bat Dark Guidance.
- Model B duoc chot la lightweight dark/illumination guidance, khong co illumination decoder rieng.
- Gain/residual heads duoc dinh nghia ro.
- V1 gain range va highlight clipping checks ro rang.
- Padding mode trong model ro rang.
- Dieu kien SDRAM cho `256x256` ro rang.
- A1/A2/B/C/D differences ro rang.
- Operator list tuong thich STM32Cube.AI.
- Blocking dummy ONNX -> STM32Cube.AI test duoc ghi ro.

## 21. D-96-Tiny-BN RAM-first candidate

Bien the nay duoc thiet ke sau khi dummy D-96 dau tien import duoc vao STM32Cube.AI nhung activation RAM FP32 qua lon.

Muc tieu:

- Ke thua Ghost/ESP Block.
- Dung depthwise separable convolution.
- Dung ReLU6 cho quantization-friendly activation.
- Dung BatchNorm khi training va bat buoc fuse BN truoc ONNX export.
- Giam channel de ha activation RAM.
- Giu input/output `1x3x96x96` de so sanh truc tiep voi paper Ghost-ESP ESP32-S3.

Kien truc:

```text
Input RGB 3x96x96
   |
   v
Stem: ConvBNReLU6 3x3, 3->4
   |
   v
Skip feature: 4x96x96
   |
   v
Downsample: DWConvBNReLU6 3x3 s2 + PWConvBNReLU6 1x1, 4->8
   |
   v
GhostESPBNBlock x3, C=8
   |
   v
Dark Guidance: fixed Conv 1x1 RGB->dark, nearest downsample, concat
   |
   v
ConvBNReLU6 1x1, 9->8
   |
   v
Nearest upsample x2
   |
   v
ConvBNReLU6 1x1, 8->4
   |
   v
Add skip
   |
   v
GhostESPBNBlock x1, C=4
   |
   v
Gain Head 1x1, 4->3
Residual Head 1x1, 4->3
   |
   v
Output = clip(Input * Gain + Residual, 0, 1)
```

GhostESPBNBlock:

```text
Input X
   |
   +--> Primary: ConvBNReLU6 1x1, C->C/2
          |
          v
       Cheap ghost: DWConvBNReLU6 3x3, C/2->C/2
          |
          v
       Concat primary + ghost
          |
          v
       Project: ConvBN 1x1, C->C
          |
          v
Output = ReLU6(X + Project)
```

BN folding rule:

```text
Conv + BatchNorm -> fused Conv
Conv + BatchNorm + ReLU6 -> fused Conv + ReLU6
```

Training graph co the co `BatchNorm2D`, nhung ONNX dua vao Cube.AI khong duoc con `BatchNormalization`.

Dummy export:

```powershell
C:\MiniForge\python.exe scripts\export_dummy_onnx.py --arch d96-tiny-bn --blocks 3 --out stm32\onnx\d_96_tiny_bn_fused_dummy.onnx
```

ONNX sau export BN folding:

```text
Input:  1x3x96x96
Output: 1x3x96x96
Train params: 919
Export params: 835
Operators: Conv, Add, Clip, Concat, Constant, Div, Mul, Resize, Sub
```

Neu STM32Cube.AI van bao activation RAM qua lon, thu theo thu tu:

1. Tat Dark Guidance.
2. Giam `blocks` tu `3` xuong `2`.
3. Giam `mid_channels` tu `8` xuong `6` neu export tool chap nhan channel le.
4. Chuyen sang INT8/QAT analyze truoc khi ket luan kien truc that bai.
