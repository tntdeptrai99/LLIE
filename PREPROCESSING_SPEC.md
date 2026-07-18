# Preprocessing Specification

## 1. Muc tieu

Tai lieu nay dinh nghia toan bo quy trinh xu ly du lieu cho:

- Student-S: baseline toc do o `128x128`.
- Student-HQ: phien ban chat luong o `256x256`.
- Dataset LOL va LOL-v2 Real.
- Camera OV5640 xuat RGB565.
- Training bang PyTorch.
- Deployment INT8 tren STM32H750VBT6.

Quy trinh phai:

- Tai lap duoc voi seed co dinh.
- Khong gay data leakage.
- Tuong thich giua training va deployment.
- Giu dung cap low-light/ground-truth.
- Ho tro QAT INT8.

## 2. Khong gian mau

Toan bo model su dung anh RGB ba kenh.

Training input:

```text
RGB float32, range [0, 1]
```

Deployment input:

```text
OV5640 RGB565
-> RGB888
-> quantized INT8 tensor
```

Khong su dung ImageNet mean/std.

Dataset image khi doc tu file:

```text
RGB uint8, range [0, 255]
```

Quy doi:

```text
image_float = image_uint8 / 255.0
```

Training tensor layout:

```text
C x H x W
```

Lab chi duoc dung de tinh CIEDE2000 trong evaluation, khong dung lam input model.

## 3. Preprocessing cho Student-S 128x128

### Training

Doi voi moi cap low-light va ground truth:

1. Doc hai anh o RGB.
2. Kiem tra hai anh co cung kich thuoc.
3. Neu chieu rong hoac chieu cao nho hon `128`:
   - Dung reflection padding.
   - Ap dung cung padding cho low-light va ground truth.
4. Random crop paired dung kich thuoc chinh xac `128x128`.
5. Khong resize sau crop trong training `128x128`.
6. Chuyen sang tensor PyTorch:

```text
C x H x W = 3 x 128 x 128
```

7. Chuan hoa ve `[0,1]`:

```text
x = pixel / 255.0
```

### Validation va test

Khong dung random crop.

Quy trinh:

```text
RGB
-> center crop vuong lon nhat
-> resize 128x128 bang bilinear
-> tensor [3,128,128]
-> normalize [0,1]
```

Muc tieu la dam bao toan bo model duoc danh gia tren cung kich thuoc va cung truong nhin voi protocol `256x256`.

## 4. Preprocessing cho Student-HQ 256x256

### Training

Doi voi moi cap low-light va ground truth:

1. Doc hai anh o RGB.
2. Kiem tra hai anh co cung kich thuoc.
3. Reflection padding neu anh nho hon `256x256`.
4. Random crop dong bo `256x256`.
5. Khong resize neu crop da dung `256x256`.
6. Chuyen sang tensor:

```text
3 x 256 x 256
```

7. Normalize ve `[0,1]`.

### Validation va test

Chon protocol co dinh:

```text
RGB
-> center crop vuong lon nhat
-> resize 256x256 bang bilinear
-> tensor [3,256,256]
-> normalize [0,1]
```

Khong dung random crop trong validation/test.

## 5. Paper-reference mode 96x96

`96x96` la evaluation-only mode de phuc vu paper-reference va Ghost-ESP-like fairness.

Quy trinh validation/test:

```text
RGB
-> center crop vuong lon nhat
-> resize 96x96 bang bilinear
-> tensor [3,96,96]
-> normalize [0,1]
```

Quy tac:

- Khong coi `96x96` la deployment resolution chinh.
- Khong bat buoc train model chinh o `96x96`.
- Neu so sanh voi Ghost-ESP paper, can ghi ro la `paper-reference mode`.
- Neu train/eval Ghost-ESP-like, nen dung cung split va preprocessing `96x96`.

## 6. Augmentation hinh hoc

Cac phep bien doi sau phai ap dung dong bo cho low-light va ground truth:

- Horizontal flip: xac suat `0.5`.
- Vertical flip: xac suat `0.2`.
- Rotation ngau nhien trong `0`, `90`, `180`, `270` do.
- Random crop nhu da dinh nghia.

Khong ap dung phep bien doi hinh hoc khac nhau giua input va ground truth.

## 7. Augmentation anh sang

Chi ap dung len anh low-light trong training.

Exposure scaling:

```text
I' = alpha * I
alpha in [0.4, 0.9]
```

Gamma darkening:

```text
I' = I^gamma
gamma in [1.2, 2.4]
```

White-balance perturbation:

```text
s_R, s_G, s_B in [0.9, 1.1]
```

Moi kenh RGB cua low-light input duoc nhan voi he so tuong ung, sau do clamp ve `[0,1]`.

Khong thay doi white balance cua ground truth.

## 8. Noise augmentation

Noise chi ap dung len low-light input.

Gaussian read noise:

```text
N_g ~ N(0, sigma^2)
sigma in [0, 0.03]
```

Poisson shot noise:

- Ap dung theo cuong do pixel.
- Muc tieu la mo phong shot noise cua sensor.

Synthetic low-light:

```text
I_syn = clip(alpha * I_GT^gamma + N_p + N_g, 0, 1)
```

Synthetic low-light phai duoc sinh tu ground-truth/normal-light image `I_GT`, khong sinh tiep tu anh low-light da toi.

Pair synthetic moi:

```text
input  = I_syn
target = I_GT
```

Ty le synthetic augmentation khong vuot qua `30%` so sample trong mot epoch.

Synthetic data chi dung nhu augmentation, khong duoc dung lam test chinh. Neu materialize synthetic image ra disk, anh goc va anh synthetic sinh tu no phai nam trong cung split.

## 9. Augmentation order

Thu tu augmentation trong training phai co dinh:

```text
paired geometric transform
-> exposure/gamma/WB tren low input
-> noise tren low input
-> clamp [0,1]
-> tensor conversion
```

Voi synthetic pair sinh tu ground truth:

```text
paired geometric transform tren I_GT
-> exposure/gamma/WB de tao low input
-> noise tren low input
-> clamp [0,1]
-> tensor conversion
```

Khong doi thu tu gamma va noise giua cac run neu dang so sanh model.

## 10. Seed va tinh tai lap

Chot:

```text
seed = 42
```

Seed phai duoc ap dung cho:

- Python random.
- NumPy.
- PyTorch.
- CUDA.
- DataLoader workers.
- Split generation.

Khi can reproducibility tuyet doi:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 11. Train/validation/test split

### Dataset cong khai

Dataset chinh:

- LOL.
- LOL-v2 Real.

Quy tac:

- Giu nguyen official test split cua LOL.
- Giu nguyen official test split cua LOL-v2 Real.
- Khong dua anh test vao train hoac validation.
- Tu official training split:
  - `90%` train.
  - `10%` validation.
- Split duoc tao mot lan bang seed `42` va luu thanh file.

### Camera dataset

Camera data chia theo scene, khong chia ngau nhien theo tung file:

- `70%` scene train.
- `15%` scene validation.
- `15%` scene test.

Tat ca anh cung canh, cung background hoac cung chuoi chup phai nam trong mot split duy nhat.

Camera test scene khong duoc dung cho train, validation, checkpoint selection, loss tuning, teacher selection hoac calibration.

## 12. Split artifacts

Khi tao source tree, luu cac file:

```text
splits/
|-- lol_train.txt
|-- lol_val.txt
|-- lol_test.txt
|-- lolv2_real_train.txt
|-- lolv2_real_val.txt
|-- lolv2_real_test.txt
|-- camera_train.txt
|-- camera_val.txt
|-- camera_test.txt
`-- calibration.txt
```

Moi dong chua duong dan den:

```text
low_light_path ground_truth_path
```

Neu anh khong co ground truth, vi du mot so frame OV5640 dung cho calibration, cot ground truth co the de `NA` va phai duoc document trong split file.

## 13. Calibration/representative subset

Calibration set khong lay tu test.

So luong:

```text
300-500 anh
```

Ty le:

- `60%` anh that tu LOL/LOL-v2 train.
- `20%` synthetic low-light.
- `20%` anh that tu OV5640.

Calibration set phai chua:

- Anh hoi toi.
- Anh rat toi.
- Anh co highlight manh.
- Anh nhieu mau.
- Anh co noise.
- Anh indoor va outdoor.

QAT input phai dung cung preprocessing voi training/evaluation.

## 14. RGB565 tu OV5640 sang RGB tensor

Moi pixel RGB565 gom:

```text
RRRRRGGG GGGBBBBB
```

Giai ma:

```text
r5 = (pixel >> 11) & 0x1F
g6 = (pixel >> 5) & 0x3F
b5 = pixel & 0x1F

r8 = round(r5 * 255 / 31)
g8 = round(g6 * 255 / 63)
b8 = round(b5 * 255 / 31)
```

Pipeline:

```text
OV5640 RGB565
-> DCMI + DMA
-> framebuffer trong SDRAM
-> crop/resize
-> RGB565 to RGB888
-> HWC to CHW
-> normalize hoac quantize
-> model input
```

Resize deployment phai dung cung nguyen tac voi training.

Neu Cube.AI model nhan INT8:

```text
q = round(x / s + z)
```

Trong do:

- `x`: gia tri float trong `[0,1]`.
- `s`: input scale.
- `z`: zero-point.

Khong hard-code scale va zero-point; lay tu model quantized.

## 15. Output preprocessing

Model output:

```text
RGB [0,1]
```

Sau inference:

1. Dequantize neu can.
2. Clamp ve `[0,1]`.
3. Chuyen ve 8-bit:

```text
I8 = round(255 * I)
```

4. Chuyen RGB888 sang RGB565.
5. Resize ve do phan giai LCD ST7735 neu can.
6. Gui qua SPI.

Evaluation dung output RGB float `[0,1]`, khong dung anh da bi quantize ve RGB565 de tinh metric nghien cuu, tru khi dang danh gia artifact camera/display rieng.

## 16. Data leakage rules

Cam:

- Dung official test de train.
- Dung official test de tune loss.
- Dung official test de chon checkpoint.
- Dung official test lam calibration.
- De cung scene camera xuat hien o ca train va test.
- Tach anh goc va anh synthetic cua no sang split khac nhau.

Bat buoc:

- Luu split artifact.
- Log seed.
- Log preprocessing config.
- Log augmentation config.

## 17. Completion criteria

Spec nay duoc xem la chot khi:

- Student-S `128x128` preprocessing ro rang.
- Student-HQ `256x256` preprocessing ro rang.
- Paper-reference `96x96` evaluation mode ro rang.
- Validation/test deu dung center-square-crop + resize.
- Training `128x128` dung exact paired crop, khong resize sau crop.
- RGB565 -> RGB tensor ro rang.
- Normalize `[0,1]` ro rang.
- Split seed `42` ro rang.
- Augmentation ranges ro rang.
- Augmentation order ro rang.
- Synthetic low-light sinh tu ground truth/normal-light ro rang.
- Calibration set ro rang.
- Data leakage rules ro rang.
