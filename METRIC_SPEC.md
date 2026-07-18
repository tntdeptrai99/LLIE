# Metric Implementation Specification

## 1. Muc tieu

Dinh nghia co dinh cach tinh chat luong anh de:

- So sanh A1/A2/B/C/D.
- So sanh FP32 va INT8.
- So sanh Ghost-ESP-like.
- Tham khao ket qua paper.
- Tranh phu thuoc setting mac dinh cua thu vien.

Metric chinh:

- RGB PSNR.
- RGB SSIM.
- LPIPS-AlexNet.
- CIEDE2000.

Deployment metrics duoc dinh nghia trong `BENCHMARK_SPEC.md`.

## 2. Du lieu truoc khi tinh metric

Prediction va ground truth phai:

1. Co cung kich thuoc.
2. Co ba kenh RGB.
3. Co gia tri float trong `[0,1]`.
4. Prediction duoc clamp:

```text
I_hat = clip(I_hat, 0, 1)
```

5. Khong dung anh da chuyen sang RGB565 de tinh metric nghien cuu.
6. Khong resize prediction va ground truth bang hai phuong phap khac nhau.

Neu phai resize de so sanh, phuong phap resize phai duoc log trong report va dung dong nhat cho tat ca model.

## 3. RGB PSNR

PSNR duoc tinh tren toan bo ba kenh RGB.

```text
MSE = (1 / (3 * H * W)) * sum_c,h,w((I_hat - I)^2)
PSNR = 10 * log10(1 / MSE)
```

Cau hinh:

```text
color space: RGB
data range: 1.0
higher is better
```

Neu `MSE = 0`, PSNR duoc ghi la infinity nhung phai xu ly rieng khi aggregate.

PSNR-Y chi duoc dung nhu metric phu neu can, khong thay the RGB PSNR.

## 4. RGB SSIM

SSIM duoc tinh tren RGB.

Cau hinh:

```text
window size: 11x11
window type: Gaussian
sigma: 1.5
data range: 1.0
channels: RGB
higher is better
```

Aggregate theo channel:

```text
SSIM_RGB = (SSIM_R + SSIM_G + SSIM_B) / 3
```

Khong chuyen sang luminance de thay the RGB SSIM.

Xu ly bien:

```text
reflect padding 5 pixels before SSIM filtering
mean over original H x W domain
```

Tinh SSIM theo tung anh, khong gop ca dataset thanh mot tensor duy nhat.

Phai su dung cung implementation cho:

- Training validation.
- Test.
- FP32.
- INT8.
- Ghost baseline.

## 5. LPIPS

Chot:

```text
LPIPS backbone: AlexNet
input range: [-1, 1]
color space: RGB
lower is better
```

Chuyen input:

```text
x_LPIPS = 2 * x - 1
```

LPIPS chi dung de danh gia, khong bat buoc dung lam loss.

Phien ban package va checkpoint phai duoc ghi lai trong requirements/environment file khi tao moi truong code.

## 6. Library/version lock

Spec khong khoa version cu the truoc khi tao environment, nhung moi run chinh thuc phai log version vao:

```text
requirements-lock.txt
environment.yaml
reports/metrics/<run>_environment.json
```

Implementation yeu cau:

| Metric | Implementation |
|---|---|
| RGB PSNR | custom implementation hoac torchmetrics, phai log version |
| RGB SSIM | custom per-channel implementation hoac pytorch-msssim, phai log version |
| LPIPS-AlexNet | `lpips`, `net='alex'`, phai log package/checkpoint |
| CIEDE2000 | scikit-image hoac colour-science, phai log version |

Neu doi library/version giua cac experiment, ket qua khong duoc gop vao cung mot bang ket luan chinh neu chua tinh lai toan bo model.

## 7. CIEDE2000

Chot:

```text
Delta E00
```

Pipeline chuyen doi:

```text
sRGB [0,1]
-> linear RGB
-> CIE XYZ
-> CIELAB
-> CIEDE2000
```

White point:

```text
D65
```

Bao cao theo pixel:

- Mean Delta E00.
- Median Delta E00.
- P95 Delta E00 tuy chon.

Lower is better.

Khong tinh Delta E truc tiep tren RGB.

## 8. Aggregate theo dataset

Metric duoc tinh theo tung anh truoc.

Moi anh tao mot record:

```text
image_id
psnr_rgb
ssim_rgb
lpips_alex
delta_e00_mean
delta_e00_median
```

Sau do aggregate toan test set:

- Mean.
- Median.
- Standard deviation.
- P95 cho LPIPS/Delta E neu can.
- Bootstrap 95% confidence interval cho PSNR, SSIM va LPIPS trong final report.

Khong lay mean theo tung batch roi trung binh cac batch co kich thuoc khac nhau.

Bootstrap CI khong bat buoc cho quick validation, nhung bat buoc cho ket luan cuoi neu tuyen bo vuot baseline/paper.

## 9. Report files

Luu hai loai file khi tao source tree:

Per-image report:

```text
reports/metrics/<model>_<dataset>_per_image.csv
```

Aggregate report:

```text
reports/metrics/<model>_<dataset>_summary.json
```

Vi du:

```json
{
  "psnr_rgb_mean": 20.21,
  "psnr_rgb_median": 20.08,
  "ssim_rgb_mean": 0.882,
  "lpips_alex_mean": 0.142,
  "delta_e00_mean": 4.81,
  "psnr_rgb_ci95": [19.85, 20.54],
  "ssim_rgb_ci95": [0.874, 0.891],
  "lpips_alex_ci95": [0.132, 0.151]
}
```

## 10. Tieu chi chat luong

Model D INT8 phai:

- Vuot A2 FP32 ve PSNR va SSIM.
- Co LPIPS thap hon A2 FP32.
- Co Delta E00 thap hon A2 FP32.
- Gan Model C FP32.

Muc suy giam toi da tu D FP32 sang D INT8:

```text
Delta PSNR <= 0.3 dB
Delta SSIM <= 0.01
LPIPS_INT8 <= 1.05 * LPIPS_FP32
DeltaE00_INT8 <= 1.05 * DeltaE00_FP32
```

De tuyen bo vuot paper, phai so sanh tren protocol tuong duong hoac ghi ro do la paper-reference comparison.

## 11. Visual evaluation

Voi cac anh dai dien, luu comparison grid:

```text
Low-light input
Ghost-ESP-like output
A2 output
C FP32 output
D INT8 output
Ground truth
```

Can co crop cho:

- Dark region.
- Highlight region.
- High texture/edge region.
- Colorful region.

Visual evaluation chi ho tro ket luan, khong thay the metric dinh luong.

## 12. Report template

| Model | Resolution | Dataset | PSNR RGB | SSIM RGB | LPIPS-AlexNet | CIEDE2000 |
|---|---:|---|---:|---:|---:|---:|
| A1 | 128x128 | LOL | | | | |
| A2 | 128x128 | LOL | | | | |
| B | 128x128 | LOL | | | | |
| C FP32 | 128x128 | LOL | | | | |
| D INT8 | 128x128 | LOL | | | | |
| D INT8 | 256x256 | LOL | | | | |
| Ghost-ESP-like | 128x128 | LOL | | | | |
| Ghost-ESP paper | 96x96 | Paper reference | | | | |

## 13. Completion criteria

Spec nay duoc xem la chot khi:

- Metric value ranges duoc dinh nghia.
- RGB SSIM channel aggregation duoc dinh nghia.
- SSIM border handling duoc dinh nghia.
- Library/version logging duoc yeu cau.
- LPIPS backbone co dinh la AlexNet.
- Delta E method co dinh la CIEDE2000.
- Aggregation rules ro rang.
- Bootstrap CI cho final report ro rang.
- Per-image va summary report ro rang.
- Tieu chi FP32 -> INT8 degradation ro rang.
- DeltaE00 threshold ro rang.
- Dieu kien paper-reference comparison ro rang.
