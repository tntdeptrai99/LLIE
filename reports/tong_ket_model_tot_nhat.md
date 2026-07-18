# Tong ket model tot nhat hien tai

Ngay cap nhat: 2026-07-16

## Ket luan nhanh

Model tot nhat hien tai la:

```text
DG-GhostESP-96
```

Ten day du de tai tao thi nghiem:

```text
Dark-Guided GhostESP 96x96, W12/M24/B3, gain_max=3.0, residual_scale=0.35, plateau best_monitor
```

Ket qua moi sau khi train tiep den gan plateau tu long80 candidate:

| Muc tieu | Ket qua hien tai | Dat? |
|---|---:|---:|
| PSNR > 17 dB | 19.6221 dB | Yes |
| SSIM > 0.7 | 0.843353 | Yes |

Checkpoint:

```text
experiments/refine/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_from_long80_best_ssim/best_monitor.pt
```

ONNX QDQ INT8:

```text
stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq.onnx
```

STM32Cube.AI Analyze:

| Thanh phan | Gia tri |
|---|---:|
| Total Flash | 55,464 B / 54.16 KiB |
| Weights | 5,944 B / 5.80 KiB |
| Library Flash | 49,520 B / 48.36 KiB |
| Total RAM | 441,820 B / 431.46 KiB |
| Activations | 425,260 B / 415.29 KiB |
| Library RAM | 16,560 B / 16.17 KiB |
| Input | 27,648 B / 27.00 KiB, included in activations |
| Output | 27,648 B / 27.00 KiB, included in activations |

Chi tiet nam trong:

```text
reports/ghost_esp_dark_plateau_score_refine.md
reports/ghost_esp_dark_trial011_long80_refine.md
```

Model tot nhat truoc plateau refinement la:

```text
DG-GhostESP-96 trial011
```

Model nay da vuot muc tieu ban dau:

| Muc tieu | Ket qua hien tai | Dat? |
|---|---:|---:|
| PSNR > 17 dB | 19.2671 dB | Yes |
| SSIM > 0.7 | 0.833423 | Yes |

Danh gia duoc chay tren `splits/lol_test.txt`, `image_size=96`, `count=15`.

## Model su dung

Checkpoint tot nhat:

```text
experiments/optuna/ghost_esp_dark_w12_m24_gain3_res035_adaptive_loss/trial_011/best.pt
```

ONNX fused:

```text
stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_fused.onnx
```

ONNX QDQ INT8:

```text
stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_qdq.onnx
```

Metric summary:

```text
reports/metrics/lol_test_ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_summary.txt
```

## Kien truc

Ten kien truc:

```text
DG-GhostESP-96
```

Cau hinh chinh:

| Thanh phan | Gia tri |
|---|---|
| Input | RGB 96x96 |
| Base channels | 12 |
| Mid channels | 24 |
| Bottleneck blocks | 3 Ghost/ESP blocks |
| Activation | ReLU6 |
| Conv chinh | Depthwise separable conv + Ghost/ESP block |
| Dark module | DarkMapGenerator |
| Dark guidance | Dua DarkMap vao bottleneck |
| Output | `input * gain + residual` |
| Gain range | `[1.0, 3.0]` |
| Residual range | `[-0.35, 0.35]` |
| MCU export | BN folding + QDQ INT8 |

Luong xu ly:

```text
low RGB 96x96
  -> DarkMapGenerator
  -> stem ConvBNReLU6, 3 -> 12
  -> DepthwiseSeparableBNReLU6 downsample, 12 -> 24
  -> GhostESPBNBlock x3 tai bottleneck
  -> concat DarkMap guidance tai bottleneck
  -> dark fusion ConvBNReLU6
  -> nearest upsample
  -> GhostESPBNBlock refine, 12 channels
  -> gain head + residual head
  -> output = clamp(input * gain + residual)
```

## Phuong phap train

Huong train hien tai:

| Thanh phan | Vai tro |
|---|---|
| Retinexformer RGB teacher | Knowledge distillation |
| GT loss | Giu anh ra gan ground truth |
| KD loss | Ke thua tri thuc tu teacher |
| SSIM loss | Tang cau truc va do tuong dong thi giac |
| Edge/color loss | Giu bien va mau on dinh hon |
| DarkMap adaptive loss | Tang trong so hoc o vung toi |
| Optuna | Tim bo tham so loss tot hon theo metric val |

Bo tham so Optuna tot nhat hien tai, trial011:

```text
lr=0.0004642590120845409
kd_weight=0.41490731938154524
kd_dark_gain=0.319409859084582
ssim_weight=0.10125968807716974
ssim_dark_gain=0.4015400599426101
teacher_ssim_weight=0.05969817766496116
edge_kd_weight=0.023192027082135695
edge_gt_weight=0.024091660352169263
color_gt_weight=0.05533961383269607
adaptive_dark_weight=0.0053751687537937284
adaptive_dark_base=0.3990482233887346
adaptive_dark_gain=1.2795608236561056
adaptive_dark_gamma=0.6383196982363735
adaptive_bright_gamma=1.9785593078550503
adaptive_bright_target=low
```

## Ket qua benchmark

Benchmark chinh tren `splits/lol_test.txt`, `image_size=96`, `count=15`.

| Model | PSNR | SSIM | Ghi chu |
|---|---:|---:|---|
| Low input | 7.8353 | 0.144682 | Anh low-light goc |
| Retinexformer teacher | 26.7819 | 0.953639 | Teacher RGB |
| D96-Tiny-BN old best | 13.3681 | 0.698348 | Model cu |
| GhostESP-Dark 8/16 Optuna trial003 | 15.0725 | 0.724715 | Ban nhe hon |
| W12/M24 Gain3 Res035 base | 18.8806 | 0.819337 | Candidate truoc Optuna |
| W12/M24 Gain3 Res035 + Optuna trial005 | 19.0805 | 0.827318 | Tang nhe |
| W12/M24 Gain3 Res035 + Optuna trial011 | 19.2671 | 0.833423 | Tot nhat hien tai |

Muc tang cua ban tot nhat so voi base W12/M24:

```text
PSNR: +0.3865 dB
SSIM: +0.014086
```

Muc tang so voi model cu D96-Tiny-BN:

```text
PSNR: +5.8990 dB
SSIM: +0.135075
```

## STM32Cube.AI Analyze

So lieu RAM/Flash da do tren ban W12/M24 Gain3 Res035 QDQ cung kien truc:

| Thanh phan | Gia tri |
|---|---:|
| Total Flash | 55,464 B / 54.16 KiB |
| Weights | 5,944 B / 5.80 KiB |
| Library Flash | 49,520 B / 48.36 KiB |
| Total RAM | 441,820 B / 431.46 KiB |
| Activations | 425,260 B / 415.29 KiB |
| Library RAM | 16,560 B / 16.17 KiB |
| Input | 27,648 B / 27.00 KiB |
| Output | 27,648 B / 27.00 KiB |

Luu y: can chay lai STM32Cube.AI Analyze tren file QDQ moi nhat de xac nhan con so cuoi cung:

```text
stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_qdq.onnx
```

## Lenh tai tao danh gia

Eval checkpoint tot nhat:

```powershell
C:\MiniForge\python.exe scripts\eval_model.py --checkpoint experiments\optuna\ghost_esp_dark_w12_m24_gain3_res035_adaptive_loss\trial_011\best.pt --split splits\lol_test.txt --image-size 96 --model-variant GHOST-ESP-DARK --metric-name ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best --name lol_test_ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best --batch-size 4 --save-images 8 --figure-dir reports\figures\ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best --base-channels 12 --mid-channels 24 --e-blocks 3 --gain-max 3.0 --residual-scale 0.35
```

Export fused ONNX:

```powershell
C:\MiniForge\python.exe scripts\export_dummy_onnx.py --arch ghost-esp-dark --blocks 3 --base-channels 12 --mid-channels 24 --image-size 96 --checkpoint experiments\optuna\ghost_esp_dark_w12_m24_gain3_res035_adaptive_loss\trial_011\best.pt --out stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_fused.onnx
```

Export QDQ INT8:

```powershell
C:\MiniForge\python.exe scripts\export_qdq_onnx.py --input stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_fused.onnx --output stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_qdq.onnx --calib-limit 64
```

## Ket luan

`GhostESP-Dark-W12-M24-B3-Gain3-Res035-96 + Optuna trial011` la candidate chinh hien tai. Model dat PSNR/SSIM vuot muc tieu, giu kich thuoc nho, va RAM van nam trong muc kha hop ly cho huong STM32H750 neu Analyze tren file QDQ moi nhat xac nhan dung muc tai nguyen.

Buoc tiep theo nen lam:

1. Chay STM32Cube.AI Analyze cho file QDQ trial011.
2. So sanh anh output FP32 vs QDQ de kiem tra drift sau quantization.
3. Do latency/FPS tren board STM32H750.
4. Neu con ngan sach thoi gian, tiep tuc Optuna quanh vung tham so trial011 voi epochs dai hon.
