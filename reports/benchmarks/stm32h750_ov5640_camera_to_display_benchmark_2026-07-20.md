# Bao cao benchmark STM32H750: model-only va camera-to-display

Ngay thuc hien: 2026-07-20

## 1. Muc tieu

Bao cao nay ghi lai ket qua benchmark tren board STM32H750VBT6 cho hai phan:

- Model-only LLIE da benchmark truoc do.
- Pipeline camera-to-display: OV5640 -> DCMI DMA -> D-cache invalidate -> ST7735 LCD.

Muc dich la tach rieng chi phi inference cua model va chi phi duong camera/display de uoc luong bottleneck khi ghep pipeline day du.

## 2. Cau hinh phan cung va firmware

| Thanh phan | Gia tri |
|---|---|
| MCU | STM32H750VBT6 |
| Camera | OV5640 |
| LCD | ST7735 0.96 inch |
| Project SDK | `08-DCMI2LCD` |
| Toolchain | Keil MDK Arm Compiler 6.24 |
| Gioi han build | Keil non-commercial / code size limit 32 KB |
| Camera input | QQVGA 160x120 RGB565 |
| Display path | ST7735 SPI LCD |

Do gioi han code size cua Keil Lite, firmware camera-to-display da duoc rut gon de giu cac phan can thiet:

- Giu OV5640 + ST7735.
- Bo logo splash va text overlay.
- Bo doc ID LCD khong can thiet.
- Giu `SCB_InvalidateDCache_by_Addr()` truoc khi CPU doc frame buffer DMA.
- Khai bao `DCMI_FrameIsReady` la `volatile` vi flag nay duoc set trong interrupt va doc trong main loop.

## 3. Ket qua model-only

Ket qua model-only da do tren STM32H750VBT6:

| Chi so | Gia tri |
|---|---:|
| Avg cycles | 91,143,787 cycles/frame |
| Avg time | 189.882 ms/frame |
| FPS | 5.26 FPS |

Ket qua nay chi bao gom inference model, chua bao gom camera capture, preprocess/resize, va display.

## 4. Ket qua camera-to-display

Pipeline da chay duoc hinh anh truc tiep tu OV5640 len LCD ST7735.

Bien benchmark trong Keil Watch:

```c
E2E_FPS
E2E_AvgMs
E2E_LastMs
```

Gia tri quan sat:

| Bien | Gia tri hex | Gia tri thap phan | Y nghia |
|---|---:|---:|---|
| `E2E_FPS` | `0x0000001A` | 26 | So frame ve len LCD moi giay |
| `E2E_AvgMs` | `0x00000011` | 17 ms | Thoi gian trung binh moi frame |
| `E2E_LastMs` | `0x00000012` | 18 ms | Thoi gian frame gan nhat |

Tom tat:

| Pipeline | Time/frame | FPS |
|---|---:|---:|
| OV5640 -> DCMI DMA -> D-cache invalidate -> ST7735 LCD | 17-18 ms/frame | ~26 FPS |

Pham vi do cua chi so nay:

```text
DCMI frame ready -> invalidate D-cache -> LCD draw done
```

Chi so nay chua bao gom toan bo thoi gian exposure/capture ben trong camera truoc khi DCMI bao frame ready.

## 5. Uoc luong pipeline day du

## 5. Ket qua preprocess/resize

Da them benchmark preprocess truc tiep tren firmware camera-to-display. Ham preprocess hien tai chuyen frame camera:

```text
160x120 RGB565 -> 96x96 RGB888 uint8
```

Phuong phap resize hien tai la nearest-neighbor, phu hop cho baseline benchmark vi code gon va toc do nhanh.

Bien benchmark trong Keil Watch:

```c
preprocess_ms
camera_display_ms
inference_ms
total_ms
total_fps
```

Gia tri quan sat:

| Bien | Gia tri hex | Gia tri thap phan | Y nghia |
|---|---:|---:|---|
| `preprocess_ms` | `0x00000001` | 1 ms | Resize/convert frame camera sang input model |
| `camera_display_ms` | `0x00000011` | 17 ms | Hien thi frame len LCD |
| `inference_ms` | `0x00000000` | 0 ms | Chua ghep model inference trong test nay |
| `total_ms` | `0x00000012` | 18 ms | Preprocess + display |
| `total_fps` | `0x0000001A` | 26 FPS | FPS cua pipeline hien tai |

Tom tat:

| Stage | Time/frame |
|---|---:|
| Preprocess/resize | ~1 ms |
| Camera/display path | ~17 ms |
| Preprocess + display | ~18 ms |

Ket qua nay cho thay preprocess dang rat nhe so voi inference model-only.

## 6. Uoc luong pipeline day du

Neu ghep model inference vao pipeline camera/display, uoc luong don gian:

```text
estimated_total_ms = preprocess_ms + model_only_ms + camera_to_display_ms
                   = 1 + 189.882 + 17
                   = 207.882 ms/frame

estimated_total_fps = 1000 / 207.882
                    ~= 4.81 FPS
```

Bang so sanh:

| Stage | Time/frame | FPS |
|---|---:|---:|
| Model-only LLIE | 189.882 ms | 5.26 FPS |
| Preprocess/resize | ~1 ms | - |
| Camera-to-display path | 17-18 ms | ~26 FPS |
| Estimated full path | ~207.9 ms | ~4.81 FPS |

## 7. Ket luan

Pipeline camera-to-display da hoat dong on dinh sau khi them D-cache invalidate cho frame buffer DMA va khai bao flag interrupt la `volatile`.

Ket qua cho thay phan camera/display va preprocess khong phai bottleneck chinh trong cau hinh hien tai. Preprocess chi mat khoang 1 ms/frame, display path mat khoang 17-18 ms/frame, trong khi model-only mat khoang 189.882 ms/frame. Khi ghep pipeline day du, bottleneck du kien la inference model LLIE.

## 8. Buoc tiep theo

1. Ghep model inference vao pipeline camera -> preprocess -> model -> display tren toolchain khong bi gioi han 32 KB, vi Keil Lite gan nhu khong du cho X-CUBE-AI + camera + LCD.
2. Do rieng cac bien:

```text
camera_display_ms
preprocess_ms
inference_ms
total_ms
total_fps
```

3. Neu total FPS thap, uu tien toi uu inference model truoc vi day la bottleneck lon nhat hien tai.
