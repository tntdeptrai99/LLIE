# Báo cáo ngắn: triển khai LLIE trên STM32H750VBT6

## 1. Model sử dụng

- Model: **DG-GhostESP-96 / GhostESP-Dark**
- Biến thể triển khai: `ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq.onnx`
- Kiểu triển khai: **INT8/QDQ ONNX**, sinh code bằng **X-CUBE-AI**
- Kích thước input/output: **RGB 96x96**, `uint8`
- Cấu hình chính: base channels `12`, mid channels `24`, `3` blocks, `gain_max=3.0`, `residual_scale=0.35`

## 2. Kết quả train hiện tại

Kết quả tốt nhất hiện tại trên tập test LOL:

| Model | PSNR | SSIM |
|---|---:|---:|
| DG-GhostESP-96 plateau best_monitor | **19.6221** | **0.843353** |

So với candidate trial011 ban đầu, model plateau tăng khoảng **+0.3550 dB PSNR** và **+0.009930 SSIM**.

## 3. Những gì đã làm được trên board

- Đã tạo firmware benchmark riêng: `LLIE_Benchmark`.
- Đã cấu hình STM32H750VBT6 chạy ở **480 MHz**.
- Đã bật **CRC**, **USART1**, **I-Cache/D-Cache** và tích hợp X-CUBE-AI.
- Đã flash firmware thành công lên board STM32H750VBT6.
- UART log đã hoạt động đúng ở `115200 baud`.
- X-CUBE-AI init thành công trên board:
  - Input: `27648 bytes`
  - Output: `27648 bytes`
  - Activation: `425260 bytes`
- Đã benchmark model-only trực tiếp trên board:
  - `avg_cycles = 91,143,787`
  - `avg_time = 189.882 ms/frame`
  - `fps = 5.26`

## 4. Kết luận

Model **GhostESP-Dark INT8/QDQ đã chạy được thật trên STM32H750VBT6** và fit bộ nhớ RAM. Tuy nhiên tốc độ hiện tại khoảng **5.26 FPS** ở chế độ model-only, nên **chưa đạt real-time 15-30 FPS**. Bước tiếp theo nên là build Release/`-O2` để benchmark lại, sau đó tối ưu model nếu FPS vẫn thấp.
