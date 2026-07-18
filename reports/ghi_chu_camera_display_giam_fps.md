# Ghi chú: xử lý giảm FPS khi triển khai camera + display

## 1. Vì sao FPS sẽ giảm?

Benchmark hiện tại mới đo **model-only**:

```text
Inference trung bình: ~189.9 ms/frame
FPS model-only: ~5.26 FPS
```

Khi thêm camera và màn hình, tổng thời gian sẽ thành:

```text
total_time = camera_capture + preprocess/resize + inference + display
```

Vì vậy FPS thực tế sẽ thấp hơn 5.26 FPS nếu phần camera, resize hoặc display tốn thêm thời gian.

## 2. Những việc cần làm trước, không làm giảm chất lượng

Ưu tiên các bước này trước vì không ảnh hưởng PSNR/SSIM của model:

1. Build firmware ở chế độ **Release** hoặc bật tối ưu `-O2`/`-Ofast`.
2. Dùng **DMA** cho camera capture nếu phần cứng hỗ trợ.
3. Dùng **DMA/SPI tốc độ cao/FSMC/LTDC** cho display tùy loại màn hình.
4. Tránh copy buffer nhiều lần giữa camera, resize, model input và display.
5. Resize trực tiếp từ frame camera về `96x96 RGB`.
6. Tắt log UART trong vòng lặp chính, chỉ log thống kê sau nhiều frame.
7. Đo riêng từng phần:

```text
camera_ms
preprocess_ms
inference_ms
display_ms
total_ms
fps_total
```

## 3. Nếu vẫn chậm thì tối ưu có đánh đổi

Các hướng này có thể tăng FPS nhưng có khả năng giảm chất lượng ảnh:

| Cách làm | Tác động FPS | Tác động chất lượng |
|---|---:|---|
| Giảm input từ `96x96` xuống `80x80` hoặc `64x64` | Tăng rõ | Giảm độ nét/chi tiết |
| Giảm số block trong model | Tăng | Có thể giảm PSNR/SSIM |
| Giảm channel, ví dụ base/mid channel nhỏ hơn | Tăng | Có thể giảm khả năng khôi phục ảnh |
| Chạy model cách frame, ví dụ 1 lần mỗi 2 frame | Tăng FPS hiển thị | Phản hồi chậm hơn |
| Dùng output frame trước cho frame hiện tại | Giảm tải | Có thể bị trễ hình |

## 4. Kết luận triển khai

Khi thêm camera + display, không nên tối ưu model ngay. Quy trình nên là:

```text
1. Đo model-only
2. Đo camera-only
3. Đo display-only
4. Đo preprocess/resize
5. Đo pipeline đầy đủ
6. Tối ưu phần chậm nhất trước
```

Chỉ nên giảm kích thước model hoặc input khi đã tối ưu build, DMA, buffer và display mà FPS vẫn chưa đạt mục tiêu.
