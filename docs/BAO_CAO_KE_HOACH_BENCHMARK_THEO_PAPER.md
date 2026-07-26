# Báo cáo kế hoạch benchmark theo paper MIWAI2025

## 1. Mục tiêu

Báo cáo này liệt kê các nhóm test cần thực hiện để benchmark hệ thống tăng cường ảnh thiếu sáng, dựa trên nội dung paper `Towards a Real-time and Compact Low-light Image Enhancement Method Using Deep Learning on ESP32-S3-EYE` và tình trạng firmware hiện tại của project.

Mục tiêu benchmark gồm:

- Đánh giá chất lượng ảnh đầu ra so với ảnh đầu vào và ground truth khi có.
- Đánh giá hiệu năng triển khai trên board: thời gian inference, full pipeline FPS, bộ nhớ và độ ổn định.
- Kiểm chứng output của board khớp với output PC/ONNX.
- Đảm bảo ảnh hiển thị LCD không bị lỗi layout, sọc nhiễu hoặc sai byte order.

## 2. Cơ sở từ paper

Paper đánh giá mô hình theo các hướng chính:

- Dataset: LOL và LOL-v2.
- Độ phân giải: 96 x 96 và 256 x 256.
- Chỉ số chất lượng ảnh: PSNR, SSIM, MAE.
- So sánh kiến trúc: Conv2D, Separable, GhostSep, Ghost-ESP, Ghost-ESP + Distill.
- So sánh loss: MAE loss và hybrid loss.
- So sánh triển khai: kích thước model FP32/INT8, code size, inference time.
- So sánh với phương pháp truyền thống: Gamma correction, Histogram Equalization, CLAHE, Log transform, Retinex.

Với firmware hiện tại, trọng tâm thực tế là mô hình 96 x 96, chạy live camera -> preprocess -> Cube.AI inference -> LCD AI output.

## 3. Danh sách benchmark cần chạy

### 3.1. Benchmark chất lượng ảnh trên dataset

Mục tiêu: so sánh chất lượng ảnh enhanced với ground truth theo chuẩn paper.

Test case cần có:

| Mã test | Input | Output | Metric |
|---|---|---|---|
| Q01 | LOL test set, 96 x 96 | AI enhanced | PSNR, SSIM, MAE |
| Q02 | LOL-v2 real test set, 96 x 96 | AI enhanced | PSNR, SSIM, MAE |
| Q03 | LOL-v2 synthetic test set, 96 x 96 | AI enhanced | PSNR, SSIM, MAE |
| Q04 | LOL test set, 256 x 256 nếu có model tương ứng | AI enhanced | PSNR, SSIM, MAE |

Kết quả cần lưu:

- Ảnh input.
- Ảnh output.
- Ảnh ground truth.
- File CSV tổng hợp metric.
- Contact sheet so sánh trực quan.

### 3.2. Benchmark kiến trúc mô hình

Mục tiêu: tái lập hướng so sánh kiến trúc trong paper.

Các model/block cần so:

- Conv2D.
- Separable.
- GhostSep.
- Ghost-ESP.
- Ghost-ESP + Distill.

Metric cần đo:

| Metric | Ý nghĩa |
|---|---|
| Số tham số | Độ lớn mô hình |
| Model size FP32 | Kích thước model trước lượng tử hóa |
| Model size INT8 | Kích thước model sau lượng tử hóa |
| PSNR | Độ trung thực so với ground truth |
| SSIM | Độ giống cấu trúc/perceptual |
| MAE | Sai số trung bình tuyệt đối |
| Inference time | Thời gian chạy một frame |

Với board STM32 hiện tại, chỉ nên deploy model đã được xác nhận output đẹp trên PC trước.

### 3.3. Benchmark loss function

Mục tiêu: kiểm chứng nhận định trong paper rằng hybrid loss tốt hơn MAE loss.

Các case:

| Mã test | Cấu hình train | Metric |
|---|---|---|
| L01 | MAE loss baseline | Loss, MAE, PSNR, SSIM |
| L02 | Hybrid loss: Charbonnier + SSIM + Perceptual + Gram | Loss, MAE, PSNR, SSIM |
| L03 | Hybrid loss + knowledge distillation | Loss, MAE, PSNR, SSIM |

Kết quả cần lưu:

- Đường train/validation loss.
- Đường PSNR/SSIM theo epoch.
- Bảng so sánh metric cuối cùng.
- Ảnh so sánh trực quan trên cùng một tập test.

### 3.4. Benchmark lượng tử hóa và tương đương PC-board

Mục tiêu: đảm bảo model deploy lên board cho output khớp với PC.

Các case:

| Mã test | So sánh | Metric |
|---|---|---|
| T01 | ONNX/FP32 output vs INT8 output | MAE, PSNR, SSIM, max diff |
| T02 | PC ONNX output vs board Cube.AI output | exact match, MAE tensor, cosine, max diff |
| T03 | Board input preprocess vs PC render | min/max, histogram, visual PNG |
| T04 | Board AI output vs PC render | min/max, histogram, visual PNG |

Với project hiện tại, case quan trọng nhất là T02 vì nó xác nhận output board không bị sai layout hoặc sai tensor.

### 3.5. Benchmark full pipeline trên board

Mục tiêu: đo hiệu năng thực tế của hệ thống live.

Các metric cần ghi:

| Metric | Ý nghĩa |
|---|---|
| Capture/preprocess ms | Thời gian lấy frame và tạo tensor input |
| Inference ms | Thời gian Cube.AI inference |
| Postprocess ms | Thời gian chuyển tensor output sang RGB565/LCD buffer |
| LCD ms | Thời gian truyền ảnh ra LCD |
| Total ms/frame | Tổng thời gian một frame |
| FPS | Tốc độ frame thực tế |
| DCMI error count | Lỗi camera/DMA |
| Display fallback count | Số lần output bất thường |

Case cần chạy:

| Mã test | Thời lượng | Điều kiện |
|---|---:|---|
| P01 | 30 giây | Cảnh cố định, ánh sáng bình thường |
| P02 | 30 giây | Cảnh thiếu sáng |
| P03 | 30 giây | Cảnh có vùng sáng mạnh |
| P04 | 5 phút | Chạy ổn định, kiểm tra treo/nhiễu |

Baseline hiện tại đã đo được:

- Full pipeline khoảng 191 ms/frame.
- Tốc độ khoảng 5 FPS.
- Inference khoảng 170 ms/frame.
- LCD khoảng 16 ms/frame.

### 3.6. Benchmark thị giác thực tế từ camera

Mục tiêu: đánh giá chất lượng ảnh như người dùng nhìn thấy, không chỉ dựa trên metric dataset.

Các cảnh cần test:

| Mã test | Cảnh | Mục tiêu quan sát |
|---|---|---|
| V01 | Rất tối | Mô hình có kéo sáng được không |
| V02 | Tối vừa | Độ tự nhiên của ảnh |
| V03 | Có nguồn sáng mạnh | Có cháy sáng không |
| V04 | Nhiều chi tiết nhỏ | Có giữ texture không |
| V05 | Vật thể nhiều màu | Có lệch màu không |
| V06 | Da tay/người | Màu da có tự nhiên không |
| V07 | Chuyển động nhẹ | Có nhòe, lag, tearing không |

Metric PC cần tính cho từng frame dump:

- Brightness mean.
- Contrast std.
- Saturation mean.
- Sharpness/Laplacian.
- Clipping ratio vùng 0.
- Clipping ratio vùng 255.
- Histogram trước/sau.

### 3.7. Benchmark phương pháp truyền thống

Mục tiêu: so AI output với các phương pháp cổ điển như paper.

Các baseline cần chạy trên cùng input:

- Gamma correction.
- Histogram Equalization.
- CLAHE.
- Log transform.
- Retinex.

Metric:

- PSNR/SSIM/MAE nếu có ground truth.
- Brightness/contrast/saturation/sharpness nếu chỉ có camera frame thật.
- Clipping ratio.
- Histogram.
- Contact sheet so sánh trực quan.

### 3.8. Benchmark LCD correctness

Mục tiêu: tránh lặp lại lỗi sọc nhiễu LCD đã gặp.

Các case:

| Mã test | Nội dung | Kết quả mong đợi |
|---|---|---|
| D01 | LCD color bar RGB565 | Màu rõ, không sọc, không lệch byte |
| D02 | Hiển thị input preprocess | Ảnh camera sạch |
| D03 | Hiển thị AI output one-shot | Output sạch, không bị ghi đè |
| D04 | Hiển thị AI output continuous | Không nhiễu sau nhiều frame |
| D05 | Dump LCD buffer rồi render PC | Buffer khớp ảnh đang hiển thị |

Case này rất quan trọng vì lỗi LCD trước đây có thể làm hiểu sai chất lượng AI output.

### 3.9. Benchmark ổn định dài hạn

Mục tiêu: đảm bảo demo không chỉ chạy được vài frame.

Các case:

| Mã test | Thời lượng | Điều cần ghi |
|---|---:|---|
| S01 | 1 phút | FPS, error count, LCD có nhiễu không |
| S02 | 5 phút | FPS avg/min/max, DCMI error |
| S03 | 30 phút nếu cần demo | Treo, memory corruption, LCD corruption |

Kết quả cần có:

- Log UART đầy đủ.
- Tổng số frame.
- FPS trung bình.
- FPS thấp nhất/cao nhất.
- Số lỗi camera/DMA.
- Số lần output bất thường.

## 4. Bộ test tối thiểu nên chạy ngay cho bản hiện tại

Để có benchmark thực dụng và nhanh, nên chạy trước các case sau:

| Thứ tự | Case | Lý do |
|---:|---|---|
| 1 | Dump input/output và tính metric PC | Đánh giá ảnh thực tế từ board |
| 2 | Full pipeline 30 giây | Có số FPS/latency chính thức |
| 3 | Board output vs PC ONNX output | Xác nhận Cube.AI output đúng |
| 4 | 5 cảnh camera thực tế | Đánh giá bằng thị giác |
| 5 | Stability 5 phút | Kiểm tra chạy lâu không nhiễu |
| 6 | Traditional baseline trên cùng frame | Có điểm so với gamma/CLAHE/Retinex |

## 5. Artifact cần xuất sau mỗi benchmark

Mỗi lần chạy benchmark nên tạo thư mục riêng trong `reports/figures/` hoặc `reports/benchmarks/`, gồm:

- Log UART gốc.
- Ảnh input PNG.
- Ảnh AI output PNG.
- Contact sheet.
- Histogram trước/sau.
- CSV metric.
- Markdown report tóm tắt.

Cấu trúc đề xuất:

```text
reports/
  benchmarks/
    current_pipeline_YYYYMMDD/
      uart.log
      metrics.csv
      report.md
  figures/
    current_pipeline_YYYYMMDD/
      input.png
      ai_output.png
      contact_sheet.png
      histogram.png
```

## 6. Kết luận

Dựa trên paper, benchmark không nên chỉ đo FPS hoặc chỉ nhìn LCD. Cần tách thành ba lớp:

1. Chất lượng model trên PC/dataset.
2. Tương đương PC-board sau khi deploy.
3. Hiệu năng và độ ổn định full pipeline trên board thật.

Với project hiện tại, ưu tiên cao nhất là benchmark frame camera thật: dump input/output, render PNG trên PC, tính brightness/contrast/saturation/sharpness/clipping/histogram, đồng thời ghi full pipeline FPS và latency.
