# Đề xuất tăng cường sáng ảnh vượt trội trên STM32H750

## 1. Mục tiêu

Xây dựng hệ thống **Low-Light Image Enhancement** chạy trên **STM32H750**, có chất lượng ảnh đầu ra vượt mô hình Ghost-ESP trong bài báo triển khai trên ESP32-S3.

Các yêu cầu chính:

- Tăng sáng rõ rệt ở vùng tối.
- Giữ chi tiết, texture và đường biên.
- Hạn chế khuếch đại nhiễu.
- Không làm cháy sáng vùng đã đủ sáng.
- Giữ màu sắc tự nhiên.
- Có thể lượng tử hóa INT8 và triển khai bằng STM32Cube.AI.
- Hướng đến xử lý ảnh hoặc video gần thời gian thực.

---

## 2. Hướng tiếp cận tổng thể

Kiến trúc đề xuất:

```text
Retinex-guided U-Net Lite
        +
Dark-region Attention
        +
Denoising Branch
        +
Gain Map + Residual Prediction
        +
Color/Edge-aware Loss
        +
Quantization-Aware Training
```

Pipeline:

```text
Ảnh tối
   ↓
Tiền xử lý
   ↓
Lightweight Encoder
   ↓
Illumination Estimation
   ↓
Dark-region Attention
   ↓
Denoising Bottleneck
   ↓
Lightweight Decoder
   ↓
Dự đoán Gain Map + Residual Map
   ↓
Hiệu chỉnh màu
   ↓
Ảnh tăng sáng
```

---

## 3. U-Net Lite thay cho Ghost-ESP

Ghost-ESP ưu tiên mô hình cực nhỏ để chạy trên ESP32-S3. Với STM32H750, nên dùng **U-Net Lite dạng encoder-decoder** để tăng khả năng phục hồi chi tiết.

```text
Input 160×160×3
        ↓
Conv 3×3
        ↓
Depthwise Separable Block
        ↓
Downsample
        ↓
Residual Blocks
        ↓
Upsample
        ↓
Skip Connection
        ↓
Conv 3×3
        ↓
Enhanced RGB
```

Lợi ích:

- Skip connection giữ cạnh và texture.
- Encoder học đặc trưng ánh sáng và nhiễu.
- Decoder phục hồi cấu trúc không gian.
- Depthwise separable convolution giảm FLOPs.
- Phù hợp với bài toán image-to-image hơn mạng residual quá nhỏ.

---

## 4. Retinex-guided enhancement

Theo Retinex:

\[
I = R \odot L
\]

Trong đó:

- \(I\): ảnh quan sát.
- \(R\): reflectance, chứa màu sắc và chi tiết vật thể.
- \(L\): illumination, biểu diễn phân bố ánh sáng.

Thay vì học trực tiếp:

\[
I_{\text{low}} \rightarrow I_{\text{bright}}
\]

mô hình học:

```text
Ảnh tối
   ↓
Ước lượng illumination
   ↓
Điều chỉnh illumination
   ↓
Phục hồi reflectance
   ↓
Fusion
   ↓
Ảnh sáng
```

Lợi ích:

- Vùng rất tối được tăng mạnh hơn.
- Vùng đủ sáng ít bị thay đổi.
- Hạn chế cháy sáng.
- Giữ màu tự nhiên hơn.
- Có cơ sở lý thuyết rõ ràng cho báo cáo.

---

## 5. Dark-region Attention

Không phải mọi vùng đều cần tăng sáng như nhau.

Có thể tạo dark map:

\[
M_{\text{dark}} = 1 - \operatorname{mean}(I_{\text{low}})
\]

Hoặc để một nhánh CNN học attention map. Áp dụng lên feature map:

\[
F' = F \odot (1 + \alpha M_{\text{dark}})
\]

Khuyến nghị:

- Đặt attention tại bottleneck hoặc decoder.
- Ưu tiên spatial attention nhẹ.
- Không cần Transformer lớn.
- Không đặt CBAM ở mọi block nếu không cần thiết.

---

## 6. Kết hợp tăng sáng và khử nhiễu

Tăng sáng thường làm nhiễu vùng tối mạnh hơn. Nên dùng kiến trúc chung encoder và hai nhánh:

```text
Shared Encoder
   ├── Illumination Branch
   └── Denoising Branch
             ↓
           Fusion
```

Khuyến nghị:

- Dùng chung encoder để giảm tham số.
- Số kênh khởi đầu: 8 → 16 → 32.
- Nhánh denoising nên nhỏ gọn.
- Có thể dự đoán noise map rồi loại khỏi ảnh.

---

## 7. Dự đoán Gain Map và Residual Map

Không nên bắt model tái tạo hoàn toàn ảnh đầu ra. Cho model dự đoán phần hiệu chỉnh.

Residual enhancement:

\[
I_{\text{out}} = \operatorname{clip}(I_{\text{in}} + \Delta I, 0, 1)
\]

Gain map kết hợp residual:

\[
I_{\text{out}} = \operatorname{clip}(I_{\text{in}} \odot G + \Delta I, 0, 1)
\]

Trong đó:

- \(G\): gain map điều chỉnh ánh sáng.
- \(\Delta I\): residual map sửa màu, chi tiết và tương phản.

Lợi ích:

- Dễ học hơn.
- Giữ nội dung gốc.
- Giảm lệch màu.
- Giảm số tham số.
- Phù hợp với MCU.

---

## 8. Độ phân giải đầu vào

| Phiên bản | Độ phân giải |
|---|---:|
| Baseline | 128×128 |
| Cân bằng | 160×160 |
| Chất lượng cao | 192×192 |
| Có SDRAM ngoài | 256×256 |

Mức đề xuất ban đầu:

\[
\boxed{160 \times 160}
\]

Mức này giữ chi tiết tốt hơn 96×96 nhưng chưa quá nặng như 256×256.

---

## 9. Hàm loss đề xuất

\[
L_{\text{total}}=
\lambda_pL_{\text{Charbonnier}}+
\lambda_sL_{\text{SSIM}}+
\lambda_{per}L_{\text{Perceptual}}+
\lambda_cL_{\text{Color}}+
\lambda_eL_{\text{Edge}}+
\lambda_iL_{\text{Illumination}}
\]

Vai trò:

- **Charbonnier:** giữ độ chính xác pixel và ổn định huấn luyện.
- **SSIM:** giữ cấu trúc và độ tương phản.
- **Perceptual:** cải thiện chất lượng nhìn bằng mắt.
- **Color:** hạn chế lệch màu và sai cân bằng trắng.
- **Edge:** giữ cạnh, chữ và texture.
- **Illumination:** giúp ánh sáng phân bố hợp lý.

Edge loss có thể viết:

\[
L_{\text{Edge}} = \|\nabla \hat I - \nabla I\|_1
\]

---

## 10. Illumination Smoothness Loss

Bản đồ illumination nên thay đổi mượt giữa các pixel lân cận.

\[
L_{\text{smooth}} = \|\nabla L\|_1
\]

Hoặc edge-aware smoothness:

\[
L_{\text{smooth}} = |\nabla L|\exp(-\beta|\nabla I|)
\]

Lợi ích:

- Giảm vùng sáng loang lổ.
- Hạn chế halo.
- Giữ cạnh vật thể.
- Làm illumination tự nhiên hơn.

---

## 11. Trọng số loss khởi đầu

\[
L = L_{\text{Charbonnier}}
+0.2L_{\text{SSIM}}
+0.01L_{\text{Perceptual}}
+0.05L_{\text{Color}}
+0.05L_{\text{Edge}}
+0.1L_{\text{Illumination}}
\]

Đây chỉ là cấu hình khởi đầu. Cần ablation study để xác định trọng số phù hợp.

### Giai đoạn 1: Baseline

\[
L_1=L_{\text{Charbonnier}}+0.2L_{\text{SSIM}}
\]

### Giai đoạn 2: Chất lượng cảm nhận

\[
L_2=L_{\text{Charbonnier}}+0.2L_{\text{SSIM}}+0.01L_{\text{Perceptual}}
\]

### Giai đoạn 3: Màu, cạnh và ánh sáng

\[
L_3=L_{\text{Charbonnier}}+0.2L_{\text{SSIM}}+0.01L_{\text{Perceptual}}+0.05L_{\text{Color}}+0.05L_{\text{Edge}}+0.1L_{\text{Illumination}}
\]

---

## 12. Knowledge Distillation

Teacher model có thể là:

- U-Net lớn.
- RetinexNet.
- Một model low-light enhancement chất lượng cao chạy trên GPU.

Student model:

- Retinex-guided U-Net Lite.
- Số kênh nhỏ.
- Operator thân thiện với INT8.
- Triển khai bằng STM32Cube.AI.

Distillation loss:

\[
L_{\text{KD}} = L_{\text{output}} + \gamma L_{\text{feature}}
\]

Feature distillation giúp student học đặc trưng trung gian mà không làm tăng chi phí inference.

---

## 13. Quantization-Aware Training

Pipeline:

```text
Train FP32
   ↓
Đánh giá FP32
   ↓
Thử PTQ INT8
   ↓
Đánh giá suy giảm chất lượng
   ↓
QAT Fine-tuning
   ↓
Export INT8
   ↓
STM32Cube.AI
```

QAT giúp:

- Giảm sai số lượng tử.
- Hạn chế banding.
- Hạn chế lệch màu sau INT8.
- Giữ PSNR và SSIM gần mô hình FP32.

---

## 14. Operator thân thiện với STM32

Nên ưu tiên:

- Conv2D.
- Depthwise Conv2D.
- Pointwise Conv2D.
- Add và Multiply.
- ReLU hoặc ReLU6.
- Resize đơn giản.
- Global average pooling.

Nên hạn chế:

- GELU.
- LayerNorm.
- Transformer lớn.
- Operator động.
- Attention nhiều nhánh.
- Operator không được STM32Cube.AI hỗ trợ tốt.

---

## 15. Dữ liệu huấn luyện

Nên sử dụng:

- LOL.
- LOL-v2 Real.
- LOL-v2 Synthetic.
- Ảnh low-light tự sinh.
- Ảnh chụp từ camera thật của dự án.

Nên chụp cùng một cảnh ở nhiều mức exposure:

```text
Cùng một cảnh
   ├── Exposure thấp
   ├── Exposure trung bình
   └── Exposure chuẩn
```

Điều này giúp model học đúng sensor noise, white balance, màu camera, lens và pipeline ISP thực tế.

---

## 16. Kiến trúc cuối cùng đề xuất

```text
Input RGB 160×160
        ↓
Shallow Conv 3×3
        ↓
Lightweight Encoder
(DWConv + Residual)
        ↓
Illumination Estimation Branch
        ↓
Dark-region Spatial Attention
        ↓
Denoising Bottleneck
        ↓
Lightweight Decoder
        ↓
Skip Connections
        ↓
Predict Gain Map + Residual Map
        ↓
Color Correction Layer
        ↓
Enhanced RGB
```

Công thức đầu ra:

\[
I_{\text{out}} = \operatorname{clip}(I_{\text{in}}\odot G+\Delta I,0,1)
\]

---

## 17. Các phiên bản thực nghiệm

### Model A — Baseline

```text
U-Net Lite
+ Charbonnier Loss
+ SSIM Loss
```

### Model B — Cải thiện ánh sáng

```text
U-Net Lite
+ Illumination Branch
+ Dark-region Attention
```

### Model C — Cải thiện chất lượng

```text
Retinex-guided U-Net Lite
+ Denoising Branch
+ Color Loss
+ Edge Loss
+ Illumination Loss
```

### Model D — Bản triển khai

```text
Distilled Student Model
+ Quantization-Aware Training
+ INT8
+ STM32Cube.AI
```

---

## 18. Ablation Study

| Thí nghiệm | Kiến trúc / Loss |
|---|---|
| A1 | U-Net Lite + Pixel Loss |
| A2 | A1 + SSIM |
| A3 | A2 + Illumination Branch |
| A4 | A3 + Dark-region Attention |
| A5 | A4 + Denoising Branch |
| A6 | A5 + Perceptual Loss |
| A7 | A6 + Color Loss |
| A8 | A7 + Edge Loss |
| A9 | A8 + Knowledge Distillation |
| A10 | A9 + QAT INT8 |

Ablation study giúp chứng minh thành phần nào thực sự cải thiện chất lượng.

---

## 19. Chỉ số đánh giá

### Chất lượng ảnh

- PSNR.
- SSIM.
- LPIPS.
- MAE.
- Delta E hoặc sai lệch màu.
- Edge preservation.
- NIQE hoặc BRISQUE khi không có ground truth.

### Khả năng triển khai

- Số tham số.
- MACs/FLOPs.
- Kích thước FP32 và INT8.
- Peak activation RAM.
- Flash usage.
- Inference time.
- FPS.
- Mức suy giảm từ FP32 sang INT8.

### Trình bày trực quan

```text
Low-light Input
Paper Baseline Output
Proposed FP32 Output
Proposed INT8 Output
Ground Truth
```

---

## 20. Ba ưu tiên quan trọng nhất

1. **Retinex/illumination-guided enhancement** để tăng sáng có kiểm soát.
2. **Dark-region attention kết hợp denoising** để tăng đúng vùng mà không khuếch đại nhiễu.
3. **QAT INT8 kết hợp dữ liệu camera thật** để giữ chất lượng khi triển khai.

---

## 21. Kết luận

Hướng đề xuất:

\[
\boxed{
\text{Retinex-guided U-Net Lite}
+
\text{Dark-region Attention}
+
\text{Denoising}
+
\text{Gain/Residual Prediction}
+
\text{QAT INT8}
}
\]

Điểm cải tiến cốt lõi không phải làm model lớn nhất, mà là giúp model hiểu:

- vùng nào cần tăng sáng;
- mức sáng cần điều chỉnh;
- cách giữ chi tiết và màu sắc;
- cách hạn chế nhiễu;
- cách bảo toàn chất lượng sau lượng tử hóa.

Đây là hướng cân bằng giữa chất lượng hình ảnh và khả năng triển khai thực tế trên STM32H750.
