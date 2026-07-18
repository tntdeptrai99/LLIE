# Success Criteria Matrix

Tai lieu nay chot ma tran tieu chi thanh cong cua project LLIE tren STM32H750. Day la bang tham chieu chinh khi danh gia viec "vuot paper Ghost-ESP".

## 1. Ma tran tieu chi

| Tieu chi | Paper Ghost-ESP | Muc tieu toi thieu | Muc tieu tot |
|---|---:|---:|---:|
| Inference time | 277 ms | < 277 ms | <= 100-200 ms |
| FPS | ~3.6 | > 3.6 | 5-10 |
| PSNR LOL | khoang 19.18 dB | > 19.18 dB | >= 20 dB |
| SSIM LOL | khoang 0.852 | > 0.852 | >= 0.88 |
| Quantization | PTQ INT8 | INT8 | QAT INT8 |
| Do phan giai | 96x96 paper reference | 128x128 speed baseline | 256x256 quality variant |
| Model output | RGB truc tiep | Gain + residual | Gain + residual |
| Nhieu | Xu ly gian tiep | Residual denoising | KD tu teacher |

## 2. Cach doc ma tran

### Muc tieu toi thieu

Project dat muc toi thieu neu:

- Model chay INT8 tren STM32H750.
- Inference time nho hon 277 ms/frame.
- FPS lon hon 3.6.
- PSNR tren LOL lon hon 19.18 dB.
- SSIM tren LOL lon hon 0.852.
- Do phan giai dat 128x128 speed baseline.
- Neu so sanh voi paper theo resolution, co ket qua paper-reference mode 96x96 hoac Ghost-ESP-like baseline ro dieu kien.
- Dau ra model dung gain map + residual map.
- Co xu ly nhieu bang residual denoising hoac co thanh phan denoising tuong duong.

### Muc tieu tot

Project dat muc tot neu:

- Inference time nam trong khoang <= 100-200 ms/frame.
- FPS dat 5-10.
- PSNR tren LOL >= 20 dB.
- SSIM tren LOL >= 0.88.
- Model cuoi dung QAT INT8.
- Co quality variant 256x256 neu external SDRAM va profiling cho phep.
- Dau ra model van la gain + residual.
- Chat luong denoising duoc cai thien bang knowledge distillation tu teacher.

## 3. Dieu kien so sanh voi paper

Khi bao cao ket qua so voi Ghost-ESP:

- Neu cung dataset, split, preprocessing, resolution va metric, co the goi la so sanh truc tiep.
- Neu chi dung so lieu paper, phai goi la so sanh tham khao.
- Neu model cua project dung 128x128 hoac 256x256 trong khi paper dung 96x96, can ghi ro khac biet resolution va dung Ghost-ESP-like baseline/paper reference cho dung muc bang chung.
- `96x96` chi la paper-reference/evaluation-only mode, khong thay the speed baseline `128x128`.

## 4. Dieu kien ket luan

Khong ket luan project thanh cong chi dua tren mot metric.

Ket luan cuoi can dong thoi xet:

- Chat luong anh: PSNR, SSIM, LPIPS, Delta E va danh gia truc quan.
- Trien khai: inference time, FPS, peak RAM, Flash va model size.
- Robustness: nhieu, chay sang, lech mau va domain camera that.
- Quantization: suy giam FP32 -> INT8 nam trong nguong chap nhan.
- Color fidelity: CIEDE2000 cua D INT8 phai thap hon A2 FP32, va DeltaE00_INT8 khong vuot qua `1.05 * DeltaE00_FP32` khi so sanh FP32 -> INT8.

Metric chat luong chinh thuc:

- RGB PSNR.
- RGB SSIM.
- LPIPS-AlexNet.
- CIEDE2000.

## 5. Ghi chu ve do hieu nang

Benchmark chinh thuc:

- Do ca model-only va end-to-end.
- 20 warmups.
- 200 measured runs.
- Bao cao mean, median, P95, min, max va std.
- Bao cao `FPS_mean = 1000 / mean_latency_ms`.
- Bao cao `FPS_P95_equivalent = 1000 / P95_latency_ms`.
- Do tung inference rieng bang DWT cycle counter va xu ly overflow 32-bit.
- Tach model-only, processing pipeline va display pipeline.
- Tach sequential va pipelined neu co double buffering.
- Log cache/DMA coherency policy.
- Cach lay peak RAM, Flash, model size va `.bss`.
- Bao cao internal SRAM va external SDRAM rieng.

## 6. Uu tien khi co trade-off

Neu co trade-off giua resolution va toc do:

1. Giu real-time/near-real-time truoc.
2. Giu chat luong vuot paper/reference o 128x128.
3. Dung 256x256 nhu quality variant, khong thay the speed baseline.

Neu co trade-off giua PSNR va cam nhan thi giac:

1. Kiem tra SSIM + LPIPS.
2. Kiem tra Delta E va vung highlight.
3. Dung crop vung toi, vung nhieu va vung sang de danh gia truc quan.
4. Khong chon model chi vi PSNR cao neu anh bi mo, lech mau hoac chay sang.
