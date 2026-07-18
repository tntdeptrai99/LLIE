# STM32H750 Benchmark Protocol

## 1. Muc tieu

Dinh nghia cach do chinh thuc:

- Latency.
- FPS.
- RAM.
- Flash.
- Model-only performance.
- End-to-end camera pipeline.

Moc paper can vuot:

```text
Ghost-ESP inference time ~= 277 ms/frame
FPS ~= 3.6
```

Muc tieu toi thieu:

```text
t_model < 277 ms
FPS_model > 3.6
```

Muc tieu chinh:

```text
7-10 FPS model-only
```

## 2. Dieu kien phan cung

- MCU: STM32H750VBT6.
- CPU clock: `480 MHz`.
- Camera: OV5640.
- Camera format: RGB565.
- Display: ST7735 SPI.
- External SDRAM: dung cho framebuffer va activation lon.
- Build mode: Release.
- Compiler optimization phai duoc ghi ro.
- Cache I/D phai duoc ghi ro la bat hay tat.
- Buffer cacheability phai duoc ghi ro cho input/output/camera/LCD buffers.
- Clock configuration phai co dinh giua cac model.

Khong thay doi clock hoac compiler flag giua cac benchmark.

## 3. Model-only benchmark

### Pham vi

Chi do:

```text
input tensor da san sang
-> AI inference
-> output tensor san sang
```

Khong tinh:

- Camera capture.
- RGB565 conversion.
- Resize.
- Quantization input.
- Dequantization output.
- LCD transfer.

### Warm-up

```text
20 lan
```

Khong ghi ket qua warm-up.

Muc dich:

- Lam nong cache.
- On dinh memory access.
- Loai bo lan chay dau bat thuong.

### Measured runs

```text
200 lan
```

Dung cung mot tensor hoac mot tap tensor co dinh.

### Timer

Dung DWT cycle counter.

Voi CPU `480 MHz`:

```text
t_ms = cycles / 480000
```

DWT CYCCNT la counter 32-bit. O `480 MHz`, counter wrap sau khoang `8.95 s`.

Quy tac:

- Do tung inference rieng, khong do mot block 200 lan bang mot khoang duy nhat.
- Tinh delta cycles bang unsigned 32-bit difference.
- Reset/log counter state neu firmware benchmark can.

### Ket qua phai bao cao

- Mean latency.
- Median latency.
- P95 latency.
- Minimum.
- Maximum.
- Standard deviation.
- Mean FPS.
- P95-equivalent FPS neu can.

FPS:

```text
FPS_mean = 1000 / mean_latency_ms
FPS_P95_equivalent = 1000 / P95_latency_ms
```

Khong lay trung binh truc tiep cac FPS tung run neu co the; uu tien tinh tu latency aggregate va bao ro cach tinh. `FPS_P95_equivalent` la throughput tuong ung voi tail latency, khong phai FPS do truc tiep.

## 4. End-to-end benchmark

Bao gom:

```text
OV5640 capture
-> DCMI/DMA
-> RGB565 framebuffer
-> crop/resize
-> RGB conversion
-> quantization
-> AI inference
-> dequantization/clamp
-> RGB565 conversion
-> LCD SPI transfer
```

Can do rieng cac stage:

- Camera capture.
- Preprocessing.
- Inference.
- Postprocessing.
- LCD transfer.
- Total frame latency.

Chay:

- `20` frame warm-up.
- It nhat `200` frame do chinh thuc.

Bao cao:

- Mean total latency.
- Median.
- P95.
- End-to-end FPS.
- Dropped frames.
- Frame jitter.
- Thoi gian tung stage.

Khong dung model-only FPS de tuyen bo FPS cua toan he thong.

Can bao cao ba muc end-to-end neu co du firmware support:

| Mode | Thanh phan tinh vao latency |
|---|---|
| Processing pipeline | camera -> preprocess -> inference -> postprocess |
| Display pipeline | camera -> preprocess -> inference -> postprocess -> LCD |
| Model-only | AI inference only |

ST7735 qua SPI co the la bottleneck; vi vay processing pipeline va display pipeline khong duoc gop chung neu khong chu thich.

## 5. Sequential va pipelined measurement

End-to-end co hai che do hop le:

### Sequential

```text
capture
-> preprocess
-> inference
-> display
-> next frame
```

### Pipelined

```text
DMA capture frame N+1 song song voi inference/display frame N
```

Quy tac:

- Bao cao sequential end-to-end FPS neu pipeline chay tung frame noi tiep.
- Bao cao pipelined steady-state FPS neu co double buffering/pipelining.
- Khong tron sequential va pipelined trong cung bang neu khong co cot `Pipeline mode`.
- Neu dung pipelined FPS, phai bao cao latency tung frame va throughput steady-state rieng.

## 6. Cache va DMA coherency

Voi Cortex-M7, OV5640 DCMI/DMA va SDRAM ngoai, cache coherency la dieu kien bat buoc de benchmark dung.

Moi benchmark phai log:

- Input/output buffer nam o vung cacheable hay non-cacheable.
- Camera framebuffer nam o internal SRAM hay external SDRAM.
- Co goi `SCB_InvalidateDCache_by_Addr` sau DMA capture hay khong.
- Co clean cache truoc DMA/LCD transfer hay khong.
- Thoi gian cache maintenance duoc tinh vao preprocessing/postprocessing hay bi loai tru.
- Cache I/D bat hay tat.

Neu cache maintenance khac nhau giua cac model, benchmark khong duoc xem la cong bang.

## 7. Ba che do benchmark theo resolution

### Mode P: 96x96

Muc dich:

- Paper-reference fairness.
- Ghost-ESP-like comparison.
- So sanh gan hon voi paper Ghost-ESP.

Quy tac:

- Khong thay the deployment speed baseline.
- Bao cao rieng voi label `paper-reference mode`.
- Neu so sanh model de xuat va Ghost-ESP-like o `96x96`, phai dung cung preprocessing, quantization va benchmark protocol.

### Mode A: 128x128

Muc tieu chinh ve toc do.

Dieu kien thanh cong:

```text
latency < 277 ms
FPS > 3.6
target 7-10 FPS
```

### Mode B: 256x256

Muc tieu chat luong cao.

Dieu kien:

- Van phai nhanh hon paper neu duoc dung lam ban deploy chinh.
- Neu khong dat, `256x256` chi duoc goi la quality variant.
- Ban deploy chinh giu o `128x128`.

## 8. RAM measurement

Phai bao ca RAM do AI tool uoc luong va RAM thuc cua firmware.

### STM32Cube.AI Analyzer

Ghi:

- Activation buffer.
- Weights RAM/Flash.
- Workspace.
- Operator memory.
- Estimated total.

### Linker map

Ghi:

- `.bss`.
- `.data`.
- Heap.
- Stack.
- Generated AI buffers.
- Static image buffers.

### Runtime

Neu co the, do:

- Stack high-water mark.
- Heap peak.
- Activation arena.
- Framebuffer.
- Temporary conversion buffer.
- Internal SRAM usage.
- External SDRAM usage.

Khong gop RAM noi va SDRAM thanh mot so duy nhat.

Neu internal SRAM budget khoang `90-100 KB`, phai ghi ro day la internal SRAM budget. Activation lon va framebuffer co the nam trong external SDRAM, nhung phai bao cao rieng va do latency voi cau hinh do.

Report phai co:

| Thanh phan | Internal SRAM | External SDRAM |
|---|---:|---:|
| Model activation | | |
| Input tensor | | |
| Output tensor | | |
| Camera buffer | | |
| LCD buffer | | |
| Stack/heap | | |
| Total | | |

## 9. Flash measurement

Tach:

- Model weights.
- Generated AI code.
- Application firmware.
- Camera driver.
- LCD driver.
- Constant tables.
- Total binary.

Nguon so lieu:

- STM32Cube.AI report.
- Linker map.
- `.elf`/`.map`.
- Binary size tool.

Khong chi bao kich thuoc file ONNX.

## 10. Benchmark fairness

Tat ca model phai dung cung:

- Board.
- CPU clock.
- Compiler flags.
- Cache setting.
- Input resolution.
- Input dtype.
- So lan warm-up.
- So lan do.
- Measurement code.
- Pipeline mode: model-only, sequential, hoac pipelined.
- Cache/DMA maintenance policy.

Khi so sanh Ghost-ESP-like va model de xuat:

- Cung preprocessing.
- Cung quantization.
- Cung input.
- Cung benchmark protocol.

Khi chi so sanh voi paper reference, phai ghi ro:

```text
Paper-reference comparison, not exact reproduction.
```

## 11. Report table

| Model | Resolution | Precision | Benchmark mode | Pipeline mode | Params | MACs | Internal RAM | SDRAM | Flash | Mean ms | Median ms | P95 ms | FPS mean | FPS P95 eq |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student | 96x96 | INT8 | model-only | paper-reference | | | | | | | | | | |
| Ghost-like | 96x96 | INT8 | model-only | paper-reference | | | | | | | | | | |
| A2 | 128x128 | FP32 | model-only | sequential | | | | | | | | | | |
| C | 128x128 | FP32 | model-only | sequential | | | | | | | | | | |
| D | 128x128 | INT8 | model-only | sequential | | | | | | | | | | |
| Ghost-like | 128x128 | INT8 | model-only | sequential | | | | | | | | | | |
| D-HQ | 256x256 | INT8 | model-only | sequential | | | | | | | | | | |

## 12. Dieu kien thanh cong cuoi

Model deploy duoc xem la dat yeu cau khi dong thoi:

- Chat luong vuot A2 FP32.
- Chat luong cao hon Ghost-ESP-like hoac cao hon paper reference trong dieu kien duoc mo ta ro.
- Model-only latency nho hon `277 ms`.
- Model-only FPS lon hon `3.6`.
- Muc tieu thuc te dat `7 FPS` tro len.
- Khong vuot ngan sach RAM/Flash.
- Khong co loi chay sang, lech mau hoac nhieu ro ret.
- INT8 gan voi FP32 theo nguong metric da chot.

## 13. Completion criteria

Spec nay duoc xem la chot khi:

- Model-only va end-to-end modes duoc tach rieng.
- Warm-up va measured run counts co dinh.
- Mean, median, P95, min, max, std duoc yeu cau.
- DWT cycle counter va cong thuc doi cycles sang ms ro rang.
- DWT 32-bit overflow handling ro rang.
- FPS_mean va FPS_P95_equivalent duoc dinh nghia ro.
- Cache/DMA coherency rules ro rang.
- Sequential va pipelined measurement duoc tach rieng.
- Processing pipeline va display pipeline duoc tach rieng.
- RAM/Flash reporting sources ro rang.
- Internal SRAM va external SDRAM duoc bao cao rieng.
- `96x96`, `128x128` va `256x256` duoc bao cao rieng.
- Benchmark fairness voi Ghost-ESP-like va paper reference ro rang.
