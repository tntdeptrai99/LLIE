# Model Provenance Audit

## Active PC Reference Model

- ONNX Runtime script default:
  `D:\LLIE_Project\stm32\onnx\ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_tail2_u8out.onnx`
- SHA-256:
  `3874C2F5621C0DA6495754943D9B3BF830C3BF866ABF2BB5943F572D416008D0`
- ONNX graph input:
  `input_rgb`, float32, `[1, 3, 96, 96]`
- ONNX graph outputs:
  `/Add_1_output_0_QuantizeLinear_Output`, uint8, `[1, 3, 96, 96]`
  `/Mul_1_output_0_QuantizeLinear_Output`, uint8, `[1, 3, 96, 96]`

## Active Cube.AI Generated Model

- Firmware project:
  `D:\LLIE_Project\stm32\firmware\LLIE_E2E_Benchmark`
- Generated source folder used by the Debug build:
  `D:\LLIE_Project\stm32\firmware\LLIE_E2E_Benchmark\X-CUBE-AI\App`
- Build source list includes:
  `X-CUBE-AI/App`
- Cube.AI report:
  `D:\LLIE_Project\stm32\firmware\LLIE_E2E_Benchmark\X-CUBE-AI\App\llieai_generate_report.txt`
- Report generation command:
  `generate --target stm32h7 --name llieai -m D:/LLIE_Project/stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_tail2_u8out.onnx --compression none`
- Cube.AI model hash:
  `0xd9fae00b20da69cb58a406c197e238c1`
- Cube.AI network signature from c-info:
  `0x7bf2e955fc0ef1ac`
- Generated timestamp:
  `2026-07-20T23:07:11+0700`
- Tool:
  `ST Edge AI Core v2.2.0-20266 2adc00962`

## Generated File Signatures

From `.ai\llieai_ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_tail2_u8out.onnx_c_info.json`:

- `llieai.c`: `0x90bff378d89cbd0458101dc554595f63`
- `llieai.h`: `0x66a8bbd35e3c2b928640752fc40cf605`
- `llieai_data.c`: `0x9b1b9e3aa989d45a15aaa93584a1a392`
- `llieai_data.h`: `0x864ec1a06cda056fd7b48ab2f750ece3`
- `llieai_data_params.c`: `0x892ba48b4fc8be2494fb96db703e2350`
- `llieai_data_params.h`: `0x7b871e2a2afea96375a628bb30d5829c`
- `llieai_config.h`: `0x54fa4b3867f1a0ff3482995b6a533663`

## Other ONNX Hashes In The Same Family

- `ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq.onnx`
  SHA-256: `10D0CDDB9D7507D51AC24E8FDBF038FA0B64CB168D907607C2233E1C05B8DD16`
- `ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_fused.onnx`
  SHA-256: `BDA9D954E56932406ECEF63CE7E7FE95B286E785B86BF226A46352B711387E7E`

## Current Firmware Identity

- Firmware probe version after audit changes:
  `7265`
- Startup log now prints:
  `fw=<version>,m=d9fae00b`
  `aio=<input_offset>/<output0_offset>/<output1_offset>,act=<activation_bytes>`

## Preliminary Conclusion

The PC comparison script and the E2E firmware project are referencing the same intended `tail2_u8out` ONNX model. The active Debug build source list points to `LLIE_E2E_Benchmark\X-CUBE-AI\App`, not the older `LLIE` or `LLIE_Benchmark` generated folders.
