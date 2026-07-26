# Cube.AI Output Metadata

## Public Input

- ONNX original input:
  `input_rgb`
- ONNX dtype:
  `float32`
- ONNX shape:
  `[1, 3, 96, 96]`
- Cube.AI public input:
  `input_rgb_output_array`
- Cube.AI dtype:
  `uint8`
- Shape:
  `[3, 96, 96]`
- Layout:
  CHW / NCHW without batch
- Scale:
  `0.0037334118969738483`
- Zero-point:
  `0`
- Activation offset from c-info:
  `279960`
- Size:
  `27648` bytes

## Public Output 0

- ONNX output name:
  `/Add_1_output_0_QuantizeLinear_Output`
- Semantic meaning:
  gain tensor
- ONNX dtype:
  `uint8`
- ONNX shape:
  `[1, 3, 96, 96]`
- Cube.AI public IO array:
  `_Add_1_output_0_QuantizeLinear_Output_Transpose_0_output_array`
- Cube.AI buffer flags:
  `STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_ACTIVATIONS`
- Cube.AI dtype:
  `uint8`
- Shape:
  `[3, 96, 96]`
- Layout:
  CHW / NCHW without batch
- Scale:
  `0.007843137718737125`
- Zero-point:
  `0`
- Activation offset from c-info:
  `0`
- Size:
  `27648` bytes

## Public Output 1

- ONNX output name:
  `/Mul_1_output_0_QuantizeLinear_Output`
- Semantic meaning:
  residual tensor
- ONNX dtype:
  `uint8`
- ONNX shape:
  `[1, 3, 96, 96]`
- Cube.AI public IO array:
  `_Mul_1_output_0_QuantizeLinear_Output_Transpose_1_output_array`
- Cube.AI buffer flags:
  `STAI_FLAG_CHANNEL_FIRST|STAI_FLAG_OUTPUTS|STAI_FLAG_PREALLOCATED|STAI_FLAG_ACTIVATIONS`
- Cube.AI dtype:
  `uint8`
- Shape:
  `[3, 96, 96]`
- Layout:
  CHW / NCHW without batch
- Scale:
  `0.0007843137136660516`
- Zero-point:
  `0`
- Activation offset from c-info:
  `138240`
- Size:
  `27648` bytes

## Important Internal Buffers

- Gain pre-transpose quantized HWC:
  `_Add_1_output_0_QuantizeLinear_Output_output_array`, offset `110592`, shape `[96, 96, 3]`
- Residual pre-transpose quantized HWC:
  `_Mul_1_output_0_QuantizeLinear_Output_output_array`, offset `110592`, shape `[96, 96, 3]`
- The pre-transpose HWC offset is reused at different epochs. It must not be treated as a stable public output.

## Signedness Finding

The generated report, header, and c-info JSON all mark both public outputs as unsigned 8-bit tensors. The PC comparison script now performs an additional `int8` reinterpret check. On the existing `v7263` dump:

- Output0 gain `uint8` MAE: `43.269`
- Output0 gain `int8` reinterpret MAE: `43.269`
- Output1 residual `uint8` MAE: `82.126`
- Output1 residual `int8` reinterpret MAE: `61.709`

The residual signed reinterpret is still very poor and has low cosine similarity. This does not explain the mismatch. The current strongest suspects are output pointer/source, stale runtime output, full-pipeline activation corruption, or an actual generated/runtime mismatch.
