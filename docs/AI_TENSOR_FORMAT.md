# AI Tensor Format

Target firmware: `stm32/firmware/LLIE_E2E_Benchmark`
Generated files:

- `X-CUBE-AI/App/llieai.h`
- `X-CUBE-AI/App/llieai.c`
- `X-CUBE-AI/App/llieai_data_params.h`

## Public IO Macros

From `llieai.h`:

| Tensor | Format | Height | Width | Channel | Size bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Input 0 | `AI_BUFFER_FORMAT_U8` | 96 | 3 | 96 | 27648 |
| Output 0 | `AI_BUFFER_FORMAT_U8` | 96 | 3 | 96 | 27648 |
| Output 1 | `AI_BUFFER_FORMAT_U8` | 96 | 3 | 96 | 27648 |

Other public macros:

- `AI_LLIEAI_IN_NUM = 1`
- `AI_LLIEAI_OUT_NUM = 2`
- `AI_LLIEAI_INPUTS_IN_ACTIVATIONS = 4`
- `AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS = 4`

The data type is confirmed as unsigned 8-bit quantized output, not signed int8 and not float32.

## Quantization

From `llieai.c` int-quant metadata:

| Tensor | Generated name | Scale | Zero point |
| --- | --- | ---: | ---: |
| Input 0 | `input_rgb_output_array_intq` | 0.0037334118969738483 | 0 |
| Output 0 | `_Add_1_output_0_QuantizeLinear_Output_Transpose_0_output_array_intq` | 0.007843137718737125 | 0 |
| Output 1 | `_Mul_1_output_0_QuantizeLinear_Output_Transpose_1_output_array_intq` | 0.0007843137136660516 | 0 |

Formula:

```text
real = scale * (q - zero_point)
```

Because all three public IO tensors are U8 with zero point 0, the current source must not reinterpret these buffers as `int8_t` or `float`.

## Layout Evidence

The header macros can be misleading for image indexing because they expose:

```text
height=96, width=3, channel=96
```

The generated public tensor declarations show the post-transpose public IO tensors:

```text
AI_SHAPE_INIT(4, 1, 96, 96, 3)
AI_STRIDE_INIT(4, 1, 1, 96, 9216)
```

Internal non-public tensors before the transpose show:

```text
AI_SHAPE_INIT(4, 1, 3, 96, 96)
AI_STRIDE_INIT(4, 1, 1, 3, 288)
```

Current interpretation:

- Public model IO is expected to be a 96 x 96 x 3 tensor after the generated transpose.
- The safest indexing for public output is the stride-backed interpretation rather than blindly trusting the `WIDTH=3, CHANNEL=96` macro names.

## Binding Offsets

Cube.AI activation map assignments in `llieai.c`:

```text
input public IO: activation + 279960
output 0 public IO: activation + 0
output 1 public IO: activation + 138240
```

The application also keeps external public output buffers in `ai_output_data`. The runtime output binding must be verified on board by checksum because the generated network can place public IO in the activation arena when `AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS` is enabled.

