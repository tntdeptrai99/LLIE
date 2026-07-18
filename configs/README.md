# Configs

Experiment configuration files live here.

Initial config:

- `a1_128.yaml`: A1 baseline at `128x128`, Dark Guidance OFF, Charbonnier loss.
- `d_96_retinexformer_distill.yaml`: D-96 paper-reference config with Retinexformer distillation, Ghost-ESP gain/residual student, adaptive dark loss, and STM32Cube.AI export gate.
- `d_96_tiny_bn_retinexformer_distill.yaml`: RAM-first D-96 variant with Ghost/ESP blocks, depthwise separable conv, ReLU6 and BN folding before ONNX export.

Do not treat a config as final unless it matches the current P0 specs.
