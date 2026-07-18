# Scripts

Command-line utilities will live here.

Planned scripts:

- Generate split files.
- Validate preprocessing.
- Train supervised student variants.
- Train distillation student variants.
- Run metrics and save visual samples.
- Export dummy ONNX for Cube.AI feasibility.
- Summarize reports.

## Supervised training

Use this for the first training stage before Retinexformer distillation:

```powershell
C:\MiniForge\python.exe scripts\train_supervised.py --model-variant D96-TINY-BN --image-size 96 --epochs 5 --batch-size 32 --train-limit 512 --val-limit 118 --out-dir experiments\d96_tiny_bn_supervised_96_stage1 --ssim-weight 0.2 --edge-weight 0.1 --color-weight 0.05 --lightness-weight 0.2 --lr 0.0005 --e-blocks 3
```

## Evaluation

```powershell
C:\MiniForge\python.exe scripts\eval_model.py --checkpoint experiments\d96_tiny_bn_supervised_96_stage1\best.pt --split splits\lol_test.txt --image-size 96 --model-variant D96-TINY-BN --metric-name d96_tiny_bn_96
```

## D-96 dummy export

Run this before long training:

```powershell
C:\MiniForge\python.exe scripts\export_dummy_onnx.py
```

Expected output:

- `stm32/onnx/d_96_dummy.onnx`
- input/output shape `1x3x96x96`
- next step is importing the ONNX file in STM32Cube.AI.

For the RAM-first BN-folded candidate:

```powershell
C:\MiniForge\python.exe scripts\export_dummy_onnx.py --arch d96-tiny-bn --blocks 3 --out stm32\onnx\d_96_tiny_bn_fused_dummy.onnx
```

To create a QDQ quantized ONNX candidate for Cube.AI INT8 testing:

```powershell
C:\MiniForge\python.exe scripts\export_qdq_onnx.py --input stm32\onnx\d_96_tiny_bn_fused_dummy.onnx --output stm32\onnx\d_96_tiny_bn_qdq_dummy.onnx
```

To export a trained D-96-Tiny-BN checkpoint:

```powershell
C:\MiniForge\python.exe scripts\export_dummy_onnx.py --arch d96-tiny-bn --blocks 3 --checkpoint experiments\d96_tiny_bn_supervised_96_stage1_resume\best.pt --out stm32\onnx\d_96_tiny_bn_supervised_96_fused.onnx
```

## Retinexformer RGB teacher

Prepare unique input files for an external Retinexformer repo:

```powershell
C:\MiniForge\python.exe scripts\prepare_teacher_rgb_inputs.py --paired-split splits\lol_train.txt --out-root data\teacher_inputs\retinexformer_rgb\lol_train --teacher-root data\teacher\retinexformer_rgb\lol_train --manifest data\teacher\retinexformer_rgb\lol_train_manifest.csv
```

After Retinexformer writes enhanced RGB images into the teacher root, create KD splits:

```powershell
C:\MiniForge\python.exe scripts\make_teacher_rgb_split.py --manifest data\teacher\retinexformer_rgb\lol_train_manifest.csv --out splits\lol_train_retinexformer_rgb.txt --strict --validate-rgb
```

See `docs/RETINEXFORMER_TEACHER_RGB.md`.
