# Retinexformer RGB Teacher Pipeline

## Goal

Generate RGB teacher images offline with Retinexformer, then use them for student knowledge distillation.

Teacher RGB is not deployed on STM32. Only the student is exported to ONNX/QDQ/Cube.AI.

## 1. Prepare Inputs For Retinexformer

For LOL train:

```powershell
C:\MiniForge\python.exe scripts\prepare_teacher_rgb_inputs.py --paired-split splits\lol_train.txt --out-root data\teacher_inputs\retinexformer_rgb\lol_train --teacher-root data\teacher\retinexformer_rgb\lol_train --manifest data\teacher\retinexformer_rgb\lol_train_manifest.csv
```

For LOL test:

```powershell
C:\MiniForge\python.exe scripts\prepare_teacher_rgb_inputs.py --paired-split splits\lol_test.txt --out-root data\teacher_inputs\retinexformer_rgb\lol_test --teacher-root data\teacher\retinexformer_rgb\lol_test --manifest data\teacher\retinexformer_rgb\lol_test_manifest.csv
```

The script creates:

```text
data/teacher_inputs/retinexformer_rgb/<split>/input/*.png
data/teacher_inputs/retinexformer_rgb/<split>/target/*.png
data/teacher/retinexformer_rgb/<split>_manifest.csv
```

The copied input filenames are unique path slugs so LOL and LOL-v2 files cannot collide.

## 2. Run Retinexformer In Its Own Repo

Use the official Retinexformer repo and pretrained weights. The official testing command style is:

```bash
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1
```

For this project, run Retinexformer on the prepared `input` folder and copy/save enhanced RGB results into the matching teacher root:

```text
D:/LLIE_Project/data/teacher/retinexformer_rgb/lol_train
D:/LLIE_Project/data/teacher/retinexformer_rgb/lol_test
```

Each output filename must match the prepared input filename, for example:

```text
data/teacher_inputs/retinexformer_rgb/lol_train/input/data_raw_LOL_train_low_10.png
data/teacher/retinexformer_rgb/lol_train/data_raw_LOL_train_low_10.png
```

In the Retinexformer option YAML, point validation paths to:

```yaml
datasets:
  val:
    dataroot_lq: D:/LLIE_Project/data/teacher_inputs/retinexformer_rgb/lol_train/input
    dataroot_gt: D:/LLIE_Project/data/teacher_inputs/retinexformer_rgb/lol_train/target
```

Then run with `--output_dir` pointing to the matching teacher folder.

## 3. Create RGB KD Splits

For LOL train:

```powershell
C:\MiniForge\python.exe scripts\make_teacher_rgb_split.py --manifest data\teacher\retinexformer_rgb\lol_train_manifest.csv --out splits\lol_train_retinexformer_rgb.txt --strict --validate-rgb
```

For LOL test:

```powershell
C:\MiniForge\python.exe scripts\make_teacher_rgb_split.py --manifest data\teacher\retinexformer_rgb\lol_test_manifest.csv --out splits\lol_test_retinexformer_rgb.txt --strict --validate-rgb
```

The output format is:

```text
low_path high_path teacher_rgb_path
```

This is directly consumed by:

```powershell
C:\MiniForge\python.exe scripts\train_distill.py --train-split splits\lol_train_retinexformer_rgb.txt --val-split splits\lol_test_retinexformer_rgb.txt --student-variant D96-TINY-BN --image-size 96 --pretrained experiments\d96_tiny_bn_supervised_96_stage1_resume\best.pt --out-dir experiments\d96_tiny_bn_rgb_kd_96_stage3
```

## Notes

- Retinexformer RGB outputs teach color, texture, denoising, and natural enhancement.
- Existing `teacher_y/*.npy` maps are useful for luma KD only, not full RGB output KD.
- Keep Retinexformer outside the STM32 deployment path.
