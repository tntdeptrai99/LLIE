# scripts/prepare_data.py
"""Prepare data for Student‑S 128×128 training.

- Reads raw image pairs (low‑light, ground‑truth) from ``data/``.
- Applies the preprocessing steps described in ``PREPROCESSING_SPEC.md`` for the 128×128 resolution:
  * Center‑square crop (largest possible square).
  * Resize to 128×128 using bilinear interpolation.
  * Convert to ``torch.Tensor`` with shape ``C x H x W`` (3 x 128 x 128).
  * Normalize pixel values to ``[0, 1]`` (float32).
- Saves the processed tensors as ``.pt`` files in ``data/cache/student_s/``.

The script also supports synthetic low‑light generation (exposure, gamma, white‑balance, noise) as required by the spec.
"""

import os
import random
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# ---------- Configuration ----------
DATA_ROOT = Path("d:/LLIE_Project/data")
CACHE_ROOT = DATA_ROOT / "cache" / "student_s"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Augmentation parameters (synthetic low‑light)
EXPOSURE_RANGE = (0.4, 0.9)
GAMMA_RANGE = (1.2, 2.4)
WB_RANGE = (0.9, 1.1)  # per channel multiplier
NOISE_STD_RANGE = (0.0, 0.03)  # Gaussian read noise std

# Ensure cache directory exists
os.makedirs(CACHE_ROOT, exist_ok=True)

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def synthetic_low_light(gt_img: Image.Image) -> Image.Image:
    """Create a synthetic low‑light version of a ground‑truth image.
    Steps follow the spec (exposure scaling, gamma, white‑balance, noise).
    """
    img = np.array(gt_img).astype(np.float32) / 255.0
    alpha = random.uniform(*EXPOSURE_RANGE)
    img = img * alpha
    gamma = random.uniform(*GAMMA_RANGE)
    img = np.power(img, gamma)
    wb = np.random.uniform(*WB_RANGE, size=3)
    img = img * wb.reshape(1, 1, 3)
    img = np.clip(img, 0.0, 1.0)
    sigma = random.uniform(*NOISE_STD_RANGE)
    noise = np.random.normal(0.0, sigma, img.shape)
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    img_uint8 = (img * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)

def process_pair(gt_path: Path, idx: int):
    gt_img = Image.open(gt_path).convert("RGB")
    low_img = synthetic_low_light(gt_img)
    transform = transforms.Compose([
        transforms.Lambda(center_crop_square),
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    low_tensor = transform(low_img)
    gt_tensor = transform(gt_img)
    save_path = CACHE_ROOT / f"pair_{idx:06d}.pt"
    torch.save({"low": low_tensor, "gt": gt_tensor}, save_path)
    return save_path

def main():
    split_file = DATA_ROOT / "splits" / "lol_train.txt"
    if not split_file.exists():
        print(f"Split file {split_file} not found. Generate splits first.")
        return
    with open(split_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    for idx, line in enumerate(lines):
        # Expected format: low_path gt_path (but we will use gt_path only for synthetic generation)
        parts = line.split()
        if len(parts) == 2:
            _, gt_path_str = parts
        else:
            gt_path_str = parts[0]
        gt_path = Path(gt_path_str)
        out = process_pair(gt_path, idx)
        print(f"Processed {idx+1}/{len(lines)} -> {out}")

if __name__ == "__main__":
    main()
