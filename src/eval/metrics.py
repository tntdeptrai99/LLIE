from __future__ import annotations

import math

import torch

from src.losses.image_losses import ssim


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim_value(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(ssim(pred, target).item())
