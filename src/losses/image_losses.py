from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


class EdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = self._edges(pred)
        target_edges = self._edges(target)
        return F.l1_loss(pred_edges, target_edges)

    def _edges(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        sobel_x = self.sobel_x[:channels]
        sobel_y = self.sobel_y[:channels]
        grad_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, sobel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)


class ColorStatsLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = (2, 3)
        pred_mean = pred.mean(dim=dims)
        target_mean = target.mean(dim=dims)
        pred_std = pred.std(dim=dims, unbiased=False)
        target_std = target.std(dim=dims, unbiased=False)
        mean_loss = F.l1_loss(pred_mean, target_mean)
        std_loss = F.l1_loss(pred_std, target_std)
        return mean_loss + std_loss


class ChromaLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_chroma = pred.amax(dim=1) - pred.amin(dim=1)
        target_chroma = target.amax(dim=1) - target.amin(dim=1)
        return F.l1_loss(pred_chroma, target_chroma)


class LocalContrastLoss(nn.Module):
    def __init__(self, window: int = 7) -> None:
        super().__init__()
        self.window = window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_luma = self._luma(pred)
        target_luma = self._luma(target)
        pred_contrast = self._local_std(pred_luma)
        target_contrast = self._local_std(target_luma)
        return F.l1_loss(pred_contrast, target_contrast)

    @staticmethod
    def _luma(image: torch.Tensor) -> torch.Tensor:
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    def _local_std(self, image: torch.Tensor) -> torch.Tensor:
        padding = self.window // 2
        mean = F.avg_pool2d(image, self.window, stride=1, padding=padding)
        mean_sq = F.avg_pool2d(image * image, self.window, stride=1, padding=padding)
        variance = (mean_sq - mean * mean).clamp_min(0.0)
        return torch.sqrt(variance + 1e-6)


class LightnessAwareLoss(nn.Module):
    def __init__(self, bins: int = 16, sigma: float = 0.08) -> None:
        super().__init__()
        centers = torch.linspace(0.0, 1.0, bins).view(1, bins, 1, 1)
        self.register_buffer("centers", centers)
        self.sigma = sigma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        low: torch.Tensor,
    ) -> torch.Tensor:
        pred_luma = self._luma(pred)
        target_luma = self._luma(target)
        low_luma = self._luma(low)
        target_hist = self._soft_hist(target_luma)
        pred_hist = self._soft_hist(pred_luma)
        hist_loss = F.l1_loss(pred_hist, target_hist)

        dark_weight = (1.0 - low_luma).detach()
        dark_recon = (dark_weight * (pred_luma - target_luma).abs()).mean()
        bright_weight = low_luma.detach()
        bright_preserve = (bright_weight * (pred_luma - target_luma).abs()).mean()
        return hist_loss + 0.5 * dark_recon + 0.25 * bright_preserve

    @staticmethod
    def _luma(image: torch.Tensor) -> torch.Tensor:
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    def _soft_hist(self, luma: torch.Tensor) -> torch.Tensor:
        distance = luma - self.centers.to(device=luma.device, dtype=luma.dtype)
        weights = torch.exp(-(distance * distance) / (2.0 * self.sigma * self.sigma))
        hist = weights.mean(dim=(2, 3))
        return hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-6)


class DarkMapAdaptiveLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        low: torch.Tensor,
        dark_map: torch.Tensor,
        dark_gamma: float = 1.0,
        bright_gamma: float = 1.0,
        bright_preserve_target: str = "low",
    ) -> torch.Tensor:
        dark_weight = dark_map.detach().clamp(0.0, 1.0).pow(dark_gamma)
        bright_weight = (1.0 - dark_map.detach().clamp(0.0, 1.0)).pow(bright_gamma)
        dark_recon = (dark_weight * (pred - target).abs()).mean()
        bright_target = low if bright_preserve_target == "low" else target
        bright_preserve = (bright_weight * (pred - bright_target).abs()).mean()
        return dark_recon + 0.5 * bright_preserve


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(pred, target)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()
