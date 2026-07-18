from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class Block5(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class GhostSepBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        intrinsic = max(1, channels // 2)
        ghost = channels - intrinsic
        self.primary = nn.Sequential(
            nn.Conv2d(channels, intrinsic, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(intrinsic, ghost, kernel_size=3, padding=1, groups=intrinsic),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        ghost = self.cheap(primary)
        fused = torch.cat([primary, ghost], dim=1)
        return self.act(x + self.project(fused))


class ConvBNReLU6(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse_for_export(self) -> nn.Sequential:
        fused = fuse_conv_bn_eval(self.conv, self.bn)
        return nn.Sequential(fused, nn.ReLU6(inplace=True))


class ConvBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

    def fuse_for_export(self) -> nn.Conv2d:
        return fuse_conv_bn_eval(self.conv, self.bn)


class DepthwiseSeparableBNReLU6(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBNReLU6(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = ConvBNReLU6(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class GhostESPBNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        intrinsic = max(1, channels // 2)
        ghost = channels - intrinsic
        self.primary = ConvBNReLU6(channels, intrinsic, kernel_size=1)
        self.cheap = ConvBNReLU6(
            intrinsic,
            ghost,
            kernel_size=3,
            padding=1,
            groups=intrinsic,
        )
        self.project = ConvBN(channels, channels, kernel_size=1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        ghost = self.cheap(primary)
        fused = torch.cat([primary, ghost], dim=1)
        return self.act(x + self.project(fused))


class PConvLiteBlock(nn.Module):
    def __init__(self, channels: int = 12, pconv_channels: int = 4) -> None:
        super().__init__()
        if not 0 < pconv_channels <= channels:
            raise ValueError("pconv_channels must be in the range [1, channels]")
        self.pconv_channels = pconv_channels
        self.spatial = ConvBNReLU6(
            pconv_channels,
            pconv_channels,
            kernel_size=3,
            padding=1,
            groups=pconv_channels,
        )
        self.mix = ConvBN(channels, channels, kernel_size=1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_part, bypass_part = torch.split(
            x,
            [self.pconv_channels, x.shape[1] - self.pconv_channels],
            dim=1,
        )
        spatial_part = self.spatial(spatial_part)
        fused = torch.cat([spatial_part, bypass_part], dim=1)
        return self.act(x + self.mix(fused))


class DarkMapGenerator(nn.Module):
    def __init__(
        self,
        threshold: float = 0.15,
        scale: float = 0.70,
        smooth_kernel: int = 5,
    ) -> None:
        super().__init__()
        if smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be odd")
        self.threshold = threshold
        self.scale = scale
        self.smooth_kernel = smooth_kernel
        luma_weight = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.register_buffer("luma_weight", luma_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        luma = (x * self.luma_weight.to(device=x.device, dtype=x.dtype)).sum(
            dim=1,
            keepdim=True,
        )
        dark = 1.0 - luma
        if self.smooth_kernel > 1:
            padding = self.smooth_kernel // 2
            dark = F.avg_pool2d(dark, self.smooth_kernel, stride=1, padding=padding)
        return ((dark - self.threshold) / self.scale).clamp(0.0, 1.0)


class FixedDarkMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avg = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.avg.weight.fill_(1.0 / 3.0)
        self.avg.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.avg(x)


def fuse_bn_modules(module: nn.Module) -> nn.Module:
    fused = copy.deepcopy(module).eval()
    for name, child in list(fused.named_children()):
        if isinstance(child, (ConvBNReLU6, ConvBN)):
            setattr(fused, name, child.fuse_for_export())
        else:
            setattr(fused, name, fuse_bn_modules(child))
    return fused


class DarknessModulation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        dark = 1.0 - low.mean(dim=1, keepdim=True)
        dark = F.interpolate(dark, size=feat.shape[-2:], mode="nearest")
        gate = self.net(torch.cat([feat, dark], dim=1))
        return feat * (0.5 + gate)


class Downsample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, groups=8),
            nn.Conv2d(8, 16, kernel_size=1),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentNet(nn.Module):
    def __init__(
        self,
        dark_guidance: bool = False,
        detail_refine: bool = False,
        detail_blocks: int = 0,
        detail_scale: float = 0.08,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.dark_guidance = dark_guidance
        self.detail_refine = detail_refine
        self.detail_scale = detail_scale
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.residual_scale = residual_scale
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = nn.Sequential(Block5(8), Block5(8))
        self.down = Downsample()
        self.stage2 = nn.Sequential(Block5(16), Block5(16), Block5(16))
        self.bottleneck = nn.Sequential(Block5(16), Block5(16))
        self.dark_fuse = nn.Sequential(
            nn.Conv2d(17, 16, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.up_proj = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.stage3 = nn.Sequential(Block5(8))
        self.detail_refine_blocks = nn.Sequential(
            *[Block5(8) for _ in range(detail_blocks)]
        )
        self.gain_head = nn.Conv2d(8, 3, kernel_size=1)
        self.residual_head = nn.Conv2d(8, 3, kernel_size=1)
        self.detail_head = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        skip = self.stage1(feat)
        feat = self.down(skip)
        feat = self.stage2(feat)
        feat = self.bottleneck(feat)
        if self.dark_guidance:
            dark = 1.0 - x.mean(dim=1, keepdim=True)
            dark = F.interpolate(dark, size=feat.shape[-2:], mode="nearest")
            feat = self.dark_fuse(torch.cat([feat, dark], dim=1))
        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = self.up_proj(feat)
        feat = self.stage3(feat + skip)
        if self.detail_refine:
            feat = self.detail_refine_blocks(feat)

        gain = self.gain_min + (self.gain_max - self.gain_min) * F.relu6(self.gain_head(feat)) / 6.0
        residual = self.residual_scale * torch.tanh(self.residual_head(feat))
        output = x * gain + residual
        if self.detail_refine:
            detail = self.detail_scale * torch.tanh(self.detail_head(feat))
            output = output + detail
        return torch.clamp(output, 0.0, 1.0)


class StudentA1(StudentNet):
    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__(
            dark_guidance=False,
            detail_refine=False,
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )


class StudentB(StudentNet):
    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__(
            dark_guidance=True,
            detail_refine=False,
            detail_blocks=0,
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )


class StudentC(StudentNet):
    def __init__(
        self,
        gain_min: float = 0.85,
        gain_max: float = 2.2,
        residual_scale: float = 0.35,
        detail_blocks: int = 2,
        detail_scale: float = 0.08,
    ) -> None:
        super().__init__(
            dark_guidance=True,
            detail_refine=True,
            detail_blocks=detail_blocks,
            detail_scale=detail_scale,
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )


class StudentE(nn.Module):
    def __init__(
        self,
        base_channels: int = 8,
        mid_channels: int = 16,
        blocks: int = 4,
        gain_min: float = 1.0,
        gain_max: float = 2.2,
        residual_scale: float = 0.25,
        dark_modulation: bool = True,
    ) -> None:
        super().__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.residual_scale = residual_scale
        self.dark_modulation = dark_modulation
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, groups=base_channels),
            nn.Conv2d(base_channels, mid_channels, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(*[GhostSepBlock(mid_channels) for _ in range(blocks)])
        self.modulate = DarknessModulation(mid_channels) if dark_modulation else nn.Identity()
        self.up = nn.Sequential(
            nn.Conv2d(mid_channels, base_channels, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.refine = GhostSepBlock(base_channels)
        self.gain_head = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)
        feat = self.down(skip)
        feat = self.blocks(feat)
        if self.dark_modulation:
            feat = self.modulate(feat, x)
        else:
            feat = self.modulate(feat)
        feat = F.interpolate(feat, size=skip.shape[-2:], mode="nearest")
        feat = self.up(feat)
        feat = self.refine(feat + skip)
        gain = self.gain_min + (self.gain_max - self.gain_min) * F.relu6(self.gain_head(feat)) / 6.0
        residual = self.residual_scale * torch.tanh(self.residual_head(feat))
        return torch.clamp(x * gain + residual, 0.0, 1.0)


class StudentD96(nn.Module):
    """D-96 paper-reference student with Cube.AI-friendly operators."""

    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
        blocks: int = 4,
        dark_guidance: bool = True,
    ) -> None:
        super().__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.residual_scale = residual_scale
        self.dark_guidance = dark_guidance
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, groups=8),
            nn.Conv2d(8, 16, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(*[GhostSepBlock(16) for _ in range(blocks)])
        self.dark_fuse = nn.Sequential(
            nn.Conv2d(17, 16, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        self.refine = GhostSepBlock(8)
        self.gain_head = nn.Conv2d(8, 3, kernel_size=1)
        self.residual_head = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)
        feat = self.down(skip)
        feat = self.blocks(feat)
        if self.dark_guidance:
            dark = 1.0 - x.mean(dim=1, keepdim=True)
            dark = F.interpolate(dark, scale_factor=0.5, mode="nearest")
            feat = self.dark_fuse(torch.cat([feat, dark], dim=1))
        feat = F.interpolate(feat, scale_factor=2.0, mode="nearest")
        feat = self.up(feat)
        feat = self.refine(feat + skip)
        gain = self.gain_min + (self.gain_max - self.gain_min) * F.relu6(self.gain_head(feat)) / 6.0
        residual_unit = F.relu6(self.residual_head(feat) + 3.0) / 3.0 - 1.0
        residual = self.residual_scale * residual_unit
        return torch.clamp(x * gain + residual, 0.0, 1.0)


class StudentD96TinyBN(nn.Module):
    """RAM-first D-96 student using Ghost/ESP, depthwise separable conv, ReLU6 and foldable BN."""

    def __init__(
        self,
        base_channels: int = 4,
        mid_channels: int = 8,
        blocks: int = 3,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
        dark_guidance: bool = True,
    ) -> None:
        super().__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.residual_scale = residual_scale
        self.dark_guidance = dark_guidance
        self.dark_map = FixedDarkMap()
        self.stem = ConvBNReLU6(3, base_channels, kernel_size=3, padding=1)
        self.down = DepthwiseSeparableBNReLU6(
            base_channels,
            mid_channels,
            stride=2,
        )
        self.blocks = nn.Sequential(*[GhostESPBNBlock(mid_channels) for _ in range(blocks)])
        self.dark_fuse = ConvBNReLU6(mid_channels + 1, mid_channels, kernel_size=1)
        self.up = ConvBNReLU6(mid_channels, base_channels, kernel_size=1)
        self.refine = GhostESPBNBlock(base_channels)
        self.gain_head = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)
        feat = self.down(skip)
        feat = self.blocks(feat)
        if self.dark_guidance:
            dark = self.dark_map(x)
            dark = F.interpolate(dark, scale_factor=0.5, mode="nearest")
            feat = self.dark_fuse(torch.cat([feat, dark], dim=1))
        feat = F.interpolate(feat, scale_factor=2.0, mode="nearest")
        feat = self.up(feat)
        feat = self.refine(feat + skip)
        gain = self.gain_min + (self.gain_max - self.gain_min) * F.relu6(self.gain_head(feat)) / 6.0
        residual_unit = F.relu6(self.residual_head(feat) + 3.0) / 3.0 - 1.0
        residual = self.residual_scale * residual_unit
        return torch.clamp(x * gain + residual, 0.0, 1.0)

    def fuse_bn_for_export(self) -> nn.Module:
        return fuse_bn_modules(self)


class StudentPConv12(nn.Module):
    """96x96 partial-convolution lite student: 12 channels, 3 blocks, gain/residual output."""

    def __init__(
        self,
        base_channels: int = 12,
        blocks: int = 3,
        pconv_channels: int = 4,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.residual_scale = residual_scale
        self.stem = ConvBNReLU6(3, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[
                PConvLiteBlock(
                    channels=base_channels,
                    pconv_channels=pconv_channels,
                )
                for _ in range(blocks)
            ]
        )
        self.gain_head = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.blocks(feat)
        gain = self.gain_min + (self.gain_max - self.gain_min) * F.relu6(self.gain_head(feat)) / 6.0
        residual_unit = F.relu6(self.residual_head(feat) + 3.0) / 3.0 - 1.0
        residual = self.residual_scale * residual_unit
        return torch.clamp(x * gain + residual, 0.0, 1.0)

    def fuse_bn_for_export(self) -> nn.Module:
        return fuse_bn_modules(self)


class StudentGhostESPDark(nn.Module):
    """Ghost/ESP student with deterministic DarkMap guidance at the bottleneck."""

    def __init__(
        self,
        base_channels: int = 8,
        mid_channels: int = 16,
        blocks: int = 3,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
        dark_threshold: float = 0.15,
        dark_scale: float = 0.70,
        dark_smooth_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.residual_scale = residual_scale
        self.dark_map = DarkMapGenerator(
            threshold=dark_threshold,
            scale=dark_scale,
            smooth_kernel=dark_smooth_kernel,
        )
        self.stem = ConvBNReLU6(3, base_channels, kernel_size=3, padding=1)
        self.down = DepthwiseSeparableBNReLU6(
            base_channels,
            mid_channels,
            stride=2,
        )
        self.blocks = nn.Sequential(
            *[GhostESPBNBlock(mid_channels) for _ in range(blocks)]
        )
        self.dark_fuse = ConvBNReLU6(mid_channels + 1, mid_channels, kernel_size=1)
        self.up = ConvBNReLU6(mid_channels, base_channels, kernel_size=1)
        self.refine = GhostESPBNBlock(base_channels)
        self.gain_head = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dark = self.dark_map(x)
        skip = self.stem(x)
        feat = self.down(skip)
        feat = self.blocks(feat)
        dark_low = F.interpolate(dark, size=feat.shape[-2:], mode="nearest")
        feat = self.dark_fuse(torch.cat([feat, dark_low], dim=1))
        feat = F.interpolate(feat, size=skip.shape[-2:], mode="nearest")
        feat = self.up(feat)
        feat = self.refine(feat + skip)
        gain = self.gain_min + (self.gain_max - self.gain_min) * F.relu6(self.gain_head(feat)) / 6.0
        residual_unit = F.relu6(self.residual_head(feat) + 3.0) / 3.0 - 1.0
        residual = self.residual_scale * residual_unit
        return torch.clamp(x * gain + residual, 0.0, 1.0)

    def compute_dark_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.dark_map(x)

    def fuse_bn_for_export(self) -> nn.Module:
        return fuse_bn_modules(self)
