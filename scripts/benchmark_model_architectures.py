from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.student import (  # noqa: E402
    ConvBN,
    ConvBNReLU6,
    DepthwiseSeparableBNReLU6,
    GhostESPBNBlock,
    GhostSepBlock,
    StudentGhostESPDark,
    fuse_bn_modules,
)


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260726"
OUT_DIR = ROOT / "reports" / "benchmarks" / f"architecture_ablation_{DATE}"
FIG_DIR = ROOT / "reports" / "figures" / f"architecture_ablation_{DATE}"
CURRENT_MODEL_BENCH = ROOT / "reports" / "benchmarks" / f"current_model_{DATE}" / "pc_dataset_split_summary.json"
CURRENT_ONNX = ROOT / "stm32" / "onnx" / "ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx"


class FullConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU6(channels, channels, kernel_size=3, padding=1),
            ConvBN(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SepBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableBNReLU6(channels, channels),
            ConvBN(channels, channels, kernel_size=1),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TinyEnhancer(nn.Module):
    def __init__(
        self,
        block_factory,
        base_channels: int = 8,
        mid_channels: int = 16,
        blocks: int = 3,
        dark_guidance: bool = False,
        gain_min: float = 1.0,
        gain_max: float = 2.0,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.dark_guidance = dark_guidance
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.residual_scale = residual_scale
        self.stem = ConvBNReLU6(3, base_channels, kernel_size=3, padding=1)
        self.down = DepthwiseSeparableBNReLU6(base_channels, mid_channels, stride=2)
        self.blocks = nn.Sequential(*[block_factory(mid_channels) for _ in range(blocks)])
        self.dark_fuse = ConvBNReLU6(mid_channels + 1, mid_channels, kernel_size=1)
        self.up = ConvBNReLU6(mid_channels, base_channels, kernel_size=1)
        self.refine = block_factory(base_channels)
        self.gain_head = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.stem(x)
        feat = self.down(skip)
        feat = self.blocks(feat)
        if self.dark_guidance:
            dark = 1.0 - x.mean(dim=1, keepdim=True)
            dark = F.interpolate(dark, size=feat.shape[-2:], mode="nearest")
            feat = self.dark_fuse(torch.cat([feat, dark], dim=1))
        feat = F.interpolate(feat, size=skip.shape[-2:], mode="nearest")
        feat = self.up(feat)
        feat = self.refine(feat + skip)
        gain = self.gain_min + (self.gain_max - self.gain_min) * F.relu6(self.gain_head(feat)) / 6.0
        residual_unit = F.relu6(self.residual_head(feat) + 3.0) / 3.0 - 1.0
        return torch.clamp(x * gain + self.residual_scale * residual_unit, 0.0, 1.0)

    def fuse_bn_for_export(self) -> nn.Module:
        return fuse_bn_modules(self)


@dataclass
class ArchSpec:
    name: str
    model: nn.Module
    quality_source: str
    note: str


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, sample: torch.Tensor) -> int:
    macs = 0
    hooks: list[Any] = []

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        nonlocal macs
        out = output
        if not isinstance(out, torch.Tensor):
            return
        batch, out_channels, out_h, out_w = out.shape
        in_channels = module.in_channels
        groups = module.groups
        kh, kw = module.kernel_size
        macs += int(batch * out_channels * out_h * out_w * (in_channels // groups) * kh * kw)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
    with torch.no_grad():
        model.eval()(sample)
    for hook in hooks:
        hook.remove()
    return macs


@torch.no_grad()
def benchmark_latency(model: nn.Module, sample: torch.Tensor, warmup: int = 20, runs: int = 100) -> dict[str, float]:
    model.eval()
    torch.set_num_threads(1)
    for _ in range(warmup):
        _ = model(sample)
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(sample)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": sorted(times)[int(0.95 * (len(times) - 1))],
    }


def read_current_quality() -> dict[str, str]:
    if not CURRENT_MODEL_BENCH.exists():
        return {}
    data = json.loads(CURRENT_MODEL_BENCH.read_text(encoding="utf-8"))
    preferred = data.get("LOL_v2_Real_val") or next(iter(data.values()))
    return {
        "psnr": f"{float(preferred['psnr_mean']):.4f}",
        "ssim": f"{float(preferred['ssim_mean']):.4f}",
        "mae": f"{float(preferred['mae_mean']):.6f}",
        "quality_split": "LOL_v2_Real_val" if "LOL_v2_Real_val" in data else "first_available",
    }


def write_bar_chart(rows: list[dict[str, str]]) -> Path:
    width, height = 1500, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 28)
        font = ImageFont.truetype("arial.ttf", 17)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font_title = font = font_small = ImageFont.load_default()

    draw.text((42, 26), "Architecture Ablation: Cost Comparison (96x96)", fill=(15, 43, 80), font=font_title)
    metrics = [
        ("params", "Parameters"),
        ("macs", "MACs"),
        ("latency_ms_mean", "CPU PyTorch ms"),
    ]
    colors = [(46, 115, 62), (39, 91, 154), (180, 87, 40)]
    panel_w = 450
    for panel_idx, (key, title) in enumerate(metrics):
        x0 = 46 + panel_idx * (panel_w + 26)
        y0 = 100
        draw.text((x0, y0), title, fill=(30, 30, 30), font=font)
        values = [float(row[key]) for row in rows]
        max_v = max(values) if values else 1.0
        for i, row in enumerate(rows):
            y = y0 + 48 + i * 74
            value = float(row[key])
            bar_x = x0 + 170
            bar_w = int((value / max_v) * 245)
            draw.rectangle((bar_x, y, bar_x + bar_w, y + 25), fill=colors[panel_idx])
            draw.text((x0, y - 2), row["architecture"], fill=(20, 20, 20), font=font_small)
            label = f"{value:,.0f}" if key != "latency_ms_mean" else f"{value:.3f}"
            draw.text((bar_x + bar_w + 6, y + 3), label, fill=(20, 20, 20), font=font_small)
    draw.text(
        (42, height - 52),
        "Quality metrics are only marked measured when a matching trained checkpoint/ONNX artifact exists.",
        fill=(90, 90, 90),
        font=font_small,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "architecture_cost_comparison.png"
    image.save(out)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = torch.rand(1, 3, 96, 96)
    quality = read_current_quality()

    specs = [
        ArchSpec(
            "Conv2D",
            TinyEnhancer(FullConvBlock, dark_guidance=False),
            "missing_checkpoint",
            "Full 3x3 conv residual blocks; cost-only benchmark.",
        ),
        ArchSpec(
            "Separable",
            TinyEnhancer(SepBlock, dark_guidance=False),
            "missing_checkpoint",
            "Depthwise separable residual blocks; cost-only benchmark.",
        ),
        ArchSpec(
            "GhostSep",
            TinyEnhancer(GhostSepBlock, dark_guidance=False),
            "missing_checkpoint",
            "Ghost-style separable block without dark-map guidance; cost-only benchmark.",
        ),
        ArchSpec(
            "Ghost-ESP",
            TinyEnhancer(GhostESPBNBlock, dark_guidance=False),
            "missing_checkpoint",
            "Ghost-ESP block without dark-map guidance; cost-only benchmark.",
        ),
        ArchSpec(
            "DarkGhost-ESPNet",
            StudentGhostESPDark(base_channels=8, mid_channels=16, blocks=3),
            "measured_current_onnx",
            "Current project model family with dark-map guidance; quality from deployed ONNX benchmark.",
        ),
    ]

    rows: list[dict[str, str]] = []
    for spec in specs:
        model = spec.model.eval()
        params = count_params(model)
        macs = estimate_macs(model, sample)
        latency = benchmark_latency(model, sample)
        fp32_size = params * 4
        int8_size = params
        row = {
            "architecture": spec.name,
            "params": str(params),
            "model_size_fp32_bytes": str(fp32_size),
            "model_size_fp32_kib": f"{fp32_size / 1024.0:.2f}",
            "model_size_int8_est_bytes": str(int8_size),
            "model_size_int8_est_kib": f"{int8_size / 1024.0:.2f}",
            "onnx_deploy_size_kib": f"{CURRENT_ONNX.stat().st_size / 1024.0:.2f}" if spec.name == "DarkGhost-ESPNet" and CURRENT_ONNX.exists() else "",
            "macs": str(macs),
            "latency_ms_mean": f"{latency['mean']:.4f}",
            "latency_ms_median": f"{latency['median']:.4f}",
            "latency_ms_p95": f"{latency['p95']:.4f}",
            "psnr": quality.get("psnr", "") if spec.quality_source == "measured_current_onnx" else "",
            "ssim": quality.get("ssim", "") if spec.quality_source == "measured_current_onnx" else "",
            "mae": quality.get("mae", "") if spec.quality_source == "measured_current_onnx" else "",
            "quality_split": quality.get("quality_split", "") if spec.quality_source == "measured_current_onnx" else "",
            "quality_source": spec.quality_source,
            "note": spec.note,
        }
        rows.append(row)

    csv_path = OUT_DIR / "architecture_ablation_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig_path = write_bar_chart(rows)
    md_lines = [
        "# Benchmark kiến trúc model",
        "",
        "Bảng này benchmark theo kiểu ablation kiến trúc tương tự paper. Các chỉ số chi phí được đo trực tiếp bằng PyTorch CPU trên input 1x3x96x96. PSNR/SSIM/MAE chỉ điền cho kiến trúc có artifact đã train/deploy trong project.",
        "",
        f"![Architecture cost comparison]({fig_path.as_posix()})",
        "",
        "| Architecture | Params | FP32 KiB | INT8 est. KiB | MACs | CPU ms | PSNR | SSIM | MAE | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['architecture']} | {int(row['params']):,} | {row['model_size_fp32_kib']} | "
            f"{row['model_size_int8_est_kib']} | {int(row['macs']):,} | {row['latency_ms_mean']} | "
            f"{row['psnr'] or '-'} | {row['ssim'] or '-'} | {row['mae'] or '-'} | {row['quality_source']} |"
        )
    md_lines.extend(
        [
            "",
            "## Ghi chú",
            "",
            "- `INT8 est. KiB` là ước lượng riêng phần weight nếu mỗi parameter lưu 1 byte; file ONNX/Cube.AI thực tế còn thêm graph metadata, scale/zero-point và tensor phụ.",
            "- `DarkGhost-ESPNet` là kiến trúc hiện tại của project; chất lượng lấy từ benchmark ONNX deploy trên split `LOL_v2_Real_val`.",
            "- Các dòng `missing_checkpoint` cần retrain/export cùng seed, cùng split và cùng loss để có PSNR/SSIM/MAE công bằng.",
        ]
    )
    md_path = OUT_DIR / "architecture_ablation_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(csv_path)
    print(md_path)
    print(fig_path)


if __name__ == "__main__":
    main()
