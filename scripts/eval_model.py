from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import PairedImageDataset
from src.eval.metrics import psnr, ssim_value
from src.models import StudentA1, StudentB, StudentC, StudentD96, StudentD96TinyBN, StudentE, StudentGhostESPDark, StudentPConv12


def make_subset(dataset: PairedImageDataset, limit: int | None) -> PairedImageDataset | Subset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def build_model(
    model_variant: str,
    gain_min: float,
    gain_max: float,
    residual_scale: float,
    detail_blocks: int,
    detail_scale: float,
    e_blocks: int,
    dark_modulation: bool | None,
    base_channels: int,
    mid_channels: int,
) -> torch.nn.Module:
    variant = model_variant.upper()
    if variant == "A1" or variant == "A2":
        return StudentA1(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )
    if variant == "B":
        return StudentB(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )
    if variant == "C":
        return StudentC(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
            detail_blocks=detail_blocks,
            detail_scale=detail_scale,
        )
    if variant in {"E", "E0", "E1", "E2"}:
        use_dark_modulation = variant in {"E", "E1", "E2"}
        if dark_modulation is not None:
            use_dark_modulation = dark_modulation
        return StudentE(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
            blocks=e_blocks,
            dark_modulation=use_dark_modulation,
        )
    if variant in {"D", "D96", "D-96"}:
        return StudentD96(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
            blocks=e_blocks,
            dark_guidance=True if dark_modulation is None else bool(dark_modulation),
        )
    if variant in {"D96TINY", "D96-TINY", "D96-TINY-BN"}:
        return StudentD96TinyBN(
            base_channels=base_channels,
            mid_channels=mid_channels,
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
            blocks=e_blocks,
            dark_guidance=True if dark_modulation is None else bool(dark_modulation),
        )
    if variant in {"PCONV12", "STUDENTPCONV12", "DSCONV-LITE-PCONV"}:
        return StudentPConv12(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
            blocks=e_blocks,
        )
    if variant in {"GHOST-ESP-DARK", "GHOSTESPDARK", "D96-GHOST-ESP-DARK"}:
        return StudentGhostESPDark(
            base_channels=base_channels,
            mid_channels=mid_channels,
            blocks=e_blocks,
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unsupported model variant: {model_variant}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/a1_128_fast/best.pt"))
    parser.add_argument("--split", type=Path, default=Path("splits/lol_test.txt"))
    parser.add_argument("--name", type=str, default="lol_test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures/a1_128_fast"))
    parser.add_argument("--metric-name", type=str, default="a1_128")
    parser.add_argument("--model-variant", type=str, default=None, choices=["A1", "A2", "B", "C", "D", "D96", "D-96", "D96TINY", "D96-TINY", "D96-TINY-BN", "PCONV12", "STUDENTPCONV12", "DSCONV-LITE-PCONV", "GHOST-ESP-DARK", "GHOSTESPDARK", "D96-GHOST-ESP-DARK", "E", "E0", "E1", "E2"])
    parser.add_argument("--gain-min", type=float, default=None)
    parser.add_argument("--gain-max", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--detail-blocks", type=int, default=None)
    parser.add_argument("--detail-scale", type=float, default=None)
    parser.add_argument("--e-blocks", type=int, default=None)
    parser.add_argument("--dark-modulation", type=int, choices=[0, 1], default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--mid-channels", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedImageDataset(args.split, image_size=args.image_size, mode="test")
    dataset = make_subset(dataset, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_variant = args.model_variant or config.get("model_variant", config.get("student_variant", "A1"))
    gain_min = args.gain_min if args.gain_min is not None else float(config.get("gain_min", 1.0))
    gain_max = args.gain_max if args.gain_max is not None else float(config.get("gain_max", 2.0))
    residual_scale = (
        args.residual_scale
        if args.residual_scale is not None
        else float(config.get("residual_scale", 0.2))
    )
    detail_blocks = (
        args.detail_blocks
        if args.detail_blocks is not None
        else int(config.get("detail_blocks", 2))
    )
    detail_scale = (
        args.detail_scale
        if args.detail_scale is not None
        else float(config.get("detail_scale", 0.08))
    )
    e_blocks = args.e_blocks if args.e_blocks is not None else int(config.get("e_blocks", 4))
    base_channels = (
        args.base_channels
        if args.base_channels is not None
        else int(config.get("base_channels", 8))
    )
    mid_channels = (
        args.mid_channels
        if args.mid_channels is not None
        else int(config.get("mid_channels", 16))
    )
    dark_modulation = (
        bool(args.dark_modulation)
        if args.dark_modulation is not None
        else (
            bool(int(config["dark_modulation"]))
            if "dark_modulation" in config and str(config["dark_modulation"]) in {"0", "1"}
            else None
        )
    )
    model = build_model(
        model_variant,
        gain_min,
        gain_max,
        residual_scale,
        detail_blocks,
        detail_scale,
        e_blocks,
        dark_modulation,
        base_channels,
        mid_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    saved = 0

    for batch_idx, batch in enumerate(loader):
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        pred = model(low).clamp(0.0, 1.0)
        for item_idx in range(low.shape[0]):
            sample_psnr = psnr(pred[item_idx : item_idx + 1], high[item_idx : item_idx + 1])
            sample_ssim = ssim_value(pred[item_idx : item_idx + 1], high[item_idx : item_idx + 1])
            rows.append({"index": len(rows), "psnr": sample_psnr, "ssim": sample_ssim})
            if saved < args.save_images:
                grid = torch.stack(
                    [
                        low[item_idx].cpu(),
                        pred[item_idx].cpu(),
                        high[item_idx].cpu(),
                    ],
                    dim=0,
                )
                save_image(grid, args.figure_dir / f"{args.name}_{saved:03d}.png", nrow=3)
                saved += 1

    metric_path = args.out_dir / f"{args.name}_{args.metric_name}.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    mean_psnr = sum(float(row["psnr"]) for row in rows) / len(rows)
    mean_ssim = sum(float(row["ssim"]) for row in rows) / len(rows)
    summary_path = args.out_dir / f"{args.name}_{args.metric_name}_summary.txt"
    summary_path.write_text(
        f"name={args.name}\ncount={len(rows)}\npsnr={mean_psnr:.4f}\nssim={mean_ssim:.6f}\n",
        encoding="utf-8",
    )
    print(f"{args.name}: count={len(rows)} psnr={mean_psnr:.2f} ssim={mean_ssim:.4f}")
    print(f"metrics={metric_path}")
    print(f"figures={args.figure_dir}")


if __name__ == "__main__":
    main()
