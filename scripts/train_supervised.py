from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import PairedImageDataset
from src.eval.metrics import psnr, ssim_value
from src.losses import (
    CharbonnierLoss,
    ChromaLoss,
    ColorStatsLoss,
    DarkMapAdaptiveLoss,
    EdgeLoss,
    LightnessAwareLoss,
    LocalContrastLoss,
    ssim_loss,
)
from src.models import StudentA1, StudentB, StudentC, StudentD96, StudentD96TinyBN, StudentE, StudentGhostESPDark, StudentPConv12
from src.utils.seed import set_seed


def make_subset(dataset: Dataset, limit: int | None) -> Dataset | Subset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    loss_fn = CharbonnierLoss()
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        pred = model(low)
        losses.append(float(loss_fn(pred, high).item()))
        psnrs.append(psnr(pred.clamp(0, 1), high))
        ssims.append(ssim_value(pred.clamp(0, 1), high))
    return {
        "loss": sum(losses) / len(losses),
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
    }


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


def load_pretrained(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    print(f"loaded_pretrained={checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-split",
        type=Path,
        nargs="+",
        default=[Path("splits/lol_train.txt"), Path("splits/lolv2_real_train.txt")],
    )
    parser.add_argument(
        "--val-split",
        type=Path,
        nargs="+",
        default=[Path("splits/lol_val.txt"), Path("splits/lolv2_real_val.txt")],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--synthetic-ratio", type=float, default=0.0)
    parser.add_argument("--light-augment-prob", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/a1_128_fast"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--color-weight", type=float, default=0.0)
    parser.add_argument("--chroma-weight", type=float, default=0.0)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument("--lightness-weight", type=float, default=0.0)
    parser.add_argument("--model-variant", type=str, default="A1", choices=["A1", "A2", "B", "C", "D", "D96", "D-96", "D96TINY", "D96-TINY", "D96-TINY-BN", "PCONV12", "STUDENTPCONV12", "DSCONV-LITE-PCONV", "GHOST-ESP-DARK", "GHOSTESPDARK", "D96-GHOST-ESP-DARK", "E", "E0", "E1", "E2"])
    parser.add_argument("--gain-min", type=float, default=1.0)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--detail-blocks", type=int, default=2)
    parser.add_argument("--detail-scale", type=float, default=0.08)
    parser.add_argument("--e-blocks", type=int, default=4)
    parser.add_argument("--dark-modulation", type=int, choices=[0, 1], default=None)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--mid-channels", type=int, default=16)
    parser.add_argument("--adaptive-dark-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-dark-base", type=float, default=0.0)
    parser.add_argument("--adaptive-dark-gain", type=float, default=1.0)
    parser.add_argument("--adaptive-dark-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-bright-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-bright-target", type=str, default="low", choices=["low", "target"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_parts = [
        PairedImageDataset(
            split,
            image_size=args.image_size,
            mode="train",
            augment=True,
            synthetic_ratio=args.synthetic_ratio,
            light_augment_prob=args.light_augment_prob,
        )
        for split in args.train_split
    ]
    train_ds = (
        train_parts[0]
        if len(train_parts) == 1
        else ConcatDataset(train_parts)
    )
    val_parts = [
        PairedImageDataset(split, image_size=args.image_size, mode="val")
        for split in args.val_split
    ]
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    train_ds = make_subset(train_ds, args.train_limit)
    val_ds = make_subset(val_ds, args.val_limit)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model_variant,
        args.gain_min,
        args.gain_max,
        args.residual_scale,
        args.detail_blocks,
        args.detail_scale,
        args.e_blocks,
        None if args.dark_modulation is None else bool(args.dark_modulation),
        args.base_channels,
        args.mid_channels,
    ).to(device)
    if args.pretrained is not None:
        load_pretrained(model, args.pretrained, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    charbonnier_loss = CharbonnierLoss()
    edge_loss = EdgeLoss().to(device)
    color_loss = ColorStatsLoss().to(device)
    chroma_loss = ChromaLoss().to(device)
    contrast_loss = LocalContrastLoss().to(device)
    lightness_loss = LightnessAwareLoss().to(device)
    adaptive_dark_loss = DarkMapAdaptiveLoss().to(device)
    log_path = args.out_dir / "train_log.csv"

    print(f"device={device} train={len(train_ds)} val={len(val_ds)}")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim"]
        )
        writer.writeheader()
        best_psnr = -1.0
        best_ssim = -1.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                low = batch["low"].to(device)
                high = batch["high"].to(device)
                pred = model(low)
                dark_map = (
                    model.compute_dark_map(low)
                    if hasattr(model, "compute_dark_map")
                    else (1.0 - low.mean(dim=1, keepdim=True)).clamp(0.0, 1.0)
                )
                dark_score = dark_map.detach().mean()
                charb = charbonnier_loss(pred, high)
                ssim_component = ssim_loss(pred, high) if args.ssim_weight > 0 else pred.new_tensor(0.0)
                edge_component = edge_loss(pred, high) if args.edge_weight > 0 else pred.new_tensor(0.0)
                color_component = color_loss(pred, high) if args.color_weight > 0 else pred.new_tensor(0.0)
                chroma_component = chroma_loss(pred, high) if args.chroma_weight > 0 else pred.new_tensor(0.0)
                contrast_component = (
                    contrast_loss(pred, high)
                    if args.contrast_weight > 0
                    else pred.new_tensor(0.0)
                )
                lightness_component = (
                    lightness_loss(pred, high, low)
                    if args.lightness_weight > 0
                    else pred.new_tensor(0.0)
                )
                adaptive_dark_component = (
                    adaptive_dark_loss(
                        pred,
                        high,
                        low,
                        dark_map,
                        dark_gamma=args.adaptive_dark_gamma,
                        bright_gamma=args.adaptive_bright_gamma,
                        bright_preserve_target=args.adaptive_bright_target,
                    )
                    if args.adaptive_dark_weight > 0
                    else pred.new_tensor(0.0)
                )
                adaptive_dark_weight = (
                    args.adaptive_dark_weight
                    * (args.adaptive_dark_base + args.adaptive_dark_gain * dark_score)
                )
                loss = (
                    charb
                    + args.ssim_weight * ssim_component
                    + args.edge_weight * edge_component
                    + args.color_weight * color_component
                    + args.chroma_weight * chroma_component
                    + args.contrast_weight * contrast_component
                    + args.lightness_weight * lightness_component
                    + adaptive_dark_weight * adaptive_dark_component
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            val = evaluate(model, val_loader, device)
            train_loss = sum(train_losses) / len(train_losses)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_psnr": val["psnr"],
                "val_ssim": val["ssim"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={val['loss']:.5f} val_psnr={val['psnr']:.2f} "
                f"val_ssim={val['ssim']:.4f}"
            )
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "config": {key: str(value) for key, value in vars(args).items()},
                "val": val,
            }
            torch.save(checkpoint, args.out_dir / "last.pt")
            if val["psnr"] > best_psnr:
                best_psnr = val["psnr"]
                torch.save(checkpoint, args.out_dir / "best.pt")
            if val["ssim"] > best_ssim:
                best_ssim = val["ssim"]
                torch.save(checkpoint, args.out_dir / "best_ssim.pt")


if __name__ == "__main__":
    main()
