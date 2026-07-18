from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import DistillImageDataset
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
from src.models import StudentA1, StudentB, StudentD96, StudentD96TinyBN, StudentGhostESPDark, StudentPConv12
from src.utils.seed import set_seed


def luma(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def teacher_aligned(pred: torch.Tensor, high: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher.shape[1] == 1:
        return luma(pred), luma(high)
    return pred, high


def make_subset(dataset: Dataset, limit: int | None) -> Dataset | Subset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def build_student(
    variant: str,
    gain_min: float,
    gain_max: float,
    residual_scale: float,
    blocks: int,
    base_channels: int,
    mid_channels: int,
) -> torch.nn.Module:
    variant = variant.upper()
    if variant == "A1":
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
    if variant in {"D", "D96", "D-96"}:
        return StudentD96(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )
    if variant in {"D96TINY", "D96-TINY", "D96-TINY-BN"}:
        return StudentD96TinyBN(
            base_channels=base_channels,
            mid_channels=mid_channels,
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
            blocks=blocks,
        )
    if variant in {"PCONV12", "STUDENTPCONV12", "DSCONV-LITE-PCONV"}:
        return StudentPConv12(
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
            blocks=blocks,
        )
    if variant in {"GHOST-ESP-DARK", "GHOSTESPDARK", "D96-GHOST-ESP-DARK"}:
        return StudentGhostESPDark(
            base_channels=base_channels,
            mid_channels=mid_channels,
            blocks=blocks,
            gain_min=gain_min,
            gain_max=gain_max,
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unsupported student variant for KD: {variant}")


def load_pretrained(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    print(f"loaded_pretrained={checkpoint_path}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = CharbonnierLoss()
    losses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    teacher_psnrs: list[float] = []
    teacher_ssims: list[float] = []
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        teacher = batch["teacher"].to(device)
        pred = model(low).clamp(0.0, 1.0)
        losses.append(float(loss_fn(pred, high).item()))
        psnrs.append(psnr(pred, high))
        ssims.append(ssim_value(pred, high))
        _, high_for_teacher = teacher_aligned(pred, high, teacher)
        teacher_psnrs.append(psnr(teacher, high_for_teacher))
        teacher_ssims.append(ssim_value(teacher, high_for_teacher))
    return {
        "loss": sum(losses) / len(losses),
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "teacher_psnr": sum(teacher_psnrs) / len(teacher_psnrs),
        "teacher_ssim": sum(teacher_ssims) / len(teacher_ssims),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=Path, nargs="+", required=True)
    parser.add_argument("--val-split", type=Path, nargs="+", required=True)
    parser.add_argument("--student-variant", type=str, default="A1", choices=["A1", "B", "D", "D96", "D-96", "D96TINY", "D96-TINY", "D96-TINY-BN", "PCONV12", "STUDENTPCONV12", "DSCONV-LITE-PCONV", "GHOST-ESP-DARK", "GHOSTESPDARK", "D96-GHOST-ESP-DARK"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/a1_kd"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--gt-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--teacher-ssim-weight", type=float, default=0.1)
    parser.add_argument("--edge-kd-weight", type=float, default=0.03)
    parser.add_argument("--edge-gt-weight", type=float, default=0.0)
    parser.add_argument("--contrast-gt-weight", type=float, default=0.0)
    parser.add_argument("--color-gt-weight", type=float, default=0.0)
    parser.add_argument("--chroma-gt-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-dark-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-dark-base", type=float, default=0.0)
    parser.add_argument("--adaptive-dark-gain", type=float, default=1.0)
    parser.add_argument("--adaptive-dark-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-bright-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-bright-target", type=str, default="low", choices=["low", "target"])
    parser.add_argument("--kd-dark-gain", type=float, default=0.0)
    parser.add_argument("--ssim-dark-gain", type=float, default=0.0)
    parser.add_argument("--gain-min", type=float, default=1.0)
    parser.add_argument("--gain-max", type=float, default=2.0)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--mid-channels", type=int, default=16)
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="none",
        choices=["none", "psnr", "ssim", "score"],
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--score-ssim-scale", type=float, default=5.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_parts = [
        DistillImageDataset(split, image_size=args.image_size, mode="train", augment=True)
        for split in args.train_split
    ]
    val_parts = [
        DistillImageDataset(split, image_size=args.image_size, mode="val", augment=False)
        for split in args.val_split
    ]
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
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

    model = build_student(
        args.student_variant,
        args.gain_min,
        args.gain_max,
        args.residual_scale,
        args.blocks,
        args.base_channels,
        args.mid_channels,
    ).to(device)
    if args.pretrained is not None:
        load_pretrained(model, args.pretrained, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    charbonnier = CharbonnierLoss()
    edge_loss = EdgeLoss().to(device)
    contrast_loss = LocalContrastLoss().to(device)
    color_loss = ColorStatsLoss().to(device)
    chroma_loss = ChromaLoss().to(device)
    lightness_loss = LightnessAwareLoss().to(device)
    adaptive_dark_loss = DarkMapAdaptiveLoss().to(device)
    log_path = args.out_dir / "train_log.csv"

    def monitored_value(val: dict[str, float]) -> float:
        if args.early_stop_metric == "psnr":
            return val["psnr"]
        if args.early_stop_metric == "ssim":
            return val["ssim"]
        if args.early_stop_metric == "score":
            return val["psnr"] + args.score_ssim_scale * val["ssim"]
        return float("-inf")

    print(f"device={device} train={len(train_ds)} val={len(val_ds)}")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_psnr",
                "val_ssim",
                "teacher_psnr",
                "teacher_ssim",
            ],
        )
        writer.writeheader()
        best_psnr = -1.0
        best_ssim = -1.0
        best_monitor = -1.0
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                low = batch["low"].to(device)
                high = batch["high"].to(device)
                teacher = batch["teacher"].to(device)
                pred = model(low)
                pred_for_teacher, _ = teacher_aligned(pred, high, teacher)
                dark_map = (
                    model.compute_dark_map(low)
                    if hasattr(model, "compute_dark_map")
                    else (1.0 - luma(low)).clamp(0.0, 1.0)
                )
                dark_score = dark_map.detach().mean()
                kd_weight = args.kd_weight * (1.0 + args.kd_dark_gain * dark_score)
                ssim_weight = args.ssim_weight * (1.0 + args.ssim_dark_gain * dark_score)
                adaptive_dark_weight = (
                    args.adaptive_dark_weight
                    * (args.adaptive_dark_base + args.adaptive_dark_gain * dark_score)
                )

                gt_charb = charbonnier(pred, high)
                kd_charb = charbonnier(pred_for_teacher, teacher)
                gt_ssim = ssim_loss(pred, high) if args.ssim_weight > 0 else pred.new_tensor(0.0)
                teacher_ssim = (
                    ssim_loss(pred_for_teacher, teacher)
                    if args.teacher_ssim_weight > 0
                    else pred.new_tensor(0.0)
                )
                edge_kd = (
                    edge_loss(pred_for_teacher, teacher)
                    if args.edge_kd_weight > 0
                    else pred.new_tensor(0.0)
                )
                edge_gt = (
                    edge_loss(pred, high)
                    if args.edge_gt_weight > 0
                    else pred.new_tensor(0.0)
                )
                contrast_gt = (
                    contrast_loss(pred, high)
                    if args.contrast_gt_weight > 0
                    else pred.new_tensor(0.0)
                )
                color_gt = (
                    color_loss(pred, high)
                    if args.color_gt_weight > 0
                    else pred.new_tensor(0.0)
                )
                chroma_gt = (
                    chroma_loss(pred, high)
                    if args.chroma_gt_weight > 0
                    else pred.new_tensor(0.0)
                )
                adaptive_dark = (
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

                loss = (
                    args.gt_weight * gt_charb
                    + kd_weight * kd_charb
                    + ssim_weight * gt_ssim
                    + args.teacher_ssim_weight * teacher_ssim
                    + args.edge_kd_weight * edge_kd
                    + args.edge_gt_weight * edge_gt
                    + args.contrast_gt_weight * contrast_gt
                    + args.color_gt_weight * color_gt
                    + args.chroma_gt_weight * chroma_gt
                    + adaptive_dark_weight * adaptive_dark
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
                "teacher_psnr": val["teacher_psnr"],
                "teacher_ssim": val["teacher_ssim"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={val['loss']:.5f} val_psnr={val['psnr']:.2f} "
                f"val_ssim={val['ssim']:.4f} teacher_psnr={val['teacher_psnr']:.2f} "
                f"teacher_ssim={val['teacher_ssim']:.4f}"
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
            if args.early_stop_metric != "none" and args.early_stop_patience > 0:
                monitor = monitored_value(val)
                if monitor > best_monitor + args.early_stop_min_delta:
                    best_monitor = monitor
                    epochs_without_improvement = 0
                    torch.save(checkpoint, args.out_dir / "best_monitor.pt")
                else:
                    epochs_without_improvement += 1
                print(
                    f"monitor_{args.early_stop_metric}={monitor:.6f} "
                    f"best_monitor={best_monitor:.6f} "
                    f"epochs_without_improvement={epochs_without_improvement}"
                )
                if epochs_without_improvement >= args.early_stop_patience:
                    print(
                        "early_stop="
                        f"metric={args.early_stop_metric} "
                        f"patience={args.early_stop_patience} "
                        f"min_delta={args.early_stop_min_delta}"
                    )
                    break


if __name__ == "__main__":
    main()
