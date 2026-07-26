from __future__ import annotations

import argparse
import csv
import json
import math
import copy
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.utils import make_grid, save_image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_model_architectures import (  # noqa: E402
    FullConvBlock,
    SepBlock,
    TinyEnhancer,
    count_params,
    estimate_macs,
)
from src.data import PairedImageDataset  # noqa: E402
from src.eval.metrics import psnr, ssim_value  # noqa: E402
from src.losses import CharbonnierLoss, DarkMapAdaptiveLoss, EdgeLoss, ChromaLoss, ssim_loss  # noqa: E402
from src.models.student import GhostESPBNBlock, GhostSepBlock, StudentGhostESPDark  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260726"
DEFAULT_OUT = ROOT / "experiments" / "architecture_ablation" / f"unified_{DATE}"
REPORT_OUT = ROOT / "reports" / "benchmarks" / f"architecture_ablation_unified_train_{DATE}"
FIG_OUT = ROOT / "reports" / "figures" / f"architecture_ablation_unified_train_{DATE}"


@dataclass
class TrainConfig:
    seed: int
    epochs: int
    batch_size: int
    image_size: int
    lr: float
    train_limit: int
    val_limit: int
    train_splits: list[str]
    val_splits: list[str]
    loss_profile: str
    ssim_weight: float
    edge_weight: float
    chroma_weight: float
    adaptive_dark_weight: float
    adaptive_dark_base: float
    adaptive_dark_gain: float
    adaptive_dark_gamma: float


def subset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def make_dataset(splits: list[Path], image_size: int, mode: str, augment: bool, limit: int) -> Dataset:
    parts = [
        PairedImageDataset(split, image_size=image_size, mode=mode, augment=augment)
        for split in splits
    ]
    data = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return subset(data, limit)


def build_architecture(name: str) -> torch.nn.Module:
    key = name.lower()
    if key == "conv2d":
        return TinyEnhancer(FullConvBlock, dark_guidance=False)
    if key == "separable":
        return TinyEnhancer(SepBlock, dark_guidance=False)
    if key == "ghostsep":
        return TinyEnhancer(GhostSepBlock, dark_guidance=False)
    if key == "ghost-esp":
        return TinyEnhancer(GhostESPBNBlock, dark_guidance=False)
    if key in {"darkghost-espnet", "darkghost"}:
        return StudentGhostESPDark(base_channels=8, mid_channels=16, blocks=3)
    raise ValueError(f"Unsupported architecture: {name}")


def compute_loss(
    model: torch.nn.Module,
    pred: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    args: argparse.Namespace,
    charbonnier: CharbonnierLoss,
    edge: EdgeLoss,
    chroma: ChromaLoss,
    adaptive: DarkMapAdaptiveLoss,
) -> torch.Tensor:
    loss = charbonnier(pred, high)
    if args.ssim_weight > 0:
        loss = loss + args.ssim_weight * ssim_loss(pred, high)
    if args.edge_weight > 0:
        loss = loss + args.edge_weight * edge(pred, high)
    if args.chroma_weight > 0:
        loss = loss + args.chroma_weight * chroma(pred, high)
    if args.adaptive_dark_weight > 0:
        if hasattr(model, "compute_dark_map"):
            dark_map = model.compute_dark_map(low)
        else:
            dark_map = 1.0 - low.mean(dim=1, keepdim=True)
        dark_score = float(dark_map.detach().mean().item())
        weight = args.adaptive_dark_weight * (args.adaptive_dark_base + args.adaptive_dark_gain * dark_score)
        loss = loss + weight * adaptive(
            pred,
            high,
            low,
            dark_map,
            dark_gamma=args.adaptive_dark_gamma,
        )
    return loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnrs: list[float] = []
    ssims: list[float] = []
    maes: list[float] = []
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        pred = model(low).clamp(0, 1)
        psnrs.append(psnr(pred, high))
        ssims.append(ssim_value(pred, high))
        maes.append(float(torch.mean(torch.abs(pred - high)).item()))
    return {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "mae": sum(maes) / len(maes),
    }


@torch.no_grad()
def save_contact_sheet(model: torch.nn.Module, loader: DataLoader, device: torch.device, out_path: Path, max_samples: int = 4) -> None:
    model.eval()
    tiles: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        pred = model(low).clamp(0, 1)
        for i in range(low.shape[0]):
            tiles.extend([low[i].cpu(), pred[i].cpu(), high[i].cpu()])
            seen += 1
            if seen >= max_samples:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(make_grid(tiles, nrow=3), out_path)
                return


def train_one(
    arch: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, str | int | float]:
    set_seed(args.seed)
    model = build_architecture(arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    charbonnier = CharbonnierLoss().to(device)
    edge = EdgeLoss().to(device)
    chroma = ChromaLoss().to(device)
    adaptive = DarkMapAdaptiveLoss().to(device)

    arch_dir = args.out_dir / arch
    arch_dir.mkdir(parents=True, exist_ok=True)
    log_path = arch_dir / "train_log.csv"
    best_path = arch_dir / "best.pt"
    last_path = arch_dir / "last.pt"

    sample = torch.rand(1, 3, args.image_size, args.image_size)
    params = count_params(model)
    macs = estimate_macs(copy.deepcopy(model).cpu(), sample)
    best_psnr = -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "mae": 0.0}
    epochs_without_improve = 0
    start_all = time.perf_counter()

    with log_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_psnr",
            "val_ssim",
            "val_mae",
            "lr",
            "epoch_seconds",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_start = time.perf_counter()
            losses: list[float] = []
            for batch in train_loader:
                low = batch["low"].to(device)
                high = batch["high"].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(low).clamp(0, 1)
                loss = compute_loss(model, pred, high, low, args, charbonnier, edge, chroma, adaptive)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            metrics = evaluate(model, val_loader, device)
            scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_start
            train_loss = sum(losses) / len(losses)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.8f}",
                    "val_psnr": f"{metrics['psnr']:.6f}",
                    "val_ssim": f"{metrics['ssim']:.8f}",
                    "val_mae": f"{metrics['mae']:.8f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.10f}",
                    "epoch_seconds": f"{epoch_seconds:.3f}",
                }
            )
            f.flush()
            print(
                f"{arch} epoch {epoch:03d}/{args.epochs} "
                f"loss={train_loss:.5f} psnr={metrics['psnr']:.3f} "
                f"ssim={metrics['ssim']:.4f} mae={metrics['mae']:.5f}"
            )
            improved = metrics["psnr"] > best_psnr + args.early_stop_delta
            if improved:
                best_psnr = metrics["psnr"]
                best_epoch = epoch
                best_metrics = metrics
                epochs_without_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "architecture": arch,
                        "config": asdict(config),
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    best_path,
                )
            else:
                epochs_without_improve += 1

            if (
                args.early_stop_patience > 0
                and epoch >= args.min_epochs
                and epochs_without_improve >= args.early_stop_patience
            ):
                print(
                    f"{arch} early_stop epoch={epoch} best_epoch={best_epoch} "
                    f"best_psnr={best_psnr:.3f} patience={args.early_stop_patience}"
                )
                break

    final_metrics = evaluate(model, val_loader, device)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": arch,
            "config": asdict(config),
            "epoch": epoch,
            "metrics": final_metrics,
        },
        last_path,
    )
    save_contact_sheet(model, val_loader, device, args.figure_dir / arch / "contact_sheet.png")

    elapsed = time.perf_counter() - start_all
    return {
        "architecture": arch,
        "params": params,
        "model_size_fp32_kib": params * 4 / 1024.0,
        "model_size_int8_est_kib": params / 1024.0,
        "macs": macs,
        "best_epoch": best_epoch,
        "best_psnr": best_metrics["psnr"],
        "best_ssim": best_metrics["ssim"],
        "best_mae": best_metrics["mae"],
        "last_psnr": final_metrics["psnr"],
        "last_ssim": final_metrics["ssim"],
        "last_mae": final_metrics["mae"],
        "epochs_ran": epoch,
        "seconds_total": elapsed,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
    }


def write_summary(rows: list[dict[str, str | int | float]], args: argparse.Namespace) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / "unified_architecture_train_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Unified architecture baseline training",
        "",
        f"- Epochs: `{args.epochs}`",
        f"- Early stop: min_epochs `{args.min_epochs}`, patience `{args.early_stop_patience}`, delta `{args.early_stop_delta}`",
        f"- Seed: `{args.seed}`",
        f"- Image size: `{args.image_size}`",
        f"- Train limit: `{args.train_limit if args.train_limit > 0 else 'full'}`",
        f"- Val limit: `{args.val_limit if args.val_limit > 0 else 'full'}`",
        f"- Loss profile: `{args.loss_profile}`",
        "",
        "| Architecture | Params | MACs | Best epoch | PSNR | SSIM | MAE | Best checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {int(row['params']):,} | {int(row['macs']):,} | "
            f"{row['best_epoch']}/{row['epochs_ran']} | {float(row['best_psnr']):.4f} | {float(row['best_ssim']):.4f} | "
            f"{float(row['best_mae']):.6f} | `{row['checkpoint_best']}` |"
        )
    lines.append("")
    lines.append("Các model được train trong cùng một lệnh, cùng split, seed, epoch, optimizer và loss profile.")
    (args.report_dir / "unified_architecture_train_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(args.report_dir / "unified_architecture_train_report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", default=["Conv2D", "Separable", "GhostSep", "Ghost-ESP", "DarkGhost-ESPNet"])
    parser.add_argument("--train-splits", nargs="+", type=Path, default=[Path("splits/lol_train.txt"), Path("splits/lolv2_real_train.txt")])
    parser.add_argument("--val-splits", nargs="+", type=Path, default=[Path("splits/lol_val.txt"), Path("splits/lolv2_real_val.txt")])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-delta", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=FIG_OUT)
    parser.add_argument("--loss-profile", choices=["baseline", "darkmap"], default="darkmap")
    parser.add_argument("--ssim-weight", type=float, default=0.10)
    parser.add_argument("--edge-weight", type=float, default=0.02)
    parser.add_argument("--chroma-weight", type=float, default=0.02)
    parser.add_argument("--adaptive-dark-weight", type=float, default=0.04)
    parser.add_argument("--adaptive-dark-base", type=float, default=0.20)
    parser.add_argument("--adaptive-dark-gain", type=float, default=1.20)
    parser.add_argument("--adaptive-dark-gamma", type=float, default=1.20)
    args = parser.parse_args()

    if args.loss_profile == "baseline":
        args.ssim_weight = 0.0
        args.edge_weight = 0.0
        args.chroma_weight = 0.0
        args.adaptive_dark_weight = 0.0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    train_ds = make_dataset(args.train_splits, args.image_size, "train", True, args.train_limit)
    val_ds = make_dataset(args.val_splits, args.image_size, "val", False, args.val_limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        train_splits=[str(p) for p in args.train_splits],
        val_splits=[str(p) for p in args.val_splits],
        loss_profile=args.loss_profile,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        chroma_weight=args.chroma_weight,
        adaptive_dark_weight=args.adaptive_dark_weight,
        adaptive_dark_base=args.adaptive_dark_base,
        adaptive_dark_gain=args.adaptive_dark_gain,
        adaptive_dark_gamma=args.adaptive_dark_gamma,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "unified_train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    rows = []
    for arch in args.architectures:
        rows.append(train_one(arch, args, train_loader, val_loader, device, config))
    write_summary(rows, args)


if __name__ == "__main__":
    main()
