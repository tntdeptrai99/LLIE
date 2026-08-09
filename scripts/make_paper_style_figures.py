from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260807"
OUT = ROOT / "reports" / "figures" / f"paper_style_{DATE}"

TRIALS = ROOT / "experiments" / "optuna" / "ghost_esp_dark_w12_m24_gain3_res035_adaptive_loss" / "trials.csv"
BASE_LOG = ROOT / "experiments" / "component_ablation" / "base_model_w12m24_300e" / "train_log.csv"
REFINE_LOG = ROOT / "experiments" / "component_ablation" / "optuna_best_w12m24_300e" / "train_log.csv"
NO_DARK_LOG = ROOT / "experiments" / "component_ablation" / "no_dark_w12m24_300e" / "train_log.csv"
NO_TEACHER_LOG = ROOT / "experiments" / "component_ablation" / "no_teacher_w12m24_300e" / "train_log.csv"
MAE_BASELINE_LOG = ROOT / "experiments" / "component_ablation" / "mae_baseline_w12m24_300e" / "train_log.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def smooth(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def annotate_last(ax, xs, ys, color: str, text_offset: tuple[float, float] = (35, 30), fmt: str = "{:.4f}") -> None:
    ax.scatter([xs[-1]], [ys[-1]], s=42, color=color, zorder=5)
    
    ax.annotate(fmt.format(ys[-1]), 
                xy=(xs[-1], ys[-1]), 
                xytext=text_offset,
                textcoords='offset points',
                color=color, fontsize=10, fontweight="bold", ha="left" if text_offset[0] >= 0 else "right", va="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=0, shrinkB=4))


def make_optuna_progress() -> Path:
    rows = [r for r in read_csv(TRIALS) if r.get("state") == "COMPLETE"]
    xs = [int(r["number"]) for r in rows]
    values = [float(r["value"]) for r in rows]
    best_idx = int(np.argmax(values))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "optuna_optimization_progress.png"

    plt.figure(figsize=(10.5, 5.2), dpi=160)
    plt.plot(xs, values, "o-", color="blue", linewidth=1.6, markersize=4, label="PSNR + 5 x SSIM")
    plt.scatter([xs[best_idx]], [values[best_idx]], s=80, color="red", zorder=6, label=f"Best trial {xs[best_idx]}")
    plt.title("Optuna Optimization Progress - DarkGhost-ESPNet")
    plt.xlabel("Trial")
    plt.ylabel("Objective score")
    plt.grid(True, alpha=0.45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def make_optuna_params() -> Path:
    rows = [r for r in read_csv(TRIALS) if r.get("state") == "COMPLETE"]
    params = [
        "kd_weight",
        "ssim_weight",
        "teacher_ssim_weight",
        "edge_kd_weight",
        "edge_gt_weight",
        "color_gt_weight",
        "adaptive_dark_weight",
    ]
    best = max(rows, key=lambda r: float(r["value"]))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "optuna_best_loss_weights.png"

    labels = ["KD", "SSIM", "Teacher SSIM", "Edge KD", "Edge GT", "Color GT", "Dark-map"]
    values = [float(best[p]) for p in params]
    plt.figure(figsize=(10.5, 4.8), dpi=160)
    bars = plt.bar(labels, values, color=["#315E9F", "#347A4D", "#8653A5", "#C56A2A", "#C56A2A", "#5D8C88", "#A8323E"])
    plt.title(f"Optuna-selected Loss Weights (Trial {best['number']})")
    plt.ylabel("Weight")
    plt.grid(axis="y", alpha=0.35)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def make_training_curves() -> Path:
    base = read_csv(BASE_LOG)
    refine = read_csv(REFINE_LOG)
    no_dark = read_csv(NO_DARK_LOG)
    no_teacher = read_csv(NO_TEACHER_LOG)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "training_validation_curves_project.png"

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), dpi=160)
    fig.suptitle("Training and Validation Curves - Project Models", fontsize=14, fontweight="bold")

    def plot_log(ax, rows, y_key, label, color, style="-", smoothed=False, epoch_offset=0):
        xs = [int(r["epoch"]) + epoch_offset for r in rows]
        ys = [f(r, y_key) for r in rows]
        if smoothed:
            ys_plot = smooth(ys, 5)
        else:
            ys_plot = ys
        ax.plot(xs, ys_plot, style, color=color, linewidth=1.7, label=label)
        return xs, ys

    ax = axes[0, 0]
    x1, y1 = plot_log(ax, base, "train_loss", "Manual weights - Train", "#4AA3DF")
    plot_log(ax, base, "val_loss", "Manual weights - Val", "#4AA3DF", "--")
    x2, y2 = plot_log(ax, refine, "train_loss", "Optuna + refine - Train", "#FF5A4E", smoothed=True, epoch_offset=0)
    plot_log(ax, refine, "val_loss", "Optuna + refine - Val", "#FF5A4E", "--", smoothed=True, epoch_offset=0)
    annotate_last(ax, x1, y1, "#4AA3DF", text_offset=(20, 25))
    annotate_last(ax, x2, y2, "#FF5A4E", text_offset=(20, -25))
    ax.set_title("Train Loss and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    plot_log(ax, base, "val_psnr", "Manual weights", "#4AA3DF")
    plot_log(ax, refine, "val_psnr", "Optuna + long refine", "#FF5A4E", smoothed=True, epoch_offset=0)
    plot_log(ax, no_dark, "val_psnr", "No adaptive dark-map loss", "#7C55B8")
    plot_log(ax, no_teacher, "val_psnr", "No teacher KD", "#D08C2E")
    # base (x=300), refine (x=222), no_dark (x=300), no_teacher (x=300)
    # Stagger x and y to prevent overlap for the 3 points at x=300
    offsets = [(20, 50), (20, 25), (55, 0), (20, -50)]
    for (rows, color, offset), toff in zip([(base, "#4AA3DF", 0), (refine, "#FF5A4E", 0), (no_dark, "#7C55B8", 0), (no_teacher, "#D08C2E", 0)], offsets):
        xs = [int(r["epoch"]) + offset for r in rows]
        ys = [f(r, "val_psnr") for r in rows]
        annotate_last(ax, xs, ys, color, text_offset=toff)
    ax.set_title("Validation PSNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    plot_log(ax, base, "val_ssim", "Manual weights", "#4AA3DF")
    plot_log(ax, refine, "val_ssim", "Optuna + long refine", "#FF5A4E", smoothed=True, epoch_offset=0)
    plot_log(ax, no_dark, "val_ssim", "No adaptive dark-map loss", "#7C55B8")
    plot_log(ax, no_teacher, "val_ssim", "No teacher KD", "#D08C2E")
    offsets = [(20, 50), (20, 25), (55, 0), (20, -50)]
    for (rows, color, offset), toff in zip([(base, "#4AA3DF", 0), (refine, "#FF5A4E", 0), (no_dark, "#7C55B8", 0), (no_teacher, "#D08C2E", 0)], offsets):
        xs = [int(r["epoch"]) + offset for r in rows]
        ys = [f(r, "val_ssim") for r in rows]
        annotate_last(ax, xs, ys, color, text_offset=toff)
    ax.set_title("Validation SSIM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    teacher_psnr_val = f(base[0], "teacher_psnr")
    max_epoch = int(refine[-1]["epoch"]) if len(refine) > 0 else 300
    ax.plot([1, max_epoch], [teacher_psnr_val, teacher_psnr_val], "--", color="#222222", linewidth=1.7, label="Teacher PSNR reference")
    plot_log(ax, base, "val_psnr", "Manual weights", "#4AA3DF")
    plot_log(ax, refine, "val_psnr", "Optuna + long refine", "#FF5A4E", smoothed=True, epoch_offset=0)
    ax.set_title("Student vs Teacher Reference")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out)
    plt.close()
    return out


def make_component_curves() -> Path:
    no_dark = read_csv(NO_DARK_LOG)
    no_teacher = read_csv(NO_TEACHER_LOG)
    mae_baseline = read_csv(MAE_BASELINE_LOG)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "component_ablation_training_curves.png"

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=160)
    fig.suptitle("Controlled 30-epoch Component Ablation", fontsize=13, fontweight="bold")
    for ax, key, title, ylabel in [
        (axes[0], "val_loss", "Validation Loss", "Loss"),
        (axes[1], "val_psnr", "Validation PSNR", "PSNR (dB)"),
        (axes[2], "val_ssim", "Validation SSIM", "SSIM"),
    ]:
        offsets = [(20, 45), (55, 0), (20, -45)]
        for (rows, label, color), toff in zip([
            (mae_baseline, "MAE Baseline (Charbonnier only)", "#4AA3DF"),
            (no_dark, "No adaptive dark-map loss", "#7C55B8"),
            (no_teacher, "No teacher KD", "#D08C2E"),
        ], offsets):
            xs = [int(r["epoch"]) for r in rows]
            ys = [f(r, key) for r in rows]
            ax.plot(xs, ys, "o-", markersize=2.8, linewidth=1.5, label=label, color=color)
            annotate_last(ax, xs, ys, color, text_offset=toff)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out)
    plt.close()
    return out


def make_mae_vs_hybrid_curves() -> Path:
    mae = read_csv(MAE_BASELINE_LOG)
    hybrid = read_csv(REFINE_LOG)  # Use Optuna best as the Hybrid proposed
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "mae_vs_hybrid_comparison.png"

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=160)
    fig.suptitle("Comparison: MAE Loss vs Proposed Hybrid Loss", fontsize=14, fontweight="bold")

    def plot_curve(ax, rows, y_key, label, color, style="-", smoothed=False):
        xs = [int(r["epoch"]) for r in rows]
        ys = [f(r, y_key) for r in rows]
        ys_plot = smooth(ys, 5) if smoothed else ys
        ax.plot(xs, ys_plot, style, color=color, linewidth=1.7, label=label)
        return xs, ys

    # 1. Loss
    ax = axes[0]
    plot_curve(ax, mae, "train_loss", "MAE Loss - Train", "#4AA3DF")
    x1, y1 = plot_curve(ax, mae, "val_loss", "MAE Loss - Val", "#4AA3DF", "--")
    plot_curve(ax, hybrid, "train_loss", "Hybrid Loss - Train", "#FF5A4E", smoothed=True)
    x2, y2 = plot_curve(ax, hybrid, "val_loss", "Hybrid Loss - Val", "#FF5A4E", "--", smoothed=True)
    annotate_last(ax, x1, y1, "#4AA3DF", text_offset=(20, -30))
    annotate_last(ax, x2, y2, "#FF5A4E", text_offset=(20, 30))
    ax.set_title("Train & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    # 2. SSIM
    ax = axes[1]
    x1, y1 = plot_curve(ax, mae, "val_ssim", "MAE Loss - Val SSIM", "#4AA3DF")
    x2, y2 = plot_curve(ax, hybrid, "val_ssim", "Hybrid Loss - Val SSIM", "#FF5A4E", smoothed=True)
    annotate_last(ax, x1, y1, "#4AA3DF", text_offset=(20, -35))
    annotate_last(ax, x2, y2, "#FF5A4E", text_offset=(20, 35))
    ax.set_title("Validation SSIM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM Score")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    # 3. PSNR
    ax = axes[2]
    x1, y1 = plot_curve(ax, mae, "val_psnr", "MAE Loss - Val PSNR", "#4AA3DF")
    x2, y2 = plot_curve(ax, hybrid, "val_psnr", "Hybrid Loss - Val PSNR", "#FF5A4E", smoothed=True)
    annotate_last(ax, x1, y1, "#4AA3DF", text_offset=(20, -35))
    annotate_last(ax, x2, y2, "#FF5A4E", text_offset=(20, 35))
    ax.set_title("Validation PSNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out)
    plt.close()
    return out


def main() -> None:
    paths = [
        make_optuna_progress(),
        make_optuna_params(),
        make_training_curves(),
        make_component_curves(),
        make_mae_vs_hybrid_curves(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
