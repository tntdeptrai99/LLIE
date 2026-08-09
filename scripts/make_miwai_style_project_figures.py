from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260809"
OUT = ROOT / "reports" / "figures" / f"miwai_style_project_{DATE}"

METRICS = ROOT / "reports" / "metrics"
BENCH = ROOT / "reports" / "benchmarks" / "current_pipeline_20260726"
ABLATION = ROOT / "reports" / "benchmarks" / "component_ablation_20260726" / "component_ablation_summary.csv"
DATASET_QUALITY = ROOT / "reports" / "figures" / "current_pipeline_20260726" / "dataset_quality"

COLORS = {
    "input": "#6B7280",
    "teacher": "#1D4ED8",
    "proposed": "#D97706",
    "old": "#9333EA",
    "ablate_a": "#059669",
    "ablate_b": "#DC2626",
    "ink": "#111827",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_column(path: Path, key: str) -> float:
    rows = read_csv(path)
    return statistics.fmean(float(r[key]) for r in rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def annotate_bars(ax, bars, fmt: str) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["ink"],
        )


def make_quality_comparison() -> Path:
    low_teacher = METRICS / "lol_test_low_teacher_96_baselines.csv"
    rows = read_csv(ABLATION)
    project = {r["variant"]: r for r in rows}

    labels = [
        "Low input",
        "Teacher",
        "D96-Tiny-BN",
        "Manual\nweights",
        "Optuna\ntrial011",
        "Proposed\nrefine",
    ]
    psnr = [
        mean_column(low_teacher, "low_psnr"),
        mean_column(low_teacher, "teacher_psnr"),
        13.3681,
        float(project["No Optuna (manual weights)"]["psnr"]),
        float(project["+ Optuna trial011"]["psnr"]),
        float(project["+ Optuna + long refine / proposed"]["psnr"]),
    ]
    ssim = [
        mean_column(low_teacher, "low_ssim"),
        mean_column(low_teacher, "teacher_ssim"),
        0.698348,
        float(project["No Optuna (manual weights)"]["ssim"]),
        float(project["+ Optuna trial011"]["ssim"]),
        float(project["+ Optuna + long refine / proposed"]["ssim"]),
    ]

    out = OUT / "quantitative_quality_lol96.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), dpi=180)
    fig.suptitle("Quantitative Performance on LOL Test Split at 96 x 96", fontweight="bold")

    bar_colors = [
        COLORS["input"],
        COLORS["teacher"],
        COLORS["old"],
        "#0891B2",
        "#EA580C",
        COLORS["proposed"],
    ]
    x = np.arange(len(labels))

    bars = axes[0].bar(x, psnr, color=bar_colors, edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, max(psnr) * 1.16)
    annotate_bars(axes[0], bars, "{:.2f}")

    bars = axes[1].bar(x, ssim, color=bar_colors, edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("SSIM")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    annotate_bars(axes[1], bars, "{:.3f}")

    fig.text(0.01, 0.01, "Source: project CSV summaries; n=15 for model evaluations.", fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    return out


def make_ablation_summary_table() -> Path:
    rows = read_csv(ABLATION)
    labels = [r["variant"].replace(" / proposed", "\n(proposed)") for r in rows]
    psnr = [float(r["psnr"]) for r in rows]
    ssim = [float(r["ssim"]) for r in rows]

    out = OUT / "component_ablation_summary.png"
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2), dpi=180)
    x = np.arange(len(labels))
    width = 0.36
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width / 2, psnr, width, label="PSNR", color="#0F766E")
    b2 = ax2.bar(x + width / 2, ssim, width, label="SSIM", color="#B45309")
    ax1.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("SSIM")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(max(0, min(psnr) - 1.0), max(psnr) + 1.0)
    ax2.set_ylim(max(0, min(ssim) - 0.08), min(1.0, max(ssim) + 0.08))
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Ablation Summary: Teacher, Dark-map Loss, and Optuna Refinement", fontweight="bold")
    annotate_bars(ax1, b1, "{:.2f}")
    for bar in b2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax1.legend([b1, b2], ["PSNR", "SSIM"], loc="upper left")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def make_deployment_breakdown() -> Path:
    labels = ["Preprocess", "AI inference", "Postprocess", "LCD", "Total"]
    ms = [2.5, 181.5, 1.5, 10.5, 196.5]
    colors = ["#0891B2", "#D97706", "#059669", "#7C3AED", "#374151"]

    out = OUT / "stm32_pipeline_latency_breakdown.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), dpi=180)
    bars = axes[0].bar(labels, ms, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_title("STM32H750 Full Pipeline Latency", fontweight="bold")
    axes[0].set_ylabel("Milliseconds per frame")
    axes[0].grid(axis="y", alpha=0.3)
    annotate_bars(axes[0], bars, "{:.1f}")

    resource_labels = ["Weights", "Runtime code", "Activations", "Runtime RAM"]
    resource_kib = [5.80, 48.36, 415.29, 16.17]
    resource_colors = ["#2563EB", "#9333EA", "#DC2626", "#059669"]
    bars = axes[1].barh(resource_labels, resource_kib, color=resource_colors, edgecolor="black", linewidth=0.4)
    axes[1].set_title("STM32Cube.AI Resource Footprint", fontweight="bold")
    axes[1].set_xlabel("KiB")
    axes[1].grid(axis="x", alpha=0.3)
    for bar in bars:
        axes[1].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}", va="center", fontsize=8)
    axes[1].set_xlim(0, max(resource_kib) * 1.25)

    fig.text(0.01, 0.01, "Latency from confirmed UART range in docs/PERFORMANCE_BASELINE.md; resources from STM32Cube.AI analyze summary.", fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return out


def make_qdq_boxplots() -> Path:
    files = [
        ("Trial011", METRICS / "qdq_drift_trial011.csv"),
        ("Long80 best SSIM", METRICS / "qdq_drift_trial011_long80_best_ssim.csv"),
        ("Plateau monitor", METRICS / "qdq_drift_plateau_score_best_monitor.csv"),
    ]
    rows_by_name = [(name, read_csv(path)) for name, path in files if path.exists()]
    out = OUT / "qdq_fp32_int8_drift_boxplots.png"

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), dpi=180)
    fig.suptitle("FP32 vs QDQ INT8 Drift on LOL Test Samples", fontweight="bold")
    for ax, key, title, ylabel in [
        (axes[0], "mean_abs", "Mean Absolute Difference", "Absolute error"),
        (axes[1], "max_abs", "Maximum Difference", "Absolute error"),
        (axes[2], "psnr_fp32_vs_qdq", "PSNR FP32 vs QDQ", "PSNR (dB)"),
    ]:
        values = [[float(r[key]) for r in rows] for _, rows in rows_by_name]
        ax.boxplot(values, patch_artist=True, labels=[name for name, _ in rows_by_name])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)
    return out


def prepare_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str = "black") -> None:
    lines = text.split("\n")
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = box[1] + (box[3] - box[1] - total_h) / 2
    for line, width, height in zip(lines, line_widths, line_heights):
        x = box[0] + (box[2] - box[0] - width) / 2
        draw.text((x, y), line, fill=fill, font=font)
        y += height + 4


def make_qualitative_grid() -> Path:
    sample_dirs = sorted([p for p in DATASET_QUALITY.iterdir() if p.is_dir()])[:6]
    columns = [("input.png", "Low input"), ("ai_output.png", "DG-GhostESP output"), ("ground_truth.png", "Ground truth")]
    cell = (180, 180)
    header_h = 34
    label_w = 120
    out_img = Image.new("RGB", (label_w + cell[0] * len(columns), header_h + cell[1] * len(sample_dirs)), "white")
    draw = ImageDraw.Draw(out_img)
    font = ImageFont.load_default(size=14)
    small_font = ImageFont.load_default(size=12)

    for i, (_, title) in enumerate(columns):
        box = (label_w + i * cell[0], 0, label_w + (i + 1) * cell[0], header_h)
        draw_centered_text(draw, box, title, font)

    for r, sample_dir in enumerate(sample_dirs):
        y = header_h + r * cell[1]
        draw_centered_text(draw, (0, y, label_w, y + cell[1]), sample_dir.name.replace("_", "\n"), small_font, "#374151")
        for c, (filename, _) in enumerate(columns):
            out_img.paste(prepare_thumb(sample_dir / filename, cell), (label_w + c * cell[0], y))

    out = OUT / "qualitative_lol_examples_grid.png"
    out_img.save(out)
    return out


def make_available_architecture_tradeoff() -> Path:
    rows = read_csv(BENCH / "architecture_benchmark.csv")
    out = OUT / "available_architecture_tradeoff.png"
    labels = [r["model"].replace(" current deployed ONNX", "\ncurrent deployed") for r in rows]
    psnr = [float(r["psnr"]) if r["psnr"] else np.nan for r in rows]
    int8 = [float(r["model_size_int8_kb"]) if r["model_size_int8_kb"] else np.nan for r in rows]
    status = [r["status"] for r in rows]

    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    x = np.arange(len(labels))
    colors = ["#D97706" if s == "measured" else "#D1D5DB" for s in status]
    bars = ax.bar(x, np.nan_to_num(psnr, nan=0.0), color=colors, edgecolor="black", linewidth=0.4)
    ax.set_title("Architecture Benchmark Availability and Measured Quality", fontweight="bold")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(np.nan_to_num(psnr, nan=0.0)) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    for i, bar in enumerate(bars):
        if status[i] == "measured":
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{psnr[i]:.2f} dB\n{int8[i]:.2f} KiB INT8", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.8, "missing\nartifact", ha="center", va="bottom", fontsize=8, color="#4B5563")
    fig.text(0.01, 0.01, "Missing bars are explicit repo state, not inferred results.", fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return out


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#374151", width: int = 3) -> None:
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        points = [(ex, ey), (ex - sign * 10, ey - 6), (ex - sign * 10, ey + 6)]
    else:
        sign = 1 if dy >= 0 else -1
        points = [(ex, ey), (ex - 6, ey - sign * 10), (ex + 6, ey - sign * 10)]
    draw.polygon(points, fill=fill)


def draw_block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    fill: str,
    outline: str,
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=2)
    x0, y0, x1, y1 = box
    draw_centered_text(draw, (x0 + 8, y0 + 10, x1 - 8, y0 + 42), title, font, "#111827")
    draw_centered_text(draw, (x0 + 8, y0 + 46, x1 - 8, y1 - 8), subtitle, small, "#374151")


def make_architecture_block_diagram() -> Path:
    out = OUT / "dark_guided_ghostesp_architecture.png"
    canvas = Image.new("RGB", (1320, 620), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=22)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    xs = [45, 215, 395, 575, 755, 935, 1110]
    y = 155
    w = 145
    h = 118

    draw_centered_text(
        draw,
        (0, 24, 1320, 58),
        "Project Architecture: Dark-Guided GhostESP Student for 96 x 96 LLIE",
        title_font,
        "#111827",
    )
    draw_centered_text(
        draw,
        (0, 60, 1320, 90),
        "Equivalent role to MIWAI Fig. 1: compact residual-block pipeline, adapted to the current STM32H750 project.",
        small,
        "#4B5563",
    )

    blocks = [
        ("RGB input", "96 x 96 x 3\nu8/float normalized", "#E5E7EB", "#6B7280"),
        ("DarkMap", "min-channel darkness\nadaptive guidance", "#FEF3C7", "#D97706"),
        ("Stem", "Conv-BN-ReLU6\n3 -> 12 channels", "#DBEAFE", "#2563EB"),
        ("Downsample", "depthwise separable\n12 -> 24 channels", "#DCFCE7", "#16A34A"),
        ("Bottleneck", "Ghost/ESP block x3\nfeature reuse", "#F3E8FF", "#9333EA"),
        ("Upsample/refine", "nearest upsample\nGhostESP refine", "#FFE4E6", "#E11D48"),
        ("Heads", "gain + residual\nclamp(input*gain+res)", "#FFEDD5", "#EA580C"),
    ]
    for i, (title, subtitle, fill, outline) in enumerate(blocks):
        draw_block(draw, (xs[i], y, xs[i] + w, y + h), title, subtitle, fill, outline, font, small)
        if i < len(blocks) - 1:
            draw_arrow(draw, (xs[i] + w, y + h // 2), (xs[i + 1] - 12, y + h // 2))

    # Show side inputs and training-only teacher branch.
    draw_block(draw, (398, 365, 560, 480), "Retinexformer teacher", "RGB distillation\ntraining only", "#EEF2FF", "#4F46E5", font, small)
    draw_block(draw, (625, 365, 845, 480), "Hybrid objective", "Charbonnier + SSIM\nedge/color + dark weight", "#FFF7ED", "#F97316", font, small)
    draw_block(draw, (910, 365, 1070, 480), "Optuna", "loss-weight search\nbest trial/refine", "#ECFDF5", "#059669", font, small)

    draw_arrow(draw, (480, 365), (650, 276), "#4F46E5", 3)
    draw_arrow(draw, (735, 365), (828, 276), "#F97316", 3)
    draw_arrow(draw, (990, 365), (1000, 276), "#059669", 3)

    draw_block(
        draw,
        (1010, 72, 1275, 148),
        "Deployment",
        "STM32H750 + Cube.AI QDQ INT8\n54.16 KiB flash, 431.46 KiB RAM",
        "#F9FAFB",
        "#9CA3AF",
        small,
        small,
    )
    draw_arrow(draw, (1180, 148), (1180, y - 10), "#6B7280", 2)

    note = "Current best: DG-GhostESP-96 W12/M24/B3, gain_max=3.0, residual_scale=0.35, plateau best_monitor."
    draw_centered_text(draw, (0, 545, 1320, 578), note, small, "#374151")
    canvas.save(out)
    return out


def write_readme(paths: list[Path]) -> Path:
    out = OUT / "README.md"
    rels = [p.relative_to(ROOT).as_posix() for p in paths]
    text = "\n".join(
        [
            "# MIWAI-style project figures",
            "",
            "Generated from the local MIWAI paper structure and this repository's available experiment artifacts.",
            "",
            "Paper figure mapping:",
            "",
            "- Fig. 1 style: compact model block diagram is generated here.",
            "- Fig. 2 style: Optuna optimization progress is already generated by `scripts/make_paper_style_figures.py`.",
            "- Fig. 3 style: MAE vs hybrid training curves are already generated by `scripts/make_paper_style_figures.py`.",
            "- Fig. 4 style: residual/block ablation curves are already generated by `scripts/make_paper_style_figures.py`.",
            "- Fig. 5 style: qualitative low-light/input/output/GT grid is generated here.",
            "- Fig. 6 style: FP32 vs INT8 drift boxplots are generated here.",
            "- Table 1/Table 2 style: quantitative quality and deployment/resource summaries are generated here.",
            "",
            "Generated files:",
            "",
            *[f"- `{rel}`" for rel in rels],
            "",
            "Notes:",
            "",
            "- Architecture baselines marked `missing_artifact` are shown as unavailable rather than filled with inferred values.",
            "- Board latency uses the confirmed UART ranges documented in `docs/PERFORMANCE_BASELINE.md`.",
        ]
    )
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        make_architecture_block_diagram(),
        make_quality_comparison(),
        make_ablation_summary_table(),
        make_deployment_breakdown(),
        make_qdq_boxplots(),
        make_qualitative_grid(),
        make_available_architecture_tradeoff(),
    ]
    paths.append(write_readme(paths))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
