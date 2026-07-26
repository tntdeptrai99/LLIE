from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260726"
CSV = ROOT / "reports" / "benchmarks" / f"architecture_ablation_unified_train_{DATE}" / "unified_architecture_train_summary.csv"
OUT_DIR = ROOT / "reports" / "figures" / f"architecture_ablation_unified_train_{DATE}"


def read_rows() -> list[dict[str, str]]:
    with CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = read_rows()
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 30)
        font = ImageFont.truetype("arial.ttf", 17)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font_title = font = font_small = ImageFont.load_default()

    draw.text((42, 26), "Unified Architecture Training Ablation (10 epochs, 96x96)", fill=(15, 43, 80), font=font_title)
    panels = [
        ("best_psnr", "PSNR higher is better", (44, 118, 87), False),
        ("best_ssim", "SSIM higher is better", (42, 95, 160), False),
        ("best_mae", "MAE lower is better", (184, 86, 46), True),
    ]
    panel_w = 450
    for panel_idx, (key, title, color, lower_better) in enumerate(panels):
        x0 = 46 + panel_idx * (panel_w + 26)
        y0 = 102
        draw.text((x0, y0), title, fill=(30, 30, 30), font=font)
        values = [float(row[key]) for row in rows]
        max_v = max(values)
        min_v = min(values)
        for i, row in enumerate(rows):
            y = y0 + 52 + i * 76
            value = float(row[key])
            if lower_better:
                normalized = min_v / value if value > 0 else 1.0
            else:
                normalized = value / max_v if max_v > 0 else 0.0
            bar_x = x0 + 170
            bar_w = int(normalized * 245)
            draw.rectangle((bar_x, y, bar_x + bar_w, y + 25), fill=color)
            draw.text((x0, y - 2), row["architecture"], fill=(20, 20, 20), font=font_small)
            label = f"{value:.4f}" if key != "best_mae" else f"{value:.6f}"
            draw.text((bar_x + bar_w + 6, y + 3), label, fill=(20, 20, 20), font=font_small)

    draw.text(
        (42, height - 58),
        "All baselines used the same split, seed, optimizer, 10 epochs and dark-map loss profile. This is a synchronized short ablation run, not the long refined final model.",
        fill=(90, 90, 90),
        font=font_small,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "unified_architecture_quality_comparison.png"
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
