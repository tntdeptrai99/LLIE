from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260726"
OUT_DIR = ROOT / "reports" / "benchmarks" / f"component_ablation_{DATE}"
FIG_DIR = ROOT / "reports" / "figures" / f"component_ablation_{DATE}"


def parse_summary(path: Path) -> dict[str, str]:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def best_from_log(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda row: float(row["val_psnr"]))
    return {
        "epoch": best["epoch"],
        "psnr": best["val_psnr"],
        "ssim": best["val_ssim"],
        "loss": best.get("val_loss", ""),
        "count": "15",
    }


def write_chart(rows: list[dict[str, str]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "component_ablation_effect.png"
    image = Image.new("RGB", (1300, 620), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 28)
        font = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        title_font = font = small = ImageFont.load_default()
    draw.text((42, 28), "Component Ablation: Optuna / Dark-map Loss / Teacher", fill=(15, 43, 80), font=title_font)
    panels = [("psnr", "PSNR", (45, 117, 75)), ("ssim", "SSIM", (43, 94, 160))]
    for panel_idx, (key, label, color) in enumerate(panels):
        x0 = 60 + panel_idx * 610
        y0 = 105
        draw.text((x0, y0), label, fill=(30, 30, 30), font=font)
        values = [float(r[key]) for r in rows]
        max_v = max(values)
        for i, row in enumerate(rows):
            y = y0 + 55 + i * 78
            name = row["variant"]
            bar_x = x0 + 230
            bar_w = int((float(row[key]) / max_v) * 260)
            draw.text((x0, y - 2), name, fill=(20, 20, 20), font=small)
            draw.rectangle((bar_x, y, bar_x + bar_w, y + 25), fill=color)
            draw.text((bar_x + bar_w + 8, y + 3), f"{float(row[key]):.4f}", fill=(20, 20, 20), font=small)
    draw.text(
        (42, 570),
        "Long restored runs and new 30-epoch controlled ablations are reported separately in the CSV notes.",
        fill=(90, 90, 90),
        font=small,
    )
    image.save(out)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = parse_summary(
        ROOT
        / "reports"
        / "metrics"
        / "lol_test_ghost_esp_dark_w12_m24_gain3_res035_96_best_ghost_esp_dark_w12_m24_gain3_res035_96_best_summary.txt"
    )
    optuna = parse_summary(
        ROOT
        / "reports"
        / "metrics"
        / "lol_test_ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_ghost_esp_dark_w12_m24_gain3_res035_optuna_trial011_best_summary.txt"
    )
    plateau = parse_summary(
        ROOT
        / "reports"
        / "metrics"
        / "lol_test_ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_summary.txt"
    )
    no_loss = best_from_log(
        ROOT / "experiments" / "component_ablation" / "no_adaptive_dark_loss_w12m24_20260726" / "train_log.csv"
    )
    no_teacher = best_from_log(
        ROOT / "experiments" / "component_ablation" / "no_teacher_kd_w12m24_20260726" / "train_log.csv"
    )
    rows = [
        {
            "variant": "No Optuna (manual weights)",
            "source_model": base["name"],
            "count": base["count"],
            "epoch": "80",
            "psnr": base["psnr"],
            "ssim": base["ssim"],
            "notes": "Restored long run; uses teacher + adaptive dark-map loss but manual weights.",
        },
        {
            "variant": "+ Optuna trial011",
            "source_model": optuna["name"],
            "count": optuna["count"],
            "epoch": "9+pretrained",
            "psnr": optuna["psnr"],
            "ssim": optuna["ssim"],
            "notes": "Restored Optuna-selected loss weights.",
        },
        {
            "variant": "+ Optuna + long refine / proposed",
            "source_model": plateau["name"],
            "count": plateau["count"],
            "epoch": "plateau/refine",
            "psnr": plateau["psnr"],
            "ssim": plateau["ssim"],
            "notes": "Best restored research checkpoint used as proposed result.",
        },
        {
            "variant": "No adaptive dark-map loss",
            "source_model": "no_adaptive_dark_loss_w12m24_20260726",
            "count": no_loss["count"],
            "epoch": no_loss["epoch"],
            "psnr": no_loss["psnr"],
            "ssim": no_loss["ssim"],
            "notes": "New 30-epoch CUDA controlled ablation; KD teacher kept, adaptive dark-map loss disabled.",
        },
        {
            "variant": "No teacher KD",
            "source_model": "no_teacher_kd_w12m24_20260726",
            "count": no_teacher["count"],
            "epoch": no_teacher["epoch"],
            "psnr": no_teacher["psnr"],
            "ssim": no_teacher["ssim"],
            "notes": "New 30-epoch CUDA controlled ablation; teacher KD disabled, GT/dark-map losses kept.",
        },
    ]
    csv_path = OUT_DIR / "component_ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    fig_path = write_chart(rows)
    md = [
        "# Component ablation: Optuna, adaptive dark-map loss, teacher",
        "",
        "| Variant | N | Epoch/source | PSNR | SSIM | Notes |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['variant']} | {row['count']} | {row['epoch']} | "
            f"{float(row['psnr']):.4f} | {float(row['ssim']):.4f} | {row['notes']} |"
        )
    md.extend(
        [
            "",
            f"![Component ablation]({fig_path.as_posix()})",
            "",
            "Important: the restored long runs and the new 30-epoch ablations are both real artifacts, but not identical training budgets.",
        ]
    )
    md_path = OUT_DIR / "component_ablation_report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)
    print(fig_path)


if __name__ == "__main__":
    main()
