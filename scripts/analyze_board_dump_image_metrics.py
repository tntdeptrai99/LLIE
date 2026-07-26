from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compare_board_tensor_dump import parse_dump, raw_to_nchw


WIDTH = 96
HEIGHT = 96
CHANNELS = 3


def nchw_to_rgb(nchw: np.ndarray) -> np.ndarray:
    return np.clip(nchw, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]


def rgb_to_hsv_saturation(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    maxc = values.max(axis=2)
    minc = values.min(axis=2)
    sat = np.zeros_like(maxc)
    nonzero = maxc > 0.0
    sat[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
    return sat


def laplacian_abs_mean(gray: np.ndarray) -> float:
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.mean(np.abs(lap)))


def laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap))


def image_metrics(name: str, rgb: np.ndarray) -> dict[str, float | str]:
    gray = rgb_to_gray(rgb)
    sat = rgb_to_hsv_saturation(rgb)
    flat = rgb.reshape(-1, 3)
    gray_flat = gray.reshape(-1)
    return {
        "name": name,
        "brightness_mean": float(gray.mean()),
        "brightness_median": float(np.median(gray)),
        "contrast_std": float(gray.std()),
        "contrast_p01_p99": float(np.percentile(gray, 99) - np.percentile(gray, 1)),
        "saturation_mean": float(sat.mean()),
        "saturation_median": float(np.median(sat)),
        "sharpness_laplacian_abs_mean": laplacian_abs_mean(gray),
        "sharpness_laplacian_variance": laplacian_variance(gray),
        "clip_0_ratio_rgb": float(np.mean(flat == 0)),
        "clip_255_ratio_rgb": float(np.mean(flat == 255)),
        "clip_0_ratio_luma": float(np.mean(gray_flat <= 1.0)),
        "clip_255_ratio_luma": float(np.mean(gray_flat >= 254.0)),
        "min_rgb": float(rgb.min()),
        "max_rgb": float(rgb.max()),
    }


def save_histogram(hist_in: np.ndarray, hist_out: np.ndarray, path: Path) -> None:
    width, height = 768, 360
    margin_l, margin_r, margin_t, margin_b = 54, 20, 24, 42
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [margin_l, margin_t, margin_l + plot_w, margin_t + plot_h],
        outline=(40, 40, 40),
    )
    max_count = max(float(hist_in.max()), float(hist_out.max()), 1.0)

    def points(hist: np.ndarray) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        for i, count in enumerate(hist):
            x = margin_l + int(i * plot_w / 255)
            y = margin_t + plot_h - int(float(count) * plot_h / max_count)
            pts.append((x, y))
        return pts

    draw.line(points(hist_in), fill=(45, 96, 190), width=2)
    draw.line(points(hist_out), fill=(205, 69, 55), width=2)
    draw.text((margin_l, 4), "Luma histogram: input blue, AI output red", fill=(20, 20, 20))
    draw.text((margin_l, height - 28), "0", fill=(20, 20, 20))
    draw.text((margin_l + plot_w - 18, height - 28), "255", fill=(20, 20, 20))
    draw.text((8, margin_t), f"max {int(max_count)}", fill=(20, 20, 20))
    image.save(path)


def save_contact_sheet(input_rgb: np.ndarray, output_rgb: np.ndarray, path: Path, scale: int = 4) -> None:
    gap = 12
    title_h = 22
    tile_w = WIDTH * scale
    tile_h = HEIGHT * scale
    canvas = Image.new("RGB", (tile_w * 2 + gap, tile_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    in_img = Image.fromarray(input_rgb).resize((tile_w, tile_h), Image.Resampling.NEAREST)
    out_img = Image.fromarray(output_rgb).resize((tile_w, tile_h), Image.Resampling.NEAREST)
    draw.text((4, 4), "input/preprocess", fill=(20, 20, 20))
    draw.text((tile_w + gap + 4, 4), "AI output", fill=(20, 20, 20))
    canvas.paste(in_img, (0, title_h))
    canvas.paste(out_img, (tile_w + gap, title_h))
    canvas.save(path)


def write_markdown(metrics: list[dict[str, float | str]], path: Path, hist_path: Path, contact_path: Path) -> None:
    fields = [
        "brightness_mean",
        "contrast_std",
        "contrast_p01_p99",
        "saturation_mean",
        "sharpness_laplacian_abs_mean",
        "sharpness_laplacian_variance",
        "clip_0_ratio_rgb",
        "clip_255_ratio_rgb",
    ]
    lines = ["# Board Frame Image Metrics", ""]
    lines.append(f"Contact sheet: `{contact_path}`")
    lines.append(f"Histogram: `{hist_path}`")
    lines.append("")
    lines.append("| metric | input/preprocess | AI output |")
    lines.append("|---|---:|---:|")
    input_m, output_m = metrics
    for field in fields:
        lines.append(f"| {field} | {float(input_m[field]):.6f} | {float(output_m[field]):.6f} |")
    lines.append("")
    lines.append("Notes: brightness/contrast/sharpness use luma; saturation uses HSV S channel; clipping ratios are fractions of RGB samples.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render STM32 tensor dump and compute image metrics.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-layout", type=int, default=None)
    parser.add_argument("--output-layout", type=int, default=None)
    args = parser.parse_args()

    tensors, meta = parse_dump(args.log)
    input_layout = int(args.input_layout if args.input_layout is not None else meta.get("in_layout", 1))
    output_layout = int(args.output_layout if args.output_layout is not None else meta.get("out_layout", 1))

    input_raw = np.frombuffer(tensors["input_runtime"], dtype=np.uint8)
    output_raw = np.frombuffer(tensors["output0_public"], dtype=np.uint8)
    input_rgb = nchw_to_rgb(raw_to_nchw(input_raw, input_layout))
    output_rgb = nchw_to_rgb(raw_to_nchw(output_raw, output_layout))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.out_dir / "board_input_preprocess.png"
    output_path = args.out_dir / "board_ai_output.png"
    contact_path = args.out_dir / "board_input_vs_ai_output_contact_x4.png"
    hist_path = args.out_dir / "luma_histogram_input_vs_ai_output.png"
    csv_path = args.out_dir / "image_metrics.csv"
    json_path = args.out_dir / "image_metrics.json"
    md_path = args.out_dir / "image_metrics.md"

    Image.fromarray(input_rgb).save(input_path)
    Image.fromarray(output_rgb).save(output_path)
    save_contact_sheet(input_rgb, output_rgb, contact_path)

    input_luma = rgb_to_gray(input_rgb).astype(np.uint8)
    output_luma = rgb_to_gray(output_rgb).astype(np.uint8)
    hist_in = np.bincount(input_luma.reshape(-1), minlength=256)
    hist_out = np.bincount(output_luma.reshape(-1), minlength=256)
    save_histogram(hist_in, hist_out, hist_path)

    metrics = [
        image_metrics("input_preprocess", input_rgb),
        image_metrics("ai_output", output_rgb),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    json_path.write_text(json.dumps({"meta": meta, "metrics": metrics}, indent=2), encoding="utf-8")
    write_markdown(metrics, md_path, hist_path, contact_path)

    print(f"saved_input={input_path}")
    print(f"saved_output={output_path}")
    print(f"saved_contact={contact_path}")
    print(f"saved_histogram={hist_path}")
    print(f"saved_metrics={csv_path}")
    for item in metrics:
        print(
            f"{item['name']}: brightness={float(item['brightness_mean']):.2f} "
            f"contrast_std={float(item['contrast_std']):.2f} "
            f"saturation={float(item['saturation_mean']):.3f} "
            f"lap_abs={float(item['sharpness_laplacian_abs_mean']):.2f} "
            f"clip0={float(item['clip_0_ratio_rgb']):.4f} "
            f"clip255={float(item['clip_255_ratio_rgb']):.4f}"
        )


if __name__ == "__main__":
    main()
