from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw


WIDTH = 96
HEIGHT = 96
CHANNELS = 3


@dataclass
class Pair:
    dataset: str
    name: str
    low: Path
    high: Path


def center_square_resize_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def output_to_hwc(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    if tuple(arr.shape) == (1, CHANNELS, HEIGHT, WIDTH):
        return arr[0].transpose(1, 2, 0)
    if tuple(arr.shape) == (1, HEIGHT, WIDTH, CHANNELS):
        return arr[0]
    if arr.size == WIDTH * HEIGHT * CHANNELS:
        return arr.reshape(CHANNELS, HEIGHT, WIDTH).transpose(1, 2, 0)
    raise ValueError(f"Unsupported output shape: {arr.shape}")


def run_onnx(session: ort.InferenceSession, input_name: str, input_rgb: np.ndarray) -> np.ndarray:
    nchw = input_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return output_to_hwc(session.run(None, {input_name: nchw})[0])


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    x = rgb_to_gray(a).astype(np.float64)
    y = rgb_to_gray(b).astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ux = x.mean()
    uy = y.mean()
    vx = x.var()
    vy = y.var()
    cov = ((x - ux) * (y - uy)).mean()
    return float(((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]


def saturation_mean(rgb: np.ndarray) -> float:
    f = rgb.astype(np.float32) / 255.0
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = np.zeros_like(mx)
    mask = mx > 0
    sat[mask] = (mx[mask] - mn[mask]) / mx[mask]
    return float(sat.mean())


def laplacian_abs(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    lap = -4 * p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return float(np.mean(np.abs(lap)))


def image_stats(prefix: str, img: np.ndarray) -> dict[str, float]:
    gray = rgb_to_gray(img)
    flat = img.reshape(-1, 3)
    return {
        f"{prefix}_brightness": float(gray.mean()),
        f"{prefix}_contrast_std": float(gray.std()),
        f"{prefix}_saturation": saturation_mean(img),
        f"{prefix}_laplacian_abs": laplacian_abs(gray),
        f"{prefix}_clip0_rgb": float(np.mean(flat == 0)),
        f"{prefix}_clip255_rgb": float(np.mean(flat == 255)),
    }


def save_contact(path: Path, input_rgb: np.ndarray, output_rgb: np.ndarray, gt_rgb: np.ndarray, scale: int = 3) -> None:
    gap = 10
    title_h = 22
    tile_w = WIDTH * scale
    tile_h = HEIGHT * scale
    canvas = Image.new("RGB", (tile_w * 3 + gap * 2, tile_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["input", "AI output", "ground truth"]
    images = [input_rgb, output_rgb, gt_rgb]
    for idx, (label, img) in enumerate(zip(labels, images)):
        x = idx * (tile_w + gap)
        draw.text((x + 4, 4), label, fill=(20, 20, 20))
        tile = Image.fromarray(img).resize((tile_w, tile_h), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, title_h))
    canvas.save(path)


def save_histogram(path: Path, input_rgb: np.ndarray, output_rgb: np.ndarray, gt_rgb: np.ndarray) -> None:
    width, height = 768, 360
    ml, mr, mt, mb = 52, 20, 24, 42
    pw, ph = width - ml - mr, height - mt - mb
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([ml, mt, ml + pw, mt + ph], outline=(40, 40, 40))
    hists = [
        np.bincount(rgb_to_gray(input_rgb).astype(np.uint8).ravel(), minlength=256),
        np.bincount(rgb_to_gray(output_rgb).astype(np.uint8).ravel(), minlength=256),
        np.bincount(rgb_to_gray(gt_rgb).astype(np.uint8).ravel(), minlength=256),
    ]
    colors = [(45, 96, 190), (205, 69, 55), (45, 150, 80)]
    mx = max(float(h.max()) for h in hists) or 1.0
    for hist, color in zip(hists, colors):
        pts = []
        for i, c in enumerate(hist):
            x = ml + int(i * pw / 255)
            y = mt + ph - int(float(c) * ph / mx)
            pts.append((x, y))
        draw.line(pts, fill=color, width=2)
    draw.text((ml, 4), "Luma histogram: input blue, AI red, GT green", fill=(20, 20, 20))
    draw.text((ml, height - 28), "0", fill=(20, 20, 20))
    draw.text((ml + pw - 18, height - 28), "255", fill=(20, 20, 20))
    image.save(path)


def collect_pairs(root: Path) -> list[Pair]:
    specs = [
        ("LOL_val", root / "data/raw/LOL/val/low", root / "data/raw/LOL/val/high"),
        ("LOL_v2_Real_val", root / "data/raw/LOL-v2-Real/val/low", root / "data/raw/LOL-v2-Real/val/high"),
    ]
    pairs: list[Pair] = []
    for dataset, low_dir, high_dir in specs:
        if not low_dir.exists() or not high_dir.exists():
            continue
        highs = {p.name: p for p in high_dir.iterdir() if p.is_file()}
        for low in sorted(p for p in low_dir.iterdir() if p.is_file()):
            high = highs.get(low.name)
            if high is None:
                stem = low.stem.replace("low", "normal").replace("Low", "Normal")
                candidates = list(high_dir.glob(stem + low.suffix))
                high = candidates[0] if candidates else None
            if high is not None:
                pairs.append(Pair(dataset, low.stem, low, high))
    return pairs


def benchmark_dataset(root: Path, session: ort.InferenceSession, onnx_path: Path, bench_dir: Path, fig_dir: Path, max_save: int) -> dict[str, dict[str, float]]:
    input_name = session.get_inputs()[0].name
    rows = []
    summary: dict[str, dict[str, float]] = {}
    pairs = collect_pairs(root)
    image_dir = fig_dir / "dataset_quality"
    image_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for idx, pair in enumerate(pairs):
        inp = center_square_resize_rgb(pair.low)
        gt = center_square_resize_rgb(pair.high)
        start = time.perf_counter()
        out = run_onnx(session, input_name, inp)
        infer_ms = (time.perf_counter() - start) * 1000.0
        row = {
            "dataset": pair.dataset,
            "sample": pair.name,
            "input_path": str(pair.low),
            "ground_truth_path": str(pair.high),
            "psnr": psnr(out, gt),
            "ssim": ssim_gray(out, gt),
            "mae": mae(out, gt),
            "inference_ms_pc": infer_ms,
        }
        row.update(image_stats("input", inp))
        row.update(image_stats("output", out))
        row.update(image_stats("gt", gt))
        rows.append(row)
        if idx < max_save:
            sample_dir = image_dir / f"{idx:03d}_{pair.dataset}_{pair.name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(inp).save(sample_dir / "input.png")
            Image.fromarray(out).save(sample_dir / "ai_output.png")
            Image.fromarray(gt).save(sample_dir / "ground_truth.png")
            save_contact(sample_dir / "contact_sheet.png", inp, out, gt)
            save_histogram(sample_dir / "histogram.png", inp, out, gt)
    csv_path = bench_dir / "dataset_quality_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    for dataset in sorted(set(r["dataset"] for r in rows)):
        ds = [r for r in rows if r["dataset"] == dataset]
        summary[dataset] = {
            "n": len(ds),
            "psnr_mean": float(np.mean([r["psnr"] for r in ds])),
            "ssim_mean": float(np.mean([r["ssim"] for r in ds])),
            "mae_mean": float(np.mean([r["mae"] for r in ds])),
            "inference_ms_pc_mean": float(np.mean([r["inference_ms_pc"] for r in ds])),
        }
    (bench_dir / "dataset_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"dataset benchmark: {len(rows)} samples in {time.perf_counter()-t0:.1f}s")
    return summary


def count_onnx_params(model_path: Path) -> int:
    model = onnx.load(model_path)
    total = 0
    for init in model.graph.initializer:
        if any(token in init.name.lower() for token in ["scale", "zero_point"]):
            continue
        n = 1
        for d in init.dims:
            n *= int(d)
        total += n
    return int(total)


def benchmark_architecture(root: Path, onnx_path: Path, session: ort.InferenceSession, bench_dir: Path, dataset_summary: dict[str, dict[str, float]]) -> None:
    rows = []
    model_size_kb = onnx_path.stat().st_size / 1024.0
    params = count_onnx_params(onnx_path)
    lol = dataset_summary.get("LOL_val", {})
    rows.append({
        "model": "Ghost-ESP current deployed ONNX",
        "artifact": str(onnx_path),
        "status": "measured",
        "params": params,
        "model_size_fp32_kb": "",
        "model_size_int8_kb": f"{model_size_kb:.2f}",
        "psnr": f"{lol.get('psnr_mean', float('nan')):.4f}" if lol else "",
        "ssim": f"{lol.get('ssim_mean', float('nan')):.4f}" if lol else "",
        "mae": f"{lol.get('mae_mean', float('nan')):.6f}" if lol else "",
        "inference_time_ms_pc": f"{lol.get('inference_ms_pc_mean', float('nan')):.4f}" if lol else "",
        "note": "Current available deployed model.",
    })
    for name in ["Conv2D", "Separable", "GhostSep", "Ghost-ESP + Distill"]:
        rows.append({
            "model": name,
            "artifact": "",
            "status": "missing_artifact",
            "params": "",
            "model_size_fp32_kb": "",
            "model_size_int8_kb": "",
            "psnr": "",
            "ssim": "",
            "mae": "",
            "inference_time_ms_pc": "",
            "note": "No ONNX/checkpoint with this exact architecture is currently present after cleanup.",
        })
    path = bench_dir / "architecture_benchmark.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_loss_logs(root: Path, bench_dir: Path) -> None:
    rows = []
    for log in sorted((root / "experiments").rglob("train_log.csv")):
        try:
            with log.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
        except Exception:
            continue
        if not data:
            continue
        def num(row: dict[str, str], key: str) -> float:
            try:
                return float(row.get(key, "nan"))
            except ValueError:
                return float("nan")
        best_psnr = max(data, key=lambda r: num(r, "val_psnr"))
        best_ssim = max(data, key=lambda r: num(r, "val_ssim"))
        last = data[-1]
        rel = log.relative_to(root)
        rows.append({
            "experiment": str(rel.parent),
            "log": str(rel),
            "epochs": len(data),
            "best_psnr": num(best_psnr, "val_psnr"),
            "best_psnr_epoch": best_psnr.get("epoch", ""),
            "best_ssim": num(best_ssim, "val_ssim"),
            "best_ssim_epoch": best_ssim.get("epoch", ""),
            "last_val_loss": num(last, "val_loss"),
            "last_val_psnr": num(last, "val_psnr"),
            "last_val_ssim": num(last, "val_ssim"),
            "loss_family_inferred": "adaptive/hybrid/dark-map experiment" if "optuna" in str(rel).lower() or "adaptive" in str(rel).lower() else "baseline/current training run",
        })
    path = bench_dir / "loss_function_benchmark_from_train_logs.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run_pc_board_equivalence(root: Path, onnx_path: Path, bench_dir: Path) -> str:
    log = root / "board_dump_current_real_input_COM3.log"
    out = bench_dir / "pc_board_equivalence.log"
    if not log.exists():
        out.write_text("Missing board_dump_current_real_input_COM3.log\n", encoding="utf-8")
        return "missing_log"
    cmd = [
        "python",
        str(root / "scripts/compare_board_tensor_dump.py"),
        "--log",
        str(log),
        "--onnx",
        str(onnx_path),
    ]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=120)
    out.write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr, encoding="utf-8", errors="replace")
    return "ok" if proc.returncode == 0 else f"failed_returncode_{proc.returncode}"


def copy_visual_board_artifacts(root: Path, bench_dir: Path, fig_dir: Path) -> None:
    src_dir = root / "reports/figures/board_dump_metrics_frame"
    dst_dir = fig_dir / "real_camera_visual"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for item in src_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dst_dir / item.name)
    log = root / "board_dump_metrics_frame_COM3.log"
    if log.exists():
        shutil.copy2(log, bench_dir / "real_camera_visual_uart.log")


def write_report(bench_dir: Path, fig_dir: Path, dataset_summary: dict[str, dict[str, float]], equivalence_status: str) -> None:
    lines = [
        "# Báo cáo benchmark hiện tại",
        "",
        "## Dataset quality",
        "",
        "| Dataset | N | PSNR mean | SSIM mean | MAE mean | PC inference ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in dataset_summary.items():
        lines.append(
            f"| {name} | {int(s['n'])} | {s['psnr_mean']:.4f} | {s['ssim_mean']:.4f} | "
            f"{s['mae_mean']:.6f} | {s['inference_ms_pc_mean']:.4f} |"
        )
    lines += [
        "",
        "Artifact dataset:",
        f"- CSV: `{bench_dir / 'dataset_quality_metrics.csv'}`",
        f"- Ảnh mẫu/contact sheet/histogram: `{fig_dir / 'dataset_quality'}`",
        "",
        "## Architecture benchmark",
        "",
        f"- CSV: `{bench_dir / 'architecture_benchmark.csv'}`",
        "- Chỉ model hiện tại có artifact ONNX để đo. Các kiến trúc Conv2D/Separable/GhostSep/Ghost-ESP+Distill được ghi là thiếu artifact nếu không có checkpoint/ONNX tương ứng.",
        "",
        "## Loss function benchmark",
        "",
        f"- CSV: `{bench_dir / 'loss_function_benchmark_from_train_logs.csv'}`",
        "- Báo cáo này tổng hợp từ các `train_log.csv` còn trong thư mục `experiments`.",
        "",
        "## Quantization và PC-board equivalence",
        "",
        f"- Trạng thái: `{equivalence_status}`",
        f"- Log: `{bench_dir / 'pc_board_equivalence.log'}`",
        "",
        "## Benchmark thị giác thực tế",
        "",
        f"- Artifact: `{fig_dir / 'real_camera_visual'}`",
        "- Dữ liệu lấy từ frame board đã dump: input preprocess, AI output, histogram và metric brightness/contrast/saturation/sharpness/clipping.",
    ]
    (bench_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--name", default="current_pipeline_20260726")
    parser.add_argument("--onnx", type=Path, default=Path("stm32/onnx/ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx"))
    parser.add_argument("--max-save", type=int, default=12)
    args = parser.parse_args()

    root = args.root.resolve()
    bench_dir = root / "reports/benchmarks" / args.name
    fig_dir = root / "reports/figures" / args.name
    bench_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = (root / args.onnx).resolve() if not args.onnx.is_absolute() else args.onnx
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    dataset_summary = benchmark_dataset(root, session, onnx_path, bench_dir, fig_dir, args.max_save)
    benchmark_architecture(root, onnx_path, session, bench_dir, dataset_summary)
    summarize_loss_logs(root, bench_dir)
    equivalence_status = run_pc_board_equivalence(root, onnx_path, bench_dir)
    copy_visual_board_artifacts(root, bench_dir, fig_dir)
    write_report(bench_dir, fig_dir, dataset_summary, equivalence_status)
    print(f"benchmark_dir={bench_dir}")
    print(f"figures_dir={fig_dir}")
    print(f"report={bench_dir / 'report.md'}")


if __name__ == "__main__":
    main()
