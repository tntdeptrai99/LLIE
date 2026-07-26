from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "stm32" / "onnx" / "ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx"
OUT_DIR = ROOT / "reports" / "benchmarks" / "current_model_20260726"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("LOL_train", ROOT / "data" / "raw" / "LOL" / "train" / "low", ROOT / "data" / "raw" / "LOL" / "train" / "high"),
    ("LOL_val", ROOT / "data" / "raw" / "LOL" / "val" / "low", ROOT / "data" / "raw" / "LOL" / "val" / "high"),
    (
        "LOL_v2_Real_train",
        ROOT / "data" / "raw" / "LOL-v2-Real" / "train" / "low",
        ROOT / "data" / "raw" / "LOL-v2-Real" / "train" / "high",
    ),
    (
        "LOL_v2_Real_val",
        ROOT / "data" / "raw" / "LOL-v2-Real" / "val" / "low",
        ROOT / "data" / "raw" / "LOL-v2-Real" / "val" / "high",
    ),
]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def center_square_resize(img: np.ndarray, size: int = 96) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0 : y0 + side, x0 : x0 + side]
    image = Image.fromarray(crop, mode="RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32) / 255.0
    b = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def run_model(sess: ort.InferenceSession, input_name: str, img: np.ndarray) -> tuple[np.ndarray, float]:
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    t0 = time.perf_counter()
    out = sess.run(None, {input_name: x})[0]
    t1 = time.perf_counter()
    if out.ndim == 4 and out.shape[1] == 3:
        out = np.transpose(out[0], (1, 2, 0))
    elif out.ndim == 4 and out.shape[-1] == 3:
        out = out[0]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, (t1 - t0) * 1000.0


def iter_pairs(low_dir: Path, high_dir: Path):
    lows = sorted([p for p in low_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    for low in lows:
        high = high_dir / low.name
        if high.exists():
            yield low, high


def main() -> None:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    rows = []
    summaries = {}
    for dataset, low_dir, high_dir in DATASETS:
        ds_rows = []
        for low, high in iter_pairs(low_dir, high_dir):
            inp = center_square_resize(read_rgb(low))
            gt = center_square_resize(read_rgb(high))
            out, ms = run_model(sess, input_name, inp)
            row = {
                "dataset": dataset,
                "sample": low.stem,
                "psnr": psnr(out, gt),
                "ssim": ssim_gray(out, gt),
                "mae": mae(out, gt),
                "inference_ms_pc": ms,
            }
            rows.append(row)
            ds_rows.append(row)
        summaries[dataset] = {
            "n": len(ds_rows),
            "psnr_mean": float(np.mean([r["psnr"] for r in ds_rows])) if ds_rows else None,
            "ssim_mean": float(np.mean([r["ssim"] for r in ds_rows])) if ds_rows else None,
            "mae_mean": float(np.mean([r["mae"] for r in ds_rows])) if ds_rows else None,
            "inference_ms_pc_mean": float(np.mean([r["inference_ms_pc"] for r in ds_rows])) if ds_rows else None,
        }
        print(dataset, summaries[dataset])

    with (OUT_DIR / "pc_dataset_split_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "sample", "psnr", "ssim", "mae", "inference_ms_pc"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "pc_dataset_split_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
