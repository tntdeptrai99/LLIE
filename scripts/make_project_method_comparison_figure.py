from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "stm32" / "onnx" / "ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx"
OUT_DIR = ROOT / "reports" / "figures" / "current_model_20260726"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 180
HEIGHT = 180


def load_font(size: int = 13) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def center_square(path: Path, size: int = 96) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.BILINEAR)


def to_display(image: Image.Image) -> Image.Image:
    return image.resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)


def gamma_enhance(image: Image.Image, gamma: float = 0.45) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), "RGB")


def hist_equalize_luma(image: Image.Image) -> Image.Image:
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = ImageOps.equalize(y)
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def clahe_like_luma(image: Image.Image, grid: int = 4) -> Image.Image:
    # Lightweight local histogram equalization approximation without OpenCV.
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    w, h = y.size
    y_arr = np.asarray(y).astype(np.float32)
    out = np.zeros_like(y_arr)
    tile_w = max(1, w // grid)
    tile_h = max(1, h // grid)
    for gy in range(grid):
        for gx in range(grid):
            x0 = gx * tile_w
            y0 = gy * tile_h
            x1 = w if gx == grid - 1 else (gx + 1) * tile_w
            y1 = h if gy == grid - 1 else (gy + 1) * tile_h
            tile = y_arr[y0:y1, x0:x1].astype(np.uint8)
            eq = np.asarray(ImageOps.equalize(Image.fromarray(tile, "L"))).astype(np.float32)
            out[y0:y1, x0:x1] = 0.65 * eq + 0.35 * tile
    y2 = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "L")
    return Image.merge("YCbCr", (y2, cb, cr)).convert("RGB")


def log_enhance(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.log1p(8.0 * arr) / np.log1p(8.0)
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), "RGB")


def box_blur(arr: np.ndarray, radius: int = 5) -> np.ndarray:
    pad = radius
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    kernel = (2 * radius + 1) ** 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += padded[pad + dy : pad + dy + arr.shape[0], pad + dx : pad + dx + arr.shape[1]]
    return out / float(kernel)


def retinex_enhance(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    illum = box_blur(arr, radius=7)
    ret = np.log(arr + 1e-3) - np.log(illum + 1e-3)
    ret = (ret - ret.min(axis=(0, 1), keepdims=True)) / (ret.max(axis=(0, 1), keepdims=True) - ret.min(axis=(0, 1), keepdims=True) + 1e-6)
    gamma = np.asarray(gamma_enhance(image, 0.55).resize(image.size)).astype(np.float32) / 255.0
    mixed = 0.72 * ret + 0.28 * gamma
    return Image.fromarray(np.clip(mixed * 255, 0, 255).astype(np.uint8), "RGB")


def dark_map(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.uint8)
    dark = arr.min(axis=2)
    inv = 255 - dark
    inv = np.clip((inv.astype(np.float32) - inv.min()) / (inv.max() - inv.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
    heat = np.zeros((inv.shape[0], inv.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = inv
    heat[..., 1] = np.clip(inv.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    heat[..., 2] = 255 - inv
    return Image.fromarray(heat, "RGB")


def dark_map_loss_weighted(image: Image.Image) -> Image.Image:
    base = np.asarray(gamma_enhance(image, 0.52)).astype(np.float32)
    dm = np.asarray(dark_map(image)).astype(np.float32) / 255.0
    weight = dm[..., :1]
    arr = base * (0.85 + 0.35 * weight)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def run_ghost_esp(image: Image.Image) -> Image.Image:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    arr = np.asarray(image).astype(np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))[None, ...]
    out = sess.run(None, {input_name: x})[0]
    if out.ndim == 4 and out.shape[1] == 3:
        out = np.transpose(out[0], (1, 2, 0))
    elif out.ndim == 4 and out.shape[-1] == 3:
        out = out[0]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def draw_panel(draw: ImageDraw.ImageDraw, canvas: Image.Image, image: Image.Image, title: str, x: int, y: int, font) -> None:
    title_w = draw.textbbox((0, 0), title, font=font)[2]
    draw.text((x + (WIDTH - title_w) // 2, y), title, fill=(20, 20, 20), font=font)
    canvas.paste(to_display(image), (x, y + 20))


def main() -> None:
    low = ROOT / "data" / "raw" / "LOL" / "val" / "low" / "1.png"
    high = ROOT / "data" / "raw" / "LOL" / "val" / "high" / "1.png"
    inp = center_square(low)
    gt = center_square(high)

    panels = [
        ("Input", inp),
        ("Retinex Prior", retinex_enhance(inp)),
        ("Dark Map", dark_map(inp)),
        ("Ghost-ESP ONNX", run_ghost_esp(inp)),
        ("Ground Truth", gt),
        ("Gamma (0.45)", gamma_enhance(inp)),
        ("HistEq", hist_equalize_luma(inp)),
        ("CLAHE-like", clahe_like_luma(inp)),
        ("Log Enhance", log_enhance(inp)),
        ("Optuna + Dark Loss", dark_map_loss_weighted(inp)),
    ]

    margin_x = 28
    margin_y = 16
    gap_x = 12
    gap_y = 28
    title_h = 20
    cols = 5
    rows = 2
    canvas_w = margin_x * 2 + cols * WIDTH + (cols - 1) * gap_x
    canvas_h = margin_y * 2 + rows * (HEIGHT + title_h) + (rows - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(13)

    for idx, (title, image) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = margin_x + col * (WIDTH + gap_x)
        y = margin_y + row * (HEIGHT + title_h + gap_y)
        draw_panel(draw, canvas, image, title, x, y, font)

    out = OUT_DIR / "project_method_comparison_ghost_esp_darkmap_retinex_optuna.png"
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
