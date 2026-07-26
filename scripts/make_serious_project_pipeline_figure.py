from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "stm32" / "onnx" / "ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx"
OUT_DIR = ROOT / "reports" / "figures" / "current_model_20260726"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_LOW = ROOT / "data" / "raw" / "LOL" / "val" / "low" / "1.png"
SOURCE_HIGH = ROOT / "data" / "raw" / "LOL" / "val" / "high" / "1.png"

MODEL_SIZE = 96
PANEL_W = 176
PANEL_H = 176


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "calibrib.ttf"] if bold else ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def center_square(path: Path, size: int = MODEL_SIZE) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)


def display(image: Image.Image) -> Image.Image:
    return image.resize((PANEL_W, PANEL_H), Image.Resampling.NEAREST)


def luma(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def normalize_gray(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return np.clip((x - x.min()) / (x.max() - x.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)


def gray_to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray, "L").convert("RGB")


def dark_map_weight(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    dark = arr.min(axis=2)
    # High value marks regions that should receive stronger low-light emphasis.
    weight = 255.0 - dark
    return gray_to_rgb(normalize_gray(weight))


def box_blur_gray(gray: np.ndarray, radius: int = 7) -> np.ndarray:
    pad = radius
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += padded[pad + dy : pad + dy + gray.shape[0], pad + dx : pad + dx + gray.shape[1]]
    return out / float((2 * radius + 1) ** 2)


def retinex_illumination(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    illum = box_blur_gray(luma(arr), radius=7)
    return gray_to_rgb(normalize_gray(illum))


def retinex_reflectance_prior(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    illum = box_blur_gray(luma(arr), radius=7)[..., None]
    refl = np.log(arr + 1e-3) - np.log(illum + 1e-3)
    refl = (refl - refl.min(axis=(0, 1), keepdims=True)) / (
        refl.max(axis=(0, 1), keepdims=True) - refl.min(axis=(0, 1), keepdims=True) + 1e-6
    )
    return Image.fromarray(np.clip(refl * 255, 0, 255).astype(np.uint8), "RGB")


def run_onnx(image: Image.Image) -> Image.Image:
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


def add_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, title: str, subtitle: str, image: Image.Image) -> None:
    title_font = font(13, bold=True)
    sub_font = font(10)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    sub_box = draw.textbbox((0, 0), subtitle, font=sub_font)
    draw.text((x + (PANEL_W - (title_box[2] - title_box[0])) // 2, y), title, fill=(18, 32, 48), font=title_font)
    draw.text((x + (PANEL_W - (sub_box[2] - sub_box[0])) // 2, y + 17), subtitle, fill=(90, 90, 90), font=sub_font)
    canvas.paste(display(image), (x, y + 34))


def main() -> None:
    inp = center_square(SOURCE_LOW)
    gt = center_square(SOURCE_HIGH)
    onnx = run_onnx(inp)

    panels = [
        ("Input", "low-light image", inp),
        ("Preprocessed Input", "96x96 RGB, NCHW", inp),
        ("Dark-map Weight", "loss emphasis map", dark_map_weight(inp)),
        ("Retinex Illumination", "estimated light layer", retinex_illumination(inp)),
        ("Retinex Prior", "reflectance cue", retinex_reflectance_prior(inp)),
        ("Ghost-ESP Output", "current ONNX deploy", onnx),
        ("Ground Truth", "paired reference", gt),
    ]

    margin_x = 28
    margin_y = 30
    gap_x = 14
    gap_y = 34
    cols = 4
    rows = 2
    title_area = 34
    width = margin_x * 2 + cols * PANEL_W + (cols - 1) * gap_x
    height = margin_y * 2 + rows * (PANEL_H + title_area) + (rows - 1) * gap_y + 36
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    header = "Current LLIE Method: Ghost-ESP + Dark-map Loss + Retinex Prior + Optuna"
    header_font = font(17, bold=True)
    hb = draw.textbbox((0, 0), header, font=header_font)
    draw.text(((width - (hb[2] - hb[0])) // 2, 8), header, fill=(11, 37, 69), font=header_font)

    for idx, (title, subtitle, image) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = margin_x + col * (PANEL_W + gap_x)
        y = margin_y + 16 + row * (PANEL_H + title_area + gap_y)
        add_panel(canvas, draw, x, y, title, subtitle, image)

    note = "Note: Dark-map/Retinex panels are priors/training signals; Ghost-ESP Output is the actual current ONNX output."
    note_font = font(11)
    nb = draw.textbbox((0, 0), note, font=note_font)
    draw.text(((width - (nb[2] - nb[0])) // 2, height - 28), note, fill=(80, 80, 80), font=note_font)

    out = OUT_DIR / "serious_current_method_ghost_esp_darkmap_retinex_optuna.png"
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
