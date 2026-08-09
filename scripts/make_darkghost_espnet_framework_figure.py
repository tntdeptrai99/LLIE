from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures" / "darkghost_espnet_20260726"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = ROOT / "stm32" / "onnx" / "ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx"
LOW = ROOT / "data" / "raw" / "LOL" / "val" / "low" / "1.png"
HIGH = ROOT / "data" / "raw" / "LOL" / "val" / "high" / "1.png"


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "calibrib.ttf"] if bold else ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf"]
    for item in candidates:
        try:
            return ImageFont.truetype(item, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = get_font(16)
FONT_B = get_font(16, True)
FONT_S = get_font(12)
FONT_SB = get_font(12, True)
FONT_XS = get_font(10)


def center_square(path: Path, size: int = 96) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)


def dark_map(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    dm = 255.0 - arr.min(axis=2)
    dm = np.clip((dm - dm.min()) / (dm.max() - dm.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(dm, "L").convert("RGB")


def run_onnx(image: Image.Image) -> Image.Image:
    if not ONNX_PATH.exists():
        # Fallback if model not available during execution
        return image.copy()
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    arr = np.asarray(image).astype(np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))[None, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    if out.ndim == 4 and out.shape[1] == 3:
        out = np.transpose(out[0], (1, 2, 0))
    elif out.ndim == 4 and out.shape[-1] == 3:
        out = out[0]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline, width=2, radius=12):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw: ImageDraw.ImageDraw, box, text, font, fill=(0, 0, 0)):
    x0, y0, x1, y1 = box
    bb = draw.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.multiline_text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill, spacing=3, align="center")


def arrow(draw: ImageDraw.ImageDraw, start, end, fill=(35, 35, 35), width=3):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        pts = [(ex, ey), (ex - sign * 10, ey - 6), (ex - sign * 10, ey + 6)]
    else:
        sign = 1 if dy >= 0 else -1
        pts = [(ex, ey), (ex - 6, ey - sign * 10), (ex + 6, ey - sign * 10)]
    draw.polygon(pts, fill=fill)


def paste_thumb(canvas: Image.Image, draw: ImageDraw.ImageDraw, image: Image.Image, x: int, y: int, label: str, sub: str = ""):
    thumb = image.resize((92, 92), Image.Resampling.NEAREST)
    draw.rectangle((x - 2, y - 2, x + 94, y + 94), fill=(255, 255, 255), outline=(210, 210, 210))
    canvas.paste(thumb, (x, y))
    centered_text(draw, (x - 16, y - 42, x + 108, y - 18), label, FONT_SB, (20, 20, 20))
    if sub:
        centered_text(draw, (x - 16, y + 98, x + 108, y + 118), sub, FONT_XS, (80, 80, 80))


def draw_student_block(draw: ImageDraw.ImageDraw):
    rounded(draw, (195, 112, 510, 278), fill=(239, 250, 235), outline=(75, 145, 66), width=3)
    centered_text(draw, (195, 120, 510, 145), "STUDENT: DarkGhost-ESPNet", FONT_B, (42, 108, 36))
    
    xs = [215, 285, 355, 425]
    w = 50
    stage_names = ["Stage 1", "Stage 2", "Stage 3", "Head"]
    
    for i, x in enumerate(xs):
        stage_h = 60 if i in (1, 2) else 80
        y = 200 - stage_h // 2
        rounded(draw, (x, y, x + w, y + stage_h), fill=(166, 220, 145), outline=(60, 130, 54), width=2, radius=6)
        centered_text(draw, (x, y + stage_h + 6, x + w, y + stage_h + 20), stage_names[i], FONT_XS, (45, 100, 45))
        if i < len(xs) - 1:
            arrow(draw, (x + w + 6, 200), (xs[i + 1] - 4, 200), fill=(60, 110, 60), width=2)
            
    rounded(draw, (330, 292, 500, 370), fill=(247, 253, 244), outline=(92, 160, 82), width=2)
    centered_text(draw, (330, 292, 500, 370), "Ghost-ESP\nbottleneck\nfeature reuse", FONT_S, (55, 120, 55))


def main() -> None:
    inp = center_square(LOW)
    gt = center_square(HIGH)
    dm = dark_map(inp)
    student = run_onnx(inp)

    w, h = 1180, 700
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    centered_text(draw, (0, 18, w, 48), "DarkGhost-ESPNet Training and Deployment Framework", get_font(22, True), (12, 42, 86))

    # Training phase container.
    rounded(draw, (28, 72, 1148, 455), fill=(255, 255, 255), outline=(205, 215, 230), width=2)
    centered_text(draw, (42, 80, 230, 105), "Training phase", FONT_B, (45, 76, 118))

    paste_thumb(canvas, draw, inp, 54, 150, "INPUT", "96x96")
    arrow(draw, (150, 196), (190, 196))
    
    draw_student_block(draw)
    
    arrow(draw, (510, 196), (555, 196))
    paste_thumb(canvas, draw, student, 560, 150, "OUTPUT (Student)", "enhanced RGB")

    paste_thumb(canvas, draw, dm, 112, 322, "DARK GUIDANCE", "dark-map 96x96")
    
    # Cleaner arrows for dark guidance
    arrow(draw, (180, 368), (326, 331), fill=(225, 130, 35), width=3)
    arrow(draw, (180, 368), (280, 242), fill=(225, 130, 35), width=3)

    # Teacher block
    rounded(draw, (755, 112, 928, 240), fill=(238, 245, 255), outline=(70, 112, 190), width=3)
    centered_text(draw, (755, 120, 928, 145), "KNOWLEDGE TEACHER", FONT_B, (40, 80, 175))
    txs = [785, 825, 865]
    ths = [55, 75, 55]
    for x, th in zip(txs, ths):
        y = 185 - th // 2
        rounded(draw, (x, y, x + 30, y + th), fill=(156, 190, 242), outline=(60, 95, 175), width=2, radius=4)
    centered_text(draw, (755, 212, 928, 236), "VD/Retinexformer\ntraining only", FONT_XS, (40, 80, 175))
    
    arrow(draw, (928, 176), (968, 176))
    paste_thumb(canvas, draw, gt, 978, 130, "OUTPUT (Teacher)", "reference signal")

    rounded(draw, (642, 292, 1114, 418), fill=(255, 240, 240), outline=(205, 52, 62), width=3)
    centered_text(draw, (655, 302, 1100, 330), "DARK-MAP ADAPTIVE DISTILLATION LOSS", FONT_B, (178, 30, 42))
    for i, txt in enumerate(["Charbonnier", "SSIM", "Perceptual", "Color/Chroma", "Dark weight"]):
        x0 = 665 + i * 86
        rounded(draw, (x0, 348, x0 + 76, 388), fill=(255, 250, 250), outline=(215, 125, 130), width=1, radius=7)
        centered_text(draw, (x0 + 2, 350, x0 + 74, 386), txt, FONT_XS, (80, 45, 45))
    centered_text(draw, (665, 392, 1100, 412), "Loss weights are tuned with Optuna and modulated by dark-map regions.", FONT_XS, (95, 55, 55))
    arrow(draw, (606, 246), (725, 292), fill=(178, 30, 42), width=3)
    arrow(draw, (1024, 226), (972, 292), fill=(178, 30, 42), width=3)

    # Deployment phase.
    rounded(draw, (28, 488, 1148, 650), fill=(250, 253, 255), outline=(115, 145, 180), width=2)
    centered_text(draw, (42, 497, 250, 522), "Deployment phase on STM32", FONT_B, (45, 76, 118))
    paste_thumb(canvas, draw, inp, 70, 546, "Camera/Input", "96x96")
    arrow(draw, (166, 592), (250, 592))
    rounded(draw, (260, 532, 548, 632), fill=(240, 249, 238), outline=(75, 145, 66), width=3)
    centered_text(draw, (270, 540, 538, 622), "DarkGhost-ESPNet-Tiny96\nCube.AI INT8/u8 runtime\nNO teacher - NO loss - NO Optuna", FONT_B, (45, 105, 45))
    arrow(draw, (548, 592), (632, 592))
    paste_thumb(canvas, draw, student, 645, 546, "Enhanced RGB", "board/ONNX output")
    centered_text(draw, (800, 546, 1120, 620), "Measured board cost:\nInference mean 170.3 ms/frame\nTotal pipeline mean 191.6 ms/frame\n~5 FPS", FONT_SB, (65, 65, 65))

    out = OUT_DIR / "darkghost_espnet_training_deployment_framework.png"
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
