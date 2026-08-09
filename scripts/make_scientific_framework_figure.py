import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
from pathlib import Path
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures" / "scientific_framework_20260726"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = ROOT / "stm32" / "onnx" / "ghost_esp_dark_w12_m24_enhanced_rgb_u8out_tail_simplified_nchw.onnx"
LOW = ROOT / "data" / "raw" / "LOL" / "val" / "low" / "1.png"
HIGH = ROOT / "data" / "raw" / "LOL" / "val" / "high" / "1.png"

def get_images():
    img_in = Image.open(LOW).convert("RGB").resize((96, 96))
    img_gt = Image.open(HIGH).convert("RGB").resize((96, 96))
    
    # dark map
    arr = np.asarray(img_in).astype(np.float32)
    dm = 255.0 - arr.min(axis=2)
    dm = np.clip((dm - dm.min()) / (dm.max() - dm.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    img_dm = Image.fromarray(dm, "L").convert("RGB")
    
    # student out
    if ONNX_PATH.exists():
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        arr = np.asarray(img_in).astype(np.float32) / 255.0
        x = np.transpose(arr, (2, 0, 1))[None, ...]
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        if out.ndim == 4 and out.shape[1] == 3:
            out = np.transpose(out[0], (1, 2, 0))
        elif out.ndim == 4 and out.shape[-1] == 3:
            out = out[0]
        img_st = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    else:
        img_st = img_in.copy()
        
    return img_in, img_gt, img_dm, img_st

def add_image(ax, img, xy, zoom=1.0, title=""):
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, xy, xycoords='data', frameon=True, bboxprops=dict(edgecolor='gray', linewidth=1))
    ax.add_artist(ab)
    if title:
        ax.text(xy[0], xy[1] - 0.9, title, ha='center', va='top', fontsize=9, fontweight='bold', color='#333333')

def add_box(ax, xy, width, height, title, facecolor, edgecolor, fontsize=10, text_color='black', dash=False):
    ls = '--' if dash else '-'
    rect = patches.Rectangle(xy, width, height, linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, linestyle=ls, zorder=1)
    ax.add_patch(rect)
    # Center text
    ax.text(xy[0] + width/2, xy[1] + height/2, title, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=3)
    return rect

def draw_arrow(ax, start, end, color='black', lw=1.5):
    arrow = patches.FancyArrowPatch(
        start, end, 
        arrowstyle='-|>', 
        mutation_scale=15, 
        color=color, 
        lw=lw, 
        zorder=2
    )
    ax.add_patch(arrow)

def main():
    img_in, img_gt, img_dm, img_st = get_images()

    fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7.5, 9.5, "DarkGhost-ESPNet Training and Deployment Framework", ha='center', va='center', fontsize=16, fontweight='bold', color='#1f497d')

    # ================= TRAINING PHASE =================
    # Training box
    add_box(ax, (0.5, 4.5), 14.0, 4.5, "", "#f8f9fa", "#ced4da", dash=True)
    ax.text(0.7, 8.7, "Training Phase", fontsize=12, fontweight='bold', color='#495057')
    
    # Images
    add_image(ax, img_in, (1.5, 7.5), zoom=0.8, title="INPUT\n(96x96)")
    add_image(ax, img_dm, (1.5, 5.5), zoom=0.6, title="DARK GUIDANCE\n(96x96)")
    add_image(ax, img_st, (8.5, 7.5), zoom=0.8, title="OUTPUT\n(Student)")
    add_image(ax, img_gt, (13.5, 7.5), zoom=0.8, title="OUTPUT\n(Teacher)")
    
    # Student Network Container
    add_box(ax, (3.0, 6.0), 4.0, 2.5, "", "#e8f5e9", "#4caf50")
    ax.text(5.0, 8.2, "STUDENT: DarkGhost-ESPNet", ha='center', va='center', fontsize=11, fontweight='bold', color='#2e7d32')
    
    # Student Blocks
    add_box(ax, (3.2, 6.5), 0.7, 1.2, "Stage 1", "#c8e6c9", "#388e3c", fontsize=9, text_color='#1b5e20')
    add_box(ax, (4.1, 6.6), 0.7, 1.0, "Stage 2", "#c8e6c9", "#388e3c", fontsize=9, text_color='#1b5e20')
    add_box(ax, (5.0, 6.6), 0.7, 1.0, "Stage 3", "#c8e6c9", "#388e3c", fontsize=9, text_color='#1b5e20')
    add_box(ax, (6.0, 6.5), 0.7, 1.2, "Head", "#c8e6c9", "#388e3c", fontsize=9, text_color='#1b5e20')
    
    draw_arrow(ax, (3.9, 7.1), (4.1, 7.1), color="#388e3c")
    draw_arrow(ax, (4.8, 7.1), (5.0, 7.1), color="#388e3c")
    draw_arrow(ax, (5.7, 7.1), (6.0, 7.1), color="#388e3c")
    
    # Feature reuse block
    add_box(ax, (4.1, 5.0), 2.0, 0.7, "Ghost-ESP Bottleneck\nFeature Reuse", "#f1f8e9", "#8bc34a", fontsize=9, text_color='#33691e')
    
    # Teacher Network
    add_box(ax, (10.0, 6.5), 2.5, 2.0, "", "#e3f2fd", "#2196f3")
    ax.text(11.25, 8.2, "KNOWLEDGE TEACHER", ha='center', va='center', fontsize=11, fontweight='bold', color='#1565c0')
    add_box(ax, (10.2, 7.0), 0.6, 1.0, "VD", "#bbdefb", "#1976d2", fontsize=9, text_color='#0d47a1')
    add_box(ax, (11.0, 6.8), 0.6, 1.4, "Retinex", "#bbdefb", "#1976d2", fontsize=9, text_color='#0d47a1')
    add_box(ax, (11.7, 7.0), 0.6, 1.0, "Tail", "#bbdefb", "#1976d2", fontsize=9, text_color='#0d47a1')
    
    # Loss Box
    add_box(ax, (7.5, 4.8), 6.5, 1.2, "", "#ffebee", "#f44336")
    ax.text(10.75, 5.7, "DARK-MAP ADAPTIVE DISTILLATION LOSS", ha='center', va='center', fontsize=11, fontweight='bold', color='#c62828')
    losses = ["Charbonnier", "SSIM", "Perceptual", "Chroma", "Dark W."]
    for i, l in enumerate(losses):
        add_box(ax, (7.7 + i*1.25, 5.0), 1.1, 0.4, l, "#ffffff", "#ef9a9a", fontsize=9, text_color='#b71c1c')
        
    # Arrows for Training
    draw_arrow(ax, (2.3, 7.5), (3.0, 7.5)) # input to student
    draw_arrow(ax, (7.0, 7.5), (7.7, 7.5)) # student to output
    draw_arrow(ax, (12.5, 7.5), (12.7, 7.5)) # teacher to output
    
    # Guidance arrows
    draw_arrow(ax, (2.1, 5.5), (4.1, 5.35), color='#ff9800', lw=2) # dark to feature reuse
    draw_arrow(ax, (2.1, 5.5), (4.1, 6.6), color='#ff9800', lw=2) # dark to stage 2
    
    # Output to Loss
    draw_arrow(ax, (8.5, 6.7), (8.5, 6.0), color='#d32f2f', lw=2)
    draw_arrow(ax, (13.5, 6.7), (12.5, 6.0), color='#d32f2f', lw=2)

    # ================= DEPLOYMENT PHASE =================
    # Deployment box
    add_box(ax, (0.5, 0.5), 14.0, 3.5, "", "#f8f9fa", "#ced4da", dash=True)
    ax.text(0.7, 3.7, "Deployment Phase (STM32)", fontsize=12, fontweight='bold', color='#495057')
    
    # Images
    add_image(ax, img_in, (1.5, 2.0), zoom=0.8, title="Camera Input\n(96x96)")
    add_image(ax, img_st, (8.5, 2.0), zoom=0.8, title="Enhanced RGB\n(LCD Output)")
    
    # Tiny model
    add_box(ax, (3.5, 1.2), 3.5, 1.5, "DarkGhost-ESPNet-Tiny96\nCube.AI INT8/u8 Runtime\n(NO teacher, NO Optuna)", "#e8f5e9", "#4caf50", fontsize=10, text_color='#2e7d32')
    
    # Metrics
    ax.text(11.5, 2.0, "Measured Board Cost:\nInference mean: 170.3 ms/frame\nTotal pipeline: 191.6 ms/frame\nSpeed: ~5 FPS", 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#424242', bbox=dict(facecolor='#f5f5f5', edgecolor='#bdbdbd', boxstyle='round,pad=1'))

    # Arrows for Deployment
    draw_arrow(ax, (2.3, 2.0), (3.5, 2.0))
    draw_arrow(ax, (7.0, 2.0), (7.7, 2.0))
    
    plt.tight_layout()
    out = OUT_DIR / "scientific_framework.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(out)

if __name__ == "__main__":
    main()
