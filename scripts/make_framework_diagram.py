import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

def add_image(ax, img_path, xy, title, zoom=0.6):
    img = cv2.imread(img_path)
    if img is None:
        img = np.ones((96, 96, 3), dtype=np.uint8) * 200
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Center crop to square
        h, w = img.shape[:2]
        c = min(h, w)
        img = img[(h-c)//2:(h+c)//2, (w-c)//2:(w+c)//2]
        img = cv2.resize(img, (96, 96))
        
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=True, pad=0.1, bboxprops=dict(edgecolor='gray'))
    ax.add_artist(ab)
    ax.text(xy[0], xy[1] + 12, title, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    return xy

def add_darkmap(ax, img_path, xy, title, zoom=0.6):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.ones((96, 96), dtype=np.uint8) * 50
    else:
        # Simulate dark map (inverted luminance threshold)
        h, w = img.shape[:2]
        c = min(h, w)
        img = img[(h-c)//2:(h+c)//2, (w-c)//2:(w+c)//2]
        img = cv2.resize(img, (96, 96))
        img = 255 - img
        img = np.clip(img * 1.5, 0, 255).astype(np.uint8)
        
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=True, pad=0.1, bboxprops=dict(edgecolor='gray'))
    ax.add_artist(ab)
    ax.text(xy[0], xy[1] + 12, title, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    return xy

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, text_color='black', fontsize=10, zorder=1):
    box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1", facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, wrap=True, zorder=zorder+1)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

def main():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(80, 98, "DG-GhostESP-96 Training and Deployment Framework", ha='center', va='center', fontsize=18, fontweight='bold', color='#0D47A1')
    
    # Outer Panels
    draw_box(ax, 2, 40, 156, 54, "", "#F8FBFF", "#90CAF9", zorder=0)
    ax.text(5, 91, "Training phase", fontsize=12, fontweight='bold', color='#1565C0')
    
    draw_box(ax, 2, 2, 156, 34, "", "#F8FBFF", "#90CAF9", zorder=0)
    ax.text(5, 33, "Deployment phase on STM32", fontsize=12, fontweight='bold', color='#1565C0')
    
    # --- TRAINING PHASE ---
    low_path = "data/raw/LOL/val/low/1.png"
    high_path = "data/raw/LOL/val/high/1.png"
    
    # Images
    add_image(ax, low_path, (15, 75), "INPUT\n96x96")
    add_darkmap(ax, low_path, (15, 52), "DARK GUIDANCE\ndark-map 96x96")
    add_image(ax, high_path, (75, 75), "OUTPUT (Student)\nenhanced RGB")
    add_image(ax, high_path, (145, 75), "OUTPUT (Teacher)\nreference signal")
    
    # Student Block
    draw_box(ax, 30, 62, 30, 26, "", "#E8F5E9", "#4CAF50", zorder=1)
    ax.text(45, 85, "STUDENT: DG-GhostESP-96", ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32', zorder=2)
    # Student internal blocks
    draw_box(ax, 32, 65, 4, 15, "Stem", "#A5D6A7", "#388E3C", fontsize=8, zorder=2)
    draw_box(ax, 40, 65, 4, 15, "Bot", "#A5D6A7", "#388E3C", fontsize=8, zorder=2)
    draw_box(ax, 48, 65, 4, 15, "Up", "#A5D6A7", "#388E3C", fontsize=8, zorder=2)
    draw_box(ax, 56, 65, 4, 15, "Head", "#A5D6A7", "#388E3C", fontsize=8, zorder=2)
    draw_arrow(ax, 36, 72.5, 40, 72.5, lw=2)
    draw_arrow(ax, 44, 72.5, 48, 72.5, lw=2)
    draw_arrow(ax, 52, 72.5, 56, 72.5, lw=2)
    
    # Student Output arrow
    draw_arrow(ax, 60, 75, 68, 75, lw=2)
    draw_arrow(ax, 23, 75, 30, 75, lw=2)
    
    # Dark Guidance Arrow
    ax.annotate("", xy=(42, 64), xytext=(23, 52), arrowprops=dict(arrowstyle="->", color="#FF8F00", lw=3))
    draw_box(ax, 42, 48, 16, 8, "Ghost-ESP\nbottleneck\nfeature reuse", "#F1F8E9", "#689F38", fontsize=8)
    ax.annotate("", xy=(44, 63), xytext=(50, 56), arrowprops=dict(arrowstyle="->", color="#FF8F00", lw=2, ls='--'))
    
    # Teacher Block
    draw_box(ax, 100, 65, 25, 20, "", "#E3F2FD", "#1976D2", zorder=1)
    ax.text(112.5, 82, "KNOWLEDGE TEACHER", ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0', zorder=2)
    ax.text(112.5, 68, "Retinexformer\n(training only)", ha='center', va='center', fontsize=9, color='#1565C0', zorder=2)
    draw_box(ax, 103, 71, 4, 8, "", "#90CAF9", "#1976D2", zorder=2)
    draw_box(ax, 110.5, 71, 4, 8, "", "#90CAF9", "#1976D2", zorder=2)
    draw_box(ax, 118, 71, 4, 8, "", "#90CAF9", "#1976D2", zorder=2)
    
    draw_arrow(ax, 125, 75, 137, 75, lw=2)
    
    # Loss Block
    draw_box(ax, 85, 42, 65, 15, "", "#FFEBEE", "#D32F2F", zorder=1)
    ax.text(117.5, 54, "DARK-MAP ADAPTIVE DISTILLATION LOSS", ha='center', va='center', fontsize=11, fontweight='bold', color='#C62828', zorder=2)
    draw_box(ax, 87, 46, 11, 4, "Charbonnier", "white", "#E57373", fontsize=8, zorder=2)
    draw_box(ax, 100, 46, 11, 4, "SSIM", "white", "#E57373", fontsize=8, zorder=2)
    draw_box(ax, 113, 46, 11, 4, "Perceptual", "white", "#E57373", fontsize=8, zorder=2)
    draw_box(ax, 126, 46, 11, 4, "Color/Chroma", "white", "#E57373", fontsize=8, zorder=2)
    draw_box(ax, 139, 46, 10, 4, "Dark weight", "white", "#E57373", fontsize=8, zorder=2)
    ax.text(117.5, 43.5, "Loss weights are tuned with Optuna and modulated by dark-map regions.", ha='center', va='center', fontsize=8, color='#C62828', zorder=2)
    
    # Arrows to loss
    draw_arrow(ax, 75, 65, 85, 50, color="#C62828", lw=2)
    draw_arrow(ax, 145, 65, 130, 57, color="#C62828", lw=2)
    
    
    # --- DEPLOYMENT PHASE ---
    add_image(ax, low_path, (15, 18), "Camera Input\n96x96")
    
    # Runtime Block
    draw_box(ax, 35, 10, 50, 18, "DG-GhostESP-96\nCube.AI QDQ INT8 runtime\nNO teacher - NO loss - NO Optuna", "#E8F5E9", "#2E7D32", fontsize=11, zorder=1)
    draw_arrow(ax, 23, 18, 35, 18, lw=2)
    
    add_image(ax, high_path, (105, 18), "Enhanced RGB\nboard/ONNX output")
    draw_arrow(ax, 85, 18, 97, 18, lw=2)
    
    # Metrics Text
    metrics_text = "Measured STM32 board cost:\nModel weights: 5.8 KiB\nInference mean 182 ms/frame\nTotal pipeline ~ 5.5 FPS"
    ax.text(140, 18, metrics_text, ha='center', va='center', fontsize=10, fontweight='bold', color='#333333')
    
    out_dir = Path("reports/figures/miwai_reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig9_training_deployment.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
