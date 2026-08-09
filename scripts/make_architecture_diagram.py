import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def draw_box(ax, cx, cy, w, h, text, facecolor, edgecolor='black', fontsize=9):
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.1", facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.2, alpha=0.9
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return cy - h/2 - 0.1, cy + h/2 + 0.1

def draw_arrow(ax, x1, y1, x2, y2, connectionstyle="arc3,rad=0", color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, connectionstyle=connectionstyle))

def draw_add_node(ax, cx, cy):
    circle = mpatches.Circle((cx, cy), radius=0.25, facecolor="#E040FB", edgecolor="black", zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy, "+", ha='center', va='center', fontsize=14, fontweight='bold')
    return cy - 0.25, cy + 0.25

def draw_concat_node(ax, cx, cy, w=1.0, h=0.4):
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.05", facecolor="#FFF176", edgecolor="black",
        linewidth=1.2, alpha=0.9, zorder=3
    )
    ax.add_patch(box)
    ax.text(cx, cy, "Concat", ha='center', va='center', fontsize=9, fontweight='bold')
    return cy - h/2 - 0.05, cy + h/2 + 0.05

def main():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    c_in = "#90CAF9"       # Input
    c_stem = "#A5D6A7"     # Stem
    c_dw = "#EF9A9A"       # Downsample
    c_ghost = "#CE93D8"    # Ghost
    c_dark = "#B0BEC5"     # DarkMap
    c_fuse = "#FFCC80"     # Fusion/Upsample
    c_head = "#80CBC4"     # Heads
    c_out = "#E0E0E0"      # Output
    
    # Background panel
    panel = mpatches.FancyBboxPatch((0.5, 0.2), 11, 11.5, boxstyle="round,pad=0.1", 
                                    facecolor="#F8BBD0", edgecolor="gray", alpha=0.15)
    ax.add_patch(panel)
    
    # Coordinates
    x_main = 5
    x_dark = 9.5
    w, h = 3.5, 0.55
    w_dark = 3.0
    
    y = 11.0
    b_in, t_in = draw_box(ax, x_main, y, w, h, "Input Image (96x96x3)", c_in, fontsize=10)
    
    y -= 1.1
    b_stem, t_stem = draw_box(ax, x_main, y, w, h, "Stem: ConvBNReLU6\n3 → 12 channels", c_stem)
    draw_arrow(ax, x_main, b_in, x_main, t_stem)
    
    # Dark Map Generator branch
    b_dark, t_dark = draw_box(ax, x_dark, y, w_dark, h, "DarkMap Generator\n3 → 1 channel", c_dark)
    # Arrow from Input to DarkMap
    draw_arrow(ax, x_main + w/2 + 0.1, y + 1.1, x_dark, t_dark, connectionstyle="angle,angleA=-90,angleB=180,rad=10", color="gray")
    
    y -= 1.1
    b_down, t_down = draw_box(ax, x_main, y, w, h, "Downsample: DW-Sep Conv\n12 → 24 channels", c_dw)
    draw_arrow(ax, x_main, b_stem, x_main, t_down)
    
    y -= 1.1
    b_bot, t_bot = draw_box(ax, x_main, y, w, h, "Bottleneck: 3x Ghost-ESP\n24 channels", c_ghost)
    draw_arrow(ax, x_main, b_down, x_main, t_bot)
    
    y -= 1.0
    b_cat, t_cat = draw_concat_node(ax, x_main, y, w=1.5)
    draw_arrow(ax, x_main, b_bot, x_main, t_cat)
    # Arrow from DarkMap to Concat
    draw_arrow(ax, x_dark, b_dark, x_main + 0.75, y, connectionstyle="angle,angleA=180,angleB=-90,rad=10", color="gray")
    ax.text(x_dark - 1.5, y + 0.2, "DarkMap (96x96x1)", color="#455A64", fontsize=9, fontweight="bold")
    
    y -= 1.0
    b_fuse, t_fuse = draw_box(ax, x_main, y, w, h, "Dark Fusion: ConvBNReLU6\n25 → 24 channels", c_fuse)
    draw_arrow(ax, x_main, b_cat, x_main, t_fuse)
    
    y -= 1.1
    b_up, t_up = draw_box(ax, x_main, y, w, h, "Upsample: Nearest + Conv\n24 → 12 channels", c_fuse)
    draw_arrow(ax, x_main, b_fuse, x_main, t_up)
    
    y -= 1.0
    b_add, t_add = draw_add_node(ax, x_main, y)
    draw_arrow(ax, x_main, b_up, x_main, t_add)
    # U-Net Skip connection from Stem to Add Node
    draw_arrow(ax, x_main - w/2 - 0.1, y + 4.2, x_main - 0.25, y, connectionstyle="arc3,rad=-0.5", color="#1976D2")
    ax.text(x_main - w/2 - 0.9, y + 2.0, "Skip Connection", color="#1976D2", fontsize=9, rotation=90, va='center')
    
    y -= 1.0
    b_ref, t_ref = draw_box(ax, x_main, y, w, h, "Refine: Ghost-ESP Block\n12 channels", c_ghost)
    draw_arrow(ax, x_main, b_add, x_main, t_ref)
    
    # Split to Heads
    y -= 1.3
    b_gain, t_gain = draw_box(ax, x_main - 1.8, y, 2.8, h, "Gain Head\n12 → 3 channels", c_head)
    b_res, t_res = draw_box(ax, x_main + 1.8, y, 2.8, h, "Residual Head\n12 → 3 channels", c_head)
    
    draw_arrow(ax, x_main, b_ref, x_main - 1.8, t_gain, connectionstyle="angle,angleA=90,angleB=0,rad=5")
    draw_arrow(ax, x_main, b_ref, x_main + 1.8, t_res, connectionstyle="angle,angleA=90,angleB=180,rad=5")
    
    # Output Equation
    y -= 1.3
    b_out, t_out = draw_box(ax, x_main, y, 6.0, 0.7, "Output = Clamp( Input × Gain + Residual )", c_out, fontsize=11)
    
    draw_arrow(ax, x_main - 1.8, b_gain, x_main - 1.0, t_out, connectionstyle="arc3,rad=0.3")
    draw_arrow(ax, x_main + 1.8, b_res, x_main + 1.0, t_out, connectionstyle="arc3,rad=-0.3")
    
    # Input straight to Output for multiplication
    draw_arrow(ax, x_main + w/2 + 0.1, y + 9.8, x_main + 3.0, y + 0.35, connectionstyle="arc3,rad=0.3", color="#1976D2")
    
    # Title
    plt.suptitle("Proposed Architecture: DG-GhostESP-96", fontsize=18, fontweight='bold', y=0.96)
    
    out_dir = Path("reports/figures/miwai_reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig8_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
