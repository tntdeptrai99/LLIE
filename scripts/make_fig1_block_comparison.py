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

def draw_arrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

def draw_skip_connection(ax, x, y_start, y_end, w):
    # Draw curved dashed line for skip connection
    rad = -0.5 if y_end < y_start else 0.5
    ax.annotate("", xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="->", color="#1976D2", lw=1.5, ls="--",
                                connectionstyle=f"arc3,rad={rad}"))

def draw_add_node(ax, cx, cy):
    circle = mpatches.Circle((cx, cy), radius=0.25, facecolor="#E040FB", edgecolor="black", zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy, "+", ha='center', va='center', fontsize=12, fontweight='bold')
    return cy - 0.25, cy + 0.25

def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    c_bg = "#F5F5F5"
    c_in = "#90CAF9"
    c_conv = "#FFCC80"
    c_dw = "#EF9A9A"
    c_relu = "#A5D6A7"
    c_ghost = "#CE93D8"
    c_bn = "#B3E5FC"
    
    titles = ["Regular Residual Block", "Separable Residual Block", 
              "Ghost Module Residual Block", "STM32 Optimized Ghost-ESP"]
    
    x_centers = [2, 6, 10, 14]
    w, h = 2.8, 0.5
    
    # Draw background panels
    panel_colors = ["#E3F2FD", "#FFF3E0", "#F3E5F5", "#E8F5E9"]
    for i, xc in enumerate(x_centers):
        panel = mpatches.FancyBboxPatch((xc - 1.8, 0.5), 3.6, 9, boxstyle="round,pad=0.1", 
                                        facecolor=panel_colors[i], edgecolor="gray", alpha=0.3)
        ax.add_patch(panel)
        ax.text(xc, 9.2, titles[i], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Block 1: Regular
    xc = x_centers[0]
    y = 8.5
    b_in, t_in = draw_box(ax, xc, y, w, h, "Input Tensor", c_in)
    
    y -= 1.2
    b_c1, t_c1 = draw_box(ax, xc, y, w, h, "Conv2D 3x3", c_conv)
    draw_arrow(ax, xc, b_in, t_c1)
    
    y -= 1.2
    b_r1, t_r1 = draw_box(ax, xc, y, w, h, "ReLU", c_relu)
    draw_arrow(ax, xc, b_c1, t_r1)
    
    y -= 1.2
    b_c2, t_c2 = draw_box(ax, xc, y, w, h, "Conv2D 3x3", c_conv)
    draw_arrow(ax, xc, b_r1, t_c2)
    
    y -= 1.2
    b_add, t_add = draw_add_node(ax, xc, y)
    draw_arrow(ax, xc, b_c2, t_add)
    draw_skip_connection(ax, xc + w/2 - 0.2, b_in, t_add, w)
    
    y -= 1.2
    b_r2, t_r2 = draw_box(ax, xc, y, w, h, "ReLU", c_relu)
    draw_arrow(ax, xc, b_add, t_r2)
    
    
    # Block 2: Separable
    xc = x_centers[1]
    y = 8.5
    b_in, t_in = draw_box(ax, xc, y, w, h, "Input Tensor", c_in)
    
    y -= 1.0
    b_d1, t_d1 = draw_box(ax, xc, y, w, h, "DepthwiseConv2D 3x3", c_dw)
    draw_arrow(ax, xc, b_in, t_d1)
    
    y -= 1.0
    b_c1, t_c1 = draw_box(ax, xc, y, w, h, "Conv2D 1x1", c_conv)
    draw_arrow(ax, xc, b_d1, t_c1)
    
    y -= 1.0
    b_r1, t_r1 = draw_box(ax, xc, y, w, h, "ReLU", c_relu)
    draw_arrow(ax, xc, b_c1, t_r1)
    
    y -= 1.0
    b_d2, t_d2 = draw_box(ax, xc, y, w, h, "DepthwiseConv2D 3x3", c_dw)
    draw_arrow(ax, xc, b_r1, t_d2)
    
    y -= 1.0
    b_c2, t_c2 = draw_box(ax, xc, y, w, h, "Conv2D 1x1", c_conv)
    draw_arrow(ax, xc, b_d2, t_c2)
    
    y -= 0.8
    b_add, t_add = draw_add_node(ax, xc, y)
    draw_arrow(ax, xc, b_c2, t_add)
    draw_skip_connection(ax, xc + w/2 - 0.2, b_in, t_add, w)
    
    y -= 0.8
    b_r2, t_r2 = draw_box(ax, xc, y, w, h, "ReLU", c_relu)
    draw_arrow(ax, xc, b_add, t_r2)
    
    
    # Block 3: Ghost Module
    xc = x_centers[2]
    y = 8.5
    b_in, t_in = draw_box(ax, xc, y, w, h, "Input Tensor", c_in)
    
    y -= 1.0
    b_g, t_g = draw_box(ax, xc, y, w, h+0.2, "Ghost Module\n(1x1 Conv + DW Conv)", c_ghost)
    draw_arrow(ax, xc, b_in, t_g)
    
    y -= 1.0
    b_d1, t_d1 = draw_box(ax, xc, y, w, h, "DepthwiseConv2D 3x3", c_dw)
    draw_arrow(ax, xc, b_g, t_d1)
    
    y -= 0.9
    b_bn1, t_bn1 = draw_box(ax, xc, y, w, h, "BatchNorm", c_bn)
    draw_arrow(ax, xc, b_d1, t_bn1)
    
    y -= 0.9
    b_c1, t_c1 = draw_box(ax, xc, y, w, h, "Conv2D 1x1", c_conv)
    draw_arrow(ax, xc, b_bn1, t_c1)
    
    y -= 0.9
    b_bn2, t_bn2 = draw_box(ax, xc, y, w, h, "BatchNorm", c_bn)
    draw_arrow(ax, xc, b_c1, t_bn2)
    
    y -= 0.75
    b_add, t_add = draw_add_node(ax, xc, y)
    draw_arrow(ax, xc, b_bn2, t_add)
    draw_skip_connection(ax, xc + w/2 - 0.2, b_in, t_add, w)
    
    y -= 0.75
    b_r2, t_r2 = draw_box(ax, xc, y, w, h, "ReLU", c_relu)
    draw_arrow(ax, xc, b_add, t_r2)
    
    
    # Block 4: Our STM32 Optimized Ghost-ESP
    xc = x_centers[3]
    y = 8.5
    b_in, t_in = draw_box(ax, xc, y, w, h, "Input Tensor", c_in)
    
    y -= 1.0
    b_c1, t_c1 = draw_box(ax, xc, y, w, h, "Conv2D 1x1 (Primary)", c_conv)
    draw_arrow(ax, xc, b_in, t_c1)
    
    y -= 1.0
    b_r1, t_r1 = draw_box(ax, xc, y, w, h, "ReLU6 (Quantization Safe)", c_relu)
    draw_arrow(ax, xc, b_c1, t_r1)
    
    y -= 1.0
    b_d1, t_d1 = draw_box(ax, xc, y, w, h, "DepthwiseConv2D 3x3", c_dw)
    draw_arrow(ax, xc, b_r1, t_d1)
    
    y -= 1.0
    b_r2, t_r2 = draw_box(ax, xc, y, w, h, "ReLU6 (Quantization Safe)", c_relu)
    draw_arrow(ax, xc, b_d1, t_r2)
    
    y -= 1.0
    b_c2, t_c2 = draw_box(ax, xc, y, w, h, "Conv2D 1x1 (Output Project)", c_conv)
    draw_arrow(ax, xc, b_r2, t_c2)
    
    y -= 0.8
    b_add, t_add = draw_add_node(ax, xc, y)
    draw_arrow(ax, xc, b_c2, t_add)
    draw_skip_connection(ax, xc + w/2 - 0.2, b_in, t_add, w)
    
    y -= 0.8
    b_r3, t_r3 = draw_box(ax, xc, y, w, h, "ReLU6 (Quantization Safe)", c_relu)
    draw_arrow(ax, xc, b_add, t_r3)
    
    
    plt.suptitle("Residual Blocks Comparison", fontsize=18, fontweight='bold', y=0.98)
    
    out_dir = Path("reports/figures/miwai_reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig1_block_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
