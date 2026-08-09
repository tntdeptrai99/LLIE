import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl

# Set seaborn style manually
plt.style.use('default')
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['font.family'] = 'sans-serif'

def main():
    out_dir = Path("reports/figures/miwai_reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels = ['Without Dark Guidance', 'With Dark Guidance']
    psnr_values = [19.45, 19.62]
    ssim_values = [0.839, 0.843]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    color_psnr = '#3498db'
    color_ssim = '#e74c3c'
    
    # Plot PSNR
    rects1 = ax1.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color=color_psnr, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('PSNR (dB)', color=color_psnr, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_psnr)
    ax1.set_ylim(19.35, 19.65) # Zoom in aggressively to show the difference
    
    # Plot SSIM on secondary axis
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, ssim_values, width, label='SSIM Score', color=color_ssim, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('SSIM Score', color=color_ssim, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_ssim)
    ax2.set_ylim(0.837, 0.845)
    
    # Add values on top of bars
    def autolabel(rects, ax, format_str):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(format_str.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
                        
    autolabel(rects1, ax1, '{:.2f}')
    autolabel(rects2, ax2, '{:.3f}')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold')
    plt.title('Impact of Dark Guidance on Enhancement Quality', fontweight='bold', pad=15)
    
    fig.tight_layout()
    plt.savefig(out_dir / "fig7_dark_guidance.png", dpi=300, bbox_inches='tight')
    print("Saved fig7_dark_guidance.png")

if __name__ == "__main__":
    main()
