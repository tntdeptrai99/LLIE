import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def make_component_ablation():
    labels = [
        "No Optuna (manual weights)",
        "+ Optuna trial011",
        "+ Optuna + long refine / proposed",
        "No adaptive dark-map loss",
        "No teacher KD"
    ]
    psnr = [18.8806, 19.2671, 19.6221, 16.4336, 16.4132]
    ssim = [0.8193, 0.8334, 0.8434, 0.7708, 0.7686]
    
    labels = labels[::-1]
    psnr = psnr[::-1]
    ssim = ssim[::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Component Ablation: Optuna / Dark-map Loss / Teacher', fontsize=18, fontweight='bold', color='#0F2B5B', x=0.05, ha='left')
    
    y = np.arange(len(labels))
    height = 0.4
    
    ax1.barh(y, psnr, height, color='#2E7D4E')
    ax1.set_title('PSNR', loc='left', fontsize=12)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlim(0, max(psnr) * 1.15)
    for i, v in enumerate(psnr):
        ax1.text(v + 0.3, i, f"{v:.4f}", va='center', fontsize=9)
        
    ax2.barh(y, ssim, height, color='#2F62A4')
    ax2.set_title('SSIM', loc='left', fontsize=12)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlim(0, max(ssim) * 1.15)
    for i, v in enumerate(ssim):
        ax2.text(v + 0.015, i, f"{v:.4f}", va='center', fontsize=9)
        
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        
    plt.figtext(0.05, 0.05, "Long restored runs and new 30-epoch controlled ablations are reported separately in the CSV notes.", fontsize=9, color="gray")
        
    plt.subplots_adjust(left=0.25, wspace=0.4, top=0.85, bottom=0.15)
    
    out_dir = Path('reports/figures/miwai_reproduction')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'fig_component_ablation_bars.png', dpi=160)
    plt.close()

def make_arch_ablation():
    labels = ["Conv2D", "Separable", "GhostSep", "Ghost-ESP", "DarkGhost-ESPNet"]
    params = [16182, 3486, 2594, 2706, 2706]
    macs = [46540800, 10450944, 8441856, 8441856, 9068544]
    cpu_ms = [4.020, 3.388, 3.489, 3.605, 3.668]
    
    labels = labels[::-1]
    params = params[::-1]
    macs = macs[::-1]
    cpu_ms = cpu_ms[::-1]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Architecture Ablation: Cost Comparison (96x96)', fontsize=18, fontweight='bold', color='#0F2B5B', x=0.05, ha='left')
    
    y = np.arange(len(labels))
    height = 0.4
    
    ax1.barh(y, params, height, color='#2E7D4E')
    ax1.set_title('Parameters', loc='left', fontsize=11)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlim(0, max(params) * 1.3)
    for i, v in enumerate(params):
        ax1.text(v + 800, i, f"{v:,}", va='center', fontsize=9)
    
    ax2.barh(y, macs, height, color='#2F62A4')
    ax2.set_title('MACs', loc='left', fontsize=11)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlim(0, max(macs) * 1.3)
    for i, v in enumerate(macs):
        ax2.text(v + 2000000, i, f"{v:,}", va='center', fontsize=9)
        
    ax3.barh(y, cpu_ms, height, color='#BA5B33')
    ax3.set_title('CPU PyTorch ms', loc='left', fontsize=11)
    ax3.set_yticks(y)
    ax3.set_yticklabels(labels, fontsize=10)
    ax3.set_xlim(0, max(cpu_ms) * 1.25)
    for i, v in enumerate(cpu_ms):
        ax3.text(v + 0.1, i, f"{v:.3f}", va='center', fontsize=9)
        
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        
    plt.subplots_adjust(left=0.1, wspace=0.4, top=0.85)
    
    out_dir = Path('reports/figures/miwai_reproduction')
    plt.savefig(out_dir / 'fig_arch_cost_bars.png', dpi=160)
    plt.close()

if __name__ == '__main__':
    make_component_ablation()
    make_arch_ablation()
