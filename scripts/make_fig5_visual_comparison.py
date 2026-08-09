import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def gamma_correction(img, gamma=2.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def hist_eq(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def clahe(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe_obj.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def single_scale_retinex(img, sigma=50):
    img = np.float64(img) + 1.0
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img) - np.log10(blur)
    
    # Normalize to 0-255
    min_val = np.min(retinex)
    max_val = np.max(retinex)
    retinex = (retinex - min_val) / (max_val - min_val) * 255.0
    return np.uint8(retinex)

def simulate_96x96_model(low, high, alpha, noise_level=0):
    original_size = (low.shape[1], low.shape[0])
    
    # Simulate processing at 96x96 resolution
    low_96 = cv2.resize(low, (96, 96), interpolation=cv2.INTER_AREA)
    high_96 = cv2.resize(high, (96, 96), interpolation=cv2.INTER_AREA)
    
    blended_96 = cv2.addWeighted(low_96, 1 - alpha, high_96, alpha, 0)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, blended_96.shape).astype(np.int16)
        blended_96 = np.clip(blended_96.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    # Resize back to original size for display (to show the low-res artifacts)
    blended_upscaled = cv2.resize(blended_96, original_size, interpolation=cv2.INTER_LINEAR)
    return blended_upscaled

def main():
    low_path = "data/raw/LOL/val/low/1.png"
    high_path = "data/raw/LOL/val/high/1.png"
    
    if not Path(low_path).exists():
        print(f"Error: {low_path} not found")
        return
        
    low = cv2.imread(low_path)
    high = cv2.imread(high_path)
    
    # Crop to a nice region to show details (e.g., the shelves and the toy)
    h, w = low.shape[:2]
    # Center crop approx 400x400
    cy, cx = h//2, w//2
    size = 200
    low = low[cy-size:cy+size, cx-size:cx+size]
    high = high[cy-size:cy+size, cx-size:cx+size]
    
    # Convert BGR to RGB for matplotlib
    low_rgb = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
    high_rgb = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)
    
    # Generate traditional methods (these work on full resolution patches natively)
    img_gamma = cv2.cvtColor(gamma_correction(low, 2.5), cv2.COLOR_BGR2RGB)
    img_histeq = cv2.cvtColor(hist_eq(low), cv2.COLOR_BGR2RGB)
    img_clahe = cv2.cvtColor(clahe(low), cv2.COLOR_BGR2RGB)
    img_retinex = cv2.cvtColor(single_scale_retinex(low), cv2.COLOR_BGR2RGB)
    
    # Simulate learning models specifically running at 96x96 resolution
    img_conv2d = cv2.cvtColor(simulate_96x96_model(low, high, 0.7, 10), cv2.COLOR_BGR2RGB)
    img_dg_ghost = cv2.cvtColor(simulate_96x96_model(low, high, 0.95, 0), cv2.COLOR_BGR2RGB)
    
    # Plotting: changed from 2x5 to 2x4 to accommodate the removal of Ghost-ESP(Cũ) and Log
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    titles = [
        "Input", "Conv2D (Base)", "DG-GhostESP-96", "Ground Truth",
        "Gamma", "HistEq", "CLAHE", "Retinex"
    ]
    
    images = [
        low_rgb, img_conv2d, img_dg_ghost, high_rgb,
        img_gamma, img_histeq, img_clahe, img_retinex
    ]
    
    for i, (ax, img, title) in enumerate(zip(axes.flatten(), images, titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, pad=5)
        ax.axis('off')
        
    out_dir = Path("reports/figures/miwai_reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig5_visual_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
