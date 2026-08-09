import base64
import requests
from pathlib import Path

mermaid_code = """graph TD
    classDef main fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef dark fill:#eceff1,stroke:#455a64,stroke-width:2px;
    classDef fusion fill:#fff8e1,stroke:#ffb300,stroke-width:2px;
    classDef head fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef output fill:#fbe9e7,stroke:#d84315,stroke-width:2px;

    In[Input RGB 96x96]:::main
    Stem[Stem: ConvBNReLU6\n3 to 12 channels]:::main
    Down[Downsample: DW-Sep\n12 to 24 channels]:::main
    Bot[Bottleneck: 3x Ghost-ESP\n24 channels]:::main
    
    Dark[DarkMap Generator\n3 to 1 channel]:::dark
    
    Cat{Concat}:::fusion
    Fuse[Dark Fusion: ConvBNReLU6\n25 to 24 channels]:::fusion
    Up[Upsample\n24 to 12 channels]:::fusion
    Add((+)):::fusion
    Refine[Refine: Ghost-ESP Block\n12 channels]:::main
    
    Gain[Gain Head\n12 to 3 channels]:::head
    Res[Residual Head\n12 to 3 channels]:::head
    
    Out[Output = Input * Gain + Residual]:::output

    In ==> Stem
    In --> Dark
    Stem ==> Down
    Down ==> Bot
    Bot ==> Cat
    Dark -.->|DarkMap Tensor| Cat
    Cat ==> Fuse
    Fuse ==> Up
    Up ==> Add
    Stem -.->|U-Net Skip| Add
    Add ==> Refine
    Refine ==> Gain
    Refine ==> Res
    Gain --> Out
    Res --> Out
"""

def generate_mermaid_image():
    # Convert string to base64
    graphbytes = mermaid_code.encode("utf-8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("utf-8")
    
    # Send request to mermaid.ink
    url = f"https://mermaid.ink/img/{base64_string}?type=png&bgColor=white"
    print(f"Requesting: {url}")
    
    response = requests.get(url)
    if response.status_code == 200:
        out_dir = Path("reports/figures/miwai_reproduction")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fig8_architecture.png"
        
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"Successfully saved to {out_path}")
    else:
        print(f"Failed to generate image. Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    generate_mermaid_image()
