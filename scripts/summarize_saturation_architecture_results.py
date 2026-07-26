from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "architecture_ablation" / "saturation_20260726"
REPORT = ROOT / "reports" / "benchmarks" / "architecture_ablation_saturation_20260726"


def read_log(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for log in sorted(EXP.glob("*/train_log.csv")):
        arch = log.parent.name
        epochs = read_log(log)
        if not epochs:
            continue
        best = max(epochs, key=lambda row: float(row["val_psnr"]))
        last = epochs[-1]
        rows.append(
            {
                "architecture": arch,
                "epochs_ran": str(len(epochs)),
                "best_epoch": best["epoch"],
                "best_psnr": best["val_psnr"],
                "best_ssim": best["val_ssim"],
                "best_mae": best["val_mae"],
                "last_epoch": last["epoch"],
                "last_psnr": last["val_psnr"],
                "last_ssim": last["val_ssim"],
                "last_mae": last["val_mae"],
                "best_checkpoint": str(log.parent / "best.pt") if (log.parent / "best.pt").exists() else "",
                "last_checkpoint": str(log.parent / "last.pt") if (log.parent / "last.pt").exists() else "",
                "status": "complete_or_early_stop" if (log.parent / "last.pt").exists() else "interrupted_partial",
            }
        )
    rows.sort(key=lambda row: float(row["best_psnr"]), reverse=True)

    csv_path = REPORT / "best_available_saturation_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Best available architecture saturation results",
        "",
        "Run này bị ngắt giữa chừng, nên bảng dưới đây chỉ lấy kết quả tốt nhất đã ghi được từ các `train_log.csv` hiện có.",
        "",
        "| Rank | Architecture | Status | Epochs ran | Best epoch | PSNR | SSIM | MAE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row['architecture']} | {row['status']} | {row['epochs_ran']} | "
            f"{row['best_epoch']} | {float(row['best_psnr']):.4f} | "
            f"{float(row['best_ssim']):.4f} | {float(row['best_mae']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Ghi chú",
            "",
            "- `complete_or_early_stop`: có `last.pt`, tức kiến trúc đã hoàn tất theo điều kiện dừng hoặc hết phần chạy hiện tại.",
            "- `interrupted_partial`: bị ngắt khi đang train, nhưng `best.pt` và log tới epoch cuối đã ghi vẫn dùng được.",
            "- DarkGhost-ESPNet không xuất hiện trong bảng vì chưa kịp bắt đầu ở run saturation này.",
        ]
    )
    md_path = REPORT / "best_available_saturation_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
