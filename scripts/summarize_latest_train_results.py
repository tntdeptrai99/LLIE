from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "benchmarks" / "darkghost_espnet_20260726"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def as_float(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return float("nan")


def classify(path: Path) -> str:
    s = str(path).lower()
    if "refine" in s:
        return "latest refinement"
    if "optuna" in s:
        return "optuna adaptive dark-map loss search"
    return "baseline/current run"


def main() -> None:
    rows = []
    for log in sorted((ROOT / "experiments").rglob("train_log.csv")):
        with log.open("r", encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        best_psnr = max(data, key=lambda r: as_float(r.get("val_psnr")))
        best_ssim = max(data, key=lambda r: as_float(r.get("val_ssim")))
        last = data[-1]
        rows.append(
            {
                "experiment": str(log.parent.relative_to(ROOT)),
                "log": str(log.relative_to(ROOT)),
                "family": classify(log),
                "mtime": f"{log.stat().st_mtime:.0f}",
                "epochs": len(data),
                "best_psnr": best_psnr.get("val_psnr", ""),
                "best_psnr_epoch": best_psnr.get("epoch", ""),
                "best_ssim": best_ssim.get("val_ssim", ""),
                "best_ssim_epoch": best_ssim.get("epoch", ""),
                "last_epoch": last.get("epoch", ""),
                "last_train_loss": last.get("train_loss", ""),
                "last_val_loss": last.get("val_loss", ""),
                "last_val_psnr": last.get("val_psnr", ""),
                "last_val_ssim": last.get("val_ssim", ""),
                "teacher_psnr": last.get("teacher_psnr", ""),
                "teacher_ssim": last.get("teacher_ssim", ""),
            }
        )

    rows_by_best = sorted(rows, key=lambda r: as_float(r["best_psnr"]), reverse=True)
    rows_by_latest = sorted(rows, key=lambda r: float(r["mtime"]), reverse=True)

    fieldnames = [
        "experiment",
        "log",
        "family",
        "mtime",
        "epochs",
        "best_psnr",
        "best_psnr_epoch",
        "best_ssim",
        "best_ssim_epoch",
        "last_epoch",
        "last_train_loss",
        "last_val_loss",
        "last_val_psnr",
        "last_val_ssim",
        "teacher_psnr",
        "teacher_ssim",
    ]
    for name, data in [
        ("train_log_summary_all.csv", rows_by_best),
        ("train_log_summary_top10.csv", rows_by_best[:10]),
        ("train_log_summary_latest10.csv", rows_by_latest[:10]),
    ]:
        with (OUT_DIR / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    best = rows_by_best[0] if rows_by_best else {}
    latest = rows_by_latest[0] if rows_by_latest else {}
    print("best=", best.get("experiment"), best.get("best_psnr"), best.get("best_ssim"))
    print("latest=", latest.get("experiment"), latest.get("last_val_psnr"), latest.get("last_val_ssim"))
    print(OUT_DIR)


if __name__ == "__main__":
    main()
