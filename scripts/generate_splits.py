"""Generate reproducible split files for paired LOL datasets.

The raw data is expected to use this layout:

data/raw/LOL/train/low/*.png
data/raw/LOL/train/high/*.png
data/raw/LOL/val/low/*.png
data/raw/LOL/val/high/*.png

and the same structure for data/raw/LOL-v2-Real.

For the official training split, this script creates a deterministic 90/10
train/validation split using seed 42. The existing ``val`` directory is treated
as the held-out official test split because that is how the current dataset
drop is organized.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DATASETS = {
    "lol": "LOL",
    "lolv2_real": "LOL-v2-Real",
}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def paired_lines(root: Path, split: str) -> list[str]:
    low_dir = root / split / "low"
    high_dir = root / split / "high"
    if not low_dir.is_dir() or not high_dir.is_dir():
        raise FileNotFoundError(f"Missing low/high directories for {root / split}")

    low_by_name = {path.name: path for path in image_files(low_dir)}
    high_by_name = {path.name: path for path in image_files(high_dir)}

    missing_high = sorted(set(low_by_name) - set(high_by_name))
    missing_low = sorted(set(high_by_name) - set(low_by_name))
    if missing_high or missing_low:
        details = []
        if missing_high:
            details.append(f"missing high: {missing_high[:5]}")
        if missing_low:
            details.append(f"missing low: {missing_low[:5]}")
        raise ValueError(f"Unmatched pairs in {root / split}: {'; '.join(details)}")

    return [
        f"{low_by_name[name].as_posix()} {high_by_name[name].as_posix()}"
        for name in sorted(low_by_name)
    ]


def split_train_val(lines: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    indices = list(range(len(lines)))
    rng.shuffle(indices)
    train_count = int(len(indices) * 0.9)
    train_indices = sorted(indices[:train_count])
    val_indices = sorted(indices[train_count:])
    return [lines[i] for i in train_indices], [lines[i] for i in val_indices]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset(prefix: str, raw_name: str, data_root: Path, split_dir: Path, seed: int) -> None:
    root = data_root / "raw" / raw_name
    train_all = paired_lines(root, "train")
    official_test = paired_lines(root, "val")
    train, val = split_train_val(train_all, seed)

    write_lines(split_dir / f"{prefix}_train.txt", train)
    write_lines(split_dir / f"{prefix}_val.txt", val)
    write_lines(split_dir / f"{prefix}_test.txt", official_test)

    print(
        f"{raw_name}: train={len(train)}, val={len(val)}, "
        f"test={len(official_test)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    for prefix, raw_name in DATASETS.items():
        generate_dataset(prefix, raw_name, args.data_root, args.split_dir, args.seed)


if __name__ == "__main__":
    main()
