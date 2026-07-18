from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def slug_for_path(path: Path) -> str:
    stem = "_".join(path.with_suffix("").parts)
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-split", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["copy", "skip-existing"], default="skip-existing")
    args = parser.parse_args()

    input_dir = args.out_root / "input"
    target_dir = args.out_root / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    args.teacher_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for line in args.paired_split.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        low_text, high_text = line.split()
        low_path = Path(low_text)
        high_path = Path(high_text)
        teacher_name = f"{slug_for_path(low_path)}.png"
        retinexformer_input = input_dir / teacher_name
        retinexformer_target = target_dir / teacher_name
        teacher_path = args.teacher_root / teacher_name
        if args.copy_mode == "copy" or not retinexformer_input.exists():
            shutil.copy2(low_path, retinexformer_input)
        if args.copy_mode == "copy" or not retinexformer_target.exists():
            shutil.copy2(high_path, retinexformer_target)
        rows.append(
            {
                "low": low_path.as_posix(),
                "high": high_path.as_posix(),
                "retinexformer_input": retinexformer_input.as_posix(),
                "retinexformer_target": retinexformer_target.as_posix(),
                "teacher": teacher_path.as_posix(),
            }
        )

    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["low", "high", "retinexformer_input", "retinexformer_target", "teacher"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"input_dir={input_dir}")
    print(f"target_dir={target_dir}")
    print(f"teacher_root={args.teacher_root}")
    print(f"manifest={args.manifest}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
