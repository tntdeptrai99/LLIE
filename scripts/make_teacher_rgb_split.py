from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def find_teacher(row_teacher: Path, retinexformer_output_root: Path | None) -> Path | None:
    candidates = [row_teacher]
    if retinexformer_output_root is not None:
        candidates.extend(
            [
                retinexformer_output_root / row_teacher.name,
                retinexformer_output_root / "visualization" / row_teacher.name,
                retinexformer_output_root / "results" / row_teacher.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def validate_rgb(path: Path) -> None:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
        image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retinexformer-output-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--validate-rgb", action="store_true")
    args = parser.parse_args()

    rows: list[str] = []
    missing: list[str] = []
    with args.manifest.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            teacher = find_teacher(Path(row["teacher"]), args.retinexformer_output_root)
            if teacher is None:
                message = f"missing teacher for {row['low']} expected {row['teacher']}"
                if args.strict:
                    raise FileNotFoundError(message)
                missing.append(message)
                continue
            if args.validate_rgb:
                validate_rgb(teacher)
            rows.append(f"{row['low']} {row['high']} {teacher.as_posix()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"wrote={args.out} rows={len(rows)} missing={len(missing)}")
    if missing[:5]:
        print("missing_examples:")
        for item in missing[:5]:
            print(item)


if __name__ == "__main__":
    main()
