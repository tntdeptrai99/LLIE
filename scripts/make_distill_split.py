from __future__ import annotations

import argparse
from pathlib import Path


def teacher_path_for_low(low_path: Path, teacher_root: Path) -> Path:
    candidates = [
        teacher_root / low_path.name,
        teacher_root / f"{low_path.stem}.png",
        teacher_root / f"{low_path.stem}.jpg",
        teacher_root / f"{low_path.stem}.jpeg",
        teacher_root / f"{low_path.stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No teacher output found for {low_path} under {teacher_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-split", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows: list[str] = []
    missing: list[str] = []
    for line in args.paired_split.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        low_text, high_text = line.split()
        low_path = Path(low_text)
        high_path = Path(high_text)
        try:
            teacher_path = teacher_path_for_low(low_path, args.teacher_root)
        except FileNotFoundError as exc:
            if args.strict:
                raise
            missing.append(str(exc))
            continue
        rows.append(f"{low_path.as_posix()} {high_path.as_posix()} {teacher_path.as_posix()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote={args.out} rows={len(rows)} missing={len(missing)}")
    if missing[:5]:
        print("missing_examples:")
        for item in missing[:5]:
            print(item)


if __name__ == "__main__":
    main()
