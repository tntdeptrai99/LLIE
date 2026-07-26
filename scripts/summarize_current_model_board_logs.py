from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "benchmarks" / "current_model_20260726"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS = [
    ROOT / "board_benchmark_full_pipeline_current_30s_COM3.log",
    ROOT / "board_full_pipeline_button_exti_final_COM3.log",
]

PATTERN = re.compile(
    r"v=(?P<fw>\d+),m=(?P<mode>\d+),y=(?P<in_layout>\d+)/(?P<out_layout>\d+),"
    r"g=(?P<gain>\d+)/(?P<residual>\d+),l=(?P<camera_display_ms>\d+),"
    r"p=(?P<preprocess_ms>\d+),a=(?P<inference_ms>\d+),q=(?P<postprocess_ms>\d+),"
    r"t=(?P<total_ms>\d+),f=(?P<fps>\d+),r=(?P<raw_min>\d+)/(?P<raw_max>\d+),"
    r"i=(?P<input_min>\d+)/(?P<input_max>\d+),o=(?P<output_min>\d+)/(?P<output_max>\d+)"
)


FIELDS = [
    "camera_display_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "total_ms",
    "fps",
    "raw_min",
    "raw_max",
    "input_min",
    "input_max",
    "output_min",
    "output_max",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return xs[max(0, min(idx, len(xs) - 1))]


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for log in LOGS:
        if not log.exists():
            continue
        for line_no, line in enumerate(log.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            m = PATTERN.search(line)
            if not m:
                continue
            row = m.groupdict()
            row["source_log"] = log.name
            row["line"] = str(line_no)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["source_log", "line", "fw", "mode", "in_layout", "out_layout", "gain", "residual"] + FIELDS
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for field in FIELDS:
        values = [float(r[field]) for r in rows]
        out.append(
            {
                "metric": field,
                "n": str(len(values)),
                "min": f"{min(values):.4f}" if values else "",
                "mean": f"{mean(values):.4f}" if values else "",
                "max": f"{max(values):.4f}" if values else "",
                "p95": f"{percentile(values, 0.95):.4f}" if values else "",
            }
        )
    return out


def main() -> None:
    rows = read_rows()
    write_csv(OUT_DIR / "board_timing_rows.csv", rows)
    summary = summarize(rows)
    with (OUT_DIR / "board_timing_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n", "min", "mean", "max", "p95"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"rows={len(rows)}")
    print(OUT_DIR / "board_timing_summary.csv")


if __name__ == "__main__":
    main()
