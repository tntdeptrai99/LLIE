from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


INPUT_SCALE = 0.0037334118969738483
GAIN_HEAD_SCALE = 0.0235294122248888
RESIDUAL_HEAD_SCALE = 0.02292412519454956
Q_1_255 = 0.003921568859368563
Q_2_255 = 0.007843137718737125
Q_RESIDUAL = 0.0007843137136660516
Q_INPUT_GAIN = 0.005081470124423504


def qdq_u8(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(x / scale), 0, 255).astype(np.uint8).astype(np.float32) * scale


def stats(name: str, a: np.ndarray) -> str:
    a = a.astype(np.float64)
    return (
        f"{name}: min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} "
        f"std={a.std():.6f} p01={np.percentile(a, 1):.6f} p99={np.percentile(a, 99):.6f}"
    )


def compare(name: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = a.astype(np.float64) - b.astype(np.float64)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff * diff))
    max_abs = np.max(np.abs(diff))
    cos = float(np.dot(a.reshape(-1), b.reshape(-1)) /
                ((np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1))) + 1e-12))
    exact = float(np.mean(a == b) * 100.0)
    print(f"{name}: exact={exact:.2f}% mae={mae:.6f} rmse={rmse:.6f} max_abs={max_abs:.6f} cos={cos:.9f}")


def make_input(case: str, rng: np.random.Generator) -> np.ndarray:
    if case.startswith("fixed"):
        value = int(case.removeprefix("fixed"))
        return np.full((1, 3, 96, 96), value, dtype=np.uint8)
    if case == "random":
        return rng.integers(0, 256, size=(1, 3, 96, 96), dtype=np.uint8)
    if case == "dark_random":
        return rng.integers(0, 65, size=(1, 3, 96, 96), dtype=np.uint8)
    if case == "gradient":
        x = np.linspace(0, 255, 96, dtype=np.uint8)
        img = np.tile(x[None, :], (96, 1))
        chw = np.stack([img, img, img], axis=0)
        return chw[None, ...].astype(np.uint8)
    raise ValueError(f"unknown case: {case}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--raw-head2", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["fixed32", "dark_random", "random", "gradient"],
    )
    args = parser.parse_args()

    sess_full = ort.InferenceSession(str(args.full), providers=["CPUExecutionProvider"])
    sess_raw = ort.InferenceSession(str(args.raw_head2), providers=["CPUExecutionProvider"])
    in_full = sess_full.get_inputs()[0].name
    in_raw = sess_raw.get_inputs()[0].name
    rng = np.random.default_rng(args.seed)

    print(f"full_input={in_full} raw_input={in_raw}")
    print("model_input_f = input_u8 * input_scale")
    print("compose_exact follows ONNX tail QDQ steps after raw gain/residual heads")
    print("compose_u8 = round(clamp(compose_exact_f, 0, 1) * 255)")
    print()

    for case in args.cases:
        x = make_input(case, rng)
        x_f = x.astype(np.float32) * INPUT_SCALE
        full_f = sess_full.run(None, {in_full: x_f})[0].astype(np.float32)
        gain_q, residual_q = sess_raw.run(None, {in_raw: x_f})

        compose_float_f = (
            x_f * (1.0 + gain_q.astype(np.float32) * GAIN_HEAD_SCALE / 6.0)
            + residual_q.astype(np.float32) * RESIDUAL_HEAD_SCALE / 15.0
        )

        gain_raw_f = gain_q.astype(np.float32) * GAIN_HEAD_SCALE
        gain_div_f = qdq_u8(gain_raw_f / 6.0, Q_1_255)
        gain_mapped_f = qdq_u8(gain_div_f + 1.0, Q_2_255)

        residual_raw_f = residual_q.astype(np.float32) * RESIDUAL_HEAD_SCALE
        residual_plus_f = qdq_u8(residual_raw_f + 3.0, GAIN_HEAD_SCALE)
        residual_div_sub_f = qdq_u8((residual_plus_f / 3.0) - 1.0, Q_1_255)
        residual_mapped_f = qdq_u8(residual_div_sub_f * 0.2, Q_RESIDUAL)

        input_gain_f = qdq_u8(x_f * gain_mapped_f, Q_INPUT_GAIN)
        compose_exact_f = qdq_u8(input_gain_f + residual_mapped_f, Q_1_255)

        compose_float_u8 = np.clip(np.rint(compose_float_f * 255.0), 0, 255).astype(np.uint8)
        compose_exact_u8 = np.clip(np.rint(compose_exact_f * 255.0), 0, 255).astype(np.uint8)
        full_u8 = np.clip(np.rint(full_f * 255.0), 0, 255).astype(np.uint8)

        print(f"case={case}")
        print(stats("  input_u8", x))
        print(stats("  full_f", full_f))
        print(stats("  gain_raw_u8", gain_q))
        print(stats("  residual_raw_u8", residual_q))
        print(stats("  compose_float_f", compose_float_f))
        print(stats("  compose_exact_f", compose_exact_f))
        compare("  compose_float_f vs full_f", compose_float_f, full_f)
        compare("  compose_float_u8 vs full_u8", compose_float_u8, full_u8)
        compare("  compose_exact_f vs full_f", compose_exact_f, full_f)
        compare("  compose_exact_u8 vs full_u8", compose_exact_u8, full_u8)
        print()


if __name__ == "__main__":
    main()
