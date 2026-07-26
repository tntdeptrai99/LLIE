from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image


WIDTH = 96
HEIGHT = 96
CHANNELS = 3
DEFAULT_INPUT_SCALE = 0.0037334118969738483
DEFAULT_INPUT_ZERO_POINT = 0


def find_input_quant_params(model_path: Path, input_name: str) -> tuple[float, int]:
    model = onnx.load(model_path)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or not node.input or node.input[0] != input_name:
            continue
        scale = float(np.asarray(initializers[node.input[1]]).reshape(-1)[0])
        zero_point = int(np.asarray(initializers[node.input[2]]).reshape(-1)[0]) if len(node.input) > 2 else 0
        return scale, zero_point
    return DEFAULT_INPUT_SCALE, DEFAULT_INPUT_ZERO_POINT


def center_square_resize_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def make_gradient() -> np.ndarray:
    ramp = np.linspace(0, 255, WIDTH, dtype=np.uint8)
    image = np.tile(ramp[None, :], (HEIGHT, 1))
    return np.stack([image, image, image], axis=2)


def stats(name: str, array: np.ndarray) -> str:
    values = array.astype(np.float64)
    return (
        f"{name}: min={values.min():.0f} max={values.max():.0f} "
        f"mean={values.mean():.2f} std={values.std():.2f} "
        f"p01={np.percentile(values, 1):.0f} p99={np.percentile(values, 99):.0f}"
    )


def save_pair(path: Path, input_hwc: np.ndarray, output_hwc: np.ndarray) -> None:
    canvas = np.concatenate([input_hwc, output_hwc], axis=1)
    Image.fromarray(canvas, mode="RGB").save(path)


def run_case(
    session: ort.InferenceSession,
    input_name: str,
    scale: float,
    zero_point: int,
    name: str,
    input_hwc: np.ndarray,
    output_dir: Path,
) -> None:
    input_chw = input_hwc.transpose(2, 0, 1)[None, ...]
    input_float = (input_chw.astype(np.float32) - float(zero_point)) * float(scale)
    output = np.asarray(session.run(None, {input_name: input_float})[0])
    if output.dtype != np.uint8:
        output = np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)
    output_hwc = output.reshape(HEIGHT, WIDTH, CHANNELS)

    out_path = output_dir / f"{name}_input_output.png"
    save_pair(out_path, input_hwc, output_hwc)

    unique = int(np.unique(output_hwc.reshape(-1, CHANNELS), axis=0).shape[0])
    print(f"case={name}")
    print(f"  {stats('input_u8', input_hwc)}")
    print(f"  {stats('enhanced_rgb_u8', output_hwc)} unique_rgb={unique}")
    print(f"  saved={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a single-output enhanced RGB uint8 ONNX model.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_mcu_u8_nhwc_out.onnx"),
    )
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures/enhanced_rgb_onnx"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    scale, zero_point = find_input_quant_params(args.onnx, input_info.name)

    if len(session.get_outputs()) != 1:
        raise SystemExit(f"Expected one output, got {len(session.get_outputs())}")
    if output_info.shape != [1, HEIGHT, WIDTH, CHANNELS]:
        raise SystemExit(f"Expected output shape [1,96,96,3], got {output_info.shape}")
    if output_info.type != "tensor(uint8)":
        raise SystemExit(f"Expected uint8 output, got {output_info.type}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"onnx={args.onnx}")
    print(f"input={input_info.name} shape={input_info.shape} type={input_info.type}")
    print(f"input_quant scale={scale:.9g} zero_point={zero_point}")
    print(f"output={output_info.name} shape={output_info.shape} type={output_info.type}")

    run_case(
        session,
        input_info.name,
        scale,
        zero_point,
        "fixed32",
        np.full((HEIGHT, WIDTH, CHANNELS), 32, dtype=np.uint8),
        args.output_dir,
    )
    run_case(session, input_info.name, scale, zero_point, "gradient", make_gradient(), args.output_dir)
    run_case(
        session,
        input_info.name,
        scale,
        zero_point,
        "dark_random",
        rng.integers(0, 65, size=(HEIGHT, WIDTH, CHANNELS), dtype=np.uint8),
        args.output_dir,
    )

    for idx, path in enumerate(args.image):
        run_case(session, input_info.name, scale, zero_point, f"image_{idx:02d}", center_square_resize_rgb(path), args.output_dir)


if __name__ == "__main__":
    main()
