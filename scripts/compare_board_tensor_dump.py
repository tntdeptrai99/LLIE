from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


WIDTH = 96
HEIGHT = 96
CHANNELS = 3
TENSOR_SIZE = WIDTH * HEIGHT * CHANNELS
DEFAULT_INPUT_SCALE = 0.003733412
DEFAULT_INPUT_ZERO_POINT = 0
OUTPUT0_SCALE = 0.007843137718737125
OUTPUT1_SCALE = 0.0007843137136660516
OUTPUT_ZERO_POINT = 0


def fnv1a_u8(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def parse_dump(path: Path) -> tuple[dict[str, bytes], dict[str, int | str]]:
    tensors: dict[str, bytearray] = {}
    expected: dict[str, tuple[int, int]] = {}
    meta: dict[str, int | str] = {}

    begin_re = re.compile(r"tdump_begin,(.*)")
    tensor_re = re.compile(r"tdump_tensor,([^,]+),(\d+),([0-9A-Fa-f]{8})")
    data_re = re.compile(r"tdump_data,([^,]+),(\d+),([0-9A-Fa-f]+)")
    inverted_data_re = re.compile(r"([0-9A-Fa-f]+)tdump_data,([^,]+),(\d+),$")

    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        begin = begin_re.match(line)
        if begin:
            for item in begin.group(1).split(","):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                try:
                    meta[key] = int(value, 0)
                except ValueError:
                    meta[key] = value
            continue

        tensor = tensor_re.match(line)
        if tensor:
            name = tensor.group(1)
            length = int(tensor.group(2))
            checksum = int(tensor.group(3), 16)
            expected[name] = (length, checksum)
            tensors[name] = bytearray(length)
            continue

        data = data_re.match(line)
        if data:
            name = data.group(1)
            offset = int(data.group(2))
            payload = bytes.fromhex(data.group(3))
            if name not in tensors:
                continue
            tensors[name][offset : offset + len(payload)] = payload
            continue

        inverted = inverted_data_re.match(line)
        if inverted:
            payload = bytes.fromhex(inverted.group(1))
            name = inverted.group(2)
            offset = int(inverted.group(3))
            if name not in tensors:
                continue
            tensors[name][offset : offset + len(payload)] = payload

    result = {name: bytes(value) for name, value in tensors.items()}
    for name, data in result.items():
        if name not in expected:
            raise ValueError(f"Missing header for tensor {name}")
        length, checksum = expected[name]
        if len(data) != length:
            raise ValueError(f"{name}: length mismatch {len(data)} != {length}")
        actual = fnv1a_u8(data)
        if actual != checksum:
            raise ValueError(f"{name}: checksum mismatch {actual:08X} != {checksum:08X}")

    return result, meta


def raw_to_nchw(raw: np.ndarray, layout: int) -> np.ndarray:
    if raw.size != TENSOR_SIZE:
        raise ValueError(f"Expected {TENSOR_SIZE} bytes, got {raw.size}")
    if layout == 0:
        return raw.reshape(HEIGHT, WIDTH, CHANNELS).transpose(2, 0, 1)
    if layout == 1:
        return raw.reshape(CHANNELS, HEIGHT, WIDTH)
    if layout == 2:
        return raw.reshape(CHANNELS, WIDTH, HEIGHT).transpose(0, 2, 1)
    raise ValueError(f"Unsupported layout {layout}")


def output_layout_candidates(raw: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "raw_nchw": raw.reshape(CHANNELS, HEIGHT, WIDTH),
        "raw_hwc": raw.reshape(HEIGHT, WIDTH, CHANNELS).transpose(2, 0, 1),
        "raw_chw_xy": raw.reshape(CHANNELS, WIDTH, HEIGHT).transpose(0, 2, 1),
        "raw_ycx": raw.reshape(HEIGHT, CHANNELS, WIDTH).transpose(1, 0, 2),
    }


def ort_output_to_nchw(output: np.ndarray) -> np.ndarray:
    shape = tuple(int(dim) for dim in output.shape)
    if shape == (1, CHANNELS, HEIGHT, WIDTH):
        return output.reshape(CHANNELS, HEIGHT, WIDTH).astype(np.uint8)
    if shape == (1, HEIGHT, WIDTH, CHANNELS):
        return output.reshape(HEIGHT, WIDTH, CHANNELS).transpose(2, 0, 1).astype(np.uint8)
    if output.size == TENSOR_SIZE:
        return output.reshape(CHANNELS, HEIGHT, WIDTH).astype(np.uint8)
    raise ValueError(f"Unsupported ONNX output shape {shape}")


def stats(name: str, array: np.ndarray) -> str:
    values = array.astype(np.float32)
    return (
        f"{name}: min={values.min():.0f} max={values.max():.0f} "
        f"mean={values.mean():.2f} std={values.std():.2f} "
        f"p01={np.percentile(values, 1):.0f} p99={np.percentile(values, 99):.0f}"
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).reshape(-1)
    bf = b.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 1.0 if np.allclose(af, bf) else 0.0
    return float(np.dot(af, bf) / denom)


def compare_quant(name: str, board: np.ndarray, ref: np.ndarray) -> tuple[float, float, int, float]:
    diff = board.astype(np.int16) - ref.astype(np.int16)
    abs_diff = np.abs(diff)
    exact = float(np.mean(abs_diff == 0) * 100.0)
    mae = float(abs_diff.mean())
    max_abs = int(abs_diff.max())
    p99 = float(np.percentile(abs_diff, 99))
    cos = cosine_similarity(board, ref)
    print(f"{name}: exact={exact:.2f}% mae_q={mae:.3f} p99_q={p99:.1f} max_abs_q={max_abs} cosine_q={cos:.6f}")
    return mae, -exact, max_abs, p99


def compare_dequant(name: str, board: np.ndarray, ref: np.ndarray, scale: float, zero_point: int) -> None:
    board_f = (board.astype(np.float32) - float(zero_point)) * float(scale)
    ref_f = (ref.astype(np.float32) - float(zero_point)) * float(scale)
    diff = np.abs(board_f - ref_f)
    print(
        f"{name}: mae_f={float(diff.mean()):.6f} "
        f"p99_f={float(np.percentile(diff, 99)):.6f} "
        f"max_abs_f={float(diff.max()):.6f} "
        f"cosine_f={cosine_similarity(board_f, ref_f):.6f}"
    )


def compare_signed_reinterpret(name: str, board: np.ndarray, ref: np.ndarray) -> None:
    board_i8 = board.astype(np.uint8).view(np.int8).astype(np.int16)
    ref_i8 = ref.astype(np.uint8).view(np.int8).astype(np.int16)
    diff = np.abs(board_i8 - ref_i8)
    print(
        f"{name}: int8_reinterpret_mae={float(diff.mean()):.3f} "
        f"p99={float(np.percentile(diff, 99)):.1f} "
        f"max_abs={int(diff.max())} "
        f"cosine_i8={cosine_similarity(board_i8, ref_i8):.6f}"
    )


def find_input_quant_params(model_path: Path, input_name: str) -> tuple[float, int]:
    try:
        import onnx
        from onnx import numpy_helper
    except Exception:
        return DEFAULT_INPUT_SCALE, DEFAULT_INPUT_ZERO_POINT

    model = onnx.load(model_path)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or not node.input or node.input[0] != input_name:
            continue
        scale = float(np.asarray(initializers[node.input[1]]).reshape(-1)[0])
        zero_point = int(np.asarray(initializers[node.input[2]]).reshape(-1)[0]) if len(node.input) > 2 else 0
        return scale, zero_point
    return DEFAULT_INPUT_SCALE, DEFAULT_INPUT_ZERO_POINT


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one STM32 tensor dump against ONNX Runtime.")
    parser.add_argument("--log", type=Path, required=True, help="UART log containing tdump_* lines.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq_tail2_u8out.onnx"),
    )
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--input-zero-point", type=int, default=None)
    parser.add_argument(
        "--fixed-input-u8",
        type=int,
        default=None,
        help="Use this constant uint8 input when input_runtime is missing from a partial UART log.",
    )
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise SystemExit("Missing onnxruntime. Install it in your Python env, then run this script again.") from exc

    tensors, meta = parse_dump(args.log)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    output_count = len(session.get_outputs())
    required_tensors = {"input_runtime", "output0_public"}
    if output_count >= 2:
        required_tensors.add("output1_public")
    missing = required_tensors - set(tensors)
    if "input_runtime" in missing and args.fixed_input_u8 is None:
        raise SystemExit(
            f"Missing tensors in log: {sorted(missing)}. "
            "For a fixed-input partial log, rerun with --fixed-input-u8 32."
        )

    input_layout = int(meta.get("in_layout", 1))
    if "input_runtime" in tensors:
        board_input_raw = np.frombuffer(tensors["input_runtime"], dtype=np.uint8)
    else:
        board_input_raw = np.full(TENSOR_SIZE, int(args.fixed_input_u8), dtype=np.uint8)
        input_layout = 1
    board_output0_raw = np.frombuffer(tensors["output0_public"], dtype=np.uint8) if "output0_public" in tensors else None
    board_output1_raw = np.frombuffer(tensors["output1_public"], dtype=np.uint8) if "output1_public" in tensors else None

    board_input_nchw = raw_to_nchw(board_input_raw, input_layout)
    input_name = session.get_inputs()[0].name
    scale, zero_point = find_input_quant_params(args.onnx, input_name)
    if args.input_scale is not None:
        scale = args.input_scale
    if args.input_zero_point is not None:
        zero_point = args.input_zero_point

    ort_input = (board_input_nchw.astype(np.float32) - float(zero_point)) * float(scale)
    ort_outputs = session.run(None, {input_name: ort_input[None, ...]})
    ref0_raw = np.asarray(ort_outputs[0])
    ref0 = ort_output_to_nchw(ref0_raw)
    ref1_raw = np.asarray(ort_outputs[1]) if output_count >= 2 else None
    ref1 = ort_output_to_nchw(ref1_raw) if ref1_raw is not None else None

    print(f"dump meta: {meta}")
    print(f"onnx input: name={input_name} scale={scale:.9g} zero_point={zero_point}")
    output0_semantic = "enhanced_rgb" if output_count == 1 else "gain"
    output0_scale = (1.0 / 255.0) if output_count == 1 else OUTPUT0_SCALE
    print(f"onnx output0: name={session.get_outputs()[0].name} shape={ref0_raw.shape} dtype={ref0_raw.dtype} scale={output0_scale:.10g} zero_point={OUTPUT_ZERO_POINT} semantic={output0_semantic}")
    if output_count >= 2 and ref1_raw is not None:
        print(f"onnx output1: name={session.get_outputs()[1].name} shape={ref1_raw.shape} dtype={ref1_raw.dtype} scale={OUTPUT1_SCALE:.10g} zero_point={OUTPUT_ZERO_POINT} semantic=residual")
    print("board output dtype per Cube.AI metadata: uint8, not int8")
    if missing:
        print(f"partial log: missing tensors={sorted(missing)}")
    print(stats("board input u8", board_input_nchw))
    if board_output0_raw is not None:
        print(stats("board output0 raw u8", board_output0_raw))
    if board_output1_raw is not None:
        print(stats("board output1 raw u8", board_output1_raw))
    extra_names = sorted(name for name in tensors if name.startswith("act_") or name.startswith("external_"))
    for name in extra_names:
        raw = np.frombuffer(tensors[name], dtype=np.uint8)
        if raw.size == TENSOR_SIZE:
            print(stats(f"board {name} raw u8", raw))
    print(stats(f"ort output0 {output0_semantic}", ref0))
    if ref1 is not None:
        print(stats("ort output1 residual", ref1))
    print()

    public_comparisons = []
    if board_output0_raw is not None:
        public_comparisons.append((f"board output0_public vs ONNX output0 {output0_semantic}", board_output0_raw, ref0, output0_scale))
    if board_output1_raw is not None and ref1 is not None:
        public_comparisons.append(("board output1_public vs ONNX output1 residual", board_output1_raw, ref1, OUTPUT1_SCALE))
    for tensor_name, board_raw, ref, scale_out in public_comparisons:
        print(tensor_name)
        ranked: list[tuple[tuple[float, float, int, float], str]] = []
        for layout_name, board in output_layout_candidates(board_raw).items():
            rank = compare_quant(f"  uint8 {layout_name}", board, ref)
            if layout_name == "raw_nchw":
                compare_signed_reinterpret(f"  signedness check {layout_name}", board, ref)
                compare_dequant(f"  dequant {layout_name}", board, ref, scale_out, OUTPUT_ZERO_POINT)
            ranked.append((rank, layout_name))
        ranked.sort()
        print(f"  best={ranked[0][1]}")
        print()

    if board_output0_raw is not None and board_output1_raw is not None and ref1 is not None:
        print("output order cross-check, using best-known CHW layout")
        board0 = board_output0_raw.reshape(CHANNELS, HEIGHT, WIDTH)
        board1 = board_output1_raw.reshape(CHANNELS, HEIGHT, WIDTH)
        compare_quant("  board output0_public vs ONNX output1 residual raw uint8", board0, ref1)
        compare_quant("  board output1_public vs ONNX output0 gain raw uint8", board1, ref0)

    if extra_names:
        print()
        print("extra tensor scan, using raw CHW layout")
        for name in extra_names:
            raw = np.frombuffer(tensors[name], dtype=np.uint8)
            if raw.size != TENSOR_SIZE:
                continue
            board = raw.reshape(CHANNELS, HEIGHT, WIDTH)
            compare_quant(f"  {name} vs ONNX output0 {output0_semantic}", board, ref0)
            if ref1 is not None:
                compare_quant(f"  {name} vs ONNX output1 residual", board, ref1)


if __name__ == "__main__":
    main()
