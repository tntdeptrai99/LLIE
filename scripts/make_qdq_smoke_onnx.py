from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def scalar(name: str, value, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(value, dtype=dtype), name=name)


def make_identity(path: Path) -> None:
    input_rgb = helper.make_tensor_value_info("input_rgb", TensorProto.FLOAT, [1, 3, 96, 96])
    output = helper.make_tensor_value_info("identity_u8", TensorProto.UINT8, [1, 3, 96, 96])
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input_rgb", "input_scale", "input_zp"],
            ["identity_u8"],
            name="identity_quant",
        )
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_smoke_identity",
        [input_rgb],
        [output],
        [scalar("input_scale", 0.0037334118969738483, np.float32), scalar("input_zp", 0, np.uint8)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_brightness(path: Path) -> None:
    input_rgb = helper.make_tensor_value_info("input_rgb", TensorProto.FLOAT, [1, 3, 96, 96])
    output = helper.make_tensor_value_info("brightness_u8", TensorProto.UINT8, [1, 3, 96, 96])
    nodes = [
        helper.make_node(
            "Mul",
            ["input_rgb", "gain"],
            ["bright_float"],
            name="brightness_mul",
        ),
        helper.make_node(
            "Clip",
            ["bright_float", "clip_min", "clip_max"],
            ["bright_clip"],
            name="brightness_clip",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["bright_clip", "output_scale", "output_zp"],
            ["brightness_u8"],
            name="brightness_quant",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_smoke_brightness",
        [input_rgb],
        [output],
        [
            scalar("gain", 2.0, np.float32),
            scalar("clip_min", 0.0, np.float32),
            scalar("clip_max", 1.0, np.float32),
            scalar("output_scale", 1.0 / 255.0, np.float32),
            scalar("output_zp", 0, np.uint8),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("stm32/onnx"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    identity = args.out_dir / "qdq_smoke_identity_u8out.onnx"
    brightness = args.out_dir / "qdq_smoke_brightness_u8out.onnx"
    make_identity(identity)
    make_brightness(brightness)
    print(f"saved={identity}")
    print(f"saved={brightness}")


if __name__ == "__main__":
    main()
