from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


def prune_unused_nodes(model: onnx.ModelProto) -> None:
    graph = model.graph
    needed = {output.name for output in graph.output}
    keep_reversed = []

    for node in reversed(graph.node):
        if any(output in needed for output in node.output):
            keep_reversed.append(node)
            needed.update(node.input)

    del graph.node[:]
    graph.node.extend(reversed(keep_reversed))


def require_tensor(graph: onnx.GraphProto, name: str) -> None:
    producers = {output: node for node in graph.node for output in node.output}
    if name not in producers:
        raise SystemExit(f"Cannot find tensor producer for {name!r}")
    if producers[name].op_type != "QuantizeLinear":
        raise SystemExit(f"{name!r} is produced by {producers[name].op_type}, expected QuantizeLinear")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gain-output-name",
        default="/Add_1_output_0_QuantizeLinear_Output",
        help="Uint8 gain tensor after gain_min/gain_max mapping.",
    )
    parser.add_argument(
        "--residual-output-name",
        default="/Mul_1_output_0_QuantizeLinear_Output",
        help="Uint8 residual tensor after residual_scale mapping.",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    require_tensor(graph, args.gain_output_name)
    require_tensor(graph, args.residual_output_name)

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                args.gain_output_name,
                TensorProto.UINT8,
                [1, 3, 96, 96],
            ),
            helper.make_tensor_value_info(
                args.residual_output_name,
                TensorProto.UINT8,
                [1, 3, 96, 96],
            ),
        ]
    )

    graph.name = f"{graph.name}_tail2_u8out"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved={args.output}")
    print(f"public_output_0_gain={args.gain_output_name}")
    print(f"public_output_1_residual={args.residual_output_name}")
    print("next=Generate this ONNX in STM32Cube.AI, then compose output in firmware")


if __name__ == "__main__":
    main()
