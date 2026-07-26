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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-name",
        default="/Clip_1_output_0_QuantizeLinear_Output",
        help="Uint8 tensor before the MCU-problematic float Div/Sub/Mul output tail.",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    producers = {out: node for node in graph.node for out in node.output}
    if args.output_name not in producers:
        raise SystemExit(f"Cannot find tensor producer for {args.output_name!r}")

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                args.output_name,
                TensorProto.UINT8,
                [1, 3, 96, 96],
            )
        ]
    )

    graph.name = f"{graph.name}_clip1_u8out"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved={args.output}")
    print(f"public_output={args.output_name}")
    print("next=Analyze/generate this ONNX in STM32Cube.AI")


if __name__ == "__main__":
    main()
