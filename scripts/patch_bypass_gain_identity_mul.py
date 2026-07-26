from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference


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
        "--old-tensor",
        default="/Mul_output_0_DequantizeLinear_Output",
        help="Tensor produced after the identity gain Mul branch.",
    )
    parser.add_argument(
        "--new-tensor",
        default="/Clip_output_0_DequantizeLinear_Output",
        help="Tensor before the identity gain Mul branch.",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)
    replaced = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == args.old_tensor:
                node.input[index] = args.new_tensor
                replaced += 1

    if replaced == 0:
        raise SystemExit(f"No consumers found for {args.old_tensor!r}")

    model.graph.name = f"{model.graph.name}_bypass_gain_identity_mul"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved={args.output}")
    print(f"replaced_consumers={replaced}")


if __name__ == "__main__":
    main()
