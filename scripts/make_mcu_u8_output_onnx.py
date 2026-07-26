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

    keep = list(reversed(keep_reversed))
    del graph.node[:]
    graph.node.extend(keep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quant-output-name",
        default="enhanced_rgb_QuantizeLinear_Output",
        help="Name of the uint8 tensor produced by the final QuantizeLinear node.",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    quant_output = args.quant_output_name
    producer = None
    for node in graph.node:
        if quant_output in node.output:
            producer = node
            break
    if producer is None:
        raise SystemExit(f"Cannot find tensor producer for {quant_output!r}")
    if producer.op_type != "QuantizeLinear":
        raise SystemExit(f"{quant_output!r} is produced by {producer.op_type}, expected QuantizeLinear")

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                quant_output,
                TensorProto.UINT8,
                [1, 3, 96, 96],
            )
        ]
    )

    graph.name = f"{graph.name}_mcu_u8out"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved={args.output}")
    print(f"public_output={quant_output}")
    print("next=Analyze this ONNX in STM32Cube.AI")


if __name__ == "__main__":
    main()
