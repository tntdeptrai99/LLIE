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


def require_producer(graph: onnx.GraphProto, tensor_name: str) -> onnx.NodeProto:
    for node in graph.node:
        if tensor_name in node.output:
            return node
    raise SystemExit(f"Cannot find tensor producer for {tensor_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quant-output-name",
        default="enhanced_rgb_QuantizeLinear_Output",
        help="NCHW uint8 tensor produced by the final QuantizeLinear node.",
    )
    parser.add_argument(
        "--public-output-name",
        default="enhanced_rgb_u8_nhwc",
        help="New public NHWC uint8 output tensor name.",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    producer = require_producer(graph, args.quant_output_name)
    if producer.op_type != "QuantizeLinear":
        raise SystemExit(
            f"{args.quant_output_name!r} is produced by {producer.op_type}, expected QuantizeLinear"
        )

    # Make the public output NHWC so ST Edge AI does not need to expose a
    # channel-first output by adding its own final output transpose.
    graph.node.append(
        helper.make_node(
            "Transpose",
            inputs=[args.quant_output_name],
            outputs=[args.public_output_name],
            name="public_output_to_nhwc",
            perm=[0, 2, 3, 1],
        )
    )

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                args.public_output_name,
                TensorProto.UINT8,
                [1, 96, 96, 3],
            )
        ]
    )

    graph.name = f"{graph.name}_mcu_u8_nhwc_out"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved={args.output}")
    print(f"public_output={args.public_output_name}")
    print("next=Run Validate on Desktop; output should be NHWC uint8.")


if __name__ == "__main__":
    main()
