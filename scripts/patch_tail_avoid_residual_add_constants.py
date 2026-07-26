from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


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


def find_node_by_output(model: onnx.ModelProto, output_name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if output_name in node.output:
            return node
    raise SystemExit(f"Cannot find node producing {output_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph
    add3 = find_node_by_output(model, "enhanced_rgb_QuantizeLinear_Input")

    graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "residual_clip_min"),
            numpy_helper.from_array(np.asarray(3.0, dtype=np.float32), "residual_clip_max"),
            numpy_helper.from_array(np.asarray(15.0, dtype=np.float32), "residual_div15_const"),
        ]
    )

    residual_clip = "residual_head_clip0_3_output"
    residual_div15 = "residual_div15_output"
    insert_at = list(graph.node).index(add3)
    graph.node.insert(
        insert_at,
        helper.make_node(
            "Clip",
            inputs=[
                "/residual_head/Conv_output_0_DequantizeLinear_Output",
                "residual_clip_min",
                "residual_clip_max",
            ],
            outputs=[residual_clip],
            name="residual_head_clip0_3",
        ),
    )
    graph.node.insert(
        insert_at + 1,
        helper.make_node(
            "Div",
            inputs=[residual_clip, "residual_div15_const"],
            outputs=[residual_div15],
            name="residual_div15",
        ),
    )

    old_residual = "/Mul_1_output_0_DequantizeLinear_Output"
    replaced = 0
    for index, name in enumerate(add3.input):
        if name == old_residual:
            add3.input[index] = residual_div15
            replaced += 1
    if replaced != 1:
        raise SystemExit(f"Expected to replace one Add_3 residual input, replaced={replaced}")

    graph.name = f"{graph.name}_avoid_residual_add_constants"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
