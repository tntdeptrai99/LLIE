from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, shape_inference


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

    mul2 = find_node_by_output(model, "/Mul_2_output_0")
    add3 = find_node_by_output(model, "enhanced_rgb_QuantizeLinear_Input")

    old_gain = "/Add_1_output_0_DequantizeLinear_Output"
    gain_div = "/Div_output_0_DequantizeLinear_Output"
    replaced = 0
    for index, name in enumerate(mul2.input):
        if name == old_gain:
            mul2.input[index] = gain_div
            replaced += 1
    if replaced != 1:
        raise SystemExit(f"Expected to replace one Mul_2 gain input, replaced={replaced}")

    input_dequant = "input_rgb_DequantizeLinear_Output"
    boost_plus_input = "input_plus_gain_boost_output"
    graph.node.insert(
        list(graph.node).index(add3),
        helper.make_node(
            "Add",
            inputs=[input_dequant, "/Mul_2_output_0_DequantizeLinear_Output"],
            outputs=[boost_plus_input],
            name="input_plus_gain_boost",
        ),
    )

    if add3.input[0] != "/Mul_2_output_0_DequantizeLinear_Output":
        raise SystemExit(f"Unexpected Add_3 first input {add3.input[0]!r}")
    add3.input[0] = boost_plus_input

    graph.name = f"{graph.name}_avoid_gain_add_constant"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
