from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import helper, shape_inference


DEFAULT_STEDGEAI = Path(
    r"C:\Users\trann\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.0\Utilities\windows\stedgeai.exe"
)


@dataclass
class Candidate:
    label: str
    tensor: str


DEFAULT_CANDIDATES = [
    Candidate("input_quant", "input_rgb_QuantizeLinear_Output"),
    Candidate("dark_mul_quant", "/dark_map/Mul_output_0_QuantizeLinear_Output"),
    Candidate("stem_quant", "/stem/stem.1/Clip_output_0_QuantizeLinear_Output"),
    Candidate("down_depthwise_quant", "/down/depthwise/depthwise.1/Clip_output_0_QuantizeLinear_Output"),
    Candidate("dark_clip_float", "/dark_map/Clip_output_0"),
    Candidate("resize_dark_quant", "/Resize_output_0_QuantizeLinear_Output"),
    Candidate("down_pointwise_quant", "/down/pointwise/pointwise.1/Clip_output_0_QuantizeLinear_Output"),
    Candidate("block0_act_quant", "/blocks/blocks.0/act/Clip_output_0_QuantizeLinear_Output"),
    Candidate("block1_act_quant", "/blocks/blocks.1/act/Clip_output_0_QuantizeLinear_Output"),
    Candidate("block2_act_quant", "/blocks/blocks.2/act/Clip_output_0_QuantizeLinear_Output"),
    Candidate("dark_fuse_quant", "/dark_fuse/dark_fuse.1/Clip_output_0_QuantizeLinear_Output"),
    Candidate("resize1_quant", "/Resize_1_output_0_QuantizeLinear_Output"),
    Candidate("up_quant", "/up/up.1/Clip_output_0_QuantizeLinear_Output"),
    Candidate("add_skip_quant", "/Add_output_0_QuantizeLinear_Output"),
    Candidate("refine_act_quant", "/refine/act/Clip_output_0_QuantizeLinear_Output"),
    Candidate("gain_head_quant", "/Clip_output_0_QuantizeLinear_Output"),
    Candidate("residual_head_quant", "/residual_head/Conv_output_0_QuantizeLinear_Output"),
    Candidate("residual_plus3_float", "/Clip_1_output_0"),
    Candidate("residual_plus3_quant", "/Clip_1_output_0_QuantizeLinear_Output"),
    Candidate("residual_div3_float", "/Div_1_output_0"),
    Candidate("residual_sub1_float", "/Sub_output_0"),
    Candidate("residual_sub1_quant", "/Sub_output_0_QuantizeLinear_Output"),
    Candidate("residual_scaled_quant", "/Mul_1_output_0_QuantizeLinear_Output"),
    Candidate("gain_mul_quant", "/Mul_output_0_QuantizeLinear_Output"),
    Candidate("gain_div_float", "/Div_output_0"),
    Candidate("gain_div_quant", "/Div_output_0_QuantizeLinear_Output"),
    Candidate("gain_add_float", "/Add_1_output_0"),
    Candidate("gain_mapped_quant", "/Add_1_output_0_QuantizeLinear_Output"),
    Candidate("mul2_gain_input_quant", "/Mul_2_output_0_QuantizeLinear_Output"),
    Candidate("enhanced_quant", "enhanced_rgb_QuantizeLinear_Output"),
]


def sanitize(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return safe[:80] or "tensor"


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


def value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = shape_inference.infer_shapes(model)
    infos = {}
    for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        infos[item.name] = item
    return infos


def make_cut_model(model_path: Path, tensor_name: str, out_path: Path) -> None:
    model = onnx.load(model_path)
    infos = value_info_map(model)
    if tensor_name not in infos:
        raise ValueError(f"No shape/type info for tensor {tensor_name!r}")

    del model.graph.output[:]
    model.graph.output.extend([infos[tensor_name]])
    model.graph.name = f"{model.graph.name}_{sanitize(tensor_name)}"
    prune_unused_nodes(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)


def parse_summary(text: str) -> tuple[str, str, str]:
    m_line = "missing"
    c_line = "missing"
    metric_line = "missing"
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("m_outputs_1:"):
            m_line = clean
        elif clean.startswith("c_outputs_1:"):
            c_line = clean
        elif clean.startswith("acc="):
            metric_line = clean
    return m_line, c_line, metric_line


def is_zero_output(c_line: str) -> bool:
    return "min/max=[0, 0]" in c_line or "min/max=[0,0]" in c_line


def run_validate(stedgeai: Path, model_path: Path, label: str, work_root: Path, output_root: Path) -> tuple[int, str]:
    workspace = work_root / label
    output = output_root / label
    cmd = [
        str(stedgeai),
        "validate",
        "--target",
        "stm32h7",
        "--name",
        f"net_{sanitize(label)}",
        "-m",
        str(model_path),
        "--compression",
        "none",
        "--verbosity",
        "1",
        "--workspace",
        str(workspace),
        "--output",
        str(output),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("stm32/onnx/ghost_esp_dark_w12_m24_gain3_res035_plateau_score_best_monitor_qdq.onnx"),
    )
    parser.add_argument("--stedgeai", type=Path, default=DEFAULT_STEDGEAI)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/stedgeai_bisect"))
    parser.add_argument("--only", nargs="*", default=None, help="Optional labels to run.")
    args = parser.parse_args()

    cut_dir = args.out_dir / "models"
    log_dir = args.out_dir / "logs"
    work_root = args.out_dir / "workspace"
    output_root = args.out_dir / "validate_output"
    log_dir.mkdir(parents=True, exist_ok=True)

    candidates = DEFAULT_CANDIDATES
    if args.only:
        selected = set(args.only)
        candidates = [c for c in candidates if c.label in selected]

    print("label,status,tensor,c_zero,m_outputs,c_outputs,metric")
    for candidate in candidates:
        cut_model = cut_dir / f"{candidate.label}.onnx"
        try:
            make_cut_model(args.model, candidate.tensor, cut_model)
        except Exception as exc:
            print(f"{candidate.label},CUT_FAIL,{candidate.tensor},n/a,{exc},,")
            continue

        code, text = run_validate(args.stedgeai, cut_model, candidate.label, work_root, output_root)
        (log_dir / f"{candidate.label}.log").write_text(text, encoding="utf-8", errors="ignore")
        m_line, c_line, metric_line = parse_summary(text)
        status = "OK" if code == 0 else f"VALIDATE_FAIL_{code}"
        c_zero = "YES" if is_zero_output(c_line) else "NO"
        print(
            f"{candidate.label},{status},{candidate.tensor},{c_zero},"
            f"{m_line.replace(',', ';')},{c_line.replace(',', ';')},{metric_line.replace(',', ';')}"
        )


if __name__ == "__main__":
    main()
