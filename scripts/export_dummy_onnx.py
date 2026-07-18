from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import onnx

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import StudentD96, StudentD96TinyBN, StudentGhostESPDark, StudentPConv12
from src.utils.seed import set_seed


def strip_identity_nodes(path: Path) -> None:
    model = onnx.load(path)
    identity_map = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
    }
    if not identity_map:
        return
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            while value in identity_map:
                value = identity_map[value]
            node.input[index] = value
    keep_nodes = [node for node in model.graph.node if node.op_type != "Identity"]
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("stm32/onnx/d_96_dummy.onnx"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dark-guidance", type=int, choices=[0, 1], default=1)
    parser.add_argument("--arch", type=str, choices=["d96", "d96-tiny-bn", "pconv12", "ghost-esp-dark"], default="d96-tiny-bn")
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--mid-channels", type=int, default=8)
    parser.add_argument("--no-bn-fold", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.arch == "d96":
        model = StudentD96(
            blocks=args.blocks,
            dark_guidance=bool(args.dark_guidance),
        ).eval()
    elif args.arch == "d96-tiny-bn":
        model = StudentD96TinyBN(
            base_channels=args.base_channels,
            mid_channels=args.mid_channels,
            blocks=args.blocks,
            dark_guidance=bool(args.dark_guidance),
        )
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state_dict = (
                checkpoint["model"]
                if isinstance(checkpoint, dict) and "model" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)
            print(f"loaded_checkpoint={args.checkpoint}")
        model = model.eval()
        if not args.no_bn_fold:
            model = model.fuse_bn_for_export().eval()
    elif args.arch == "pconv12":
        model = StudentPConv12(
            base_channels=12,
            blocks=args.blocks,
            pconv_channels=4,
        )
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state_dict = (
                checkpoint["model"]
                if isinstance(checkpoint, dict) and "model" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)
            print(f"loaded_checkpoint={args.checkpoint}")
        model = model.eval()
        if not args.no_bn_fold:
            model = model.fuse_bn_for_export().eval()
    else:
        model = StudentGhostESPDark(
            base_channels=args.base_channels,
            mid_channels=args.mid_channels,
            blocks=args.blocks,
        )
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state_dict = (
                checkpoint["model"]
                if isinstance(checkpoint, dict) and "model" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)
            print(f"loaded_checkpoint={args.checkpoint}")
        model = model.eval()
        if not args.no_bn_fold:
            model = model.fuse_bn_for_export().eval()
    if args.arch == "d96" and args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.eval()
        print(f"loaded_checkpoint={args.checkpoint}")
    dummy = torch.rand(1, 3, args.image_size, args.image_size)

    with torch.no_grad():
        output = model(dummy)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input_rgb"],
        output_names=["enhanced_rgb"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    strip_identity_nodes(args.out)
    params = sum(param.numel() for param in model.parameters())
    print(f"exported={args.out}")
    print(f"input_shape={tuple(dummy.shape)} output_shape={tuple(output.shape)}")
    print(f"params={params}")
    print("next=import this ONNX into STM32Cube.AI before long training")


if __name__ == "__main__":
    main()
