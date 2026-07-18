from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        image_paths: list[Path],
        image_size: int,
        random_count: int,
    ) -> None:
        self.input_name = input_name
        self.image_size = image_size
        self.samples = self._load_samples(image_paths)
        if not self.samples:
            self.samples = self._make_random_samples(random_count)
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return {self.input_name: sample}

    def _load_samples(self, image_paths: list[Path]) -> list[np.ndarray]:
        samples: list[np.ndarray] = []
        for path in image_paths:
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            tensor = TF.to_tensor(image)
            tensor = self._center_square_crop(tensor)
            tensor = TF.resize(tensor, [self.image_size, self.image_size], antialias=True)
            samples.append(tensor.unsqueeze(0).numpy().astype(np.float32))
        return samples

    def _make_random_samples(self, random_count: int) -> list[np.ndarray]:
        rng = np.random.default_rng(42)
        samples: list[np.ndarray] = []
        for _ in range(random_count):
            sample = rng.uniform(
                low=0.0,
                high=0.35,
                size=(1, 3, self.image_size, self.image_size),
            ).astype(np.float32)
            samples.append(sample)
        return samples

    @staticmethod
    def _center_square_crop(tensor):
        _, height, width = tensor.shape
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return tensor[:, top : top + side, left : left + side]


def read_low_paths(split_files: list[Path], limit: int) -> list[Path]:
    paths: list[Path] = []
    for split_file in split_files:
        if not split_file.exists():
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            paths.append(Path(parts[0]))
            if len(paths) >= limit:
                return paths
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("stm32/onnx/d_96_tiny_bn_fused_dummy.onnx"))
    parser.add_argument("--output", type=Path, default=Path("stm32/onnx/d_96_tiny_bn_qdq_dummy.onnx"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--calib-split", type=Path, nargs="*", default=[Path("splits/lol_train.txt")])
    parser.add_argument("--calib-limit", type=int, default=32)
    parser.add_argument("--random-count", type=int, default=32)
    args = parser.parse_args()

    image_paths = read_low_paths(args.calib_split, args.calib_limit)
    reader = ImageCalibrationReader(
        input_name="input_rgb",
        image_paths=image_paths,
        image_size=args.image_size,
        random_count=args.random_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(args.input),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"exported={args.output}")
    print(f"calibration_images={len(image_paths)}")
    print("next=import this QDQ ONNX into STM32Cube.AI and run Analyze")


if __name__ == "__main__":
    main()
