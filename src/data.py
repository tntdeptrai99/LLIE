from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]


class PairedImageDataset(Dataset):
    def __init__(
        self,
        split: Path | str,
        image_size: int = 96,
        mode: str = "train",
        augment: bool = False,
        synthetic_ratio: float = 0.0,
        light_augment_prob: float = 0.0,
    ) -> None:
        self.split = Path(split)
        if not self.split.is_absolute():
            self.split = ROOT / self.split
        self.image_size = image_size
        self.mode = mode
        self.augment = augment
        self.synthetic_ratio = synthetic_ratio
        self.light_augment_prob = light_augment_prob
        self.pairs = self._read_split(self.split)

    @staticmethod
    def _read_split(split: Path) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for raw in split.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            low = Path(parts[0])
            high = Path(parts[1])
            if not low.is_absolute():
                low = ROOT / low
            if not high.is_absolute():
                high = ROOT / high
            pairs.append((low, high))
        if not pairs:
            raise ValueError(f"No image pairs found in split: {split}")
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        low_path, high_path = self.pairs[index]
        low = Image.open(low_path).convert("RGB")
        high = Image.open(high_path).convert("RGB")

        low = TF.resize(low, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BICUBIC)
        high = TF.resize(high, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BICUBIC)

        if self.augment:
            if random.random() < 0.5:
                low = TF.hflip(low)
                high = TF.hflip(high)
            if random.random() < 0.5:
                low = TF.vflip(low)
                high = TF.vflip(high)

        low_t = TF.to_tensor(low)
        high_t = TF.to_tensor(high)

        if self.augment and self.light_augment_prob > 0 and random.random() < self.light_augment_prob:
            factor = random.uniform(0.75, 1.10)
            low_t = (low_t * factor).clamp(0.0, 1.0)

        return {
            "low": low_t,
            "high": high_t,
            "low_path": str(low_path),
            "high_path": str(high_path),
        }


class DistillImageDataset(Dataset):
    def __init__(
        self,
        split: Path | str,
        image_size: int = 96,
        mode: str = "train",
        augment: bool = False,
    ) -> None:
        self.split = Path(split)
        if not self.split.is_absolute():
            self.split = ROOT / self.split
        self.image_size = image_size
        self.mode = mode
        self.augment = augment
        self.rows = self._read_split(self.split)

    @staticmethod
    def _read_split(split: Path) -> list[tuple[Path, Path, Path]]:
        rows: list[tuple[Path, Path, Path]] = []
        for raw in split.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            low, high, teacher = (Path(parts[0]), Path(parts[1]), Path(parts[2]))
            if not low.is_absolute():
                low = ROOT / low
            if not high.is_absolute():
                high = ROOT / high
            if not teacher.is_absolute():
                teacher = ROOT / teacher
            rows.append((low, high, teacher))
        if not rows:
            raise ValueError(f"No distillation rows found in split: {split}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        low_path, high_path, teacher_path = self.rows[index]
        low = Image.open(low_path).convert("RGB")
        high = Image.open(high_path).convert("RGB")
        teacher = Image.open(teacher_path).convert("RGB")

        low = TF.resize(low, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BICUBIC)
        high = TF.resize(high, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BICUBIC)
        teacher = TF.resize(teacher, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BICUBIC)

        if self.augment:
            if random.random() < 0.5:
                low = TF.hflip(low)
                high = TF.hflip(high)
                teacher = TF.hflip(teacher)
            if random.random() < 0.5:
                low = TF.vflip(low)
                high = TF.vflip(high)
                teacher = TF.vflip(teacher)

        return {
            "low": TF.to_tensor(low),
            "high": TF.to_tensor(high),
            "teacher": TF.to_tensor(teacher),
            "low_path": str(low_path),
            "high_path": str(high_path),
            "teacher_path": str(teacher_path),
        }
