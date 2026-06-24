from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def grayscale_loader(path: str) -> Image.Image:
    with open(path, "rb") as image_file:
        return Image.open(image_file).convert("L")


def get_loader(
        split_path: Path,
        transform: transforms.Compose,
        batch_size: int = 128,
        shuffle: bool = False,
        seed: int = 42,
    ) -> DataLoader:
    """Создает DataLoader сплита и возвращает результат"""

    dataset = datasets.ImageFolder(
        root=split_path,
        transform=transform,
        loader=grayscale_loader,
    )
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
