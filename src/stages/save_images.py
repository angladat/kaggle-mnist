import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


IMAGE_SIZE = 28
PIXELS_COUNT = IMAGE_SIZE * IMAGE_SIZE


def save_grayscale_png(path: Path, pixels: list[int]) -> None:
    if len(pixels) != PIXELS_COUNT:
        raise ValueError(f"Expected {PIXELS_COUNT} pixels, got {len(pixels)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.array(pixels, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
    Image.fromarray(image, mode="L").save(path)


def save_dataset_images(csv_path: Path, output_path: Path, has_labels: bool) -> None:
    with csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        for row_number, row in enumerate(reader, start=1):
            if has_labels:
                label = row[0]
                pixels = [int(value) for value in row[1:]]
                image_path = output_path / label / f"{row_number:05d}.png"
            else:
                pixels = [int(value) for value in row]
                image_path = output_path / f"{row_number:05d}.png"

            save_grayscale_png(image_path, pixels)


def save_images(config_path: str) -> None:
    """
    Сохранение данных в виде изображений

    :param config_path: Путь к конфигу
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    raw_data_path = Path(config["data"]["raw"])
    processed_data_path = Path(config["data"]["saved"])

    if processed_data_path.exists():
        shutil.rmtree(processed_data_path)

    save_dataset_images(
        csv_path=raw_data_path / "train.csv",
        output_path=processed_data_path / "train",
        has_labels=True,
    )
    save_dataset_images(
        csv_path=raw_data_path / "test.csv",
        output_path=processed_data_path / "test",
        has_labels=False,
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    save_images(config_path=args.config)
