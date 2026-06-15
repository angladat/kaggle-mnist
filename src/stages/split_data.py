import argparse
import random
import shutil
from pathlib import Path

import yaml


def validate_split_sizes(val_size: float, test_size: float) -> None:
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be in [0, 1)")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")


def copy_images(images: list[Path], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, output_path / image_path.name)


def split_label_images(
    label_path: Path,
    splits_path: Path,
    val_size: float,
    test_size: float,
    rng: random.Random,
) -> None:
    images = sorted(label_path.glob("*.png"))
    rng.shuffle(images)

    val_count = int(len(images) * val_size)
    test_count = int(len(images) * test_size)
    train_count = len(images) - val_count - test_count

    split_images = {
        "train": images[:train_count],
        "val": images[train_count:train_count + val_count],
        "test": images[train_count + val_count:],
    }

    for split_name, image_paths in split_images.items():
        copy_images(image_paths, splits_path / split_name / label_path.name)


def split_data(config_path: str) -> None:
    """
    Разделение данных на train, val и test

    :param config_path: Путь к конфигу
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    processed_train_path = Path(config["data"]["saved"]) / "train"
    splits_path = Path(config["data"]["splits"])
    seed = config["split"]["seed"]
    val_size = config["split"]["val_size"]
    test_size = config["split"]["test_size"]

    validate_split_sizes(val_size=val_size, test_size=test_size)

    if not processed_train_path.exists():
        raise FileNotFoundError(f"Processed train directory not found: {processed_train_path}")

    if splits_path.exists():
        shutil.rmtree(splits_path)

    rng = random.Random(seed)
    label_paths = sorted(path for path in processed_train_path.iterdir() if path.is_dir())

    for label_path in label_paths:
        split_label_images(
            label_path=label_path,
            splits_path=splits_path,
            val_size=val_size,
            test_size=test_size,
            rng=rng,
        )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    split_data(config_path=args.config)
