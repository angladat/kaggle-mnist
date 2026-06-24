import argparse
import csv
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.data import get_default_transform
from src.models import BaselineCNN


class KaggleTestDataset(Dataset):
    def __init__(self, test_path: Path) -> None:
        self.image_paths = sorted(test_path.rglob("*.png"))
        self.transform = get_default_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        image_path = self.image_paths[index]
        image_id = int(image_path.stem)

        with image_path.open("rb") as image_file:
            image = Image.open(image_file).convert("L")

        return image_id, self.transform(image)


def predict(config_path: str) -> None:
    """
    Предсказания для каггла

    :param config_path: Путь к конфигу
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)


    test_dir = Path(config['data']['saved']) / "test"
    dataset = KaggleTestDataset(test_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineCNN(num_classes=10)
    model.load_state_dict(
        torch.load(
            config['train']['path'],
            map_location=device
        )
    )
    model.to(device)

    predictions: list[tuple[int, int]] = []
    with torch.no_grad():
        for image_ids, images in dataloader:
            images = images.to(device)
            labels = model(images).argmax(dim=1).cpu().tolist()

            for image_id, label in zip(image_ids.tolist(), labels):
                predictions.append((image_id, label))

    predictions.sort(key=lambda prediction: prediction[0])

    submission_path = Path(config['predict']['submission_path'])
    with submission_path.open("w", newline="") as submission_file:
        writer = csv.writer(submission_file)
        writer.writerow(["ImageId", "Label"])
        writer.writerows(predictions)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    predict(config_path=args.config)
