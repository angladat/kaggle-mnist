import argparse
from pathlib import Path

import yaml
import json
import torch
import torch.nn as nn

from src.data import get_loader
from src.transforms import get_test_transform
from src.models import load_config_model
from src.train import Trainer


def evaluate(config_path: str) -> None:
    """
    Оценка модели

    :param config_path: Путь к конфигу
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    splits_dir = Path(config['data']['splits'])
    test_loader = get_loader(
        split_path=splits_dir/ "test",
        transform=get_test_transform(),
        batch_size=config['train']['batch_size'],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_config_model(config, device)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=None,
        val_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=None,
    )

    test_loss, test_acc = trainer.evaluate()

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }

    json.dump(
        obj=metrics,
        fp=open(config['evaluate']['metrics_file'], 'w')
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)
