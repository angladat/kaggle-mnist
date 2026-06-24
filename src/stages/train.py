import argparse
import random
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import numpy as np

from src.data import get_loader, get_default_transform
from src.models import BaselineCNN
from src.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(config) -> tuple:
    splits_dir = Path(config['data']['splits'])
    train_loader = get_loader(
        split_path=splits_dir/ "train",
        transform=get_default_transform(),
        batch_size=config['train']['batch_size'],
        shuffle=True,
        seed=config['train']['seed']
    )
    val_loader = get_loader(
        split_path=splits_dir/ "val",
        transform=get_default_transform(),
        batch_size=config['train']['batch_size'],
        seed=config['train']['seed']
    )
    return train_loader, val_loader



def train(config_path: str) -> None:
    """
    Обучение модели

    :param config_path: Путь к конфигу
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    set_seed(config['train']['seed'])

    train_loader, val_loader = get_loaders(config)

    model = BaselineCNN(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_params = config['train']['optimizer']['params']
    optimizer = torch.optim.Adam(model.parameters(), float(optimizer_params['lr']))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    trainer.fit(epochs=config['train']['epochs'], progress=True)
    trainer.save_checkpoint(config['train']['path'])


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
