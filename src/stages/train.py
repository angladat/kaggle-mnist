import argparse
import random
from pathlib import Path

import yaml
import torch
import numpy as np

from src.data import get_loader
from src.transforms import get_test_transform
from src.models import init_model
from src.train import Trainer, get_optimizer, get_loss_fn, get_scheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(config) -> tuple:
    splits_dir = Path(config['data']['splits'])
    train_loader = get_loader(
        split_path=splits_dir/ "train",
        transform=get_test_transform(),
        batch_size=config['train']['batch_size'],
        shuffle=True,
        seed=config['train']['seed']
    )
    val_loader = get_loader(
        split_path=splits_dir/ "val",
        transform=get_test_transform(),
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

    model_name = config['train']['model']
    model = init_model(model_name)
    loss_fn_name = config['train']['loss_fn']['name']
    loss_fn_params = config['train']['loss_fn']['params']
    loss_fn = get_loss_fn(loss_fn_name, loss_fn_params)

    optimizer_name = config['train']['optimizer']['name']
    optimizer_params = config['train']['optimizer']['params']
    optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)
    
    scheduler = None
    scheduler_name = config['train']['scheduler']['name']
    if scheduler_name is not None:
        scheduler = get_scheduler(scheduler_name, config['train']['scheduler']['params'], optimizer)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        best_checkpoint_path=Path(config['train']['best_path']),
    )

    trainer.fit(epochs=config['train']['epochs'], progress=True)
    trainer.save_checkpoint(config['train']['path'])


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
