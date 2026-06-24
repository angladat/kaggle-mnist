from typing import Any

import torch


OPTIMIZERS = {
    "Adam" : torch.optim.Adam,
}


LOSS_FNS = {
    "CrossEntropyLoss" : torch.nn.CrossEntropyLoss,
}


def get_optimizer(name: str, model_params: Any, optimizer_params: dict) -> Any:
    if name not in OPTIMIZERS:
        raise ValueError(f"There is no optimizer named {name}") 
    return OPTIMIZERS[name](model_params, **optimizer_params)


def get_loss_fn(name: str, params: dict | None = None) -> Any:
    params = params or {}
    if name not in LOSS_FNS:
        raise ValueError(f"There is no loss function named {name}") 
    return LOSS_FNS[name](**params)
