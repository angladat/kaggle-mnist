from typing import Any

import torch


OPTIMIZERS = {
    "Adam" : torch.optim.Adam,
    "AdamW" : torch.optim.AdamW,
}


LOSS_FNS = {
    "CrossEntropyLoss" : torch.nn.CrossEntropyLoss,
}

SCHEDULERS = {
    "CosineAnnealingLR" : torch.optim.lr_scheduler.CosineAnnealingLR,
}


def get_optimizer(
        name: str,
        model_params: Any,
        optimizer_params: dict,
        scheduler_name: str | None = None,
        scheduler_params: dict[str, Any] | None = None,
    ) -> Any:
    if name not in OPTIMIZERS:
        raise ValueError(f"There is no optimizer named {name}") 
    optimizer = OPTIMIZERS[name](model_params, **optimizer_params)
    return optimizer


def get_scheduler(name: str, params: dict[str, Any], optimizer: Any) -> Any:
    if name not in SCHEDULERS:
        raise ValueError(f"There is no scheduler named {name}")
    scheduler = SCHEDULERS[name](
        optimizer,
        params["T_max"],
        float(params["eta_min"]),
    )
    return scheduler


def get_loss_fn(name: str, params: dict | None = None) -> Any:
    params = params or {}
    if name not in LOSS_FNS:
        raise ValueError(f"There is no loss function named {name}") 
    return LOSS_FNS[name](**params)
