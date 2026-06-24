import torch

from .baseline import BaselineCNN


MODELS = {
    'BaselineCNN' : BaselineCNN,
}
NUM_CLASSES = 10


def init_model(name: str, params: dict | None = None) -> torch.nn.Module:
    if name not in MODELS:
        raise ValueError(f"There is no model with name '{name}'")
    params = params or {}
    params["num_classes"] = NUM_CLASSES
    model = MODELS[name](**params)
    return model


def load_model(name: str, weights_path: str, device: str, params: dict | None = None) -> torch.nn.Module:
    model = init_model(name, params)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    return model


def load_config_model(config: dict, device) -> torch.nn.Module:
    model_name = config['train']['model']
    weights_path = config['train']['path']
    model = load_model(model_name, weights_path, device)
    return model
