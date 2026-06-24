from torchvision import transforms

from .common import DEFAULT_TRANSFORMS


def get_train_transform() -> transforms.Compose:
    transforms_ = [
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5,),
    ]
    return transforms.Compose(transforms_.extend(DEFAULT_TRANSFORMS))
