from torchvision import transforms

from .common import DEFAULT_TRANSFORMS


def get_train_transform() -> transforms.Compose:
    transforms_ = [
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5,),
        transforms.ElasticTransform(alpha=20.0, sigma=4.0,),
    ]
    transforms_.extend(DEFAULT_TRANSFORMS)
    return transforms.Compose(transforms_)
