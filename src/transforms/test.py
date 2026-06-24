from torchvision import transforms

from .common import DEFAULT_TRANSFORMS


def get_test_transform() -> transforms.Compose:
    return transforms.Compose(DEFAULT_TRANSFORMS)
