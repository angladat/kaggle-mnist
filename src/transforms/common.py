from torchvision import transforms


DEFAULT_TRANSFORMS = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
