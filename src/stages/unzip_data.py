import argparse
import zipfile
from pathlib import Path

import yaml


def unzip_data(config_path: str) -> None:
    """
    Распаковка данных

    :param config_path: Путь к конфигу
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data_root = Path(config["data"]["root"])
    raw_data_path = Path(config["data"]["raw"])
    zip_path = data_root / "digit-recognizer.zip"

    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    raw_data_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(raw_data_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    unzip_data(config_path=args.config)
