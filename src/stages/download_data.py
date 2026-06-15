import argparse

import yaml
from dotenv import load_dotenv

load_dotenv()


def download_data(config_path: str) -> None:
    """
    Создание данных

    :param config_path: Путь к конфигу
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files(
        competition="digit-recognizer",
        path=config["data"]["root"],
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()
    download_data(config_path=args.config)
