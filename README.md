# Kaggle MNIST — Digit Recognizer

A DVC-orchestrated PyTorch pipeline for the Kaggle [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) (MNIST) competition: downloads the data, trains a model, evaluates it, and produces a submission file.

## Pipeline

The pipeline is defined in [`dvc.yaml`](dvc.yaml) and parameterized via [`params.yaml`](params.yaml):

| Stage | Description | Output |
|---|---|---|
| `download_data` | Downloads the competition data via the Kaggle API | `data/digit-recognizer.zip` |
| `unzip_data` | Unzips the raw CSVs | `data/raw/` |
| `save_images` | Converts CSV pixel rows into PNG images | `data/saved/` |
| `split_data` | Splits into train/val/test sets | `data/splits/` |
| `train` | Trains the CNN model | `models/weights.pt` |
| `evaluate` | Evaluates the trained model on the test split | `reports/metrics.json` |
| `predict` | Generates predictions for the Kaggle test set | `reports/submission.csv` |

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Downloading data requires Kaggle API credentials. Create a `.env` file in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

You also need to have joined the [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition on Kaggle to download its data.

## Usage

Run the full pipeline:

```bash
uv run dvc repro
```

Run a single stage (and its dependencies):

```bash
uv run dvc repro train
```

Adjust hyperparameters (model, optimizer, batch size, epochs, train/val/test split, etc.) in [`params.yaml`](params.yaml).

## Results

Current metrics ([`reports/metrics.json`](reports/metrics.json)):

- Test accuracy: **99.38%**
- Test loss: 0.0281

Predictions for submission are written to `reports/submission.csv`.

## Project structure

```
src/
  stages/        # DVC pipeline stage entry points
  models/        # model definitions
  trainer.py     # training/evaluation loop
  data.py        # dataset utilities
params.yaml       # pipeline parameters
dvc.yaml          # pipeline stage definitions
dvc.lock          # pinned stage outputs/hashes
```
