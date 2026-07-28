import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET,
    TRAIN_DATA,
    TRAIN_SPLIT_FILE,
    VALID_SPLIT_FILE,
)
from src.settings import load_params
from src.utils import create_directory


def prepare_data(
    input_path,
    train_output,
    valid_output,
    *,
    test_size,
    random_state,
):
    data = pd.read_csv(input_path)
    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing")

    train_data, valid_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET],
    )

    train_output = Path(train_output)
    valid_output = Path(valid_output)
    create_directory(train_output.parent)
    create_directory(valid_output.parent)
    train_data.to_csv(train_output, index=False)
    valid_data.to_csv(valid_output, index=False)

    print(
        f"Prepared {len(train_data)} train rows and "
        f"{len(valid_data)} validation rows."
    )
    return train_output, valid_output


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Titanic train split.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=str(TRAIN_DATA))
    parser.add_argument("--train-output", default=str(TRAIN_SPLIT_FILE))
    parser.add_argument("--valid-output", default=str(VALID_SPLIT_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params(args.params)
    data_params = params["data"]
    prepare_data(
        args.input,
        args.train_output,
        args.valid_output,
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
    )


if __name__ == "__main__":
    main()
