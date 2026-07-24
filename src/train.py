import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    TRAIN_DATA,
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_FILE,
)

from feature_engineering import add_features
from preprocessing import drop_columns
from model import build_pipeline
from evaluate import evaluate_model
from utils import save_model
from tuning import optuna_tuning

def load_data():

    data = pd.read_csv(TRAIN_DATA)

    X = data.drop(columns=[TARGET])

    y = data[TARGET]

    return X, y


def prepare_data():

    X, y = load_data()

    X = add_features(X)

    X = drop_columns(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_valid, y_train, y_valid


def train():

    X_train, X_valid, y_train, y_valid = prepare_data()

    pipeline = build_pipeline(
        "random_forest",
    )

    best_model, study = optuna_tuning(
        X_train,
        y_train,
        n_trials=100,
    )

    print()

    print("=" * 60)
    print("Validation")
    print("=" * 60)

    evaluate_model(
        best_model,
        X_valid,
        y_valid,
    )

    save_model(
        best_model,
        MODEL_FILE,
    )

    print()

    print("Model saved successfully.")


def main():

    train()


if __name__ == "__main__":
    main()