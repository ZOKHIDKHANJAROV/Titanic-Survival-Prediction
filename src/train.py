import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    TRAIN_DATA,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_FILE,
)

from feature_engineering import add_features
from preprocessing import drop_columns
from model import build_pipeline
from evaluate import evaluate_model
from utils import save_model


def load_data():
    """
    Загружает данные.
    """
    data = pd.read_csv(TRAIN_DATA)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    return X, y


def prepare_data():
    """
    Подготавливает данные.
    """

    X, y = load_data()

    # Feature Engineering
    X = add_features(X)

    # Удаляем ненужные признаки
    X = drop_columns(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_valid, y_train, y_valid


def train_model():

    X_train, X_valid, y_train, y_valid = prepare_data()

    pipeline = build_pipeline()

    pipeline.fit(
        X_train,
        y_train,
    )

    evaluate_model(
        pipeline,
        X_valid,
        y_valid,
    )

    save_model(
        pipeline,
        MODEL_FILE,
    )


def main():

    train_model()


if __name__ == "__main__":
    main()