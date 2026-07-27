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

from tuning import optuna_tuning

from evaluate import (
    calculate_metrics,
    print_metrics,
)

from mlflow_utils import (
    setup_mlflow,
    start_run,
    end_run,
    log_params,
    log_metrics,
    log_model,
)

from utils import save_model

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

    setup_mlflow(
        experiment_name="Titanic RandomForest",
    )

    run = start_run(
        run_name="Optuna RandomForest",
    )

    try:

        X_train, X_valid, y_train, y_valid = prepare_data()

        best_model, study = optuna_tuning(
            X_train,
            y_train,
            n_trials=100,
        )

        metrics, prediction = calculate_metrics(
            best_model,
            X_valid,
            y_valid,
        )

        print_metrics(
            metrics,
            y_valid,
            prediction,
        )

        log_params(
            study.best_params,
        )

        log_metrics(
            metrics,
        )

        log_model(
            best_model,
        )

        save_model(
            best_model,
            MODEL_FILE,
        )

        print()

        print("Model saved successfully.")

    finally:

        end_run()

def main():

    train()


if __name__ == "__main__":
    main()