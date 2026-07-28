import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    CANDIDATE_METRICS_DIR,
    CANDIDATE_MODEL_DIR,
    TARGET,
    TRAIN_SPLIT_FILE,
    VALID_SPLIT_FILE,
)
from src.evaluate import calculate_metrics, print_metrics
from src.model import MODEL_NAMES
from src.settings import load_params
from src.tuning import tune_model
from src.utils import save_json, save_model


def load_split(path):
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing in {path}")
    return data.drop(columns=[TARGET]), data[TARGET]


def enabled_models(params):
    configured_models = params["models"]
    unknown = set(configured_models).difference(MODEL_NAMES)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown models in params.yaml: {names}")

    return [
        name
        for name in MODEL_NAMES
        if configured_models.get(name, {}).get("enabled", False)
    ]


def train_candidate(
    model_name,
    *,
    params,
    train_path=TRAIN_SPLIT_FILE,
    valid_path=VALID_SPLIT_FILE,
    model_dir=CANDIDATE_MODEL_DIR,
    metrics_dir=CANDIDATE_METRICS_DIR,
    n_trials_override=None,
):
    if model_name not in enabled_models(params):
        raise ValueError(f"Model '{model_name}' is not enabled")

    X_train, y_train = load_split(train_path)
    X_valid, y_valid = load_split(valid_path)

    data_params = params["data"]
    training_params = params["training"]
    n_trials = (
        n_trials_override
        if n_trials_override is not None
        else training_params["n_trials"]
    )

    pipeline, study = tune_model(
        model_name,
        X_train,
        y_train,
        random_state=data_params["random_state"],
        n_trials=n_trials,
        cv_folds=training_params["cv_folds"],
        scoring=training_params["scoring"],
        n_jobs=training_params["n_jobs"],
    )

    metrics = calculate_metrics(pipeline, X_valid, y_valid)
    metrics.update(
        {
            "model": model_name,
            "cv_score": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": n_trials,
        }
    )

    model_path = Path(model_dir) / f"{model_name}.pkl"
    metrics_path = Path(metrics_dir) / f"{model_name}.json"
    save_model(pipeline, model_path)
    save_json(metrics, metrics_path)
    print_metrics(model_name, metrics)
    return metrics


def train_all(params, **kwargs):
    model_names = enabled_models(params)
    if not model_names:
        raise ValueError("At least one model must be enabled")

    results = {}
    for model_name in model_names:
        results[model_name] = train_candidate(
            model_name,
            params=params,
            **kwargs,
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune and evaluate Titanic model candidates."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=MODEL_NAMES)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--train-data", default=str(TRAIN_SPLIT_FILE))
    parser.add_argument("--valid-data", default=str(VALID_SPLIT_FILE))
    parser.add_argument("--model-dir", default=str(CANDIDATE_MODEL_DIR))
    parser.add_argument("--metrics-dir", default=str(CANDIDATE_METRICS_DIR))
    parser.add_argument("--n-trials", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params(args.params)
    common_args = {
        "train_path": args.train_data,
        "valid_path": args.valid_data,
        "model_dir": args.model_dir,
        "metrics_dir": args.metrics_dir,
        "n_trials_override": args.n_trials,
    }

    if args.model:
        train_candidate(args.model, params=params, **common_args)
    else:
        train_all(params, **common_args)


if __name__ == "__main__":
    main()
