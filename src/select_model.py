import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    CANDIDATE_METRICS_DIR,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    SUMMARY_METRICS_FILE,
    TARGET,
    TRAIN_SPLIT_FILE,
    VALID_SPLIT_FILE,
)
from src.model import build_pipeline
from src.mlflow_utils import (
    end_run,
    log_artifact,
    log_metrics,
    log_model,
    log_params,
    setup_mlflow,
    start_run,
)
from src.settings import load_params
from src.train import enabled_models
from src.utils import load_json, save_json, save_model


METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def track_selected_model(
    *,
    params,
    model,
    X_example,
    metadata,
    best_result,
    metadata_path,
    summary_path,
):
    tracking = params.get("mlflow", {})
    if not tracking.get("enabled", False):
        return None

    setup_mlflow(
        experiment_name=tracking["experiment_name"],
        tracking_uri=tracking["tracking_uri"],
    )
    run = start_run(run_name=f"selected-{metadata['model']}")
    try:
        log_params(
            {
                "model": metadata["model"],
                "selection_metric": metadata["selection_metric"],
                "training_rows": metadata["training_rows"],
                **{
                    f"best_{key}": value
                    for key, value in metadata["best_params"].items()
                },
            }
        )
        log_metrics(
            {
                **{
                    f"validation_{metric}": best_result[metric]
                    for metric in METRIC_NAMES
                },
                "cv_score": metadata["cv_score"],
            }
        )
        log_model(model, X_example)
        log_artifact(metadata_path)
        log_artifact(summary_path)
        return run.info.run_id
    finally:
        end_run()


def select_and_refit(
    *,
    params,
    train_path=TRAIN_SPLIT_FILE,
    valid_path=VALID_SPLIT_FILE,
    metrics_dir=CANDIDATE_METRICS_DIR,
    model_output=MODEL_FILE,
    metadata_output=MODEL_METADATA_FILE,
    summary_output=SUMMARY_METRICS_FILE,
):
    model_names = enabled_models(params)
    if not model_names:
        raise ValueError("At least one model must be enabled")

    candidate_results = {}
    for model_name in model_names:
        metrics_path = Path(metrics_dir) / f"{model_name}.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics for '{model_name}' are missing: {metrics_path}"
            )
        candidate_results[model_name] = load_json(metrics_path)

    selection_metric = params["training"]["selection_metric"]
    if selection_metric not in METRIC_NAMES:
        raise ValueError(
            f"Unsupported selection metric: {selection_metric}. "
            f"Choose one of {', '.join(METRIC_NAMES)}."
        )

    best_name = max(
        model_names,
        key=lambda name: candidate_results[name][selection_metric],
    )
    best_result = candidate_results[best_name]

    full_data = pd.concat(
        [
            pd.read_csv(train_path),
            pd.read_csv(valid_path),
        ],
        ignore_index=True,
    )
    X_full = full_data.drop(columns=[TARGET])
    y_full = full_data[TARGET]

    best_pipeline = build_pipeline(
        best_name,
        random_state=params["data"]["random_state"],
        model_params=best_result["best_params"],
    )
    best_pipeline.fit(X_full, y_full)
    save_model(best_pipeline, model_output)

    summary = {
        "best_score": float(best_result[selection_metric]),
        "models": {
            model_name: {
                metric: float(candidate_results[model_name][metric])
                for metric in METRIC_NAMES
            }
            for model_name in model_names
        },
    }
    metadata = {
        "model": best_name,
        "selection_metric": selection_metric,
        "selection_score": float(best_result[selection_metric]),
        "cv_score": float(best_result["cv_score"]),
        "best_params": best_result["best_params"],
        "training_rows": int(len(full_data)),
    }

    save_json(summary, summary_output)
    save_json(metadata, metadata_output)
    run_id = track_selected_model(
        params=params,
        model=best_pipeline,
        X_example=X_full,
        metadata=metadata,
        best_result=best_result,
        metadata_path=metadata_output,
        summary_path=summary_output,
    )
    print(
        f"Selected '{best_name}' with "
        f"{selection_metric}={best_result[selection_metric]:.4f}. "
        f"Refit on {len(full_data)} rows."
    )
    if run_id:
        print(f"Logged selected model to MLflow run {run_id}.")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the best candidate and refit it on all data."
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--train-data", default=str(TRAIN_SPLIT_FILE))
    parser.add_argument("--valid-data", default=str(VALID_SPLIT_FILE))
    parser.add_argument("--metrics-dir", default=str(CANDIDATE_METRICS_DIR))
    parser.add_argument("--model-output", default=str(MODEL_FILE))
    parser.add_argument("--metadata-output", default=str(MODEL_METADATA_FILE))
    parser.add_argument("--summary-output", default=str(SUMMARY_METRICS_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    select_and_refit(
        params=load_params(args.params),
        train_path=args.train_data,
        valid_path=args.valid_data,
        metrics_dir=args.metrics_dir,
        model_output=args.model_output,
        metadata_output=args.metadata_output,
        summary_output=args.summary_output,
    )


if __name__ == "__main__":
    main()
