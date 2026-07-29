from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def setup_mlflow(experiment_name: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def start_run(run_name: str | None = None):
    return mlflow.start_run(run_name=run_name)


def end_run():
    mlflow.end_run()


def log_params(params: dict):
    if params:
        mlflow.log_params(
            {
                key: "None" if value is None else value
                for key, value in params.items()
            }
        )


def log_metrics(metrics: dict):
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def log_model(model, X_example):
    prediction = model.predict(X_example)
    signature = infer_signature(X_example, prediction)

    return mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_example.head(5),
        serialization_format="cloudpickle",
    )


def log_artifact(path):
    mlflow.log_artifact(str(Path(path)))
