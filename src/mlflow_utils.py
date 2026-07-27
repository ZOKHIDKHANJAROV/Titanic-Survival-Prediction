import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def setup_mlflow(
    experiment_name: str,
):
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        experiment_name,
    )


def start_run(run_name: str | None = None):
    """
    Начать эксперимент.
    """

    return mlflow.start_run(run_name=run_name)


def end_run():
    """
    Завершить эксперимент.
    """

    mlflow.end_run()


def log_params(params: dict):
    """
    Логирование параметров.
    """

    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict):
    """
    Логирование метрик.
    """

    if metrics:
        mlflow.log_metrics(metrics)


def log_model(
    model,
    X_example,
):
    prediction = model.predict(X_example)

    signature = infer_signature(
        X_example,
        prediction,
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_example.head(5),
        serialization_format="cloudpickle",
    )

    return model_info


def log_artifact(path):
    """
    Сохранение файла.
    """

    mlflow.log_artifact(path)