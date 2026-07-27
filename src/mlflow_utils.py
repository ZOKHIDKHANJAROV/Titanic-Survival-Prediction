import mlflow
import mlflow.sklearn


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


def log_model(model, artifact_path="model"):
    """
    Сохранение модели.
    """

    mlflow.sklearn.log_model(
        sk_model=model,
        name=artifact_path,
        serialization_format="pickle",
    )


def log_artifact(path):
    """
    Сохранение файла.
    """

    mlflow.log_artifact(path)