from pathlib import Path
import joblib


def create_directory(path):
    """
    Создает директорию, если ее нет.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )

def save_model(model, path):
    """
    Сохраняет модель.
    """

    joblib.dump(
        model,
        path,
    )

def load_model(path):
    """
    Загружает модель.
    """

    return joblib.load(path)