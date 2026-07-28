import json
from pathlib import Path

import joblib


def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model, path):
    path = Path(path)
    create_directory(path.parent)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_json(data, path):
    path = Path(path)
    create_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, sort_keys=True)


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)
