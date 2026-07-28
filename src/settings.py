from pathlib import Path

import yaml

from src.config import PARAMS_FILE


def load_params(path=PARAMS_FILE):
    params_path = Path(path)
    with params_path.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)

    required_sections = {"data", "training", "models"}
    missing_sections = required_sections.difference(params)
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"Missing params.yaml sections: {missing}")

    return params
