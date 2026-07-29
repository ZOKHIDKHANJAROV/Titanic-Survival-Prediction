from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import TitanicFeatureEngineer
from src.preprocessing import create_preprocessor


MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
)


def build_estimator(
    model_name: str,
    *,
    random_state: int,
    model_params: dict | None = None,
):
    params = dict(model_params or {})

    if model_name == "logistic_regression":
        return LogisticRegression(
            random_state=random_state,
            max_iter=2000,
            solver="liblinear",
            **params,
        )

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=random_state,
            **params,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=1,
            **params,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            random_state=random_state,
            n_jobs=1,
            **params,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=random_state,
            **params,
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(
    model_name: str,
    *,
    random_state: int,
    model_params: dict | None = None,
):
    estimator = build_estimator(
        model_name,
        random_state=random_state,
        model_params=model_params,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", TitanicFeatureEngineer()),
            ("preprocessor", create_preprocessor()),
            ("model", estimator),
        ]
    )
