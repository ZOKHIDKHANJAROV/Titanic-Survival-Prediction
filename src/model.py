from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import create_preprocessor
from config import RANDOM_STATE


MODELS = {
    "logistic": LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    ),

    "decision_tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
    ),

    "random_forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
    ),
}


def build_pipeline(model_name: str):

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                MODELS[model_name],
            ),
        ]
    )