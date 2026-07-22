from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocessing import create_preprocessor

from config import RANDOM_STATE

def build_pipeline():
    """
    Создает ML Pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )

    return pipeline