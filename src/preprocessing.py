from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    NUMERIC_FEATURES,
)


def drop_columns(df):
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def create_numeric_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def create_categorical_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                create_numeric_pipeline(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                create_categorical_pipeline(),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
