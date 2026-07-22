from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from config import (
    DROP_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

def drop_columns(df):
    return df.drop(columns=DROP_COLUMNS)

def create_numeric_pipeline():
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return pipeline

def create_categorical_pipeline():
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return pipeline

# Создаем общий препроцессор.
def create_preprocessor():
    preprocessor = ColumnTransformer(
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

    return preprocessor