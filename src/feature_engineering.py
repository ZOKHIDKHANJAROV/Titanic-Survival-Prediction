import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def add_family_size(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def add_is_alone(df):
    df = df.copy()
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def extract_title(df):
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    title = title.replace(
        {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    )
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    df["Title"] = title.where(title.isin(common_titles), "Rare").fillna("Unknown")
    return df


def extract_deck(df):
    df = df.copy()
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    return df


def add_fare_per_person(df):
    df = df.copy()
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    return df


def add_age_group(df):
    df = df.copy()
    bins = [float("-inf"), 12, 18, 35, 60, float("inf")]
    labels = ["Child", "Teen", "Adult", "Middle", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df


def add_features(df):
    df = add_family_size(df)
    df = add_is_alone(df)
    df = extract_title(df)
    df = extract_deck(df)
    df = add_fare_per_person(df)
    df = add_age_group(df)
    return df


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add deterministic Titanic-specific features inside an sklearn pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(X)
