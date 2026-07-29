from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"
PARAMS_FILE = BASE_DIR / "params.yaml"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_SPLIT_FILE = PROCESSED_DATA_DIR / "train.csv"
VALID_SPLIT_FILE = PROCESSED_DATA_DIR / "valid.csv"

CANDIDATE_MODEL_DIR = MODEL_DIR / "candidates"
MODEL_FILE = MODEL_DIR / "best_model.pkl"
MODEL_METADATA_FILE = MODEL_DIR / "best_model.json"

METRICS_DIR = REPORT_DIR / "metrics"
CANDIDATE_METRICS_DIR = METRICS_DIR / "candidates"
SUMMARY_METRICS_FILE = METRICS_DIR / "summary.json"

TARGET = "Survived"

DROP_COLUMNS = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin",
]

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "Pclass",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "AgeGroup",
]
