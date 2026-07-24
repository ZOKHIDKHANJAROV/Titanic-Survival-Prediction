from pathlib import Path

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "reports"

TRAIN_DATA = DATA_DIR / "train.csv"

TEST_DATA = DATA_DIR / "test.csv"

MODEL_FILE = MODEL_DIR / "titanic_model.pkl"

# ==========================================
# Training
# ==========================================

RANDOM_STATE = 42

TEST_SIZE = 0.2

TARGET = "Survived"

# ==========================================
# Columns
# ==========================================

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