import pandas as pd

from config import MODEL_FILE

from feature_engineering import add_features
from preprocessing import drop_columns
from utils import load_model


def predict(data: pd.DataFrame):
    """
    Выполняет предсказание для новых данных.
    """

    # Feature Engineering
    data = add_features(data)

    # Удаляем ненужные признаки
    data = drop_columns(data)

    model = load_model(MODEL_FILE)

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    return prediction, probability


def main():

    passenger = pd.DataFrame(
        [
            {
                "PassengerId": 892,
                "Pclass": 3,
                "Name": "Ivanov, Mr. Ivan",
                "Sex": "male",
                "Age": 28,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "A/5 21171",
                "Fare": 7.25,
                "Cabin": None,
                "Embarked": "S",
            }
        ]
    )

    prediction, probability = predict(passenger)

    print("=" * 50)
    print(f"Prediction : {prediction[0]}")
    print(f"Probability: {probability[0][1]:.4f}")

    if prediction[0] == 1:
        print("Passenger is predicted to survive.")
    else:
        print("Passenger is predicted not to survive.")


if __name__ == "__main__":
    main()