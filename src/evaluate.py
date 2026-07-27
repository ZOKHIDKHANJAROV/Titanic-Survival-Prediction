from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(
    model,
    X,
    y,
):

    prediction = model.predict(X)

    probability = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction),
        "recall": recall_score(y, prediction),
        "f1": f1_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
    }

    return metrics, prediction


def print_metrics(
    metrics,
    y_true,
    y_pred,
):

    print("=" * 60)
    print("Validation")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key:10}: {value:.4f}")

    print()

    print("Confusion Matrix")

    print(confusion_matrix(
        y_true,
        y_pred,
    ))

    print()

    print(classification_report(
        y_true,
        y_pred,
    ))