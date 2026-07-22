from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_valid, y_valid):
    """
    Оценивает модель и выводит основные метрики.
    """

    predictions = model.predict(X_valid)

    probabilities = model.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(y_valid, predictions)

    precision = precision_score(y_valid, predictions)

    recall = recall_score(y_valid, predictions)

    f1 = f1_score(y_valid, predictions)

    roc_auc = roc_auc_score(y_valid, probabilities)

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_valid, predictions))

    print("\nClassification Report")
    print(classification_report(y_valid, predictions))