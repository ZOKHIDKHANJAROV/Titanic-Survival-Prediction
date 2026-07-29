from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(model, X_valid, y_valid):
    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(
            precision_score(y_valid, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_valid, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_valid,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def print_metrics(model_name, metrics):
    print("=" * 50)
    print(f"Model Evaluation: {model_name}")
    print("=" * 50)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"CV score : {metrics['cv_score']:.4f}")
