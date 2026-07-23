from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


def cross_validate_model(
    pipeline,
    X,
    y,
):
    """
    Выполняет Cross Validation и возвращает средние значения метрик.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
    )

    metrics = {}

    for metric in [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]:
        metrics[metric] = scores[metric].mean()

    return metrics
