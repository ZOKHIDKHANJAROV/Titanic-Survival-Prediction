from sklearn.model_selection import GridSearchCV


def grid_search_tuning(
    pipeline,
    X_train,
    y_train,
):
    """
    Подбирает лучшие гиперпараметры
    для модели.
    """

    param_grid = {

        "model__n_estimators": [
            100,
            200,
            300,
        ],

        "model__max_depth": [
            None,
            5,
            10,
            20,
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4,
        ],

        "model__max_features": [
            "sqrt",
            "log2",
            None,
        ],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print("=" * 60)
    print("Grid Search Finished")
    print("=" * 60)

    print()

    print(f"Best Score : {grid_search.best_score_:.4f}")

    print()

    print("Best Parameters")

    for key, value in grid_search.best_params_.items():
        print(f"{key}: {value}")

    return grid_search.best_estimator_