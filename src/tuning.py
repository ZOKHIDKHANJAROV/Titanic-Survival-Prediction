import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor
from config import RANDOM_STATE


def optuna_tuning(X_train, y_train, n_trials=100):
    """
    Подбор гиперпараметров RandomForest с помощью Optuna.
    """

    def objective(trial):

        model = RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators",
                100,
                500,
            ),

            max_depth=trial.suggest_int(
                "max_depth",
                3,
                15,
            ),

            min_samples_split=trial.suggest_int(
                "min_samples_split",
                2,
                8,
            ),

            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf",
                1,
                10,
            ),

            max_features=trial.suggest_categorical(
                "max_features",
                [
                    "sqrt",
                    "log2",
                ],
            ),

            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
        )

        return scores.mean()

    sampler = TPESampler(
        seed=RANDOM_STATE,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("=" * 60)
    print("Optuna finished")
    print("=" * 60)
    print(f"Best score : {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_model = RandomForestClassifier(
        **study.best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                best_model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline, study