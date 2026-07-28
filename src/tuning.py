import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.model import build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_hyperparameters(trial, model_name: str) -> dict:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"],
            ),
        }

    if model_name == "decision_tree":
        return {
            "criterion": trial.suggest_categorical(
                "criterion",
                ["gini", "entropy"],
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 30
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 15
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"],
            ),
        }

    if model_name in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 500, step=50
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"],
            ),
        }

    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 300, step=25
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 15
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
        }

    raise ValueError(f"Unknown model: {model_name}")


def tune_model(
    model_name,
    X_train,
    y_train,
    *,
    random_state,
    n_trials,
    cv_folds,
    scoring,
    n_jobs,
):
    """Tune one model with deterministic Optuna sampling and stratified CV."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    def objective(trial):
        model_params = suggest_hyperparameters(trial, model_name)
        pipeline = build_pipeline(
            model_name,
            random_state=random_state,
            model_params=model_params,
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            error_score="raise",
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=f"titanic-{model_name}",
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_pipeline = build_pipeline(
        model_name,
        random_state=random_state,
        model_params=study.best_params,
    )
    best_pipeline.fit(X_train, y_train)
    return best_pipeline, study
