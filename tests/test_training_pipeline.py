import unittest

import pandas as pd

from src.model import MODEL_NAMES, build_pipeline
from src.tuning import tune_model


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.read_csv("data/train.csv").head(200)
        cls.X = data.drop(columns=["Survived"])
        cls.y = data["Survived"]

    def test_every_registered_model_supports_probabilities(self):
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                pipeline = build_pipeline(
                    model_name,
                    random_state=42,
                )
                pipeline.fit(self.X, self.y)
                probabilities = pipeline.predict_proba(self.X.head(3))
                self.assertEqual(probabilities.shape, (3, 2))

    def test_engineered_features_reach_preprocessor(self):
        pipeline = build_pipeline(
            "logistic_regression",
            random_state=42,
        )
        pipeline.fit(self.X, self.y)
        feature_names = set(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )

        for feature in ("FamilySize", "IsAlone", "FarePerPerson"):
            self.assertIn(f"numeric__{feature}", feature_names)

        self.assertTrue(any("Title_" in name for name in feature_names))
        self.assertTrue(any("Deck_" in name for name in feature_names))
        self.assertTrue(any("AgeGroup_" in name for name in feature_names))

    def test_optuna_sampler_is_reproducible(self):
        kwargs = {
            "random_state": 42,
            "n_trials": 1,
            "cv_folds": 3,
            "scoring": "roc_auc",
            "n_jobs": 1,
        }
        _, first_study = tune_model(
            "logistic_regression",
            self.X,
            self.y,
            **kwargs,
        )
        _, second_study = tune_model(
            "logistic_regression",
            self.X,
            self.y,
            **kwargs,
        )

        self.assertEqual(first_study.best_params, second_study.best_params)
        self.assertEqual(first_study.best_value, second_study.best_value)


if __name__ == "__main__":
    unittest.main()
