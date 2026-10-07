"""Tests for ML pipeline engine: feature engineering, model selection, hyperparameter tuning."""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

from src.ml.pipeline import (
    MLPipelineEngine,
    FeatureEngineer,
    ModelSelector,
    HyperparameterTuner,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_redundant=1,
        n_classes=2, random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=200, n_features=5, n_informative=3, noise=10, random_state=42,
    )
    return X, y


@pytest.fixture
def config():
    return PipelineConfig(
        task="classification",
        cv_folds=3,
        random_state=42,
        test_size=0.25,
    )


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Test feature engineering transformations."""

    def test_polynomial_features(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        fe = FeatureEngineer(polynomial_degree=2)
        X_out = fe.fit_transform(X)
        # Original 2 features + squared + interaction = 5 features
        assert X_out.shape[1] == 5
        assert X_out.shape[0] == 3

    def test_binning(self):
        X = np.array([[1], [10], [20], [30], [100], [200], [300], [400], [500], [600]])
        fe = FeatureEngineer(n_bins=3)
        X_out = fe.fit_transform(X)
        # Binned into 3 categories, one-hot encoded
        assert X_out.shape[0] == 10
        assert X_out.shape[1] == 3

    def test_standard_scaling(self):
        X = np.array([[1, 100], [2, 200], [3, 300]], dtype=float)
        fe = FeatureEngineer(scale_features=True)
        X_out = fe.fit_transform(X)
        # After scaling, mean ≈ 0 and std ≈ 1
        np.testing.assert_allclose(X_out.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(X_out.std(axis=0), 1, atol=1e-10)

    def test_imputation(self):
        X = np.array([[1, 2], [np.nan, 3], [7, np.nan]], dtype=float)
        fe = FeatureEngineer(impute_missing=True)
        X_out = fe.fit_transform(X)
        assert not np.isnan(X_out).any()

    def test_correlation_filter(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 4] = X[:, 0] * 0.99 + rng.randn(100) * 0.01  # highly correlated with col 0
        fe = FeatureEngineer(correlation_threshold=0.95)
        X_out = fe.fit_transform(X)
        assert X_out.shape[1] < X.shape[1]

    def test_fit_transform_consistency(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(80, 4)
        X_test = rng.randn(20, 4)
        fe = FeatureEngineer(scale_features=True)
        fe.fit(X_train)
        X_train_out = fe.transform(X_train)
        X_test_out = fe.transform(X_test)
        assert X_train_out.shape[1] == X_test_out.shape[1]
        assert X_train_out.shape[0] == 80
        assert X_test_out.shape[0] == 20


# ---------------------------------------------------------------------------
# Model Selection Tests
# ---------------------------------------------------------------------------

class TestModelSelection:
    """Test model selection with cross-validation."""

    def test_selects_best_model(self, classification_data):
        X, y = classification_data
        selector = ModelSelector(task="classification", cv_folds=3, random_state=42)
        candidates = [
            LogisticRegression(max_iter=1000),
            DecisionTreeClassifier(random_state=42),
            RandomForestClassifier(n_estimators=10, random_state=42),
        ]
        best_name, best_model, scores = selector.select(X, y, candidates)
        assert best_name in ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"]
        assert scores[best_name] == max(scores.values())
        assert best_model is not None

    def test_all_candidates_scored(self, classification_data):
        X, y = classification_data
        selector = ModelSelector(task="classification", cv_folds=3, random_state=42)
        candidates = [
            LogisticRegression(max_iter=1000),
            DecisionTreeClassifier(random_state=42),
        ]
        _, _, scores = selector.select(X, y, candidates)
        assert len(scores) == 2
        assert all(isinstance(v, float) for v in scores.values())

    def test_regression_model_selection(self, regression_data):
        X, y = regression_data
        selector = ModelSelector(task="regression", cv_folds=3, random_state=42)
        candidates = [Ridge(), GradientBoostingRegressor(n_estimators=10, random_state=42)]
        best_name, best_model, scores = selector.select(X, y, candidates)
        assert best_name in ["Ridge", "GradientBoostingRegressor"]
        assert best_model is not None


# ---------------------------------------------------------------------------
# Hyperparameter Tuning Tests
# ---------------------------------------------------------------------------

class TestHyperparameterTuning:
    """Test hyperparameter tuning with grid and random search."""

    def test_grid_search_improves_or_matches(self, classification_data):
        X, y = classification_data
        tuner = HyperparameterTuner(search_type="grid", cv_folds=3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
        best_model, best_params, best_score = tuner.tune(X, y, model, param_grid)
        assert best_params["C"] in [0.01, 0.1, 1.0, 10.0]
        assert best_score > 0.5  # should beat random on this dataset

    def test_random_search(self, classification_data):
        X, y = classification_data
        tuner = HyperparameterTuner(search_type="random", cv_folds=3, random_state=42, n_iter=5)
        model = RandomForestClassifier(random_state=42)
        param_dist = {"n_estimators": [5, 10, 20], "max_depth": [3, 5, 10]}
        best_model, best_params, best_score = tuner.tune(X, y, model, param_dist)
        assert best_params["n_estimators"] in [5, 10, 20]
        assert best_params["max_depth"] in [3, 5, 10]
        assert best_score > 0.5

    def test_tuned_model_predicts(self, classification_data):
        X, y = classification_data
        tuner = HyperparameterTuner(search_type="grid", cv_folds=3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        param_grid = {"C": [0.1, 1.0]}
        best_model, _, _ = tuner.tune(X, y, model, param_grid)
        preds = best_model.predict(X[:5])
        assert len(preds) == 5
        assert all(p in [0, 1] for p in preds)


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestMLPipelineEngine:
    """Test the full ML pipeline engine end-to-end."""

    def test_classification_pipeline(self, classification_data, config):
        X, y = classification_data
        engine = MLPipelineEngine(config=config)
        engine.fit(X, y)
        metrics = engine.evaluate(X, y)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] > 0.7

    def test_regression_pipeline(self, regression_data):
        X, y = regression_data
        config = PipelineConfig(
            task="regression", cv_folds=3, random_state=42, test_size=0.25,
        )
        engine = MLPipelineEngine(config=config)
        engine.fit(X, y)
        metrics = engine.evaluate(X, y)
        assert "r2" in metrics
        assert "mse" in metrics
        assert metrics["r2"] > 0.5

    def test_pipeline_predict(self, classification_data, config):
        X, y = classification_data
        engine = MLPipelineEngine(config=config)
        engine.fit(X, y)
        preds = engine.predict(X[:10])
        assert len(preds) == 10
        assert all(p in [0, 1] for p in preds)

    def test_pipeline_with_feature_engineering(self, classification_data):
        X, y = classification_data
        config = PipelineConfig(
            task="classification",
            cv_folds=3,
            random_state=42,
            test_size=0.25,
            scale_features=True,
            polynomial_degree=2,
        )
        engine = MLPipelineEngine(config=config)
        engine.fit(X, y)
        metrics = engine.evaluate(X, y)
        assert metrics["accuracy"] > 0.7

    def test_pipeline_with_tuning(self, classification_data):
        X, y = classification_data
        config = PipelineConfig(
            task="classification",
            cv_folds=3,
            random_state=42,
            test_size=0.25,
            tune_hyperparams=True,
            param_grid={"n_estimators": [10, 50], "max_depth": [3, 5, None]},
        )
        engine = MLPipelineEngine(config=config)
        engine.fit(X, y)
        metrics = engine.evaluate(X, y)
        assert metrics["accuracy"] > 0.7

    def test_pipeline_feature_importance(self, classification_data, config):
        X, y = classification_data
        engine = MLPipelineEngine(config=config)
        engine.fit(X, y)
        importance = engine.feature_importance()
        assert importance is not None
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())

    def test_pipeline_summary(self, classification_data, config):
        X, y = classification_data
        engine = MLPipelineEngine(config=config)
        engine.fit(X, y)
        summary = engine.summary()
        assert "task" in summary
        assert "best_model" in summary
        assert "metrics" in summary
        assert "n_features" in summary
        assert summary["task"] == "classification"
