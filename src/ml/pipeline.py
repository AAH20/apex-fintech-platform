"""ML pipeline engine: feature engineering, model selection, hyperparameter tuning."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, StandardScaler


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline engine."""

    task: str = "classification"  # "classification" or "regression"
    cv_folds: int = 5
    random_state: int = 42
    test_size: float = 0.2
    scale_features: bool = True
    polynomial_degree: int = 1
    n_bins: int = 5
    impute_missing: bool = True
    correlation_threshold: float = 0.95
    tune_hyperparams: bool = False
    search_type: str = "grid"  # "grid" or "random"
    n_iter: int = 10
    param_grid: Optional[Dict[str, List[Any]]] = None
    feature_selection: bool = True


class FeatureEngineer:
    """Handles feature engineering: imputation, scaling, polynomial features, binning."""

    def __init__(
        self,
        scale_features: bool = True,
        polynomial_degree: int = 1,
        n_bins: Optional[int] = None,
        impute_missing: bool = True,
        correlation_threshold: float = 0.95,
    ):
        self.scale_features = scale_features
        self.polynomial_degree = polynomial_degree
        self.n_bins = n_bins
        self.impute_missing = impute_missing
        self.correlation_threshold = correlation_threshold
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._poly: Optional[PolynomialFeatures] = None
        self._binner: Optional[KBinsDiscretizer] = None
        self._selected_indices: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureEngineer":
        X = np.asarray(X, dtype=float)
        if self.impute_missing:
            self._imputer = SimpleImputer(strategy="median")
            self._imputer.fit(X)
        if self.scale_features:
            self._scaler = StandardScaler()
            self._scaler.fit(X)
        if self.polynomial_degree > 1:
            self._poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
            self._poly.fit(X)
        if self.n_bins is not None and self.n_bins > 1:
            X_pre = self._apply_transforms(X)
            self._binner = KBinsDiscretizer(
                n_bins=self.n_bins, encode="onehot-dense", strategy="uniform"
            )
            self._binner.fit(X_pre)
        # Compute correlation filter on the fully transformed output
        if self.correlation_threshold < 1.0:
            X_transformed = self._apply_transforms(X)
            if X_transformed.shape[0] > X_transformed.shape[1]:
                self._selected_indices = self._compute_correlation_filter(X_transformed)
        self._is_fitted = True
        return self

    def _apply_transforms(self, X: np.ndarray) -> np.ndarray:
        """Apply all transformations except correlation filter."""
        if self._imputer is not None:
            X = self._imputer.transform(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        if self._poly is not None:
            X = self._poly.transform(X)
        if self._binner is not None:
            X = self._binner.transform(X)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        X = np.asarray(X, dtype=float)
        X = self._apply_transforms(X)
        if self._selected_indices is not None:
            X = X[:, self._selected_indices]
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _compute_correlation_filter(self, X: np.ndarray) -> np.ndarray:
        """Remove highly correlated features, keeping the first of each correlated pair."""
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        n = corr.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if np.isfinite(corr[i, j]) and abs(corr[i, j]) > self.correlation_threshold:
                    keep[j] = False
        return np.where(keep)[0]


class ModelSelector:
    """Selects the best model from candidates using cross-validation."""

    def __init__(self, task: str = "classification", cv_folds: int = 5, random_state: int = 42):
        self.task = task
        self.cv_folds = cv_folds
        self.random_state = random_state

    def select(
        self, X: np.ndarray, y: np.ndarray, candidates: List[BaseEstimator]
    ) -> Tuple[str, BaseEstimator, Dict[str, float]]:
        scores: Dict[str, float] = {}
        for model in candidates:
            name = type(model).__name__
            if self.task == "classification":
                metric = cross_val_score(model, X, y, cv=self.cv_folds, scoring="accuracy")
            else:
                metric = cross_val_score(model, X, y, cv=self.cv_folds, scoring="r2")
            scores[name] = float(metric.mean())
        best_name = max(scores, key=scores.get)
        best_model = next(m for m in candidates if type(m).__name__ == best_name)
        best_model.fit(X, y)
        return best_name, best_model, scores


class HyperparameterTuner:
    """Hyperparameter tuning with grid or random search."""

    def __init__(
        self,
        search_type: str = "grid",
        cv_folds: int = 5,
        random_state: int = 42,
        n_iter: int = 10,
    ):
        self.search_type = search_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_iter = n_iter

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: BaseEstimator,
        param_grid: Dict[str, List[Any]],
    ) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        valid_params = model.get_params()
        filtered_grid = {k: v for k, v in param_grid.items() if k in valid_params}
        if not filtered_grid:
            model.fit(X, y)
            return model, {}, 0.0
        if self.search_type == "grid":
            search = GridSearchCV(
                model, filtered_grid, cv=self.cv_folds, scoring="accuracy", n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                model,
                filtered_grid,
                n_iter=min(self.n_iter, len(filtered_grid)),
                cv=self.cv_folds,
                scoring="accuracy",
                random_state=self.random_state,
                n_jobs=-1,
            )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, float(search.best_score_)


class MLPipelineEngine:
    """End-to-end ML pipeline: feature engineering → model selection → tuning → evaluation."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.feature_engineer = FeatureEngineer(
            scale_features=self.config.scale_features,
            polynomial_degree=self.config.polynomial_degree,
            n_bins=self.config.n_bins,
            impute_missing=self.config.impute_missing,
            correlation_threshold=self.config.correlation_threshold,
        )
        self.model_selector = ModelSelector(
            task=self.config.task,
            cv_folds=self.config.cv_folds,
            random_state=self.config.random_state,
        )
        self.tuner = HyperparameterTuner(
            search_type=self.config.search_type,
            cv_folds=self.config.cv_folds,
            random_state=self.config.random_state,
            n_iter=self.config.n_iter,
        )
        self.model: Optional[BaseEstimator] = None
        self.best_model_name: str = ""
        self.cv_scores: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}
        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPipelineEngine":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        X_train = self.feature_engineer.fit_transform(X_train, y_train)
        X_test = self.feature_engineer.transform(X_test)
        self._X_train, self._X_test = X_train, X_test
        self._y_train, self._y_test = y_train, y_test
        candidates = self._default_candidates()
        best_name, best_model, scores = self.model_selector.select(X_train, y_train, candidates)
        self.best_model_name = best_name
        self.cv_scores = scores
        if self.config.tune_hyperparams and self.config.param_grid:
            best_model, best_params, best_score = self.tuner.tune(
                X_train, y_train, best_model, self.config.param_grid
            )
            self.cv_scores["tuned"] = best_score
        self.model = best_model
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Pipeline must be fitted before predict")
        X = np.asarray(X, dtype=float)
        X = self.feature_engineer.transform(X)
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Pipeline must be fitted before evaluate")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        X = self.feature_engineer.transform(X)
        preds = self.model.predict(X)
        if self.config.task == "classification":
            self.metrics = {
                "accuracy": float(accuracy_score(y, preds)),
                "f1": float(f1_score(y, preds, average="weighted")),
            }
            if len(np.unique(y)) == 2:
                try:
                    proba = self.model.predict_proba(X)[:, 1]
                    self.metrics["roc_auc"] = float(roc_auc_score(y, proba))
                except (AttributeError, IndexError):
                    pass
        else:
            self.metrics = {
                "mse": float(mean_squared_error(y, preds)),
                "r2": float(r2_score(y, preds)),
            }
        return self.metrics

    def feature_importance(self) -> Optional[Dict[str, float]]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Pipeline must be fitted first")
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return None
        return {f"feature_{i}": float(v) for i, v in enumerate(importances)}

    def summary(self) -> Dict[str, Any]:
        return {
            "task": self.config.task,
            "best_model": self.best_model_name,
            "metrics": self.metrics,
            "cv_scores": self.cv_scores,
            "n_features": self._X_train.shape[1] if self._X_train is not None else 0,
            "n_train": len(self._y_train) if self._y_train is not None else 0,
            "n_test": len(self._y_test) if self._y_test is not None else 0,
        }

    def _default_candidates(self) -> List[BaseEstimator]:
        if self.config.task == "classification":
            return [
                LogisticRegression(max_iter=1000, random_state=self.config.random_state),
                RandomForestClassifier(n_estimators=50, random_state=self.config.random_state),
                GradientBoostingClassifier(n_estimators=50, random_state=self.config.random_state),
            ]
        return [
            Ridge(random_state=self.config.random_state),
            RandomForestRegressor(n_estimators=50, random_state=self.config.random_state),
            GradientBoostingRegressor(n_estimators=50, random_state=self.config.random_state),
        ]
