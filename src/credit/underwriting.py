"""AI Credit Underwriting Engine.

Implements:
- CreditUnderwritingEngine: RandomForest-based credit scoring with SHAP-like explainability
- Feature importance and prediction contributions

References:
- CFA Institute: AI in Credit Underwriting
- SR 11-7: Model Risk Management
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


class CreditUnderwritingEngine:
    """AI-powered credit underwriting engine using RandomForest.

    Provides training, prediction, explainability, and feature importance
    for credit default assessment.
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 200, max_depth: int = 8):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model: RandomForestClassifier | None = None
        self.imputer: SimpleImputer | None = None
        self.is_fitted = False
        self.feature_names_: list[str] = []
        self.n_features_in_: int = 0
        self._X_train: pd.DataFrame | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> "CreditUnderwritingEngine":
        """Train the credit underwriting model.

        Args:
            X: Feature matrix (traditional + alternative data)
            y: Target variable (0 = non-default, 1 = default)

        Returns:
            self
        """
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = list(X.columns)

        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        # Train RandomForest classifier
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_imputed, y)

        # Store training data for explainability
        self._X_train = X_imputed
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict default class (0 or 1).

        Args:
            X: Feature matrix

        Returns:
            Binary predictions
        """
        self._check_fitted()
        X_imputed = self._impute(X)
        predictions = self.model.predict(X_imputed)
        return pd.Series(predictions, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict default probability.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with columns [prob_non_default, prob_default]
        """
        self._check_fitted()
        X_imputed = self._impute(X)
        proba = self.model.predict_proba(X_imputed)
        return pd.DataFrame(
            proba,
            columns=["prob_non_default", "prob_default"],
            index=X.index,
        )

    def explain(self, X: pd.DataFrame) -> dict:
        """Generate SHAP-like feature contributions for predictions.

        Uses tree-based feature importance weighted by prediction confidence
        to approximate per-sample contributions.

        Args:
            X: Feature matrix to explain

        Returns:
            Dictionary with contributions per sample
        """
        self._check_fitted()
        X_imputed = self._impute(X)
        proba = self.model.predict_proba(X_imputed)

        # Feature importance from the trained model
        importance = self.model.feature_importances_

        # Compute per-sample contributions: importance * feature values
        # Normalize features to [0, 1] for comparable contributions
        X_norm = self._normalize(X_imputed)
        contributions = X_norm.values * importance

        return {
            "contributions": contributions,
            "feature_names": self.feature_names_,
            "base_value": float(np.mean(proba[:, 1])),
            "predictions": proba[:, 1].tolist(),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores (sorted descending)
        """
        self._check_fitted()
        importance = self.model.feature_importances_
        result = dict(zip(self.feature_names_, importance, strict=True))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using fitted imputer."""
        if self.imputer is None:
            raise RuntimeError("Imputer not fitted. Call train() first.")
        return pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index,
        )

    def _normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0, 1] range using training data stats."""
        if self._X_train is None:
            return X
        mins = self._X_train.min()
        maxs = self._X_train.max()
        ranges = maxs - mins
        ranges = ranges.replace(0, 1.0)  # Avoid division by zero
        return (X - mins) / ranges

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
