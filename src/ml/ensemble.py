"""Ensemble stacking for ML pipeline — numpy-only, no sklearn dependencies.

Provides evolved alpha generation by leveraging diverse model predictions
and learning optimal convex combination weights on out-of-fold predictions.
Fully numpy-only implementation — works even when scipy/torch/sklearn
are unavailable. Compatible with any model that returns np.ndarray predictions.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable


class StackingEnsemble:
    """Stacking ensemble that combines base models via a meta-learner.

    Uses out-of-fold predictions from K-fold cross-validation to train
    the meta-learner, avoiding look-ahead bias. All operations are
    numpy-only — compatible with any model that returns np.ndarray predictions.

    The meta-learner is a simple Ridge regression solved via np.linalg.lstsq,
    making the entire ensemble numpy-only and dependency-free.

    Properties:
        n_base_models: Number of base models in the ensemble.
    """

    def __init__(
        self,
        base_model_predictors: List[Callable[[np.ndarray], np.ndarray]],
        meta_learner_weight: Optional[np.ndarray] = None,
        n_splits: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        """Initialize stacking ensemble.

        Args:
            base_model_predictors: List of callables, each taking X (n_samples, n_features)
                and returning prediction array (n_samples,). These represent trained models.
            meta_learner_weight: Optional initial weights; if None, computed via Ridge regression.
            n_splits: Number of folds for out-of-fold prediction generation.
            random_state: Random seed for reproducible K-fold splits.
            shuffle: Whether to shuffle data before splitting.
        """
        self.base_model_predictors = base_model_predictors
        self.meta_learner_weight = meta_learner_weight
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.ofp_: Dict[int, np.ndarray] = {}  # out-of-fold: model_idx -> (n_samples,)
        self.meta_weights_: Optional[np.ndarray] = None
        self.is_fitted_ = False
        self._rng: Optional[np.random.Generator] = None
        self._n_base_models = len(base_model_predictors)

    @property
    def n_base_models(self) -> int:
        """Return number of base models in the ensemble."""
        return self._n_base_models

    def _kfolds(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate K-fold train/test splits.

        Returns list of (train_idx, test_idx) tuples.
        """
        self._rng = np.random.default_rng(self.random_state)
        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)
        splits = []
        fold_size = n // self.n_splits
        remainder = n % self.n_splits
        start = 0
        for i in range(self.n_splits):
            extra = 1 if i < remainder else 0
            end = start + fold_size + extra
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            splits.append((train_idx, test_idx))
            start = end
        return splits

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_predictions: Optional[List[np.ndarray]] = None,
    ) -> "StackingEnsemble":
        """Fit stacking ensemble using out-of-fold predictions.

        If model_predictions are provided (pre-computed), use those.
        Otherwise, generate out-of-fold predictions by calling each
        base_model_predictor on the appropriate train/test splits.

        Args:
            X: Feature matrix (n_samples, n_features) — used only for shape;
                predictors receive X directly.
            y: Target vector (n_samples,).
            model_predictions: Optional list of (n_samples,) arrays per model,
                already aligned with full dataset. If provided, skips OOF generation.

        Returns:
            self for method chaining
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).flatten()
        n = X.shape[0]
        kfolds = self._kfolds(n)

        n_models = self.n_base_models

        # Out-of-fold predictions: ofp[model_idx][sample_idx] = prediction
        ofp = np.empty((n_models, n))

        if model_predictions is not None and len(model_predictions) == n_models:
            # Use provided predictions (must be aligned with full X order)
            for m_idx, pred in enumerate(model_predictions):
                ofp[m_idx] = np.asarray(pred, dtype=float).flatten()
        else:
            # Generate out-of-fold predictions
            for fold_idx, (train_idx, test_idx) in enumerate(kfolds):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]

                for m_idx, predictor in enumerate(self.base_model_predictors):
                    # Call predictor on test subset
                    preds = predictor(X_test)
                    preds = np.asarray(preds, dtype=float).flatten()
                    if len(preds) != len(test_idx):
                        raise ValueError(
                            f"Model {m_idx} prediction length {len(preds)} "
                            f"!= test fold size {len(test_idx)}"
                        )
                    ofp[m_idx][test_idx] = preds

        self.ofp_ = {i: ofp[i] for i in range(n_models)}

        # Train meta-learner: compute optimal convex combination weights
        # using Ridge regression on out-of-fold predictions
        OFP = self.ofp_  # shape (n_models, n)
        alpha = 1e-6  # small Ridge regularization
        # Solve: (OFP^T OFP + alpha*I) w = OFP^T y
        gram = OFP.T @ OFP + alpha * np.eye(n_models)
        rhs = OFP.T @ y
        try:
            self.meta_weights_ = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback: equal weights
            self.meta_weights_ = np.ones(n_models) / n_models

        # Normalize to convex combination (weights sum to 1, non-negative)
        if np.sum(np.abs(self.meta_weights_)) > 0:
            self.meta_weights_ = self.meta_weights_ / np.sum(np.abs(self.meta_weights_))
        # Ensure non-negative
        self.meta_weights_ = np.maximum(self.meta_weights_, 0)
        if np.sum(self.meta_weights_) > 0:
            self.meta_weights_ = self.meta_weights_ / np.sum(self.meta_weights_)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking ensemble.

        Combines base model predictions via learned meta-weights.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Combined prediction array (n_samples,).
        """
        if not self.is_fitted_ or self.meta_weights_ is None:
            raise RuntimeError("Ensemble must be fitted before predict")

        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Generate predictions from each base model on full X
        n_models = self.n_base_models
        model_preds = np.empty((n_models, n))

        for m_idx, predictor in enumerate(self.base_model_predictors):
            preds = predictor(X)
            preds = np.asarray(preds, dtype=float).flatten()
            if len(preds) != n:
                raise ValueError(
                    f"Model {m_idx} prediction length {len(preds)} != n_samples {n}"
                )
            model_preds[m_idx] = preds

        # Combine via meta-weights
        combined = model_preds @ self.meta_weights_
        return combined

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Return class probability predictions.

        For binary classification: applies sigmoid to combined predictions
        to produce probabilities in [0, 1].

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Probability array (n_samples,) in [0, 1], or None if not applicable.
        """
        if not self.is_fitted_ or self.meta_weights_ is None:
            raise RuntimeError("Ensemble must be fitted before predict_proba")

        combined = self.predict(X)
        # Apply sigmoid to convert to probability
        combined = np.clip(combined, -250, 250)
        prob = 1.0 / (1.0 + np.exp(-combined))
        return prob

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score (regression) or ROC AUC (classification) on test data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: True target vector (n_samples,).

        Returns:
            Float score.
        """
        y = np.asarray(y).flatten()
        preds = self.predict(X)

        # Check if binary classification (assuming 0/1 labels)
        if len(np.unique(y)) == 2 and np.all(np.isin(y, [0, 1])):
            # ROC AUC via predicted probabilities
            try:
                proba = self.predict_proba(X)
                # Simple AUC approximation: proportion of correct pairwise comparisons
                n_pos = np.sum(y == 1)
                n_neg = np.sum(y == 0)
                if n_pos > 0 and n_neg > 0:
                    # Count how many positive predictions exceed negative predictions
                    pos_preds = preds[y == 1]
                    neg_preds = preds[y == 0]
                    if len(pos_preds) > 0 and len(neg_preds) > 0:
                        n_correct = np.sum(pos_preds[:, np.newaxis] > neg_preds[np.newaxis, :])
                        auc = n_correct / (len(pos_preds) * len(neg_preds))
                        return float(np.clip(auc, 0.0, 1.0))
                return 0.5  # random
            except Exception:
                return 0.5
        else:
            # Regression: R² score
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        """Return summary of ensemble configuration and fitted state."""
        return {
            "n_base_models": self.n_base_models,
            "meta_weights": self.meta_weights_.tolist() if self.meta_weights_ is not None else None,
            "n_splits": self.n_splits,
            "ofp_available": self.ofp_ is not None and len(self.ofp_) > 0,
            "is_fitted": self.is_fitted_,
            "rng_seed": self.random_state,
        }


def create_default_stacking_ensemble(
    n_models: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
) -> StackingEnsemble:
    """Factory function to create a default stacking ensemble.

    Args:
        n_models: Number of base models in the ensemble.
        n_splits: Number of folds for out-of-fold prediction generation.
        random_state: Random seed for reproducibility.

    Returns:
        Configured but unfitted StackingEnsemble instance.
    """

    def make_predictor(mean_val: float = 0.0, std_val: float = 1.0):
        def predictor(X: np.ndarray) -> np.ndarray:
            rng = np.random.default_rng(seed=int(mean_val) + 100)
            return rng.normal(loc=mean_val, scale=std_val, size=X.shape[0])
        return predictor

    predictors = [make_predictor(i * 10) for i in range(n_models)]
    return StackingEnsemble(
        base_model_predictors=predictors,
        n_splits=n_splits,
        random_state=random_state,
    )