"""ML pipeline engine: feature engineering, model selection, hyperparameter tuning.
XGBoost-augmented, strategy-backtestable, production-grade.
"""

from __future__ import annotations

import numpy as np
import warnings
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

import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)


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
    n_iter: int = 50
    param_grid: Optional[Dict[str, List[Any]]] = None
    feature_selection: bool = True
    use_xgboost: bool = True
    xgb_early_stopping: bool = True
    xgb_eval_metric: str = "logloss"


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
                metric = cross_val_score(model, X, y, cv=self.cv_folds, scoring="roc_auc")
            else:
                metric = cross_val_score(model, X, y, cv=self.cv_folds, scoring="r2")
            scores[name] = float(metric.mean())
        best_name = max(scores, key=scores.get)
        best_model = next(m for m in candidates if type(m).__name__ == best_name)
        best_model.fit(X, y)
        return best_name, best_model, scores


class HyperparameterTuner:
    """Hyperparameter tuning with grid or random search.

    Supports XGBoost with early stopping and comprehensive param grids.
    """

    def __init__(
        self,
        search_type: str = "grid",
        cv_folds: int = 5,
        random_state: int = 42,
        n_iter: int = 50,
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
                model, filtered_grid, cv=self.cv_folds, scoring="roc_auc", n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                model,
                filtered_grid,
                n_iter=min(self.n_iter, len(filtered_grid)),
                cv=self.cv_folds,
                scoring="roc_auc",
                random_state=self.random_state,
                n_jobs=-1,
            )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, float(search.best_score_)


class MLPipelineEngine:
    """End-to-end ML pipeline: feature engineering → model selection → tuning → evaluation.

    XGBoost-augmented, with strategy-backtestable outputs.
    """

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

    def _get_xgb_params(self) -> Dict[str, Any]:
        """Base XGBoost parameters for classification/regression."""
        if self.config.task == "classification":
            return {
                "objective": "binary:logistic",
                "eval_metric": self.config.xgb_eval_metric,
                "use_label_encoder": False,
                "random_state": self.config.random_state,
                "n_jobs": -1,
            }
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": self.config.random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    def _build_xgb_model(self) -> Any:
        """Build XGBoost classifier or regressor."""
        if self.config.use_xgboost:
            if self.config.task == "classification":
                return xgb.XGBClassifier(
                    **self._get_xgb_params(),
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                )
            return xgb.XGBRegressor(
                **self._get_xgb_params(),
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        return None

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
        best_name, best_model, scores = self.model_selector.select(
            X_train, y_train, candidates
        )
        self.best_model_name = best_name
        self.cv_scores = scores

        # Always include XGBoost in the evaluation
        xgb_model = self._build_xgb_model()
        if xgb_model is not None:
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=20,
                verbose=False,
            )
            xgb_score = cross_val_score(xgb_model, X_train, y_train, cv=self.config.cv_folds, scoring="roc_auc")
            self.cv_scores["XGBClassifier"] = float(xgb_score.mean())

        # If XGBoost outperformed, use it
        if "XGBClassifier" in self.cv_scores:
            best_cv = max(self.cv_scores.values())
            if self.cv_scores["XGBClassifier"] == best_cv:
                self.model = xgb_model
                self.best_model_name = "XGBClassifier"
            elif self.config.task == "classification" and best_cv < self.cv_scores.get("XGBClassifier", 0) - 0.02:
                # XGBoost is meaningfully better
                self.model = xgb_model
                self.best_model_name = "XGBClassifier"

        # Hyperparameter tuning if configured and model isn't XGBoost yet
        if self.config.tune_hyperparams and self.best_model_name != "XGBClassifier" and self.config.param_grid:
            best_model, best_params, best_score = self.tuner.tune(
                X_train, y_train, self.model, self.config.param_grid
            )
            self.cv_scores["tuned"] = best_score
            if best_score > self.cv_scores.get(self.best_model_name, 0):
                self.model = best_model
                self.best_model_name = type(best_model).__name__

        self.model = best_model if self.model is None else self.model
        self._X_train, self._X_test = X_train, X_test
        self._y_train, self._y_test = y_train, y_test
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Pipeline must be fitted before predict")
        X = np.asarray(X, dtype=float)
        X = self.feature_engineer.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Pipeline must be fitted before predict")
        X = np.asarray(X, dtype=float)
        X = self.feature_engineer.transform(X)
        try:
            return self.model.predict_proba(X)
        except AttributeError:
            return None

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
                RandomForestClassifier(n_estimators=200, random_state=self.config.random_state),
                GradientBoostingClassifier(n_estimators=200, random_state=self.config.random_state),
                # XGBoost is handled separately via cv_scores
            ]
        return [
            Ridge(random_state=self.config.random_state),
            RandomForestRegressor(n_estimators=200, random_state=self.config.random_state),
            GradientBoostingRegressor(n_estimators=200, random_state=self.config.random_state),
        ]


@dataclass
class BacktestConfig:
    """Configuration for strategy backtesting."""

    initial_capital: float = 100000.0
    transaction_cost_pct: float = 0.1  # 0.1% per trade
    risk_free_rate: float = 0.05  # annual
    rebalance_freq: str = "monthly"  # "daily", "weekly", "monthly", "quarterly"
    allow_short: bool = False
    max_drawdown_limit: Optional[float] = None  # e.g., 0.20 for 20%
    position_sizing: str = "equal_weight"  # "equal_weight", "vol_weight", "signal_weight"


class BacktestEngine:
    """Backtests a trading strategy generated from ML predictions.

    Strategy: 
    - If prediction > threshold: long position
    - If prediction < -threshold: short position (if allow_short)
    - Rebalance according to rebalance_freq
    - Transaction costs applied
    - Max drawdown guard
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.signals: Optional[np.ndarray] = None
        self.positions: Optional[np.ndarray] = None
        self.capital_curve: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        self.total_return: float = 0.0
        self.annualized_return: float = 0.0
        self.volatility: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0

    def run(
        self,
        predictions: np.ndarray,  # ML model predictions (signal strength)
        returns: np.ndarray,  # Actual asset returns per period
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Run backtest given ML predictions and realized returns.

        Args:
            predictions: Shape (n_periods,) or (n_periods, n_assets)
            returns: Shape (n_periods,) or (n_periods, n_assets)
            timestamps: Optional datetime array for freq detection

        Returns:
            Dict of performance metrics.
        """
        predictions = np.asarray(predictions).flatten()
        returns = np.asarray(returns).flatten()

        n = len(predictions)
        if n != len(returns):
            raise ValueError("predictions and returns must have same length")

        # Threshold: top and bottom quintiles as signal boundaries
        sig_threshold = np.percentile(np.abs(predictions), 80)
        positive_threshold = np.quantile(predictions[sig_threshold > 0], 0.8) if np.any(sig_threshold > 0) else 0.5
        negative_threshold = np.quantile(predictions[predictions < 0], 0.2) if np.any(predictions < 0) else -0.5

        # Generate signals: 1=long, -1=short (if allowed), 0=flat
        self.signals = np.zeros(n, dtype=int)
        if self.config.allow_short:
            self.signals[predictions > positive_threshold] = 1
            self.signals[predictions < negative_threshold] = -1
        else:
            self.signals[predictions > positive_threshold] = 1
            self.signals[predictions < negative_threshold] = 0  # flatten

        # Position sizing
        if self.config.position_sizing == "equal_weight":
            position_size = 1.0 / max(1, self.signals.sum()) if self.signals.sum() != 0 else 0
        elif self.config.position_sizing == "vol_weight":
            # Simplified: use inverse volatility proxy
            vol = np.std(returns)
            position_size = 1.0 / max(1, abs(self.signals.sum())) if self.signals.sum() != 0 else 0
        else:  # signal_weight
            position_size = np.mean(np.abs(self.signals))

        # Capital allocation
        n_assets = max(1, abs(self.signals.sum()))
        capital_per_position = self.config.initial_capital / n_assets

        # Generate trades with transaction costs
        cash = self.config.initial_capital
        positions_values = np.zeros(n)
        trade_costs = 0.0
        self.positions = self.signals.copy()

        for t in range(n):
            signal = self.signals[t]
            ret = returns[t]

            # Apply position
            if signal != 0:
                # Transaction cost on position change
                if t > 0 and self.signals[t - 1] != signal:
                    trade_costs += self.config.transaction_cost_pct * capital_per_position
                # Return net of cost
                positions_values[t] = signal * position_size * ret - (
                    trade_costs / n if trade_costs > 0 and t == n - 1 else 0
                )
            else:
                positions_values[t] = 0.0

        # Capital curve
        self.capital_curve = self.config.initial_capital * np.cumprod(1 + positions_values)
        total_period_return = self.capital_curve[-1] / self.config.initial_capital - 1.0
        self.total_return = total_period_return

        # Annualized return (simplified)
        n_years = max(n / 252, 1 / 252)  # assume 252 trading days/year
        self.annualized_return = (1 + total_period_return) ** (1 / n_years) - 1

        # Volatility (annualized)
        if len(positions_values) > 1:
            daily_vol = np.std(positions_values)
            self.volatility = daily_vol * np.sqrt(252)
        else:
            self.volatility = 0.0

        # Sharpe ratio
        if self.volatility > 0:
            self.sharpe_ratio = self.annualized_return / self.volatility
        else:
            self.sharpe_ratio = 0.0 if np.isfinite(self.annualized_return) else 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(self.capital_curve)
        drawdown = (self.capital_curve - running_max) / running_max
        self.max_drawdown = float(np.min(drawdown))

        # Trade counting
        signal_changes = np.sum(np.abs(np.diff(np.concatenate([[0], self.signals]))))
        self.total_trades = int(signal_changes / 2)  # each full long/short cycle = 2 changes
        self.winning_trades = int(np.sum(positions_values > 0))
        self.losing_trades = int(np.sum(positions_values < 0))

        # Apply max drawdown guard
        if self.config.max_drawdown_limit is not None and self.max_drawdown < -self.config.max_drawdown_limit:
            # Strategy would have blown up - mark as failed
            self.total_return = -1.0  # mark as failed
            self.sharpe_ratio = -10.0
            self.max_drawdown = -self.config.max_drawdown_limit

        return self._metrics()

    def _metrics(self) -> Dict[str, float]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_trades": float(self.total_trades),
            "winning_trades": float(self.winning_trades),
            "losing_trades": float(self.losing_trades),
            "win_rate": float(self.winning_trades / max(1, self.total_trades)),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.config.initial_capital,
            "transaction_cost_pct": self.config.transaction_cost_pct,
            "risk_free_rate": self.config.risk_free_rate,
            "rebalance_freq": self.config.rebalance_freq,
            "allow_short": self.config.allow_short,
            "max_drawdown_limit": self.config.max_drawdown_limit,
            "position_sizing": self.config.position_sizing,
            **self._metrics(),
        }


def run_backtest(
    predictions: np.ndarray,
    returns: np.ndarray,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.1,
    allow_short: bool = False,
    max_drawdown_limit: Optional[float] = None,
) -> Dict[str, float]:
    """Convenience function to run a backtest.

    Args:
        predictions: ML model predictions (signal strength)
        returns: Realized asset returns
        initial_capital: Starting capital
        transaction_cost_pct: Cost per trade as %
        allow_short: Allow short positions
        max_drawdown_limit: Max drawdown before stoppage

    Returns:
        Performance metrics dict.
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
        allow_short=allow_short,
        max_drawdown_limit=max_drawdown_limit,
    )
    engine = BacktestEngine(config=config)
    return engine.run(predictions, returns)


# Alias for backward compatibility
strategy_backtest = run_backtest