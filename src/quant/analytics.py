"""Quantitative Analytics Engine.

Implements ARIMA, GARCH, factor models, regime switching, and signal generation
for quantitative trading strategies.

References:
    - Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis: Forecasting
      and Control. Holden-Day.
    - Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity with
      Estimates of the Variance of United Kingdom Inflation. Econometrica, 50(4),
      987-1007.
    - Bollerslev, T. (1986). Generalized Autoregressive Conditional
      Heteroscedasticity. Journal of Econometrics, 31(3), 307-327.
    - Hamilton, J.D. (1989). A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle. Econometrica, 57(2),
      357-384.
    - CFA Institute Quantitative Methods curriculum.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller


class SignalType(Enum):
    """Trading signal types."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class QuantitativeAnalyticsEngine:
    """Quantitative analytics engine for model fitting and signal generation.

    Supports:
    - ARIMA time series modeling
    - GARCH volatility modeling
    - PCA-based factor models
    - Markov regime switching models
    - Signal generation from price series
    """

    def __init__(self) -> None:
        """Initialize the analytics engine."""
        self.models: dict[str, Any] = {}
        self.signals: list[SignalType] = []

    # ------------------------------------------------------------------
    # ARIMA
    # ------------------------------------------------------------------

    def fit_arima(self, series: pd.Series, order: tuple[int, int, int]) -> None:
        """Fit an ARIMA model to a time series.

        Args:
            series: Time series data (e.g., prices or returns).
            order: (p, d, q) order of the ARIMA model.

        Raises:
            ValueError: If order is not a tuple of 3 integers.
        """
        if not isinstance(order, tuple) or len(order) != 3:
            raise ValueError("order must be a tuple of 3 integers (p, d, q)")

        model = ARIMA(series, order=order)
        result = model.fit()
        self.models["arima"] = result

    def forecast_arima(self, steps: int) -> np.ndarray:
        """Generate forecasts from the fitted ARIMA model.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            Array of forecasted values.

        Raises:
            ValueError: If steps is not positive.
            RuntimeError: If ARIMA model has not been fitted.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")
        if "arima" not in self.models:
            raise RuntimeError("ARIMA model not fitted")

        result = self.models["arima"]
        forecast = result.forecast(steps=steps)
        return np.asarray(forecast)

    # ------------------------------------------------------------------
    # GARCH
    # ------------------------------------------------------------------

    def fit_garch(
        self, series: pd.Series, p: int = 1, q: int = 1, dist: str = "normal"
    ) -> None:
        """Fit a GARCH(p, q) model to a return series.

        Args:
            series: Return series (not prices).
            p: GARCH lag order.
            q: ARCH lag order.
            dist: Error distribution ('normal', 't', 'skewt').
        """
        from arch import arch_model

        # Demean the series for GARCH fitting
        returns = series.dropna()
        model = arch_model(returns, vol="Garch", p=p, q=q, dist=dist, rescale=False)
        result = model.fit(disp="off", show_warning=False)
        self.models["garch"] = result

    def forecast_garch(self, steps: int) -> np.ndarray:
        """Generate volatility forecasts from the fitted GARCH model.

        Args:
            steps: Number of steps ahead to forecast.

        Returns:
            Array of forecasted volatility values.

        Raises:
            ValueError: If steps is not positive.
            RuntimeError: If GARCH model has not been fitted.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")
        if "garch" not in self.models:
            raise RuntimeError("GARCH model not fitted")

        result = self.models["garch"]
        forecast = result.forecast(horizon=steps)
        # variance is shape (1, steps) — take the last row
        variance = np.asarray(forecast.variance.values[-1])
        volatility = np.sqrt(variance)
        return volatility

    # ------------------------------------------------------------------
    # Factor Model (PCA)
    # ------------------------------------------------------------------

    def fit_factor_model(
        self, returns: pd.DataFrame, n_factors: int = 3
    ) -> None:
        """Fit a PCA-based factor model to multi-asset returns.

        Args:
            returns: DataFrame of asset returns (T x N).
            n_factors: Number of factors to extract.

        Raises:
            ValueError: If n_factors is not positive.
        """
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")

        from sklearn.decomposition import PCA

        # Standardize returns
        clean_returns = returns.dropna()
        mean = clean_returns.mean()
        std = clean_returns.std()
        standardized = (clean_returns - mean) / std

        pca = PCA(n_components=n_factors)
        factor_returns = pca.fit_transform(standardized.values)

        # Loadings: components_ shape (n_factors, n_assets)
        loadings = pca.components_.T  # (n_assets, n_factors)

        # Reconstruct returns
        reconstructed_standardized = factor_returns @ pca.components_
        reconstructed = reconstructed_standardized * std.values + mean.values
        reconstructed_df = pd.DataFrame(
            reconstructed, index=clean_returns.index, columns=clean_returns.columns
        )

        self.models["factor_model"] = {
            "loadings": loadings,
            "factor_returns": pd.DataFrame(
                factor_returns,
                index=clean_returns.index,
                columns=[f"factor_{i+1}" for i in range(n_factors)],
            ),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "reconstructed_returns": reconstructed_df,
            "pca": pca,
        }

    # ------------------------------------------------------------------
    # Regime Switching
    # ------------------------------------------------------------------

    def fit_regime_switching(
        self, series: pd.Series, n_regimes: int = 2
    ) -> None:
        """Fit a Markov regime switching model.

        Args:
            series: Time series data.
            n_regimes: Number of regimes (must be >= 2).

        Raises:
            ValueError: If n_regimes < 2.
        """
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")

        model = MarkovRegression(
            series.values, k_regimes=n_regimes, trend="c", switching_variance=True
        )
        result = model.fit(disp=False, maxiter=200)

        smoothed_probs = result.smoothed_marginal_probabilities
        # Ensure 2D array
        if smoothed_probs.ndim == 1:
            smoothed_probs = smoothed_probs.reshape(-1, 1)

        self.models["regime_switching"] = {
            "result": result,
            "smoothed_probs": smoothed_probs,
            "n_regimes": n_regimes,
        }

    # ------------------------------------------------------------------
    # Signal Generation
    # ------------------------------------------------------------------

    def generate_signal(self, series: pd.Series) -> SignalType:
        """Generate a trading signal from a price series.

        Uses a combination of trend (linear regression slope) and
        mean-reversion (z-score of last price vs rolling mean) to
        determine the signal.

        Args:
            series: Price series.

        Returns:
            SignalType: BULLISH, BEARISH, or NEUTRAL.

        Raises:
            ValueError: If series has fewer than 20 observations.
        """
        if len(series) < 20:
            raise ValueError("series must have at least 20 observations")

        # Compute log returns
        log_prices = np.log(series.values)
        returns = np.diff(log_prices)

        # Trend: linear regression on log prices
        n = len(log_prices)
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = log_prices.mean()
        slope = np.sum((x - x_mean) * (log_prices - y_mean)) / np.sum(
            (x - x_mean) ** 2
        )

        # Normalize slope by volatility
        vol = np.std(returns) * np.sqrt(252)  # annualized vol
        if vol < 1e-10:
            signal = SignalType.NEUTRAL
        else:
            normalized_slope = slope / (vol / np.sqrt(252))

            # Mean reversion: z-score of last price vs 20-day rolling mean
            window = min(20, len(series))
            rolling_mean = series.rolling(window).mean().iloc[-1]
            rolling_std = series.rolling(window).std().iloc[-1]
            if rolling_std < 1e-10:
                z_score = 0.0
            else:
                z_score = (series.iloc[-1] - rolling_mean) / rolling_std

            # Combined signal
            if normalized_slope > 0.3 and z_score < 2.0:
                signal = SignalType.BULLISH
            elif normalized_slope < -0.3 and z_score > -2.0:
                signal = SignalType.BEARISH
            else:
                signal = SignalType.NEUTRAL

        self.signals.append(signal)
        return signal

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def adf_test(self, series: pd.Series) -> dict[str, float]:
        """Run Augmented Dickey-Fuller test for stationarity.

        Args:
            series: Time series to test.

        Returns:
            Dictionary with test statistic and p-value.
        """
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "test_statistic": result[0],
            "p_value": result[1],
            "used_lag": result[2],
            "n_obs": result[3],
        }

    def clear_signals(self) -> None:
        """Clear all stored signals."""
        self.signals.clear()

    def clear_models(self) -> None:
        """Clear all fitted models."""
        self.models.clear()
