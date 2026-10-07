"""Tests for the Quantitative Analytics Engine.

TDD: These tests define the expected behavior of QuantitativeAnalyticsEngine.
Covers ARIMA, GARCH, factor models, regime switching, and signal generation.

References:
    - Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis.
    - Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity.
    - Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroscedasticity.
    - Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary
      Time Series and the Business Cycle.
    - CFA Institute Quantitative Methods curriculum.
"""

import numpy as np
import pandas as pd
import pytest

from quant.analytics import QuantitativeAnalyticsEngine, SignalType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh engine instance for each test."""
    return QuantitativeAnalyticsEngine()


@pytest.fixture
def price_series():
    """Deterministic price series with trend and noise (252 trading days)."""
    np.random.seed(42)
    n = 252
    returns = np.random.normal(0.0005, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="price")


@pytest.fixture
def stationary_series():
    """Stationary AR(1) series for ARIMA testing."""
    np.random.seed(123)
    n = 200
    e = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + e[i]
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(y, index=dates, name="value")


@pytest.fixture
def multi_asset_returns():
    """Multi-asset return matrix for factor model testing."""
    np.random.seed(99)
    n = 300
    # 3 latent factors drive 5 assets
    f1 = np.random.normal(0, 0.01, n)
    f2 = np.random.normal(0, 0.008, n)
    f3 = np.random.normal(0, 0.005, n)
    loadings = np.array([
        [0.8, 0.3, 0.1],
        [0.7, 0.4, 0.2],
        [0.6, 0.5, 0.1],
        [0.5, 0.6, 0.3],
        [0.4, 0.7, 0.2],
    ])
    factors = np.column_stack([f1, f2, f3])
    returns = factors @ loadings.T + np.random.normal(0, 0.005, (n, 5))
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    return pd.DataFrame(returns, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Engine Initialization
# ---------------------------------------------------------------------------


class TestEngineInit:
    """Test engine initialization and basic properties."""

    def test_engine_initializes(self):
        """Engine can be instantiated."""
        engine = QuantitativeAnalyticsEngine()
        assert engine is not None

    def test_engine_has_no_models_initially(self):
        """Engine starts with no fitted models."""
        engine = QuantitativeAnalyticsEngine()
        assert engine.models == {}

    def test_engine_has_no_signals_initially(self):
        """Engine starts with no signals."""
        engine = QuantitativeAnalyticsEngine()
        assert engine.signals == []


# ---------------------------------------------------------------------------
# ARIMA Model
# ---------------------------------------------------------------------------


class TestARIMA:
    """Tests for ARIMA model fitting and forecasting."""

    def test_arima_fit_returns_model(self, engine, stationary_series):
        """ARIMA fit stores a fitted model."""
        engine.fit_arima(stationary_series, order=(1, 0, 0))
        assert "arima" in engine.models
        assert engine.models["arima"] is not None

    def test_arima_forecast_returns_values(self, engine, stationary_series):
        """ARIMA forecast returns correct number of steps."""
        engine.fit_arima(stationary_series, order=(1, 0, 0))
        forecast = engine.forecast_arima(steps=10)
        assert len(forecast) == 10
        assert all(np.isfinite(forecast))

    def test_arima_forecast_positive_steps(self, engine, stationary_series):
        """ARIMA forecast requires positive steps."""
        engine.fit_arima(stationary_series, order=(1, 0, 0))
        with pytest.raises(ValueError, match="steps must be positive"):
            engine.forecast_arima(steps=0)

    def test_arima_aic_available(self, engine, stationary_series):
        """ARIMA model exposes AIC for model selection."""
        engine.fit_arima(stationary_series, order=(1, 0, 0))
        result = engine.models["arima"]
        assert hasattr(result, "aic")
        assert np.isfinite(result.aic)

    def test_arima_order_validation(self, engine, stationary_series):
        """ARIMA rejects invalid orders."""
        with pytest.raises(ValueError, match="order must be a tuple of 3"):
            engine.fit_arima(stationary_series, order=(1, 0))

    def test_arima_forecast_before_fit_raises(self, engine):
        """Forecasting before fitting raises an error."""
        with pytest.raises(RuntimeError, match="ARIMA model not fitted"):
            engine.forecast_arima(steps=5)


# ---------------------------------------------------------------------------
# GARCH Model
# ---------------------------------------------------------------------------


class TestGARCH:
    """Tests for GARCH volatility model fitting and forecasting."""

    def test_garch_fit_returns_model(self, engine, stationary_series):
        """GARCH fit stores a fitted model."""
        engine.fit_garch(stationary_series, p=1, q=1)
        assert "garch" in engine.models
        assert engine.models["garch"] is not None

    def test_garch_forecast_returns_volatility(self, engine, stationary_series):
        """GARCH forecast returns volatility estimates."""
        engine.fit_garch(stationary_series, p=1, q=1)
        forecast = engine.forecast_garch(steps=5)
        assert len(forecast) == 5
        assert all(v > 0 for v in forecast)
        assert all(np.isfinite(forecast))

    def test_garch_forecast_positive_steps(self, engine, stationary_series):
        """GARCH forecast requires positive steps."""
        engine.fit_garch(stationary_series, p=1, q=1)
        with pytest.raises(ValueError, match="steps must be positive"):
            engine.forecast_garch(steps=-1)

    def test_garch_forecast_before_fit_raises(self, engine):
        """GARCH forecasting before fitting raises an error."""
        with pytest.raises(RuntimeError, match="GARCH model not fitted"):
            engine.forecast_garch(steps=5)

    def test_garch_variance_positive(self, engine, stationary_series):
        """GARCH conditional variance is always positive."""
        engine.fit_garch(stationary_series, p=1, q=1)
        result = engine.models["garch"]
        cond_vol = result.conditional_volatility
        assert all(cond_vol > 0)


# ---------------------------------------------------------------------------
# Factor Model
# ---------------------------------------------------------------------------


class TestFactorModel:
    """Tests for PCA-based factor model."""

    def test_factor_model_fit(self, engine, multi_asset_returns):
        """Factor model fit stores results."""
        engine.fit_factor_model(multi_asset_returns, n_factors=3)
        assert "factor_model" in engine.models
        assert engine.models["factor_model"] is not None

    def test_factor_model_returns_factor_loadings(self, engine, multi_asset_returns):
        """Factor model produces loadings matrix."""
        engine.fit_factor_model(multi_asset_returns, n_factors=3)
        result = engine.models["factor_model"]
        assert "loadings" in result
        assert result["loadings"].shape == (5, 3)

    def test_factor_model_returns_factor_returns(self, engine, multi_asset_returns):
        """Factor model produces factor return series."""
        engine.fit_factor_model(multi_asset_returns, n_factors=3)
        result = engine.models["factor_model"]
        assert "factor_returns" in result
        assert result["factor_returns"].shape[1] == 3

    def test_factor_model_explained_variance(self, engine, multi_asset_returns):
        """Factor model reports explained variance ratio."""
        engine.fit_factor_model(multi_asset_returns, n_factors=3)
        result = engine.models["factor_model"]
        assert "explained_variance_ratio" in result
        evr = result["explained_variance_ratio"]
        assert len(evr) == 3
        assert all(0 <= v <= 1 for v in evr)
        assert sum(evr) <= 1.0

    def test_factor_model_reconstruct_returns(self, engine, multi_asset_returns):
        """Factor model can reconstruct approximate returns."""
        engine.fit_factor_model(multi_asset_returns, n_factors=3)
        result = engine.models["factor_model"]
        reconstructed = result["reconstructed_returns"]
        assert reconstructed.shape == multi_asset_returns.shape

    def test_factor_model_n_factors_validation(self, engine, multi_asset_returns):
        """Factor model rejects too many factors."""
        with pytest.raises(ValueError, match="n_factors must be positive"):
            engine.fit_factor_model(multi_asset_returns, n_factors=0)


# ---------------------------------------------------------------------------
# Regime Switching
# ---------------------------------------------------------------------------


class TestRegimeSwitching:
    """Tests for Markov regime switching model."""

    def test_regime_switching_fit(self, engine, stationary_series):
        """Regime switching model fits successfully."""
        engine.fit_regime_switching(stationary_series, n_regimes=2)
        assert "regime_switching" in engine.models
        assert engine.models["regime_switching"] is not None

    def test_regime_switching_smoothed_probs(self, engine, stationary_series):
        """Regime switching produces smoothed probabilities."""
        engine.fit_regime_switching(stationary_series, n_regimes=2)
        result = engine.models["regime_switching"]
        assert "smoothed_probs" in result
        probs = result["smoothed_probs"]
        assert probs.shape[1] == 2
        # Probabilities sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_regime_switching_n_regimes_validation(self, engine, stationary_series):
        """Regime switching requires at least 2 regimes."""
        with pytest.raises(ValueError, match="n_regimes must be >= 2"):
            engine.fit_regime_switching(stationary_series, n_regimes=1)


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------


class TestSignalGeneration:
    """Tests for trading signal generation."""

    def test_generate_signal_returns_signal_type(self, engine, price_series):
        """Signal generation returns a valid signal type."""
        signal = engine.generate_signal(price_series)
        assert isinstance(signal, SignalType)

    def test_generate_signal_bullish_on_uptrend(self, engine):
        """Strong uptrend generates BULLISH signal."""
        np.random.seed(7)
        n = 100
        returns = np.random.normal(0.005, 0.01, n)  # strong positive drift
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        series = pd.Series(prices, index=dates)
        signal = engine.generate_signal(series)
        assert signal == SignalType.BULLISH

    def test_generate_signal_bearish_on_downtrend(self, engine):
        """Strong downtrend generates BEARISH signal."""
        np.random.seed(7)
        n = 100
        returns = np.random.normal(-0.005, 0.01, n)  # strong negative drift
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        series = pd.Series(prices, index=dates)
        signal = engine.generate_signal(series)
        assert signal == SignalType.BEARISH

    def test_generate_signal_neutral_on_flat(self, engine):
        """Flat/mean-reverting series generates NEUTRAL signal."""
        np.random.seed(42)
        n = 100
        returns = np.random.normal(0, 0.005, n)  # no drift
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        series = pd.Series(prices, index=dates)
        signal = engine.generate_signal(series)
        assert signal == SignalType.NEUTRAL

    def test_signals_stored_in_engine(self, engine, price_series):
        """Generated signals are stored in engine state."""
        initial_count = len(engine.signals)
        engine.generate_signal(price_series)
        assert len(engine.signals) == initial_count + 1

    def test_signal_with_short_series_raises(self, engine):
        """Signal generation requires sufficient data."""
        short_series = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 20 observations"):
            engine.generate_signal(short_series)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_arima_garch_signals(self, engine, price_series):
        """Full pipeline: fit ARIMA, GARCH, generate signal."""
        engine.fit_arima(price_series, order=(1, 1, 1))
        engine.fit_garch(price_series, p=1, q=1)
        signal = engine.generate_signal(price_series)
        assert isinstance(signal, SignalType)
        assert "arima" in engine.models
        assert "garch" in engine.models

    def test_full_pipeline_factor_model(self, engine, multi_asset_returns, price_series):
        """Full pipeline: factor model + signal generation."""
        engine.fit_factor_model(multi_asset_returns, n_factors=3)
        signal = engine.generate_signal(price_series)
        assert isinstance(signal, SignalType)
        assert "factor_model" in engine.models

    def test_multiple_signals_accumulate(self, engine, price_series):
        """Multiple signal calls accumulate in engine state."""
        engine.generate_signal(price_series)
        engine.generate_signal(price_series)
        engine.generate_signal(price_series)
        assert len(engine.signals) == 3
