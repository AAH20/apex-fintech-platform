"""Tests for time series forecasting engine.

Covers ARIMA, Prophet, LSTM, ensemble methods, and backtesting.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting import ForecastingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_series():
    """Generate a synthetic financial time series with trend + seasonality + noise."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    trend = np.linspace(100, 150, 200)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 30)
    noise = np.random.normal(0, 2, 200)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name="price")


@pytest.fixture
def short_series():
    """Generate a short time series for edge case testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
    values = np.random.normal(100, 5, 30)
    return pd.Series(values, index=dates, name="price")


# ---------------------------------------------------------------------------
# ARIMA Tests
# ---------------------------------------------------------------------------


class TestARIMAForecasting:
    """Tests for ARIMA model fitting and forecasting."""

    def test_arima_fit_and_predict(self, sample_series):
        """ARIMA model can be fitted and produces a forecast."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        forecast = engine.forecast(steps=5, model="arima")
        assert len(forecast) == 5
        assert isinstance(forecast, pd.Series)

    def test_arima_forecast_length(self, sample_series):
        """ARIMA forecast has the requested number of steps."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        forecast = engine.forecast(steps=10, model="arima")
        assert len(forecast) == 10

    def test_arima_confidence_intervals(self, sample_series):
        """ARIMA forecast includes confidence intervals."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        forecast, lower, upper = engine.forecast_with_intervals(steps=5, model="arima")
        assert len(forecast) == 5
        assert len(lower) == 5
        assert len(upper) == 5
        assert (lower <= forecast).all()
        assert (forecast <= upper).all()

    def test_arima_with_exog(self, sample_series):
        """ARIMA model can be fitted with exogenous variables."""
        engine = ForecastingEngine(sample_series)
        exog = pd.Series(
            np.random.normal(0, 1, len(sample_series)),
            index=sample_series.index,
            name="exog",
        )
        engine.fit_arima(order=(1, 1, 1), exog=exog)
        # Forecasting with exog requires future exog values
        future_exog = pd.Series(np.random.normal(0, 1, 5))
        forecast = engine.forecast(steps=5, model="arima", exog=future_exog)
        assert len(forecast) == 5

    def test_arima_different_order(self, sample_series):
        """ARIMA model works with different (p, d, q) orders."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(2, 1, 2))
        forecast = engine.forecast(steps=5, model="arima")
        assert len(forecast) == 5


# ---------------------------------------------------------------------------
# Prophet Tests
# ---------------------------------------------------------------------------


class TestProphetForecasting:
    """Tests for Prophet model fitting and forecasting."""

    def test_prophet_fit_and_predict(self, sample_series):
        """Prophet model can be fitted and produces a forecast."""
        engine = ForecastingEngine(sample_series)
        engine.fit_prophet()
        forecast = engine.forecast(steps=5, model="prophet")
        assert len(forecast) == 5
        assert isinstance(forecast, pd.Series)

    def test_prophet_forecast_length(self, sample_series):
        """Prophet forecast has the requested number of steps."""
        engine = ForecastingEngine(sample_series)
        engine.fit_prophet()
        forecast = engine.forecast(steps=10, model="prophet")
        assert len(forecast) == 10

    def test_prophet_with_seasonality_mode(self, sample_series):
        """Prophet model works with custom seasonality mode."""
        engine = ForecastingEngine(sample_series)
        engine.fit_prophet(seasonality_mode="multiplicative")
        forecast = engine.forecast(steps=5, model="prophet")
        assert len(forecast) == 5


# ---------------------------------------------------------------------------
# LSTM Tests
# ---------------------------------------------------------------------------


class TestLSTMForecasting:
    """Tests for LSTM model fitting and forecasting."""

    def test_lstm_fit_and_predict(self, sample_series):
        """LSTM model can be fitted and produces a forecast."""
        engine = ForecastingEngine(sample_series)
        engine.fit_lstm(hidden_size=20, epochs=50, seq_length=10)
        forecast = engine.forecast(steps=5, model="lstm")
        assert len(forecast) == 5
        assert isinstance(forecast, pd.Series)

    def test_lstm_forecast_length(self, sample_series):
        """LSTM forecast has the requested number of steps."""
        engine = ForecastingEngine(sample_series)
        engine.fit_lstm(hidden_size=20, epochs=50, seq_length=10)
        forecast = engine.forecast(steps=10, model="lstm")
        assert len(forecast) == 10


# ---------------------------------------------------------------------------
# Ensemble Tests
# ---------------------------------------------------------------------------


class TestEnsembleForecasting:
    """Tests for ensemble forecasting methods."""

    def test_ensemble_simple_average(self, sample_series):
        """Ensemble forecast combines multiple models with simple average."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        engine.fit_prophet()
        forecast = engine.ensemble_forecast(steps=5, models=["arima", "prophet"])
        assert len(forecast) == 5
        assert isinstance(forecast, pd.Series)

    def test_ensemble_weighted_average(self, sample_series):
        """Ensemble forecast supports weighted average."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        engine.fit_prophet()
        forecast = engine.ensemble_forecast(
            steps=5, models=["arima", "prophet"], weights=[0.7, 0.3]
        )
        assert len(forecast) == 5

    def test_ensemble_multiple_models(self, sample_series):
        """Ensemble forecast can combine more than two models."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        engine.fit_prophet()
        engine.fit_lstm(hidden_size=20, epochs=50, seq_length=10)
        forecast = engine.ensemble_forecast(
            steps=5, models=["arima", "prophet", "lstm"]
        )
        assert len(forecast) == 5


# ---------------------------------------------------------------------------
# Backtesting Tests
# ---------------------------------------------------------------------------


class TestBacktesting:
    """Tests for walk-forward backtesting."""

    def test_backtest_walk_forward(self, sample_series):
        """Backtest returns error metrics for a fitted model."""
        engine = ForecastingEngine(sample_series)
        results = engine.backtest(model="arima", train_size=0.8, steps=5)
        assert "mae" in results
        assert "rmse" in results
        assert "mape" in results
        assert results["mae"] >= 0
        assert results["rmse"] >= 0

    def test_backtest_rmse_gte_mae(self, sample_series):
        """RMSE is always greater than or equal to MAE."""
        engine = ForecastingEngine(sample_series)
        results = engine.backtest(model="arima", train_size=0.8, steps=5)
        assert results["rmse"] >= results["mae"]


# ---------------------------------------------------------------------------
# Model Comparison Tests
# ---------------------------------------------------------------------------


class TestModelComparison:
    """Tests for comparing multiple models."""

    def test_compare_models(self, sample_series):
        """Model comparison returns metrics for all fitted models."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        engine.fit_prophet()
        comparison = engine.compare_models(steps=5)
        assert "arima" in comparison
        assert "prophet" in comparison
        assert "mae" in comparison["arima"]
        assert "rmse" in comparison["prophet"]


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_short_series(self, short_series):
        """Engine handles short time series."""
        engine = ForecastingEngine(short_series)
        engine.fit_arima(order=(1, 1, 1))
        forecast = engine.forecast(steps=3, model="arima")
        assert len(forecast) == 3

    def test_single_step_forecast(self, sample_series):
        """Engine can produce a single-step forecast."""
        engine = ForecastingEngine(sample_series)
        engine.fit_arima(order=(1, 1, 1))
        forecast = engine.forecast(steps=1, model="arima")
        assert len(forecast) == 1

    def test_numpy_array_input(self):
        """Engine accepts numpy array input."""
        np.random.seed(42)
        data = np.random.normal(100, 5, 100)
        engine = ForecastingEngine(data)
        engine.fit_arima(order=(1, 1, 1))
        forecast = engine.forecast(steps=5, model="arima")
        assert len(forecast) == 5

    def test_list_input(self):
        """Engine accepts list input."""
        np.random.seed(42)
        data = list(np.random.normal(100, 5, 100))
        engine = ForecastingEngine(data)
        engine.fit_arima(order=(1, 1, 1))
        forecast = engine.forecast(steps=5, model="arima")
        assert len(forecast) == 5
