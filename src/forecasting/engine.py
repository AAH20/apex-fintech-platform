"""Time series forecasting engine.

Provides ARIMA, Prophet, LSTM, and ensemble forecasting methods
for financial time series analysis.

References:
    Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis: Forecasting and Control.
    Taylor, S.J. & Letham, B. (2018). Forecasting at scale. The American Statistician.
    Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from prophet import Prophet
    from statsmodels.tsa.arima.model import ARIMA

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Container for forecast results with metadata."""

    forecast: pd.Series
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None
    model_name: str = ""
    confidence_level: float = 0.95


@dataclass
class BacktestResult:
    """Container for backtesting results."""

    mae: float
    rmse: float
    mape: float
    predictions: pd.Series = field(repr=False)
    actuals: pd.Series = field(repr=False)


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------


class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ---------------------------------------------------------------------------
# Forecasting Engine
# ---------------------------------------------------------------------------


class ForecastingEngine:
    """Time series forecasting engine for financial data.

    Supports ARIMA, Prophet, LSTM, and ensemble forecasting methods
    with walk-forward backtesting and model comparison.
    """

    def __init__(self, data: Union[pd.Series, np.ndarray, List[float]]) -> None:
        """Initialize the forecasting engine.

        Args:
            data: Time series data as pandas Series, numpy array, or list.
        """
        self.data = self._prepare_data(data)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Dict[str, float]] = {}

    def _prepare_data(self, data: Union[pd.Series, np.ndarray, List[float]]) -> pd.Series:
        """Convert input data to pandas Series."""
        if isinstance(data, pd.Series):
            return data.copy()
        elif isinstance(data, np.ndarray):
            return pd.Series(data)
        elif isinstance(data, list):
            return pd.Series(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    # -----------------------------------------------------------------------
    # ARIMA
    # -----------------------------------------------------------------------

    def fit_arima(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        exog: Optional[pd.Series] = None,
    ) -> None:
        """Fit an ARIMA model.

        Args:
            order: (p, d, q) order of the ARIMA model.
            exog: Optional exogenous variables.
        """
        exog_data = None
        if exog is not None:
            exog_data = np.asarray(exog)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(self.data, order=order, exog=exog_data)
            fitted = model.fit()
        self.models["arima"] = fitted

    def _forecast_arima(self, steps: int, exog: Optional[pd.Series] = None) -> pd.Series:
        """Generate ARIMA forecast."""
        if "arima" not in self.models:
            raise ValueError("ARIMA model not fitted. Call fit_arima() first.")
        exog_data = None
        if exog is not None:
            exog_data = np.asarray(exog)
        result = self.models["arima"].forecast(steps=steps, exog=exog_data)
        return pd.Series(result.values if hasattr(result, "values") else result)

    def _forecast_arima_with_intervals(
        self, steps: int, alpha: float = 0.05
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate ARIMA forecast with confidence intervals."""
        if "arima" not in self.models:
            raise ValueError("ARIMA model not fitted. Call fit_arima() first.")
        result = self.models["arima"].get_forecast(steps=steps)
        forecast = result.predicted_mean
        conf_int = result.conf_int(alpha=alpha)
        return (
            pd.Series(forecast.values),
            pd.Series(conf_int.iloc[:, 0].values),
            pd.Series(conf_int.iloc[:, 1].values),
        )

    # -----------------------------------------------------------------------
    # Prophet
    # -----------------------------------------------------------------------

    def fit_prophet(
        self,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
    ) -> None:
        """Fit a Prophet model.

        Args:
            seasonality_mode: 'additive' or 'multiplicative'.
            changepoint_prior_scale: Flexibility of the trend.
        """
        df = pd.DataFrame(
            {"ds": self.data.index, "y": self.data.values}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
            )
            model.fit(df)
        self.models["prophet"] = model

    def _forecast_prophet(self, steps: int) -> pd.Series:
        """Generate Prophet forecast."""
        if "prophet" not in self.models:
            raise ValueError("Prophet model not fitted. Call fit_prophet() first.")
        model = self.models["prophet"]
        future = model.make_future_dataframe(periods=steps, freq="D")
        forecast = model.predict(future)
        return pd.Series(forecast["yhat"].values[-steps:])

    # -----------------------------------------------------------------------
    # LSTM
    # -----------------------------------------------------------------------

    def fit_lstm(
        self,
        hidden_size: int = 50,
        epochs: int = 100,
        seq_length: int = 10,
        learning_rate: float = 0.001,
        num_layers: int = 1,
    ) -> None:
        """Fit an LSTM model.

        Args:
            hidden_size: Number of hidden units in LSTM layer.
            epochs: Number of training epochs.
            seq_length: Length of input sequences.
            learning_rate: Learning rate for optimizer.
            num_layers: Number of LSTM layers.
        """
        # Normalize data
        mean = float(self.data.mean())
        std = float(self.data.std())
        if std == 0:
            std = 1.0
        self.scalers["lstm"] = {"mean": mean, "std": std}
        normalized = (self.data.values - mean) / std

        # Create sequences
        X, y = self._create_sequences(normalized, seq_length)
        if len(X) == 0:
            raise ValueError("Not enough data for the given sequence length.")

        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)

        # Build and train model
        model = LSTMModel(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        self.models["lstm"] = {
            "model": model,
            "seq_length": seq_length,
            "hidden_size": hidden_size,
        }

    def _create_sequences(
        self, data: np.ndarray, seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for LSTM."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def _forecast_lstm(self, steps: int) -> pd.Series:
        """Generate LSTM forecast."""
        if "lstm" not in self.models:
            raise ValueError("LSTM model not fitted. Call fit_lstm() first.")

        model_info = self.models["lstm"]
        model = model_info["model"]
        seq_length = model_info["seq_length"]
        scaler = self.scalers["lstm"]

        model.eval()
        with torch.no_grad():
            # Use the last seq_length values as input
            last_values = self.data.values[-seq_length:]
            normalized = (last_values - scaler["mean"]) / scaler["std"]
            current_seq = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(-1)

            predictions = []
            for _ in range(steps):
                pred = model(current_seq)
                pred_val = pred.item()
                predictions.append(pred_val)

                # Update sequence: remove first element, append prediction
                new_val = torch.FloatTensor([[[pred_val]]])
                current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

        # Denormalize
        predictions = np.array(predictions) * scaler["std"] + scaler["mean"]
        return pd.Series(predictions)

    # -----------------------------------------------------------------------
    # Generic forecast interface
    # -----------------------------------------------------------------------

    def forecast(
        self, steps: int, model: str = "arima", exog: Optional[pd.Series] = None
    ) -> pd.Series:
        """Generate forecast using specified model.

        Args:
            steps: Number of steps to forecast.
            model: Model to use ('arima', 'prophet', 'lstm').
            exog: Optional exogenous variables for ARIMA forecasting.

        Returns:
            Forecast as pandas Series.
        """
        if model == "arima":
            return self._forecast_arima(steps, exog=exog)
        elif model == "prophet":
            return self._forecast_prophet(steps)
        elif model == "lstm":
            return self._forecast_lstm(steps)
        else:
            raise ValueError(f"Unknown model: {model}")

    def forecast_with_intervals(
        self, steps: int, model: str = "arima", confidence_level: float = 0.95
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate forecast with confidence intervals.

        Args:
            steps: Number of steps to forecast.
            model: Model to use.
            confidence_level: Confidence level (e.g., 0.95 for 95%).

        Returns:
            Tuple of (forecast, lower_bound, upper_bound).
        """
        alpha = 1.0 - confidence_level
        if model == "arima":
            return self._forecast_arima_with_intervals(steps, alpha)
        else:
            # For non-ARIMA models, return forecast with NaN intervals
            forecast = self.forecast(steps, model)
            nan_series = pd.Series([np.nan] * steps)
            return forecast, nan_series, nan_series

    # -----------------------------------------------------------------------
    # Ensemble methods
    # -----------------------------------------------------------------------

    def ensemble_forecast(
        self,
        steps: int,
        models: List[str],
        weights: Optional[List[float]] = None,
    ) -> pd.Series:
        """Generate ensemble forecast by combining multiple models.

        Args:
            steps: Number of steps to forecast.
            models: List of model names to combine.
            weights: Optional weights for weighted average. If None, uses equal weights.

        Returns:
            Ensemble forecast as pandas Series.
        """
        if not models:
            raise ValueError("At least one model must be specified.")

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models.")

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        forecasts = []
        for model_name in models:
            forecast = self.forecast(steps, model_name)
            forecasts.append(forecast.values)

        forecasts_array = np.array(forecasts)
        ensemble = np.average(forecasts_array, axis=0, weights=weights)
        return pd.Series(ensemble)

    # -----------------------------------------------------------------------
    # Backtesting
    # -----------------------------------------------------------------------

    def backtest(
        self,
        model: str = "arima",
        train_size: float = 0.8,
        steps: int = 1,
    ) -> Dict[str, float]:
        """Perform walk-forward backtesting.

        Args:
            model: Model to backtest.
            train_size: Proportion of data to use for training.
            steps: Number of steps to forecast in each iteration.

        Returns:
            Dictionary with 'mae', 'rmse', 'mape' metrics.
        """
        n = len(self.data)
        train_end = int(n * train_size)

        if train_end < 10:
            raise ValueError("Not enough data for backtesting with given train_size.")

        predictions = []
        actuals = []

        for i in range(train_end, n, steps):
            train_data = self.data.iloc[:i]
            test_data = self.data.iloc[i : i + steps]

            if len(test_data) == 0:
                break

            # Fit model on training data
            engine = ForecastingEngine(train_data)
            if model == "arima":
                engine.fit_arima(order=(1, 1, 1))
            elif model == "prophet":
                engine.fit_prophet()
            elif model == "lstm":
                engine.fit_lstm(hidden_size=20, epochs=50, seq_length=10)
            else:
                raise ValueError(f"Unknown model: {model}")

            forecast = engine.forecast(steps=len(test_data), model=model)
            predictions.extend(forecast.values)
            actuals.extend(test_data.values)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = float(np.mean(np.abs(predictions - actuals)))
        rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        mape = float(np.mean(np.abs((predictions - actuals) / actuals)) * 100)

        return {"mae": mae, "rmse": rmse, "mape": mape}

    # -----------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------

    def compare_models(self, steps: int = 5) -> Dict[str, Dict[str, float]]:
        """Compare all fitted models using backtesting.

        Args:
            steps: Number of steps to forecast in each backtest iteration.

        Returns:
            Dictionary mapping model names to their error metrics.
        """
        results = {}
        for model_name in self.models.keys():
            metrics = self.backtest(model=model_name, train_size=0.8, steps=steps)
            results[model_name] = metrics
        return results
