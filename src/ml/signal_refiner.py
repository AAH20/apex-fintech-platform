"""Signal refinement module for ML pipeline — filters and transforms ML predictions
into optimized trading signals for evolved alpha generation.

Numpy-only, no sklearn/torch dependencies. Compatible with existing pipeline
outputs. Provides technical filters, regime detection, and signal enhancement.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class SignalRefiner:
    """Refines ML model predictions into trading signals via technical filters
    and regime-aware adjustments.

    Filters include: momentum threshold, mean-reversion adjustment,
    volatility scaling, and regime detection. All operations are numpy-only
    for deployment portability.

    Example:
        >>> import numpy as np
        >>> from ml.pipeline import run_backtest
        >>> # Assume predictions and returns from ML model
        >>> predictions = np.random.randn(100)
        >>> returns = np.random.randn(100) * 0.02
        >>> # Refine signals
        >>> refiner = SignalRefiner(vol_target=0.15, momentum_lookback=5)
        >>> signals = refiner.refine(predictions, returns)
        >>> metrics = run_backtest(signals, returns, allow_short=True)
    """

    def __init__(
        self,
        vol_target: float = 0.15,  # annual volatility target for scaling
        momentum_lookback: int = 5,
        mean_reversion_speed: float = 0.1,
        regime_lookback: int = 20,
        min_signal_strength: float = 0.3,
    ):
        """Initialize signal refiner.

        Args:
            vol_target: Target annual volatility for position scaling.
            momentum_lookback: Lookback periods for momentum filter.
            mean_reversion_speed: Speed parameter for mean-reversion adjustment.
            regime_lookback: Periods for regime (high/low volatility) detection.
            min_signal_strength: Minimum absolute prediction to generate signal.
        """
        self.vol_target = vol_target
        self.momentum_lookback = momentum_lookback
        self.mean_reversion_speed = mean_reversion_speed
        self.regime_lookback = regime_lookback
        self.min_signal_strength = min_signal_strength

    def _compute_volatility(self, returns: np.ndarray, lookback: int) -> np.ndarray:
        """Rolling volatility over specified lookback periods."""
        n = len(returns)
        volatilities = np.zeros(n)
        for i in range(n):
            start = max(0, i - lookback + 1)
            r = returns[start : i + 1]
            if len(r) > 1:
                volatilities[i] = np.std(r) * np.sqrt(252)  # annualized
            else:
                volatilities[i] = np.std(returns) * np.sqrt(252)
        return volatilities

    def _compute_momentum(self, predictions: np.ndarray, lookback: int) -> np.ndarray:
        """Simple momentum filter on predictions."""
        n = len(predictions)
        momentum = np.zeros(n)
        for i in range(n):
            start = max(0, i - lookback + 1)
            if i > 0 and start < i:
                # Momentum = change in prediction over lookback
                mom = predictions[i] - predictions[start]
                momentum[i] = mom
            else:
                momentum[i] = 0.0
        return momentum

    def _detect_regime(self, vol_series: np.ndarray) -> np.ndarray:
        """Detect high/low volatility regime."""
        # Compute rolling mean of volatility
        rolling_mean = np.convolve(vol_series, np.ones(self.regime_lookback) / self.regime_lookback, mode="same")
        # Regime: 1 = high vol, 0 = low vol, -1 = transition
        threshold = np.percentile(vol_series, 70)
        regime = np.where(vol_series > threshold, 1, np.where(vol_series < np.percentile(vol_series, 30), -1, 0))
        return regime

    def refine(
        self,
        predictions: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Refine ML predictions into trading signals.

        Args:
            predictions: ML model predictions (n_periods,).
            returns: Realized asset returns (n_periods,).

        Returns:
            Refined signal array (n_periods,), values typically in [-1, 1]
            where sign indicates direction and magnitude indicates strength.
        """
        predictions = np.asarray(predictions).flatten()
        returns = np.asarray(returns).flatten()
        n = len(predictions)

        # 1. Basic thresholding
        basic_signal = np.sign(predictions)
        abs_pred = np.abs(predictions)

        # 2. Volatility scaling — scale signal by inverse volatility
        vol = self._compute_volatility(returns, self.regime_lookback)
        vol_safe = np.where(vol == 0, np.std(returns) * np.sqrt(252), vol)
        vol_scaling = self.vol_target / vol_safe
        # Cap scaling to avoid extreme positions
        vol_scaling = np.clip(vol_scaling, 0.5, 2.0)

        # 3. Mean-reversion adjustment
        # If prediction and recent return have same sign, dampen; opposite sign, amplify
        recent_returns = returns[max(0, len(returns) - self.momentum_lookback) :]
        mean_reversion = np.ones(n)
        if len(recent_returns) > 0:
            # Simple proxy: use overall mean reversion speed
            # Sign of prediction*return: same sign -> dampen, opposite -> amplify
            signal_return_product = predictions * returns
            mean_reversion = 1 - self.mean_reversion_speed * np.sign(signal_return_product)

        # 4. Momentum filter
        momentum = self._compute_momentum(predictions, self.momentum_lookback)
        momentum_factor = 1 + 0.1 * np.tanh(momentum / (np.abs(predictions) + 1e-8))

        # 5. Regime adjustment
        vol_series = self._compute_volatility(returns, self.regime_lookback)
        regime = self._detect_regime(vol_series)
        regime_factor = np.where(regime == 1, 0.8, np.where(regime == -1, 1.2, 1.0))

        # 6. Minimum signal strength filter
        signal = (
            basic_signal
            * np.clip(abs_pred, self.min_signal_strength, 1.0)
            * vol_scaling
            * mean_reversion
            * momentum_factor
            * regime_factor
        )

        # Normalize signals to prevent runaway positions
        signal_norm = np.sqrt(np.mean(signal ** 2)) if np.mean(signal ** 2) > 0 else 1.0
        signal = signal / signal_norm

        return signal


def refine_signals(
    predictions: np.ndarray,
    returns: np.ndarray,
    vol_target: float = 0.15,
    momentum_lookback: int = 5,
    mean_reversion_speed: float = 0.1,
    regime_lookback: int = 20,
    min_signal_strength: float = 0.3,
) -> np.ndarray:
    """Convenience function to refine signals.

    Args:
        predictions: ML model predictions (n_periods,).
        returns: Realized asset returns (n_periods,).
        vol_target: Annual volatility target for scaling.
        momentum_lookback: Lookback periods for momentum filter.
        mean_reversion_speed: Speed parameter for mean-reversion adjustment.
        regime_lookback: Periods for regime detection.
        min_signal_strength: Minimum absolute prediction to generate signal.

    Returns:
        Refined signal array (n_periods,).
    """
    refiner = SignalRefiner(
        vol_target=vol_target,
        momentum_lookback=momentum_lookback,
        mean_reversion_speed=mean_reversion_speed,
        regime_lookback=regime_lookback,
        min_signal_strength=min_signal_strength,
    )
    return refiner.refine(predictions, returns)