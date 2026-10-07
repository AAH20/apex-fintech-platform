"""Monte Carlo simulation engine for stochastic processes and option pricing.

References:
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    Black, F. & Scholes, M. (1973). The pricing of options and corporate
    liabilities. Journal of Political Economy, 81(3), 637-654.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    """Result of a Monte Carlo simulation."""

    price: float
    std_error: float
    confidence_interval: tuple[float, float]
    n_paths: int


class MonteCarloEngine:
    """Monte Carlo simulation engine for stochastic processes and option pricing."""

    def __init__(self, dt: float = 0.01, seed: int | None = None) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _reset_rng(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def simulate_gbm(
        self, s0: float, mu: float, sigma: float, T: float, n_paths: int
    ) -> np.ndarray:
        """Generate geometric Brownian motion paths.

        Args:
            s0: Initial asset price.
            mu: Drift (expected return).
            sigma: Volatility.
            T: Time horizon.
            n_paths: Number of paths to simulate.

        Returns:
            Array of shape (n_paths, n_steps + 1) with GBM paths.
        """
        if s0 <= 0:
            raise ValueError("s0 must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if T < 0:
            raise ValueError("T must be non-negative")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        self._reset_rng()
        n_steps = max(1, int(round(T / self.dt)))
        dt_actual = T / n_steps

        W = self._rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt_actual)
        t = np.arange(n_steps + 1) * dt_actual
        drift = (mu - 0.5 * sigma**2) * t
        diffusion = sigma * np.concatenate([np.zeros((n_paths, 1)), np.cumsum(W, axis=1)], axis=1)
        return s0 * np.exp(drift + diffusion)

    def price_european_option(
        self,
        s0: float,
        k: float,
        r: float,
        sigma: float,
        T: float,
        n_paths: int,
        option_type: str = "call",
    ) -> SimulationResult:
        """Price a European option using Monte Carlo.

        Args:
            s0: Current asset price.
            k: Strike price.
            r: Risk-free rate.
            sigma: Volatility.
            T: Time to maturity.
            n_paths: Number of Monte Carlo paths.
            option_type: "call" or "put".

        Returns:
            SimulationResult with price estimate and statistics.
        """
        if s0 <= 0:
            raise ValueError("s0 must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if T < 0:
            raise ValueError("T must be non-negative")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        self._reset_rng()

        if T == 0:
            if option_type == "call":
                payoff = max(s0 - k, 0.0)
            else:
                payoff = max(k - s0, 0.0)
            return SimulationResult(price=payoff, std_error=0.0, confidence_interval=(payoff, payoff), n_paths=n_paths)

        n_steps = max(1, int(round(T / self.dt)))
        dt_actual = T / n_steps

        W = self._rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt_actual)
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sum(W, axis=1)
        S_T = s0 * np.exp(drift + diffusion)

        if option_type == "call":
            payoffs = np.exp(-r * T) * np.maximum(S_T - k, 0.0)
        else:
            payoffs = np.exp(-r * T) * np.maximum(k - S_T, 0.0)

        price = float(np.mean(payoffs))
        std_error = float(np.std(payoffs, ddof=1) / math.sqrt(n_paths))
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return SimulationResult(
            price=price,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            n_paths=n_paths,
        )

    def price_american_option(
        self,
        s0: float,
        k: float,
        r: float,
        sigma: float,
        T: float,
        n_paths: int,
        option_type: str = "call",
    ) -> SimulationResult:
        """Price an American option using Least Squares Monte Carlo (LSM).

        Args:
            s0: Current asset price.
            k: Strike price.
            r: Risk-free rate.
            sigma: Volatility.
            T: Time to maturity.
            n_paths: Number of Monte Carlo paths.
            option_type: "call" or "put".

        Returns:
            SimulationResult with price estimate and statistics.
        """
        if s0 <= 0:
            raise ValueError("s0 must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if T < 0:
            raise ValueError("T must be non-negative")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        self._reset_rng()

        if T == 0:
            if option_type == "call":
                payoff = max(s0 - k, 0.0)
            else:
                payoff = max(k - s0, 0.0)
            return SimulationResult(price=payoff, std_error=0.0, confidence_interval=(payoff, payoff), n_paths=n_paths)

        n_steps = max(1, int(round(T / self.dt)))
        dt_actual = T / n_steps

        # Generate paths
        W = self._rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt_actual)
        t = np.arange(n_steps + 1) * dt_actual
        drift = (r - 0.5 * sigma**2) * t
        diffusion = sigma * np.concatenate([np.zeros((n_paths, 1)), np.cumsum(W, axis=1)], axis=1)
        paths = s0 * np.exp(drift + diffusion)

        # Payoff at maturity
        if option_type == "call":
            cash_flows = np.maximum(paths[:, -1] - k, 0.0)
        else:
            cash_flows = np.maximum(k - paths[:, -1], 0.0)

        # Backward induction (LSM)
        for step in range(n_steps - 1, 0, -1):
            if option_type == "call":
                payoff = np.maximum(paths[:, step] - k, 0.0)
            else:
                payoff = np.maximum(k - paths[:, step], 0.0)

            itm = payoff > 0
            if not np.any(itm):
                continue

            X = paths[itm, step]
            Y = cash_flows[itm] * np.exp(-r * dt_actual * (n_steps - step))

            # Polynomial regression (degree 2)
            A = np.vstack([np.ones_like(X), X, X**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2

            exercise = payoff[itm] > continuation
            indices = np.where(itm)[0][exercise]
            cash_flows[indices] = payoff[indices]

        price = float(np.mean(cash_flows * np.exp(-r * dt_actual * n_steps)))
        std_error = float(np.std(cash_flows * np.exp(-r * dt_actual * n_steps), ddof=1) / math.sqrt(n_paths))
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return SimulationResult(
            price=price,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            n_paths=n_paths,
        )
