"""Derivatives Pricing Engine — Black-Scholes, Monte Carlo, Greeks, exotics."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


class DerivativesPricingEngine:
    """High-performance derivatives pricing engine.

    Supports:
    - Black-Scholes closed-form pricing
    - Monte Carlo simulation (with optional dividend yield)
    - Greeks (delta, gamma, vega, theta, rho)
    - Implied volatility
    - Exotic options (Asian, barrier, digital)
    """

    def __init__(self):
        self._price_count = 0
        self._total_time_ms = 0.0

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
        if S <= 0:
            raise ValueError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Time to maturity must be positive, got {T}")
        if sigma < 0:
            raise ValueError(f"Volatility must be non-negative, got {sigma}")

    @staticmethod
    def _validate_option_type(option_type: str) -> None:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    # ------------------------------------------------------------------
    # Black-Scholes closed-form
    # ------------------------------------------------------------------
    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """Black-Scholes closed-form price for European options.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield (default 0)

        Returns:
            Option price
        """
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        if sigma == 0:
            forward = S * np.exp((r - q) * T)
            discount = np.exp(-r * T)
            if option_type == "call":
                return max(forward - K, 0.0) * discount
            else:
                return max(K - forward, 0.0) * discount

        d1, d2 = self._d1_d2(S, K, T, r, sigma, q)

        if option_type == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return float(price)

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------
    def delta(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType, q: float = 0.0) -> float:
        """Delta: dPrice/dS."""
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        if sigma == 0:
            forward = S * np.exp((r - q) * T)
            if option_type == "call":
                return float(np.exp(-q * T)) if forward > K else 0.0
            else:
                return float(-np.exp(-q * T)) if forward < K else 0.0

        d1, _ = self._d1_d2(S, K, T, r, sigma, q)
        if option_type == "call":
            return float(np.exp(-q * T) * norm.cdf(d1))
        else:
            return float(np.exp(-q * T) * (norm.cdf(d1) - 1.0))

    def gamma(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType | None = None, q: float = 0.0) -> float:
        """Gamma: d²Price/dS² (same for call and put)."""
        self._validate_inputs(S, K, T, r, sigma)

        if sigma == 0:
            return 0.0

        d1, _ = self._d1_d2(S, K, T, r, sigma, q)
        return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    def vega(self, S: float, K: float, T: float, r: float, sigma: float,
             option_type: OptionType | None = None, q: float = 0.0) -> float:
        """Vega: dPrice/dSigma (same for call and put)."""
        self._validate_inputs(S, K, T, r, sigma)

        if sigma == 0:
            return 0.0

        d1, _ = self._d1_d2(S, K, T, r, sigma, q)
        return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))

    def theta(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType, q: float = 0.0) -> float:
        """Theta: dPrice/dT (per year)."""
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        if sigma == 0:
            return 0.0

        d1, d2 = self._d1_d2(S, K, T, r, sigma, q)
        n_d1 = norm.pdf(d1)

        term1 = -S * np.exp(-q * T) * n_d1 * sigma / (2.0 * np.sqrt(T))

        if option_type == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)

        return float(term1 + term2 + term3)

    def rho(self, S: float, K: float, T: float, r: float, sigma: float,
            option_type: OptionType, q: float = 0.0) -> float:
        """Rho: dPrice/dr."""
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        if sigma == 0:
            forward = S * np.exp((r - q) * T)
            if option_type == "call":
                return float(-T * max(forward - K, 0.0) * np.exp(-r * T))
            else:
                return float(-T * max(K - forward, 0.0) * np.exp(-r * T))

        _, d2 = self._d1_d2(S, K, T, r, sigma, q)
        if option_type == "call":
            return float(K * T * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))

    def all_greeks(self, S: float, K: float, T: float, r: float, sigma: float,
                   option_type: OptionType, q: float = 0.0) -> dict[str, float]:
        """Compute all Greeks at once."""
        return {
            "delta": self.delta(S, K, T, r, sigma, option_type, q),
            "gamma": self.gamma(S, K, T, r, sigma, q=q),
            "vega": self.vega(S, K, T, r, sigma, q=q),
            "theta": self.theta(S, K, T, r, sigma, option_type, q),
            "rho": self.rho(S, K, T, r, sigma, option_type, q),
        }

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        n_paths: int = 100_000,
        n_steps: int = 1,
        q: float = 0.0,
        seed: int | None = None,
        return_std_err: bool = False,
    ) -> float | tuple[float, float]:
        """Monte Carlo price for European options.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path
            q: Dividend yield
            seed: Random seed for reproducibility
            return_std_err: If True, return (price, std_err)

        Returns:
            Price, or (price, std_err) if return_std_err=True
        """
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        rng = np.random.default_rng(seed)
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol_sqrt_dt = sigma * np.sqrt(dt)

        # Simulate paths
        log_S = np.full(n_paths, np.log(S))
        for _ in range(n_steps):
            Z = rng.standard_normal(n_paths)
            log_S += drift + vol_sqrt_dt * Z

        S_T = np.exp(log_S)

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        discount = np.exp(-r * T)
        price = float(discount * np.mean(payoffs))

        if return_std_err:
            std_err = float(discount * np.std(payoffs, ddof=1) / np.sqrt(n_paths))
            return price, std_err
        return price

    # ------------------------------------------------------------------
    # Implied Volatility
    # ------------------------------------------------------------------
    def implied_volatility(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        q: float = 0.0,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """Compute implied volatility using Newton-Raphson with bisection fallback.

        Args:
            price: Observed option price
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            q: Dividend yield
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Implied volatility
        """
        self._validate_inputs(S, K, T, r, 0.2)  # dummy sigma for validation
        self._validate_option_type(option_type)

        # Check price bounds
        forward = S * np.exp((r - q) * T)
        discount = np.exp(-r * T)
        if option_type == "call":
            intrinsic = max(forward - K, 0.0) * discount
            upper_bound = S * np.exp(-q * T)
        else:
            intrinsic = max(K - forward, 0.0) * discount
            upper_bound = K * np.exp(-r * T)

        if price < intrinsic - 1e-12:
            raise ValueError(f"Price {price} below intrinsic value {intrinsic}")
        if price > upper_bound + 1e-12:
            raise ValueError(f"Price {price} above upper bound {upper_bound}")

        # Newton-Raphson
        sigma = 0.2  # initial guess
        for _ in range(max_iter):
            try:
                price_est = self.black_scholes_price(S, K, T, r, sigma, option_type, q)
            except ValueError:
                break

            diff = price_est - price
            if abs(diff) < tol:
                return float(sigma)

            vega = self.vega(S, K, T, r, sigma, q=q)
            if vega < 1e-12:
                break

            sigma_new = sigma - diff / vega
            if sigma_new <= 0 or sigma_new > 10:
                break
            sigma = sigma_new

        # Bisection fallback
        lo, hi = 1e-6, 10.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            try:
                price_est = self.black_scholes_price(S, K, T, r, mid, option_type, q)
            except ValueError:
                lo = mid
                continue

            if abs(price_est - price) < tol:
                return float(mid)

            if price_est > price:
                hi = mid
            else:
                lo = mid

        return float((lo + hi) / 2.0)

    # ------------------------------------------------------------------
    # Exotic Options
    # ------------------------------------------------------------------
    def asian_option_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        n_paths: int = 50_000,
        n_steps: int = 50,
        q: float = 0.0,
        seed: int | None = None,
    ) -> float:
        """Asian option price (arithmetic average) via Monte Carlo.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            q: Dividend yield
            seed: Random seed

        Returns:
            Asian option price
        """
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        rng = np.random.default_rng(seed)
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol_sqrt_dt = sigma * np.sqrt(dt)

        # Simulate paths and compute arithmetic average
        log_S = np.full(n_paths, np.log(S))
        S_sum = np.zeros(n_paths)

        for _ in range(n_steps):
            Z = rng.standard_normal(n_paths)
            log_S += drift + vol_sqrt_dt * Z
            S_sum += np.exp(log_S)

        S_avg = S_sum / n_steps

        if option_type == "call":
            payoffs = np.maximum(S_avg - K, 0.0)
        else:
            payoffs = np.maximum(K - S_avg, 0.0)

        discount = np.exp(-r * T)
        return float(discount * np.mean(payoffs))

    def barrier_option_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        barrier: float,
        barrier_type: str,
        n_paths: int = 50_000,
        n_steps: int = 50,
        q: float = 0.0,
        seed: int | None = None,
    ) -> float:
        """Barrier option price via Monte Carlo.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            barrier: Barrier level
            barrier_type: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            q: Dividend yield
            seed: Random seed

        Returns:
            Barrier option price
        """
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        valid_types = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")
        if barrier_type not in valid_types:
            raise ValueError(f"barrier_type must be one of {valid_types}")

        rng = np.random.default_rng(seed)
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol_sqrt_dt = sigma * np.sqrt(dt)

        log_S = np.full(n_paths, np.log(S))
        if barrier_type.startswith("up"):
            hit_barrier = np.zeros(n_paths, dtype=bool)
        else:
            hit_barrier = np.zeros(n_paths, dtype=bool)

        for _ in range(n_steps):
            Z = rng.standard_normal(n_paths)
            log_S += drift + vol_sqrt_dt * Z
            S_current = np.exp(log_S)

            if barrier_type.startswith("up"):
                hit_barrier |= S_current >= barrier
            else:
                hit_barrier |= S_current <= barrier

        S_T = np.exp(log_S)

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        if "out" in barrier_type:
            payoffs = np.where(hit_barrier, 0.0, payoffs)
        else:
            payoffs = np.where(hit_barrier, payoffs, 0.0)

        discount = np.exp(-r * T)
        return float(discount * np.mean(payoffs))

    def digital_option_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """Digital (binary) option price — closed form.

        Pays 1 if S_T > K (call) or S_T < K (put) at expiry.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield

        Returns:
            Digital option price
        """
        self._validate_inputs(S, K, T, r, sigma)
        self._validate_option_type(option_type)

        if sigma == 0:
            forward = S * np.exp((r - q) * T)
            if option_type == "call":
                return float(np.exp(-r * T)) if forward > K else 0.0
            else:
                return float(np.exp(-r * T)) if forward < K else 0.0

        _, d2 = self._d1_d2(S, K, T, r, sigma, q)

        if option_type == "call":
            return float(np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(np.exp(-r * T) * norm.cdf(-d2))

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------
    @property
    def average_pricing_time_ms(self) -> float:
        """Average pricing time in milliseconds."""
        if self._price_count == 0:
            return 0.0
        return self._total_time_ms / self._price_count

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._price_count = 0
        self._total_time_ms = 0.0
