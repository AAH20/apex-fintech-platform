"""Commodities analytics engine.

Implements cost-of-carry pricing, futures curve construction,
roll yield analysis, and contango/backwardation detection.

References:
    CFA Institute, "Commodities" (Reading 48).
    Hull, J. "Options, Futures, and Other Derivatives", 10th ed.
"""

from __future__ import annotations

import numpy as np


class CommoditiesEngine:
    """Commodity futures pricing and curve analysis engine.

    Cost-of-carry model:
        F = S * exp((r + u - y) * T)

    where:
        F = futures price
        S = spot price
        r = risk-free rate
        u = storage cost (proportional)
        y = convenience yield
        T = time to maturity (years)
    """

    def __init__(
        self,
        spot_price: float,
        risk_free_rate: float = 0.05,
        storage_cost: float = 0.02,
        convenience_yield: float = 0.0,
    ) -> None:
        """Initialize the commodities engine.

        Args:
            spot_price: Current spot price of the commodity (must be > 0).
            risk_free_rate: Annualized risk-free rate (default 5%).
            storage_cost: Annualized proportional storage cost (default 2%).
            convenience_yield: Annualized convenience yield (default 0%).

        Raises:
            ValueError: If any parameter is invalid.
        """
        if spot_price <= 0:
            raise ValueError("spot_price must be positive")
        if risk_free_rate < 0:
            raise ValueError("risk_free_rate must be non-negative")
        if storage_cost < 0:
            raise ValueError("storage_cost must be non-negative")
        if convenience_yield < 0:
            raise ValueError("convenience_yield must be non-negative")

        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.storage_cost = storage_cost
        self.convenience_yield = convenience_yield

    # ------------------------------------------------------------------
    # Futures pricing
    # ------------------------------------------------------------------

    def futures_price(self, time_to_maturity: float) -> float:
        """Compute the theoretical futures price via cost-of-carry.

        F = S * exp((r + u - y) * T)

        Args:
            time_to_maturity: Time to maturity in years (must be >= 0).

        Returns:
            Theoretical futures price.

        Raises:
            ValueError: If time_to_maturity is negative.
        """
        if time_to_maturity < 0:
            raise ValueError("time_to_maturity must be non-negative")
        carry = self.risk_free_rate + self.storage_cost - self.convenience_yield
        return self.spot_price * np.exp(carry * time_to_maturity)

    def futures_curve(self, maturities: np.ndarray) -> np.ndarray:
        """Construct the futures curve for given maturities.

        Args:
            maturities: Array of times to maturity in years.

        Returns:
            Array of futures prices corresponding to each maturity.
        """
        maturities = np.asarray(maturities, dtype=float)
        carry = self.risk_free_rate + self.storage_cost - self.convenience_yield
        return self.spot_price * np.exp(carry * maturities)

    # ------------------------------------------------------------------
    # Roll yield
    # ------------------------------------------------------------------

    def roll_yield(self, near_price: float, far_price: float) -> float:
        """Compute the roll yield between two contract months.

        Roll yield = (far_price - near_price) / near_price

        Positive in contango (far > near), negative in backwardation.

        Args:
            near_price: Price of the near-month contract.
            far_price: Price of the far-month contract.

        Returns:
            Roll yield as a decimal (e.g., 0.02 = 2%).
        """
        return (far_price - near_price) / near_price

    def annualized_roll_yield(
        self, near_price: float, far_price: float, time_between: float
    ) -> float:
        """Compute the annualized roll yield.

        Annualized = (1 + roll_yield)^(1/T) - 1

        Args:
            near_price: Price of the near-month contract.
            far_price: Price of the far-month contract.
            time_between: Time between contracts in years (must be > 0).

        Returns:
            Annualized roll yield as a decimal.

        Raises:
            ValueError: If time_between is not positive.
        """
        if time_between <= 0:
            raise ValueError("time_between must be positive")
        ry = self.roll_yield(near_price, far_price)
        return (1.0 + ry) ** (1.0 / time_between) - 1.0

    def roll_yield_series(
        self, near_prices: np.ndarray, far_prices: np.ndarray
    ) -> np.ndarray:
        """Compute roll yields for a series of contract pairs.

        Args:
            near_prices: Array of near-month prices.
            far_prices: Array of far-month prices (same length).

        Returns:
            Array of roll yields, one per pair.

        Raises:
            ValueError: If arrays have different lengths.
        """
        near_prices = np.asarray(near_prices, dtype=float)
        far_prices = np.asarray(far_prices, dtype=float)
        if len(near_prices) != len(far_prices):
            raise ValueError("near_prices and far_prices must have same length")
        return (far_prices - near_prices) / near_prices

    # ------------------------------------------------------------------
    # Contango / Backwardation
    # ------------------------------------------------------------------

    def is_contango(self, maturities: np.ndarray, curve: np.ndarray) -> bool:
        """Detect contango: futures curve slopes upward.

        Args:
            maturities: Array of maturities.
            curve: Array of futures prices at those maturities.

        Returns:
            True if the curve is in contango (strictly increasing).
        """
        maturities = np.asarray(maturities, dtype=float)
        curve = np.asarray(curve, dtype=float)
        if len(maturities) < 2:
            return False
        return bool(np.all(np.diff(curve) > 0))

    def is_backwardation(self, maturities: np.ndarray, curve: np.ndarray) -> bool:
        """Detect backwardation: futures curve slopes downward.

        Args:
            maturities: Array of maturities.
            curve: Array of futures prices at those maturities.

        Returns:
            True if the curve is in backwardation (strictly decreasing).
        """
        maturities = np.asarray(maturities, dtype=float)
        curve = np.asarray(curve, dtype=float)
        if len(maturities) < 2:
            return False
        return bool(np.all(np.diff(curve) < 0))

    def curve_slope(self, maturities: np.ndarray, curve: np.ndarray) -> float:
        """Compute the slope of the futures curve via linear regression.

        Args:
            maturities: Array of maturities.
            curve: Array of futures prices.

        Returns:
            Slope of the best-fit line (price change per year).
        """
        maturities = np.asarray(maturities, dtype=float)
        curve = np.asarray(curve, dtype=float)
        if len(maturities) < 2:
            return 0.0
        # Linear regression: slope = cov(T, F) / var(T)
        slope, _ = np.polyfit(maturities, curve, 1)
        return float(slope)

    # ------------------------------------------------------------------
    # Basis analysis
    # ------------------------------------------------------------------

    def basis(self, futures_price: float) -> float:
        """Compute the basis: Futures - Spot.

        Args:
            futures_price: Current futures price.

        Returns:
            Basis (positive in contango, negative in backwardation).
        """
        return futures_price - self.spot_price

    def annualized_basis(self, futures_price: float, time_to_maturity: float) -> float:
        """Compute the annualized basis.

        Annualized basis = (F - S) / (S * T)

        Args:
            futures_price: Current futures price.
            time_to_maturity: Time to maturity in years.

        Returns:
            Annualized basis as a decimal.
        """
        return (futures_price - self.spot_price) / (self.spot_price * time_to_maturity)

    # ------------------------------------------------------------------
    # Implied convenience yield
    # ------------------------------------------------------------------

    def implied_convenience_yield(
        self, market_futures_price: float, time_to_maturity: float
    ) -> float:
        """Back out the implied convenience yield from an observed futures price.

        From F = S * exp((r + u - y) * T):
            y = r + u - ln(F/S) / T

        Args:
            market_futures_price: Observed market futures price.
            time_to_maturity: Time to maturity in years (must be > 0).

        Returns:
            Implied convenience yield as a decimal.

        Raises:
            ValueError: If time_to_maturity is not positive.
        """
        if time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be positive")
        return (
            self.risk_free_rate
            + self.storage_cost
            - np.log(market_futures_price / self.spot_price) / time_to_maturity
        )

    # ------------------------------------------------------------------
    # Curve interpolation
    # ------------------------------------------------------------------

    def interpolate_curve(
        self, maturities: np.ndarray, curve: np.ndarray, target_T: float
    ) -> float:
        """Linearly interpolate the futures curve at a target maturity.

        Args:
            maturities: Array of known maturities (sorted).
            curve: Array of known futures prices.
            target_T: Target maturity to interpolate at.

        Returns:
            Interpolated futures price at target_T.

        Raises:
            ValueError: If target_T is outside the range of maturities.
        """
        maturities = np.asarray(maturities, dtype=float)
        curve = np.asarray(curve, dtype=float)
        if len(maturities) == 0:
            raise ValueError("maturities array is empty")
        if target_T < maturities[0] or target_T > maturities[-1]:
            raise ValueError(
                f"target maturity {target_T} out of range "
                f"[{maturities[0]}, {maturities[-1]}]"
            )
        return float(np.interp(target_T, maturities, curve))

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def curve_summary(self, maturities: np.ndarray, curve: np.ndarray) -> dict:
        """Generate a summary of the futures curve.

        Args:
            maturities: Array of maturities.
            curve: Array of futures prices.

        Returns:
            Dictionary with keys: slope, is_contango, is_backwardation,
            min_price, max_price.
        """
        maturities = np.asarray(maturities, dtype=float)
        curve = np.asarray(curve, dtype=float)
        return {
            "slope": self.curve_slope(maturities, curve),
            "is_contango": self.is_contango(maturities, curve),
            "is_backwardation": self.is_backwardation(maturities, curve),
            "min_price": float(np.min(curve)) if len(curve) > 0 else None,
            "max_price": float(np.max(curve)) if len(curve) > 0 else None,
        }
