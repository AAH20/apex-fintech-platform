"""Avellaneda-Stoikov optimal market making engine.

Implements the Avellaneda & Stoikov (2008) framework for optimal quote placement
with inventory risk management and adverse selection handling.

References:
    Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit
    order book. Quantitative Finance, 8(3), 217-224.
    Guéant, O. (2017). The Financial Mathematics of Market Liquidity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Quote:
    """Optimal bid/ask quotes from the market making engine."""

    bid: float
    ask: float
    mid: float
    spread: float
    reservation_price: float


class MarketMakingEngine:
    """Avellaneda-Stoikov optimal market making engine.

    Computes optimal bid/ask quotes based on:
    - Current mid price
    - Inventory position
    - Time remaining
    - Market volatility
    - Risk aversion
    - Order arrival parameters
    - Adverse selection

    The optimal quotes are:
        reservation_price = mid - inventory * gamma * sigma^2 * time_remaining
        spread = gamma * sigma^2 * time_remaining + (2/gamma) * ln(1 + gamma/k)
        bid = reservation_price - spread/2
        ask = reservation_price + spread/2
    """

    def __init__(
        self,
        gamma: float,
        sigma: float,
        k: float,
        A: float,
        dt: float,
        max_inventory: int,
        adverse_selection: float = 0.0,
    ) -> None:
        """Initialize the market making engine.

        Args:
            gamma: Risk aversion coefficient (higher = more risk averse).
            sigma: Volatility of the underlying asset.
            k: Order arrival rate decay parameter.
            A: Order arrival rate scaling parameter.
            dt: Time step for discretization.
            max_inventory: Maximum allowed inventory.
            adverse_selection: Adverse selection cost parameter.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if A <= 0:
            raise ValueError("A must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if max_inventory <= 0:
            raise ValueError("max_inventory must be positive")
        if adverse_selection < 0:
            raise ValueError("adverse_selection must be non-negative")

        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.A = A
        self.dt = dt
        self.max_inventory = max_inventory
        self.adverse_selection = adverse_selection

    def compute_quotes(
        self,
        mid_price: float,
        inventory: int,
        time_remaining: float,
    ) -> Quote:
        """Compute optimal bid and ask quotes.

        Args:
            mid_price: Current mid price of the asset.
            inventory: Current inventory position (positive = long).
            time_remaining: Time remaining until horizon (T - t).

        Returns:
            Quote object with optimal bid, ask, and metadata.
        """
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")
        if time_remaining < 0:
            raise ValueError("time_remaining must be non-negative")

        # Clamp inventory to max_inventory
        q = float(np.clip(inventory, -self.max_inventory, self.max_inventory))

        # Avellaneda-Stoikov reservation price
        # r(t) = s(t) - q * gamma * sigma^2 * (T - t)
        reservation_price = mid_price - q * self.gamma * self.sigma**2 * time_remaining

        # Avellaneda-Stoikov optimal spread
        # delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/(k*A))
        # Higher A (more aggressive arrivals) tightens the spread
        spread = self.gamma * self.sigma**2 * time_remaining + (2.0 / self.gamma) * math.log(
            1.0 + self.gamma / (self.k * self.A)
        )

        # Add adverse selection cost to spread
        if self.adverse_selection > 0:
            spread += self.adverse_selection * abs(q)

        # Optimal quotes around reservation price
        bid = reservation_price - spread / 2.0
        ask = reservation_price + spread / 2.0

        return Quote(
            bid=bid,
            ask=ask,
            mid=mid_price,
            spread=spread,
            reservation_price=reservation_price,
        )

    def compute_quotes_batch(
        self,
        mid_prices: np.ndarray,
        inventories: np.ndarray,
        times_remaining: np.ndarray,
    ) -> list[Quote]:
        """Compute optimal quotes for a batch of states.

        Args:
            mid_prices: Array of mid prices.
            inventories: Array of inventory positions.
            times_remaining: Array of time remaining values.

        Returns:
            List of Quote objects.
        """
        if not (len(mid_prices) == len(inventories) == len(times_remaining)):
            raise ValueError("All input arrays must have the same length")

        return [
            self.compute_quotes(mid, inv, t)
            for mid, inv, t in zip(mid_prices, inventories, times_remaining)
        ]

    def expected_profit(self, quote: Quote, inventory: int) -> float:
        """Estimate expected profit from current quotes.

        Args:
            quote: Current quote.
            inventory: Current inventory.

        Returns:
            Estimated expected profit.
        """
        half_spread = quote.spread / 2.0
        # Expected profit from spread minus inventory risk
        inventory_risk = self.gamma * self.sigma**2 * inventory**2
        return half_spread - inventory_risk
