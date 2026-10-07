"""Trading analytics engine — market impact, liquidity analysis, and TCA.

Implements standard models from market microstructure literature:
- Square-root impact (Almgren et al. 2005, Frazzini et al. 2018)
- Linear impact (Kyle 1985)
- Temporary/permanent impact decomposition (Obizhaeva & Wang 2013)
- Spread metrics and order book imbalance
- Amihud illiquidity (Amihud 2002)
- Kyle's lambda (Kyle 1985)
- VWAP slippage and implementation shortfall (Perold 1988)
- Full transaction cost analysis (TCA)

References:
    Almgren, R. et al. (2005). Direct estimation of equity market impact.
    Journal of Risk, 8(3), 5-24.
    Amihud, Y. (2002). Illiquidity and stock returns. JFM, 5(3), 31-56.
    Frazzini, A. et al. (2018). Trading costs. SSRN 2994029.
    Kyle, A.S. (1985). Continuous auctions and insider trading. Econometrica.
    Obizhaeva, A. & Wang, J. (2013). Optimal trading strategy and supply/demand dynamics.
    Perold, A. (1988). The implementation shortfall. JPM, 6(3), 3-9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImpactDecomposition:
    """Temporary/permanent impact decomposition."""

    temporary: float
    permanent: float
    total: float


@dataclass
class SpreadMetrics:
    """Bid-ask spread metrics."""

    spread: float
    spread_bps: float


@dataclass
class TotalCostResult:
    """Full transaction cost analysis result."""

    implementation_shortfall_bps: float
    commission_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    total_cost_bps: float
    total_cost: float


class TradingAnalyticsEngine:
    """Trading analytics engine for market impact, liquidity, and TCA.

    Provides methods for:
    - Market impact estimation (sqrt and linear models)
    - Temporary/permanent impact decomposition
    - Spread and order book imbalance analysis
    - Amihud illiquidity and Kyle's lambda
    - VWAP slippage and implementation shortfall
    - Full transaction cost analysis (TCA)
    """

    def __init__(
        self,
        sqrt_coefficient: float = 0.5,
        linear_coefficient: float = 0.1,
    ) -> None:
        """Initialize the trading analytics engine.

        Args:
            sqrt_coefficient: Y in the sqrt impact model (default 0.5).
            linear_coefficient: Coefficient for linear impact model (default 0.1).
        """
        if sqrt_coefficient <= 0:
            raise ValueError("sqrt_coefficient must be positive")
        if linear_coefficient <= 0:
            raise ValueError("linear_coefficient must be positive")

        self.sqrt_coefficient = sqrt_coefficient
        self.linear_coefficient = linear_coefficient

    # ------------------------------------------------------------------
    # Market impact models
    # ------------------------------------------------------------------

    def sqrt_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        """Square-root market impact model.

        Impact = Y * sigma * sqrt(Q / V)

        Args:
            quantity: Order quantity (shares).
            daily_volume: Average daily volume (shares).
            volatility: Daily volatility (decimal, e.g. 0.02 for 2%).

        Returns:
            Estimated price impact as a decimal fraction.
        """
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        if daily_volume <= 0:
            raise ValueError("daily_volume must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        return self.sqrt_coefficient * volatility * np.sqrt(quantity / daily_volume)

    def linear_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        """Linear market impact model.

        Impact = coefficient * sigma * (Q / V)

        Args:
            quantity: Order quantity (shares).
            daily_volume: Average daily volume (shares).
            volatility: Daily volatility (decimal).

        Returns:
            Estimated price impact as a decimal fraction.
        """
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        if daily_volume <= 0:
            raise ValueError("daily_volume must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        return self.linear_coefficient * volatility * (quantity / daily_volume)

    def decompose_impact(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        decay: float,
    ) -> ImpactDecomposition:
        """Decompose impact into temporary and permanent components.

        Temporary impact decays with rate `decay`; permanent impact persists.

        Args:
            quantity: Order quantity (shares).
            daily_volume: Average daily volume (shares).
            volatility: Daily volatility (decimal).
            decay: Fraction of impact that is temporary (0 to 1).

        Returns:
            ImpactDecomposition with temporary, permanent, and total.
        """
        if not 0 <= decay <= 1:
            raise ValueError("decay must be between 0 and 1")

        total = self.sqrt_impact(quantity, daily_volume, volatility)
        temporary = total * decay
        permanent = total * (1.0 - decay)

        return ImpactDecomposition(
            temporary=temporary,
            permanent=permanent,
            total=total,
        )

    # ------------------------------------------------------------------
    # Liquidity analysis
    # ------------------------------------------------------------------

    def spread_metrics(self, bid: float, ask: float, mid: float) -> SpreadMetrics:
        """Compute bid-ask spread metrics.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            mid: Mid price.

        Returns:
            SpreadMetrics with absolute spread and spread in bps.
        """
        if bid <= 0 or ask <= 0 or mid <= 0:
            raise ValueError("prices must be positive")
        if bid > ask:
            raise ValueError("bid must be <= ask")

        spread = ask - bid
        spread_bps = (spread / mid) * 10_000

        return SpreadMetrics(spread=spread, spread_bps=spread_bps)

    def book_imbalance(self, bid_vols: np.ndarray, ask_vols: np.ndarray) -> float:
        """Compute order book imbalance.

        Imbalance = (sum(bid_vols) - sum(ask_vols)) / (sum(bid_vols) + sum(ask_vols))

        Args:
            bid_vols: Array of bid volumes at each level.
            ask_vols: Array of ask volumes at each level.

        Returns:
            Imbalance in [-1, 1]. Positive = more bid volume.
        """
        bid_vols = np.asarray(bid_vols)
        ask_vols = np.asarray(ask_vols)

        if bid_vols.shape != ask_vols.shape:
            raise ValueError("bid_vols and ask_vols must have the same shape")

        total_bid = np.sum(bid_vols)
        total_ask = np.sum(ask_vols)
        total = total_bid + total_ask

        if total == 0:
            return 0.0

        return float((total_bid - total_ask) / total)

    def amihud_illiquidity(self, returns: np.ndarray, volumes: np.ndarray) -> float:
        """Amihud illiquidity measure.

        Amihud = mean(|return| / volume)

        Args:
            returns: Array of returns (decimal).
            volumes: Array of volumes (shares).

        Returns:
            Amihud illiquidity (higher = more illiquid).
        """
        returns = np.asarray(returns)
        volumes = np.asarray(volumes)

        if returns.shape != volumes.shape:
            raise ValueError("returns and volumes must have the same shape")
        if len(returns) == 0:
            raise ValueError("returns and volumes must not be empty")
        if np.any(volumes <= 0):
            raise ValueError("volumes must be positive")

        return float(np.mean(np.abs(returns) / volumes))

    def kyle_lambda(self, price_changes: np.ndarray, order_flow: np.ndarray) -> float:
        """Estimate Kyle's lambda via OLS regression.

        delta_price = lambda * order_flow + epsilon

        Args:
            price_changes: Array of price changes.
            order_flow: Array of signed order flow (shares).

        Returns:
            Kyle's lambda (price impact per unit of order flow).
        """
        price_changes = np.asarray(price_changes)
        order_flow = np.asarray(order_flow)

        if price_changes.shape != order_flow.shape:
            raise ValueError("price_changes and order_flow must have the same shape")
        if len(price_changes) < 2:
            raise ValueError("at least 2 observations required")

        # OLS: lambda = cov(x, y) / var(x)
        x = order_flow
        y = price_changes
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    # ------------------------------------------------------------------
    # Transaction cost analysis
    # ------------------------------------------------------------------

    def vwap_slippage(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        arrival_price: float,
        side: str,
    ) -> float:
        """Compute VWAP slippage in bps.

        Buy: slippage = (VWAP - arrival) / arrival * 10000
        Sell: slippage = (arrival - VWAP) / arrival * 10000

        Args:
            prices: Array of execution prices.
            volumes: Array of execution volumes.
            arrival_price: Arrival (decision) price.
            side: "buy" or "sell".

        Returns:
            VWAP slippage in basis points.
        """
        prices = np.asarray(prices)
        volumes = np.asarray(volumes)

        if prices.shape != volumes.shape:
            raise ValueError("prices and volumes must have the same shape")
        if len(prices) == 0:
            raise ValueError("prices and volumes must not be empty")
        if arrival_price <= 0:
            raise ValueError("arrival_price must be positive")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        vwap = np.sum(prices * volumes) / np.sum(volumes)

        if side == "buy":
            return float((vwap - arrival_price) / arrival_price * 10_000)
        else:
            return float((arrival_price - vwap) / arrival_price * 10_000)

    def implementation_shortfall(
        self,
        arrival_price: float,
        execution_prices: np.ndarray,
        quantities: np.ndarray,
        side: str,
    ) -> float:
        """Compute implementation shortfall in bps.

        Buy: IS = (avg_exec - arrival) / arrival * 10000
        Sell: IS = (arrival - avg_exec) / arrival * 10000

        Args:
            arrival_price: Arrival (decision) price.
            execution_prices: Array of execution prices.
            quantities: Array of execution quantities.
            side: "buy" or "sell".

        Returns:
            Implementation shortfall in basis points.
        """
        execution_prices = np.asarray(execution_prices)
        quantities = np.asarray(quantities)

        if execution_prices.shape != quantities.shape:
            raise ValueError("execution_prices and quantities must have the same shape")
        if len(execution_prices) == 0:
            raise ValueError("execution_prices and quantities must not be empty")
        if arrival_price <= 0:
            raise ValueError("arrival_price must be positive")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        avg_exec = np.sum(execution_prices * quantities) / np.sum(quantities)

        if side == "buy":
            return float((avg_exec - arrival_price) / arrival_price * 10_000)
        else:
            return float((arrival_price - avg_exec) / arrival_price * 10_000)

    def total_cost(
        self,
        arrival_price: float,
        execution_prices: np.ndarray,
        quantities: np.ndarray,
        commission: float,
        side: str,
        daily_volume: float | None = None,
        volatility: float | None = None,
    ) -> TotalCostResult:
        """Full transaction cost analysis.

        Decomposes total cost into:
        - Implementation shortfall (IS)
        - Commission
        - Market impact (estimated via sqrt model if volume/vol provided)
        - Timing cost (IS - market impact)

        Args:
            arrival_price: Arrival (decision) price.
            execution_prices: Array of execution prices.
            quantities: Array of execution quantities.
            commission: Total commission in currency units.
            side: "buy" or "sell".
            daily_volume: Average daily volume (for market impact estimation).
            volatility: Daily volatility (for market impact estimation).

        Returns:
            TotalCostResult with all cost components.
        """
        total_qty = float(np.sum(quantities))
        notional = arrival_price * total_qty

        is_bps = self.implementation_shortfall(
            arrival_price, execution_prices, quantities, side
        )
        commission_bps = (commission / notional) * 10_000 if notional > 0 else 0.0

        if daily_volume is not None and volatility is not None:
            market_impact_bps = self.expected_impact_bps(
                quantity=total_qty,
                daily_volume=daily_volume,
                volatility=volatility,
            )
            timing_cost_bps = is_bps - market_impact_bps
        else:
            market_impact_bps = 0.0
            timing_cost_bps = is_bps

        total_cost_bps = is_bps + commission_bps
        total_cost = (total_cost_bps / 10_000) * notional

        return TotalCostResult(
            implementation_shortfall_bps=is_bps,
            commission_bps=commission_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            total_cost_bps=total_cost_bps,
            total_cost=total_cost,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def participation_rate(self, quantity: float, daily_volume: float) -> float:
        """Compute participation rate.

        Args:
            quantity: Order quantity (shares).
            daily_volume: Average daily volume (shares).

        Returns:
            Participation rate (Q / V).
        """
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        if daily_volume <= 0:
            raise ValueError("daily_volume must be positive")

        return quantity / daily_volume

    def expected_impact_bps(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        """Expected market impact in basis points.

        Args:
            quantity: Order quantity (shares).
            daily_volume: Average daily volume (shares).
            volatility: Daily volatility (decimal).

        Returns:
            Expected impact in basis points.
        """
        return self.sqrt_impact(quantity, daily_volume, volatility) * 10_000
