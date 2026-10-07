"""
Private Equity Analytics Engine.

Provides tools for PE deal structuring, waterfall analysis, carried interest
calculations, and fund performance metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaterfallResult:
    """Immutable result of a waterfall distribution calculation."""
    lp_return_of_capital: float
    lp_preferred_return: float
    lp_excess_share: float
    gp_carry: float

    @property
    def lp_total(self) -> float:
        """Total amount distributed to LPs."""
        return self.lp_return_of_capital + self.lp_preferred_return + self.lp_excess_share

    @property
    def gp_total(self) -> float:
        """Total amount distributed to GP (equals carry)."""
        return self.gp_carry


class PEAnalyticsEngine:
    """Engine for private equity analytics.

    Handles waterfall distributions, carried interest, IRR/MOIC calculations,
    and fund performance metrics.
    """

    def __init__(
        self,
        committed_capital: float,
        management_fee: float = 0.02,
        carried_interest: float = 0.20,
        hurdle_rate: float = 0.08,
        catch_up: float = 1.0,
    ):
        self.committed_capital = committed_capital
        self.management_fee = management_fee
        self.carried_interest = carried_interest
        self.hurdle_rate = hurdle_rate
        self.catch_up = catch_up

    # ─── IRR / MOIC ─────────────────────────────────────────────────────────

    def compute_irr(self, cash_flows: np.ndarray) -> float:
        """Calculate Internal Rate of Return using numpy.roots.

        Args:
            cash_flows: Array of cash flows (negative = outflow, positive = inflow).
                       First element is typically the initial investment (negative).

        Returns:
            IRR as a decimal (e.g., 0.10 for 10%).

        Raises:
            ValueError: If cash flows have no sign change or fewer than 2 elements.
        """
        cash_flows = np.asarray(cash_flows, dtype=float)
        if len(cash_flows) < 2:
            raise ValueError("IRR requires at least 2 cash flows")
        if not (np.any(cash_flows > 0) and np.any(cash_flows < 0)):
            raise ValueError("IRR undefined: no sign change in cash flows")

        # NPV(r) = sum(cf_t / (1+r)^t) = 0
        # Let x = 1+r, then sum(cf_t * x^(-t)) = 0
        # Multiply by x^n: sum(cf_t * x^(n-t)) = 0
        # This is a polynomial in x: cf_0*x^n + cf_1*x^(n-1) + ... + cf_n = 0
        # numpy.roots expects [a_n, a_{n-1}, ..., a_0] for a_n*x^n + ... + a_0
        # So coeffs = [cf_0, cf_1, ..., cf_n]
        roots = np.roots(cash_flows)
        real_roots = roots[np.isreal(roots)].real
        positive_roots = real_roots[real_roots > 0]

        if len(positive_roots) == 0:
            raise ValueError("IRR undefined: no positive real root found")

        x = np.min(positive_roots)
        return float(x - 1.0)

    def compute_moic(self, total_distributions: float, invested_capital: float) -> float:
        """Calculate Multiple of Invested Capital (MOIC).

        Args:
            total_distributions: Total distributions received.
            invested_capital: Total capital invested.

        Returns:
            MOIC as a float.
        """
        if invested_capital == 0:
            return 0.0
        return total_distributions / invested_capital

    # ─── Waterfall Distribution ────────────────────────────────────────────

    def waterfall_distribution(
        self,
        total_proceeds: float,
        invested_capital: float,
        preferred_return: float = 0.0,
    ) -> WaterfallResult:
        """Calculate waterfall distribution between LPs and GP.

        Waterfall tiers:
        1. Return of capital to LPs
        2. Preferred return (hurdle) to LPs
        3. GP catch-up (GP gets 100% of remaining until GP has carry% of total profit)
        4. 80/20 split of remaining excess

        Args:
            total_proceeds: Total proceeds from exit.
            invested_capital: Total capital invested.
            preferred_return: Preferred return amount (hurdle * invested_capital).

        Returns:
            WaterfallResult with distribution breakdown.
        """
        remaining = total_proceeds

        # Tier 1: Return of capital to LPs
        lp_return_of_capital = min(remaining, invested_capital)
        remaining -= lp_return_of_capital

        # Tier 2: Preferred return to LPs
        lp_preferred_return = min(remaining, preferred_return)
        remaining -= lp_preferred_return

        # Tier 3: GP catch-up
        total_profit = max(0.0, total_proceeds - invested_capital)
        gp_target = self.carried_interest * total_profit
        gp_carry = 0.0

        if remaining > 0 and gp_target > 0:
            catch_up_amount = min(remaining, self.catch_up * (gp_target - gp_carry))
            gp_carry += catch_up_amount
            remaining -= catch_up_amount

        # Tier 4: 80/20 split of remaining
        lp_excess_share = remaining * (1 - self.carried_interest)
        gp_carry += remaining * self.carried_interest

        return WaterfallResult(
            lp_return_of_capital=lp_return_of_capital,
            lp_preferred_return=lp_preferred_return,
            lp_excess_share=lp_excess_share,
            gp_carry=gp_carry,
        )

    # ─── Carried Interest ──────────────────────────────────────────────────

    def calculate_carried_interest(
        self,
        total_proceeds: float,
        invested_capital: float,
        preferred_return: float = 0.0,
    ) -> float:
        """Calculate carried interest (GP's share of profits).

        Args:
            total_proceeds: Total proceeds from exit.
            invested_capital: Total capital invested.
            preferred_return: Preferred return amount.

        Returns:
            Carried interest amount.
        """
        result = self.waterfall_distribution(
            total_proceeds=total_proceeds,
            invested_capital=invested_capital,
            preferred_return=preferred_return,
        )
        return result.gp_carry
