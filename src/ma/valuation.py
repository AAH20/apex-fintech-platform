"""M&A valuation engine.

Implements three core valuation methods used in M&A:
    1. Discounted Cash Flow (DCF) — intrinsic value based on projected FCFs
    2. Comparable Company Analysis (Comps) — relative value from peer multiples
    3. Precedent Transactions — relative value from historical M&A deal multiples

References:
    CFA Institute. (2023). Corporate Finance: Capital Budgeting and Valuation.
    Damodaran, A. (2012). Investment Valuation: Tools and Techniques.
    Rosenbaum, J. & Pearl, J. (2013). Investment Banking: Valuation, LBOs, M&A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DCFResult:
    """Result of a DCF valuation."""

    enterprise_value: float
    equity_value: float
    pv_of_fcfs: float
    pv_of_terminal_value: float
    terminal_value: float
    implied_ev_ebitda: float | None = None


@dataclass
class CompsResult:
    """Result of comparable company analysis."""

    low_value: float
    mid_value: float
    high_value: float
    median_multiple: float
    metric_name: str = "EBITDA"


@dataclass
class PrecedentResult:
    """Result of precedent transaction analysis."""

    low_value: float
    mid_value: float
    high_value: float
    median_multiple: float
    control_premium: float


class ValuationEngine:
    """M&A valuation engine supporting DCF, comps, and precedent transactions."""

    # -----------------------------------------------------------------------
    # DCF
    # -----------------------------------------------------------------------

    def dcf_valuation(
        self,
        free_cash_flows: list[float],
        wacc: float,
        terminal_growth: float,
        net_debt: float = 0.0,
        ebitda: float | None = None,
    ) -> DCFResult:
        """Discounted Cash Flow valuation.

        Args:
            free_cash_flows: Projected unlevered free cash flows.
            wacc: Weighted average cost of capital (discount rate).
            terminal_growth: Perpetuity growth rate for terminal value.
            net_debt: Total debt minus cash (for equity value bridge).
            ebitda: Current EBITDA (for implied multiple calculation).

        Returns:
            DCFResult with enterprise value, equity value, and components.

        Raises:
            ValueError: If terminal_growth >= wacc or inputs are invalid.
        """
        if not free_cash_flows:
            raise ValueError("free_cash_flows must not be empty")
        if wacc <= 0:
            raise ValueError("wacc must be positive")
        if terminal_growth < 0:
            raise ValueError("terminal_growth must be non-negative")
        if terminal_growth >= wacc:
            raise ValueError("terminal_growth must be less than wacc")
        if net_debt < 0:
            raise ValueError("net_debt must be non-negative")

        fcfs = np.asarray(free_cash_flows, dtype=float)
        periods = np.arange(1, len(fcfs) + 1)

        # PV of explicit forecast period
        discount_factors = (1.0 + wacc) ** periods
        pv_of_fcfs = float(np.sum(fcfs / discount_factors))

        # Gordon Growth terminal value
        terminal_value = float(fcfs[-1] * (1.0 + terminal_growth) / (wacc - terminal_growth))
        pv_of_terminal = float(terminal_value / discount_factors[-1])

        enterprise_value = pv_of_fcfs + pv_of_terminal
        equity_value = enterprise_value - net_debt

        implied_ev_ebitda = None
        if ebitda is not None and ebitda > 0:
            implied_ev_ebitda = enterprise_value / ebitda

        return DCFResult(
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            pv_of_fcfs=pv_of_fcfs,
            pv_of_terminal_value=pv_of_terminal,
            terminal_value=terminal_value,
            implied_ev_ebitda=implied_ev_ebitda,
        )

    # -----------------------------------------------------------------------
    # Comparable Company Analysis
    # -----------------------------------------------------------------------

    def comps_valuation(
        self,
        metric: float,
        peer_multiples: list[float],
        metric_name: str = "EBITDA",
    ) -> CompsResult:
        """Comparable company analysis valuation.

        Args:
            metric: Target company's financial metric (e.g., EBITDA, Revenue).
            peer_multiples: Trading multiples from comparable companies.
            metric_name: Name of the metric for labeling.

        Returns:
            CompsResult with low, mid, and high valuation estimates.

        Raises:
            ValueError: If fewer than 3 peer multiples provided.
        """
        if len(peer_multiples) < 3:
            raise ValueError("at least 3 peer multiples required")
        if metric <= 0:
            raise ValueError("metric must be positive")

        multiples = np.asarray(peer_multiples, dtype=float)
        p25 = float(np.percentile(multiples, 25))
        median = float(np.median(multiples))
        p75 = float(np.percentile(multiples, 75))

        return CompsResult(
            low_value=metric * p25,
            mid_value=metric * median,
            high_value=metric * p75,
            median_multiple=median,
            metric_name=metric_name,
        )

    # -----------------------------------------------------------------------
    # Precedent Transactions
    # -----------------------------------------------------------------------

    def precedent_transactions_valuation(
        self,
        metric: float,
        deal_multiples: list[float],
        control_premium: float = 0.0,
    ) -> PrecedentResult:
        """Precedent transaction analysis valuation.

        Args:
            metric: Target company's financial metric.
            deal_multiples: Multiples paid in comparable M&A transactions.
            control_premium: Control premium to apply (e.g., 0.25 for 25%).

        Returns:
            PrecedentResult with low, mid, and high valuation estimates.

        Raises:
            ValueError: If fewer than 2 deal multiples provided.
        """
        if len(deal_multiples) < 2:
            raise ValueError("at least 2 deal multiples required")
        if metric <= 0:
            raise ValueError("metric must be positive")
        if control_premium < 0:
            raise ValueError("control_premium must be non-negative")

        multiples = np.asarray(deal_multiples, dtype=float)
        p25 = float(np.percentile(multiples, 25))
        median = float(np.median(multiples))
        p75 = float(np.percentile(multiples, 75))

        # Apply control premium
        premium_factor = 1.0 + control_premium

        return PrecedentResult(
            low_value=metric * p25 * premium_factor,
            mid_value=metric * median * premium_factor,
            high_value=metric * p75 * premium_factor,
            median_multiple=median,
            control_premium=control_premium,
        )

    # -----------------------------------------------------------------------
    # Combined Valuation
    # -----------------------------------------------------------------------

    def combined_valuation(
        self,
        dcf_value: float,
        comps_mid: float,
        precedent_mid: float,
        weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ) -> float:
        """Weighted average of DCF, comps, and precedent valuations.

        Args:
            dcf_value: DCF enterprise or equity value.
            comps_mid: Comps mid-case value.
            precedent_mid: Precedent transactions mid-case value.
            weights: (dcf_weight, comps_weight, precedent_weight). Must sum to 1.

        Returns:
            Weighted average valuation.

        Raises:
            ValueError: If weights don't sum to 1.
        """
        if not np.isclose(sum(weights), 1.0, atol=1e-9):
            raise ValueError("Weights must sum to 1")
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative")

        return weights[0] * dcf_value + weights[1] * comps_mid + weights[2] * precedent_mid

    # -----------------------------------------------------------------------
    # Sensitivity Analysis
    # -----------------------------------------------------------------------

    def dcf_sensitivity(
        self,
        free_cash_flows: list[float],
        wacc_range: np.ndarray,
        terminal_growth_range: np.ndarray,
    ) -> np.ndarray:
        """DCF sensitivity analysis over WACC and terminal growth.

        Args:
            free_cash_flows: Projected unlevered free cash flows.
            wacc_range: Array of WACC values to test.
            terminal_growth_range: Array of terminal growth rates to test.

        Returns:
            2D array of enterprise values with shape (len(wacc_range), len(tg_range)).
        """
        fcfs = np.asarray(free_cash_flows, dtype=float)
        periods = np.arange(1, len(fcfs) + 1)

        grid = np.zeros((len(wacc_range), len(terminal_growth_range)))

        for i, wacc in enumerate(wacc_range):
            for j, tg in enumerate(terminal_growth_range):
                if tg >= wacc:
                    grid[i, j] = np.nan
                    continue
                discount_factors = (1.0 + wacc) ** periods
                pv_fcfs = float(np.sum(fcfs / discount_factors))
                tv = float(fcfs[-1] * (1.0 + tg) / (wacc - tg))
                pv_tv = float(tv / discount_factors[-1])
                grid[i, j] = pv_fcfs + pv_tv

        return grid
