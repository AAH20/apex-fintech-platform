"""Fundamental financial statement analysis engine.

Implements ratio analysis, cash flow analysis, earnings quality assessment,
and forensic anomaly detection per CFA Institute financial statement
analysis standards.

References:
    CFA Institute. (2020). Financial Statement Analysis.
    CFA Institute. (2020). Financial Reporting Quality.
    Nigrini, M. J. (2012). Benford's Law: Applications for Forensic Accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Anomaly:
    """A detected financial anomaly."""

    category: str
    metric: str
    period: str
    value: float
    threshold: float
    severity: str  # "low", "medium", "high"
    description: str


def benford_chi2(data: np.ndarray) -> tuple[float, float, dict]:
    """Compute chi-square statistic for Benford's law conformity.

    Args:
        data: Array of positive numerical values.

    Returns:
        Tuple of (chi2_statistic, p_value, details_dict).
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]

    if len(data) < 50:
        return 0.0, 1.0, {}

    # Extract first digits
    log10_data = np.log10(data)
    first_digits = np.floor(data / 10 ** np.floor(log10_data)).astype(int)
    first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]

    # Observed frequencies
    observed = np.array([np.sum(first_digits == d) for d in range(1, 10)], dtype=float)
    n = np.sum(observed)

    # Expected frequencies (Benford's law)
    expected_probs = np.log10(1 + 1 / np.arange(1, 10))
    expected = n * expected_probs

    # Chi-square statistic
    chi2 = float(np.sum((observed - expected) ** 2 / expected))

    # p-value (df = 8)
    p_value = float(stats.chi2.sf(chi2, 8))

    details = {
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "n": int(n),
    }

    return chi2, p_value, details


class FundamentalAnalysisEngine:
    """Fundamental financial statement analysis engine.

    Analyzes income statement, balance sheet, and cash flow data to compute
    financial ratios, assess earnings quality, and detect anomalies.
    """

    def __init__(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> None:
        """Initialize the engine with financial statement data.

        Args:
            income_statement: DataFrame with line items as index and periods as columns.
            balance_sheet: DataFrame with line items as index and periods as columns.
            cash_flow: DataFrame with line items as index and periods as columns.
        """
        self.income = income_statement
        self.balance = balance_sheet
        self.cashflow = cash_flow
        self.periods = list(income_statement.columns)

    @classmethod
    def from_dicts(
        cls,
        income_dict: dict,
        balance_dict: dict,
        cashflow_dict: dict,
    ) -> "FundamentalAnalysisEngine":
        """Build engine from plain nested dicts.

        Args:
            income_dict: {period: {line_item: value}}
            balance_dict: {period: {line_item: value}}
            cashflow_dict: {period: {line_item: value}}

        Returns:
            FundamentalAnalysisEngine instance.
        """
        income = pd.DataFrame.from_dict(income_dict, orient="index").T
        balance = pd.DataFrame.from_dict(balance_dict, orient="index").T
        cashflow = pd.DataFrame.from_dict(cashflow_dict, orient="index").T
        return cls(income, balance, cashflow)

    def _get(self, df: pd.DataFrame, item: str, period: str, default: float = 0.0) -> float:
        """Safely get a value from a statement DataFrame."""
        if item in df.index and period in df.columns:
            val = df.loc[item, period]
            if pd.notna(val):
                return float(val)
        return default

    def _safe_div(self, numerator: float, denominator: float) -> float:
        """Safe division returning 0.0 for zero denominator."""
        if denominator == 0:
            return 0.0
        return numerator / denominator

    # -----------------------------------------------------------------------
    # Ratio analysis
    # -----------------------------------------------------------------------

    def profitability_ratios(self) -> pd.DataFrame:
        """Compute profitability ratios.

        Returns:
            DataFrame with ratios as index and periods as columns.
        """
        ratios: dict[str, dict[str, float]] = {}
        for period in self.periods:
            revenue = self._get(self.income, "revenue", period)
            gross_profit = self._get(self.income, "gross_profit", period)
            operating_income = self._get(self.income, "operating_income", period)
            net_income = self._get(self.income, "net_income", period)
            ebitda = self._get(self.income, "ebitda", period)
            total_assets = self._get(self.balance, "total_assets", period)
            total_equity = self._get(self.balance, "total_equity", period)

            ratios.setdefault("gross_margin", {})[period] = self._safe_div(gross_profit, revenue)
            ratios.setdefault("operating_margin", {})[period] = self._safe_div(
                operating_income, revenue
            )
            ratios.setdefault("net_margin", {})[period] = self._safe_div(net_income, revenue)
            ratios.setdefault("ebitda_margin", {})[period] = self._safe_div(ebitda, revenue)
            ratios.setdefault("roa", {})[period] = self._safe_div(net_income, total_assets)
            ratios.setdefault("roe", {})[period] = self._safe_div(net_income, total_equity)

        return pd.DataFrame(ratios).T

    def liquidity_ratios(self) -> pd.DataFrame:
        """Compute liquidity ratios."""
        ratios: dict[str, dict[str, float]] = {}
        for period in self.periods:
            current_assets = self._get(self.balance, "current_assets", period)
            current_liabilities = self._get(self.balance, "current_liabilities", period)
            inventory = self._get(self.balance, "inventory", period)
            cash = self._get(self.balance, "cash", period)

            ratios.setdefault("current_ratio", {})[period] = self._safe_div(
                current_assets, current_liabilities
            )
            ratios.setdefault("quick_ratio", {})[period] = self._safe_div(
                current_assets - inventory, current_liabilities
            )
            ratios.setdefault("cash_ratio", {})[period] = self._safe_div(cash, current_liabilities)
            ratios.setdefault("working_capital", {})[period] = current_assets - current_liabilities

        return pd.DataFrame(ratios).T

    def leverage_ratios(self) -> pd.DataFrame:
        """Compute leverage ratios."""
        ratios: dict[str, dict[str, float]] = {}
        for period in self.periods:
            total_debt = self._get(self.balance, "total_debt", period)
            total_equity = self._get(self.balance, "total_equity", period)
            total_assets = self._get(self.balance, "total_assets", period)
            operating_income = self._get(self.income, "operating_income", period)
            interest_expense = self._get(self.income, "interest_expense", period)

            ratios.setdefault("debt_to_equity", {})[period] = self._safe_div(
                total_debt, total_equity
            )
            ratios.setdefault("debt_to_assets", {})[period] = self._safe_div(
                total_debt, total_assets
            )
            ratios.setdefault("interest_coverage", {})[period] = self._safe_div(
                operating_income, interest_expense
            )

        return pd.DataFrame(ratios).T

    def efficiency_ratios(self) -> pd.DataFrame:
        """Compute efficiency / activity ratios."""
        ratios: dict[str, dict[str, float]] = {}
        for period in self.periods:
            revenue = self._get(self.income, "revenue", period)
            cogs = self._get(self.income, "cogs", period)
            total_assets = self._get(self.balance, "total_assets", period)
            receivables = self._get(self.balance, "receivables", period)
            inventory = self._get(self.balance, "inventory", period)

            ratios.setdefault("asset_turnover", {})[period] = self._safe_div(
                revenue, total_assets
            )
            ratios.setdefault("dso", {})[period] = (
                self._safe_div(receivables, revenue) * 365
            )
            ratios.setdefault("inventory_turnover", {})[period] = self._safe_div(cogs, inventory)
            ratios.setdefault("receivables_turnover", {})[period] = self._safe_div(
                revenue, receivables
            )

        return pd.DataFrame(ratios).T

    # -----------------------------------------------------------------------
    # Cash flow analysis
    # -----------------------------------------------------------------------

    def cash_flow_analysis(self) -> pd.DataFrame:
        """Compute cash flow metrics."""
        metrics: dict[str, dict[str, float]] = {}
        for period in self.periods:
            ocf = self._get(self.cashflow, "operating_cash_flow", period)
            capex = self._get(self.cashflow, "capex", period)
            net_income = self._get(self.income, "net_income", period)
            revenue = self._get(self.income, "revenue", period)

            fcf = ocf - capex
            metrics.setdefault("free_cash_flow", {})[period] = fcf
            metrics.setdefault("cash_conversion", {})[period] = self._safe_div(ocf, net_income)
            metrics.setdefault("fcf_margin", {})[period] = self._safe_div(fcf, revenue)

        return pd.DataFrame(metrics).T

    def cash_flow_trends(self) -> pd.DataFrame:
        """Compute year-over-year growth trends."""
        trends: dict[str, dict[str, float]] = {}
        for i, period in enumerate(self.periods):
            if i == 0:
                trends.setdefault("operating_cash_flow_growth", {})[period] = 0.0
                trends.setdefault("revenue_growth", {})[period] = 0.0
                continue

            prev = self.periods[i - 1]
            ocf = self._get(self.cashflow, "operating_cash_flow", period)
            prev_ocf = self._get(self.cashflow, "operating_cash_flow", prev)
            revenue = self._get(self.income, "revenue", period)
            prev_revenue = self._get(self.income, "revenue", prev)

            trends.setdefault("operating_cash_flow_growth", {})[period] = self._safe_div(
                ocf - prev_ocf, prev_ocf
            )
            trends.setdefault("revenue_growth", {})[period] = self._safe_div(
                revenue - prev_revenue, prev_revenue
            )

        return pd.DataFrame(trends).T

    # -----------------------------------------------------------------------
    # Earnings quality
    # -----------------------------------------------------------------------

    def earnings_quality(self) -> pd.DataFrame:
        """Assess earnings quality via accruals and cash conversion."""
        metrics: dict[str, dict[str, float]] = {}
        for period in self.periods:
            net_income = self._get(self.income, "net_income", period)
            ocf = self._get(self.cashflow, "operating_cash_flow", period)
            total_assets = self._get(self.balance, "total_assets", period)

            accruals = net_income - ocf
            accruals_ratio = self._safe_div(accruals, total_assets)
            cash_conversion = self._safe_div(ocf, net_income)

            # Quality score: weighted combination of cash conversion and accruals
            cc_score = min(cash_conversion, 1.5) / 1.5
            accr_score = 1.0 - min(abs(accruals_ratio) / 0.20, 1.0)
            quality_score = 0.5 * cc_score + 0.5 * accr_score

            metrics.setdefault("accruals_ratio", {})[period] = accruals_ratio
            metrics.setdefault("cash_conversion", {})[period] = cash_conversion
            metrics.setdefault("quality_score", {})[period] = quality_score

        return pd.DataFrame(metrics).T

    # -----------------------------------------------------------------------
    # Anomaly detection
    # -----------------------------------------------------------------------

    def detect_anomalies(self) -> list[Anomaly]:
        """Detect financial anomalies and red flags.

        Returns:
            List of Anomaly objects.
        """
        anomalies: list[Anomaly] = []

        for i, period in enumerate(self.periods):
            if i == 0:
                continue

            prev = self.periods[i - 1]

            # 1. Revenue-CF divergence
            rev_growth = self._safe_div(
                self._get(self.income, "revenue", period)
                - self._get(self.income, "revenue", prev),
                self._get(self.income, "revenue", prev),
            )
            ocf_growth = self._safe_div(
                self._get(self.cashflow, "operating_cash_flow", period)
                - self._get(self.cashflow, "operating_cash_flow", prev),
                self._get(self.cashflow, "operating_cash_flow", prev),
            )

            if rev_growth > 0.15 and ocf_growth < -0.20:
                divergence = rev_growth - ocf_growth
                severity = (
                    "high" if divergence > 0.50 else "medium" if divergence > 0.30 else "low"
                )
                anomalies.append(
                    Anomaly(
                        category="revenue_cf_divergence",
                        metric="revenue_vs_ocf",
                        period=period,
                        value=divergence,
                        threshold=0.30,
                        severity=severity,
                        description=(
                            f"Revenue grew {rev_growth:.1%} but operating cash flow "
                            f"declined {ocf_growth:.1%}"
                        ),
                    )
                )

            # 2. DSO spike
            dso = (
                self._safe_div(
                    self._get(self.balance, "receivables", period),
                    self._get(self.income, "revenue", period),
                )
                * 365
            )
            prev_dso = (
                self._safe_div(
                    self._get(self.balance, "receivables", prev),
                    self._get(self.income, "revenue", prev),
                )
                * 365
            )
            dso_change = self._safe_div(dso - prev_dso, prev_dso)

            if dso_change > 0.30:
                severity = "high" if dso_change > 0.50 else "medium"
                anomalies.append(
                    Anomaly(
                        category="dso_spike",
                        metric="dso",
                        period=period,
                        value=dso,
                        threshold=prev_dso * 1.3,
                        severity=severity,
                        description=(
                            f"DSO increased {dso_change:.1%} from {prev_dso:.0f} to {dso:.0f} days"
                        ),
                    )
                )

            # 3. Accruals spike
            net_income = self._get(self.income, "net_income", period)
            ocf = self._get(self.cashflow, "operating_cash_flow", period)
            total_assets = self._get(self.balance, "total_assets", period)
            accruals_ratio = self._safe_div(net_income - ocf, total_assets)

            if accruals_ratio > 0.10:
                severity = "high" if accruals_ratio > 0.20 else "medium"
                anomalies.append(
                    Anomaly(
                        category="accruals_spike",
                        metric="accruals_ratio",
                        period=period,
                        value=accruals_ratio,
                        threshold=0.10,
                        severity=severity,
                        description=(
                            f"Accruals ratio of {accruals_ratio:.1%} exceeds 10% threshold"
                        ),
                    )
                )

            # 4. Negative FCF
            fcf = ocf - self._get(self.cashflow, "capex", period)
            if fcf < 0:
                severity = "high" if fcf < -100 else "medium" if fcf < -50 else "low"
                anomalies.append(
                    Anomaly(
                        category="negative_fcf",
                        metric="free_cash_flow",
                        period=period,
                        value=fcf,
                        threshold=0.0,
                        severity=severity,
                        description=f"Free cash flow is negative at {fcf:.1f}",
                    )
                )

        return anomalies

    # -----------------------------------------------------------------------
    # Benford's law analysis
    # -----------------------------------------------------------------------

    def benford_analysis(self, statement: str = "income") -> dict[str, Any]:
        """Apply Benford's law analysis to a financial statement.

        Args:
            statement: "income", "balance", or "cashflow"

        Returns:
            Dictionary with chi2, p_value, and conformity assessment.
        """
        df = getattr(self, statement)
        values = df.values.flatten()
        values = values[values > 0]
        chi2, p_value, details = benford_chi2(values)
        return {
            "chi2": chi2,
            "p_value": p_value,
            "conforms": p_value > 0.05,
            "details": details,
        }

    # -----------------------------------------------------------------------
    # Full analysis
    # -----------------------------------------------------------------------

    def full_analysis(self) -> dict[str, Any]:
        """Run complete fundamental analysis.

        Returns:
            Dictionary with all analysis results.
        """
        return {
            "ratios": {
                "profitability": self.profitability_ratios(),
                "liquidity": self.liquidity_ratios(),
                "leverage": self.leverage_ratios(),
                "efficiency": self.efficiency_ratios(),
            },
            "cash_flow": self.cash_flow_analysis(),
            "cash_flow_trends": self.cash_flow_trends(),
            "earnings_quality": self.earnings_quality(),
            "anomalies": self.detect_anomalies(),
        }
