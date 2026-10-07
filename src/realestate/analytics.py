"""Real Estate Analytics Engine.

Provides property valuation, REIT analysis, and cap rate calculations
following CFA Institute real estate investment standards.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


class RealEstateEngine:
    """Engine for real estate analytics and REIT analysis."""

    # ------------------------------------------------------------------
    # Property Valuation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_noi(
        gross_potential_rent: float,
        vacancy_rate: float,
        operating_expenses: float,
    ) -> float:
        """Calculate Net Operating Income.

        NOI = GPR × (1 − Vacancy Rate) − Operating Expenses
        """
        return gross_potential_rent * (1.0 - vacancy_rate) - operating_expenses

    @staticmethod
    def value_property_income_approach(noi: float, cap_rate: float) -> float:
        """Value property using the income approach.

        Value = NOI / Cap Rate
        """
        if cap_rate <= 0:
            raise ValueError("Cap rate must be positive")
        return noi / cap_rate

    @staticmethod
    def value_property_dcf(
        cash_flows: list[float],
        discount_rate: float,
        terminal_cap_rate: float | None = None,
    ) -> float:
        """Value property using Discounted Cash Flow analysis.

        NPV = Σ CF_t / (1 + r)^t

        If terminal_cap_rate is provided, a terminal value is computed as
        TV = CF_n / terminal_cap_rate and discounted back n periods.
        """
        if not cash_flows:
            raise ValueError("Cash flows cannot be empty")
        if discount_rate < 0:
            raise ValueError("Discount rate must be non-negative")

        npv = 0.0
        for t, cf in enumerate(cash_flows):
            npv += cf / (1.0 + discount_rate) ** t

        if terminal_cap_rate is not None:
            if terminal_cap_rate <= 0:
                raise ValueError("Terminal cap rate must be positive")
            final_noi = cash_flows[-1]
            terminal_value = final_noi / terminal_cap_rate
            n_periods = len(cash_flows) - 1
            npv += terminal_value / (1.0 + discount_rate) ** n_periods

        return npv

    # ------------------------------------------------------------------
    # Cap Rate Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_going_in_cap_rate(noi: float, purchase_price: float) -> float:
        """Going-in cap rate = NOI / Purchase Price."""
        if purchase_price <= 0:
            raise ValueError("Price must be positive")
        return noi / purchase_price

    @staticmethod
    def calculate_exit_cap_rate(noi: float, exit_price: float) -> float:
        """Exit cap rate = Year N NOI / Exit Price."""
        if exit_price <= 0:
            raise ValueError("Exit price must be positive")
        return noi / exit_price

    @staticmethod
    def calculate_implied_cap_rate(
        noi: float, market_cap: float, total_debt: float
    ) -> float:
        """Implied cap rate = NOI / (Market Cap + Total Debt)."""
        total_value = market_cap + total_debt
        if total_value <= 0:
            raise ValueError("Total value must be positive")
        return noi / total_value

    @staticmethod
    def calculate_cap_rate_spread(exit_cap: float, going_in_cap: float) -> float:
        """Cap rate spread = Exit Cap − Going-in Cap."""
        return exit_cap - going_in_cap

    # ------------------------------------------------------------------
    # REIT Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ffo(
        net_income: float,
        depreciation: float,
        gains_on_sales: float,
    ) -> float:
        """Funds From Operations.

        FFO = Net Income + Depreciation − Gains on Sales
        """
        return net_income + depreciation - gains_on_sales

    @staticmethod
    def calculate_affo(
        ffo: float,
        recurring_capex: float,
        straight_line_rent: float,
    ) -> float:
        """Adjusted Funds From Operations.

        AFFO = FFO − Recurring Capex − Straight-line Rent
        """
        return ffo - recurring_capex - straight_line_rent

    @staticmethod
    def calculate_nav(total_assets: float, total_liabilities: float) -> float:
        """Net Asset Value = Total Assets − Total Liabilities."""
        return total_assets - total_liabilities

    @staticmethod
    def calculate_dividend_yield(annual_dividend: float, price: float) -> float:
        """Dividend Yield = Annual Dividend / Price."""
        if price <= 0:
            raise ValueError("Price must be positive")
        return annual_dividend / price

    @staticmethod
    def calculate_payout_ratio(annual_dividend: float, ffo_per_share: float) -> float:
        """Payout Ratio = Annual Dividend / FFO per Share."""
        if ffo_per_share <= 0:
            raise ValueError("FFO per share must be positive")
        return annual_dividend / ffo_per_share

    @staticmethod
    def calculate_price_to_ffo(price: float, ffo_per_share: float) -> float:
        """P/FFO = Price / FFO per Share."""
        if ffo_per_share <= 0:
            raise ValueError("FFO per share must be positive")
        return price / ffo_per_share

    @staticmethod
    def calculate_debt_to_assets(total_debt: float, total_assets: float) -> float:
        """Debt / Total Assets."""
        if total_assets <= 0:
            raise ValueError("Total assets must be positive")
        return total_debt / total_assets

    def analyze_reit(
        self,
        *,
        ticker: str,
        price: float,
        shares_outstanding: float,
        net_income: float,
        depreciation: float,
        gains_on_sales: float,
        ffo: float,
        total_debt: float,
        cash: float,
        total_assets: float,
        total_liabilities: float,
        annual_dividend: float,
        property_count: int,
        portfolio_noi: float,
    ) -> dict[str, float]:
        """Full REIT analysis returning key metrics.

        Returns a dictionary with:
        - ffo_per_share
        - affo
        - nav_per_share
        - dividend_yield
        - payout_ratio
        - implied_cap_rate
        - debt_to_ebitda
        """
        ffo_per_share = ffo / shares_outstanding

        # Estimate recurring capex and straight-line rent as % of NOI
        recurring_capex = 0.15 * portfolio_noi
        straight_line_rent = 0.02 * portfolio_noi
        affo = self.calculate_affo(ffo, recurring_capex, straight_line_rent)

        nav = self.calculate_nav(total_assets, total_liabilities)
        nav_per_share = nav / shares_outstanding

        dividend_yield = self.calculate_dividend_yield(annual_dividend, price)
        payout_ratio = self.calculate_payout_ratio(annual_dividend, ffo_per_share)

        market_cap = price * shares_outstanding
        implied_cap_rate = self.calculate_implied_cap_rate(
            portfolio_noi, market_cap, total_debt
        )

        # Simplified EBITDA = Net Income + Depreciation
        ebitda = net_income + depreciation
        debt_to_ebitda = total_debt / ebitda if ebitda > 0 else float("inf")

        return {
            "ffo_per_share": ffo_per_share,
            "affo": affo,
            "nav_per_share": nav_per_share,
            "dividend_yield": dividend_yield,
            "payout_ratio": payout_ratio,
            "implied_cap_rate": implied_cap_rate,
            "debt_to_ebitda": debt_to_ebitda,
        }

    # ------------------------------------------------------------------
    # Investment Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_irr(
        cash_flows: list[float],
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> float:
        """Calculate Internal Rate of Return using bisection method.

        Returns NaN if no real IRR exists (e.g., all negative cash flows).
        """
        if not cash_flows:
            return float("nan")

        # Trivial cases: no sign change → no IRR
        if all(cf <= 0 for cf in cash_flows):
            return float("nan")
        if all(cf >= 0 for cf in cash_flows):
            return float("nan")

        def npv(rate: float) -> float:
            return sum(cf / (1.0 + rate) ** i for i, cf in enumerate(cash_flows))

        low, high = -0.99, 10.0
        npv_low = npv(low)
        npv_high = npv(high)

        if npv_low * npv_high > 0:
            return float("nan")

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            npv_mid = npv(mid)

            if abs(npv_mid) < tol:
                return mid

            if npv_low * npv_mid < 0:
                high = mid
            else:
                low = mid
                npv_low = npv_mid

        return (low + high) / 2.0

    @staticmethod
    def calculate_equity_multiple(cash_flows: list[float]) -> float:
        """Equity Multiple = Total Distributions / Total Invested."""
        invested = sum(cf for cf in cash_flows if cf < 0)
        distributions = sum(cf for cf in cash_flows if cf > 0)
        if invested == 0:
            return float("inf")
        return distributions / abs(invested)

    @staticmethod
    def calculate_loan_constant(annual_rate: float, years: int) -> float:
        """Mortgage constant (annual debt service / loan amount).

        MC = (r × (1+r)^n) / ((1+r)^n − 1)
        """
        if years <= 0:
            raise ValueError("Years must be positive")
        if annual_rate == 0:
            return 1.0 / years
        r = annual_rate
        n = years
        return (r * (1.0 + r) ** n) / ((1.0 + r) ** n - 1.0)

    @staticmethod
    def calculate_ltv(loan_amount: float, property_value: float) -> float:
        """Loan-to-Value = Loan Amount / Property Value."""
        if property_value <= 0:
            raise ValueError("Property value must be positive")
        return loan_amount / property_value

    @staticmethod
    def calculate_dscr(noi: float, debt_service: float) -> float:
        """Debt Service Coverage Ratio = NOI / Debt Service."""
        if debt_service <= 0:
            raise ValueError("Debt service must be positive")
        return noi / debt_service

    # ------------------------------------------------------------------
    # Portfolio Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def portfolio_summary(properties: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate portfolio metrics across multiple properties."""
        if not properties:
            return {
                "total_value": 0.0,
                "total_noi": 0.0,
                "weighted_avg_cap_rate": 0.0,
                "property_count": 0,
            }

        total_value = sum(p["value"] for p in properties)
        total_noi = sum(p["noi"] for p in properties)
        weighted_avg_cap_rate = total_noi / total_value if total_value > 0 else 0.0

        return {
            "total_value": total_value,
            "total_noi": total_noi,
            "weighted_avg_cap_rate": weighted_avg_cap_rate,
            "property_count": len(properties),
        }

    @staticmethod
    def portfolio_by_type(properties: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Group portfolio metrics by property type."""
        result: dict[str, dict[str, Any]] = {}
        for p in properties:
            ptype = p["type"]
            if ptype not in result:
                result[ptype] = {
                    "count": 0,
                    "total_value": 0.0,
                    "total_noi": 0.0,
                }
            result[ptype]["count"] += 1
            result[ptype]["total_value"] += p["value"]
            result[ptype]["total_noi"] += p["noi"]

        # Add cap rate per type
        for ptype in result:
            tv = result[ptype]["total_value"]
            result[ptype]["cap_rate"] = (
                result[ptype]["total_noi"] / tv if tv > 0 else 0.0
            )

        return result

    # ------------------------------------------------------------------
    # DataFrame Integration
    # ------------------------------------------------------------------

    @staticmethod
    def properties_to_dataframe(properties: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert property list to DataFrame with computed cap_rate column."""
        df = pd.DataFrame(properties)
        if "noi" in df.columns and "value" in df.columns:
            df["cap_rate"] = df["noi"] / df["value"]
        return df

    @staticmethod
    def reits_to_dataframe(reits: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert REIT data to DataFrame with computed metrics."""
        df = pd.DataFrame(reits)
        if "dividend" in df.columns and "price" in df.columns:
            df["dividend_yield"] = df["dividend"] / df["price"]
        if "ffo" in df.columns and "shares_outstanding" in df.columns:
            df["ffo_per_share"] = df["ffo"] / df["shares_outstanding"]
        if "dividend" in df.columns and "ffo_per_share" in df.columns:
            df["payout_ratio"] = df["dividend"] / df["ffo_per_share"]
        return df
