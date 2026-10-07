"""Wealth Management Engine for private banking and wealth planning.

Provides client profiling, portfolio construction, tax optimization,
and retirement projection capabilities based on CFA Institute principles.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RiskTolerance(str, Enum):
    """Client risk tolerance levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class ClientProfile:
    """Client demographic and financial profile."""

    age: int
    annual_income: float
    net_worth: float
    risk_tolerance: RiskTolerance
    investment_horizon_years: int
    liquidity_needs: float


class WealthManagementEngine:
    """Engine for generating comprehensive wealth management plans.

    Implements client profiling, portfolio construction, tax optimization,
    and retirement projections using mean-variance optimization and
    tax-efficient strategies.
    """

    ASSET_CLASSES = ["us_equity", "intl_equity", "bonds", "cash", "reits"]

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Client Profiling
    # ------------------------------------------------------------------

    def assess_client_profile(self, client: ClientProfile) -> dict:
        """Assess client risk profile and return a risk score (0-100).

        The score incorporates risk tolerance, age, investment horizon,
        and liquidity needs. Higher scores indicate greater risk capacity.
        """
        base_scores = {
            RiskTolerance.CONSERVATIVE: 20,
            RiskTolerance.MODERATE: 50,
            RiskTolerance.AGGRESSIVE: 80,
        }
        score = float(base_scores[client.risk_tolerance])

        # Age adjustment: younger clients have higher risk capacity
        age_factor = max(0.0, (65 - client.age) * 0.5)
        score += age_factor

        # Horizon adjustment: longer horizons allow more risk
        horizon_factor = min(10.0, client.investment_horizon_years * 0.3)
        score += horizon_factor

        # Liquidity adjustment: higher liquidity needs reduce risk capacity
        liq_ratio = client.liquidity_needs / max(client.net_worth, 1.0)
        liq_penalty = min(15.0, liq_ratio * 30.0)
        score -= liq_penalty

        score = max(0.0, min(100.0, score))

        return {
            "risk_score": round(score, 2),
            "risk_tolerance": client.risk_tolerance.value,
            "age": client.age,
            "investment_horizon": client.investment_horizon_years,
            "liquidity_ratio": round(liq_ratio, 4),
        }

    # ------------------------------------------------------------------
    # Portfolio Construction
    # ------------------------------------------------------------------

    def construct_portfolio(self, client: ClientProfile) -> dict:
        """Construct a strategic asset allocation based on client profile.

        Returns a dictionary mapping asset class names to target weights.
        Weights sum to 1.0 and are all non-negative.
        """
        profile = self.assess_client_profile(client)
        score = profile["risk_score"]

        if score <= 30:
            # Conservative: capital preservation focus
            allocation = {
                "us_equity": 0.25,
                "intl_equity": 0.05,
                "bonds": 0.50,
                "cash": 0.15,
                "reits": 0.05,
            }
        elif score <= 70:
            # Moderate: balanced growth and income
            allocation = {
                "us_equity": 0.40,
                "intl_equity": 0.15,
                "bonds": 0.25,
                "cash": 0.05,
                "reits": 0.15,
            }
        else:
            # Aggressive: growth maximization
            allocation = {
                "us_equity": 0.50,
                "intl_equity": 0.20,
                "bonds": 0.10,
                "cash": 0.00,
                "reits": 0.20,
            }

        return allocation

    def optimize_portfolio(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: Optional[float] = None,
    ) -> dict:
        """Mean-variance optimization for a given target return.

        Uses the analytical solution to the minimum-variance problem
        with a target return constraint. Weights are clipped to be
        non-negative and renormalized.

        Parameters
        ----------
        returns : np.ndarray
            Array of shape (n_assets, n_periods) with historical returns.
        cov_matrix : np.ndarray
            Covariance matrix of shape (n_assets, n_assets).
        target_return : float, optional
            Target portfolio return. If None, uses the maximum
            individual asset return.

        Returns
        -------
        dict with keys 'weights', 'expected_return', 'volatility'.
        """
        mean_returns = np.mean(returns, axis=1)
        n = len(mean_returns)

        if target_return is None:
            target_return = float(np.max(mean_returns))

        # Clip target to achievable range (long-only, no leverage)
        target_return = float(np.clip(target_return, np.min(mean_returns), np.max(mean_returns)))

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_matrix)

        ones = np.ones(n)
        A = float(ones @ inv_cov @ ones)
        B = float(ones @ inv_cov @ mean_returns)
        C = float(mean_returns @ inv_cov @ mean_returns)

        denom = A * C - B**2
        if abs(denom) < 1e-10:
            weights = np.ones(n) / n
        else:
            lambda1 = (2 * C - 2 * B * target_return) / denom
            lambda2 = (2 * A * target_return - 2 * B) / denom
            weights = 0.5 * inv_cov @ (lambda1 * ones + lambda2 * mean_returns)

        # Enforce long-only constraint
        weights = np.maximum(weights, 0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n) / n

        expected_return = float(weights @ mean_returns)
        volatility = float(np.sqrt(weights @ cov_matrix @ weights))

        return {
            "weights": weights,
            "expected_return": expected_return,
            "volatility": volatility,
        }

    def compute_efficient_frontier(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 10,
    ) -> list[dict]:
        """Compute the efficient frontier by varying target return.

        Returns a list of dictionaries with 'return', 'volatility',
        and 'sharpe' for each point on the frontier.
        """
        mean_returns = np.mean(returns, axis=1)
        min_ret = float(np.min(mean_returns))
        max_ret = float(np.max(mean_returns))

        frontier = []
        for target in np.linspace(min_ret, max_ret, n_points):
            result = self.optimize_portfolio(returns, cov_matrix, target_return=float(target))
            sharpe = (
                (result["expected_return"] - self.risk_free_rate) / result["volatility"]
                if result["volatility"] > 0
                else 0.0
            )
            frontier.append(
                {
                    "return": result["expected_return"],
                    "volatility": result["volatility"],
                    "sharpe": sharpe,
                }
            )
        return frontier

    # ------------------------------------------------------------------
    # Tax Optimization
    # ------------------------------------------------------------------

    def identify_tax_losses(self, holdings: list[dict]) -> list[dict]:
        """Identify holdings with unrealized losses for tax-loss harvesting.

        Parameters
        ----------
        holdings : list of dict
            Each dict must have 'symbol', 'shares', 'cost_basis', 'current_price'.

        Returns
        -------
        list of dict with 'symbol', 'unrealized_loss', 'shares' for each
        holding with a negative unrealized P&L.
        """
        losses = []
        for h in holdings:
            unrealized = (h["current_price"] - h["cost_basis"]) * h["shares"]
            if unrealized < 0:
                losses.append(
                    {
                        "symbol": h["symbol"],
                        "unrealized_loss": round(unrealized, 2),
                        "shares": h["shares"],
                    }
                )
        return losses

    def recommend_asset_location(self, client: ClientProfile) -> dict:
        """Recommend asset location across account types.

        Places tax-inefficient assets in tax-advantaged accounts and
        tax-efficient assets in taxable accounts to minimize tax drag.
        """
        return {
            "taxable": ["us_equity", "intl_equity", "reits"],
            "tax_deferred": ["bonds", "cash"],
            "tax_free": ["municipal_bonds"],
        }

    def tax_efficient_rebalance(self, current: dict, target: dict) -> dict:
        """Generate tax-efficient rebalancing trades.

        Identifies the trades needed to move from current to target
        allocation and estimates the tax impact of selling appreciated
        assets.

        Returns
        -------
        dict with 'trades' (list of trade dicts) and 'estimated_tax_impact'.
        """
        trades = []
        tax_impact = 0.0

        all_assets = set(current.keys()) | set(target.keys())
        for asset in all_assets:
            current_weight = current.get(asset, 0.0)
            target_weight = target.get(asset, 0.0)
            diff = target_weight - current_weight

            if abs(diff) > 0.001:
                action = "buy" if diff > 0 else "sell"
                trades.append(
                    {
                        "asset": asset,
                        "action": action,
                        "amount": round(abs(diff), 4),
                    }
                )
                # Estimate capital gains tax on sells (15% long-term rate)
                if action == "sell":
                    tax_impact += abs(diff) * 0.15

        return {
            "trades": trades,
            "estimated_tax_impact": round(tax_impact, 2),
        }

    # ------------------------------------------------------------------
    # Retirement Projection
    # ------------------------------------------------------------------

    def future_value(
        self,
        present_value: float,
        annual_return: float,
        years: int,
        annual_contribution: float = 0.0,
        inflation_rate: float = 0.0,
    ) -> float:
        """Calculate future value with optional contributions and inflation.

        Uses the real rate of return when inflation is specified.
        """
        real_return = (1 + annual_return) / (1 + inflation_rate) - 1

        fv = present_value * (1 + real_return) ** years

        if annual_contribution > 0 and abs(real_return) > 1e-10:
            fv += annual_contribution * ((1 + real_return) ** years - 1) / real_return
        elif annual_contribution > 0:
            fv += annual_contribution * years

        return fv

    def safe_withdrawal_amount(self, portfolio_value: float, withdrawal_rate: float = 0.04) -> float:
        """Calculate annual safe withdrawal amount using the 4% rule."""
        return portfolio_value * withdrawal_rate

    # ------------------------------------------------------------------
    # Risk Metrics
    # ------------------------------------------------------------------

    def portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return float(np.sqrt(weights @ cov_matrix @ weights))

    def sharpe_ratio(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """Calculate the Sharpe ratio for a portfolio."""
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        port_return = float(weights @ returns)
        port_vol = self.portfolio_volatility(weights, cov_matrix)
        if port_vol < 1e-10:
            return 0.0
        return (port_return - rf) / port_vol

    def max_drawdown(self, values: np.ndarray) -> float:
        """Calculate the maximum drawdown from a series of portfolio values."""
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return float(np.max(drawdown))

    # ------------------------------------------------------------------
    # Wealth Plan Generation
    # ------------------------------------------------------------------

    def generate_wealth_plan(
        self,
        client: ClientProfile,
        holdings: Optional[list[dict]] = None,
    ) -> dict:
        """Generate a comprehensive wealth management plan.

        Combines client profiling, portfolio construction, tax optimization,
        and retirement projections into a single plan.
        """
        profile = self.assess_client_profile(client)
        allocation = self.construct_portfolio(client)

        # Tax strategy
        tax_losses = self.identify_tax_losses(holdings) if holdings else []
        asset_location = self.recommend_asset_location(client)

        # Retirement projections
        retirement_age = 65
        years_to_retirement = max(0, retirement_age - client.age)
        annual_savings = client.annual_income * 0.15
        retirement_value = self.future_value(
            present_value=client.net_worth,
            annual_return=0.06,
            years=years_to_retirement,
            annual_contribution=annual_savings,
        )

        # Recommendations
        recommendations = []
        if profile["risk_score"] < 30:
            recommendations.append(
                "Consider increasing equity allocation for long-term growth"
            )
        if tax_losses:
            recommendations.append(
                f"Harvest tax losses in {len(tax_losses)} position(s)"
            )
        if client.liquidity_needs > client.net_worth * 0.1:
            recommendations.append(
                "Maintain higher cash reserves for liquidity needs"
            )
        if client.annual_income > 0 and annual_savings < client.annual_income * 0.1:
            recommendations.append(
                "Increase savings rate to at least 10% of income"
            )
        if not recommendations:
            recommendations.append(
                "Review and rebalance portfolio annually to maintain target allocation"
            )

        return {
            "client_profile": profile,
            "portfolio": allocation,
            "tax_strategy": {
                "tax_loss_harvesting": tax_losses,
                "asset_location": asset_location,
            },
            "projections": {
                "retirement_age_value": round(retirement_value, 2),
                "years_to_retirement": years_to_retirement,
            },
            "recommendations": recommendations,
        }
