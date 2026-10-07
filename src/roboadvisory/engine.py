"""Robo-advisory engine: risk profiling, goal-based investing, tax-loss harvesting.

Implements a CFA Institute-style robo-advisor:
- Risk profiling via capacity/willingness questionnaire scoring
- Goal-based allocation via mean-variance optimization (Markowitz)
- Tax-loss harvesting with wash-sale detection
- Rebalancing trade generation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import brentq


class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class RiskProfile:
    tolerance: RiskTolerance
    score: float
    capacity_score: float
    willingness_score: float


@dataclass(frozen=True)
class Goal:
    name: str
    target_amount: float
    years: int
    current_savings: float = 0.0
    monthly_contribution: float = 0.0


@dataclass(frozen=True)
class GoalPlan:
    goal: Goal
    required_return: float
    feasible: bool
    expected_return: float
    volatility: float
    allocation: dict[str, float]
    monthly_contribution_needed: float


@dataclass(frozen=True)
class TaxLot:
    asset: str
    shares: float
    cost_basis: float
    current_price: float
    purchase_date: date


@dataclass(frozen=True)
class TaxLossOpportunity:
    asset: str
    unrealized_loss: float
    tax_savings: float
    wash_sale_blocked: bool


@dataclass(frozen=True)
class RebalanceTrade:
    asset: str
    trade: float  # positive = buy, negative = sell


@dataclass(frozen=True)
class InvestmentPlan:
    risk_profile: RiskProfile
    goal_plans: list[GoalPlan]
    blended_expected_return: float
    blended_volatility: float
    blended_weights: dict[str, float]
    rebalancing_trades: list[RebalanceTrade]
    generation_time_ms: float


# Asset class universe: expected returns and annual covariance matrix
ASSET_CLASSES = ("US Equity", "Intl Equity", "EM Equity", "REITs", "Bonds", "Cash")
EXPECTED_RETURNS = np.array([0.08, 0.07, 0.09, 0.06, 0.03, 0.01])
COVARIANCE = np.array([
    [0.0225, 0.0150, 0.0180, 0.0120, 0.0015, 0.0000],
    [0.0150, 0.0256, 0.0160, 0.0110, 0.0012, 0.0000],
    [0.0180, 0.0160, 0.0400, 0.0130, 0.0010, 0.0000],
    [0.0120, 0.0110, 0.0130, 0.0289, 0.0020, 0.0000],
    [0.0015, 0.0012, 0.0010, 0.0020, 0.0025, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
])
# Regularize to ensure positive-definiteness (Cash has zero variance)
COVARIANCE += np.eye(len(ASSET_CLASSES)) * 1e-8

# Max equity exposure per risk tolerance
MAX_EQUITY = {
    RiskTolerance.CONSERVATIVE: 0.25,
    RiskTolerance.MODERATE: 0.60,
    RiskTolerance.AGGRESSIVE: 0.90,
}

EQUITY_ASSETS = ("US Equity", "Intl Equity", "EM Equity", "REITs")

QUESTIONNAIRE_FIELDS = (
    "age", "horizon_years", "income_stability", "net_worth",
    "loss_reaction", "experience", "objective",
)

VALID_OBJECTIVES = {"preservation", "income", "balanced", "growth", "speculation"}


class RoboAdvisoryEngine:
    """Personalized investment plan generator.

    Combines risk profiling, goal-based mean-variance optimization,
    tax-loss harvesting, and rebalancing into a single plan.
    """

    def __init__(self, tax_rate: float = 0.25, wash_sale_window_days: int = 30):
        self.tax_rate = tax_rate
        self.wash_sale_window_days = wash_sale_window_days

    # ------------------------------------------------------------------
    # Risk profiling
    # ------------------------------------------------------------------

    def assess_risk_profile(self, answers: dict) -> RiskProfile:
        """Score investor risk tolerance from questionnaire answers.

        Capacity (objective ability to take risk): age, horizon, income
        stability, net worth. Willingness (subjective comfort): loss
        reaction, experience, objective. Final score is 50/50 blend.
        """
        missing = [f for f in QUESTIONNAIRE_FIELDS if f not in answers]
        if missing:
            raise ValueError(f"Missing questionnaire answers: {missing}")

        objective = answers["objective"]
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective!r}")

        # Capacity: 0-100
        age = answers["age"]
        age_score = max(0.0, min(100.0, (100 - age) * 1.5))
        horizon_score = max(0.0, min(100.0, answers["horizon_years"] * 4.0))
        income_score = answers["income_stability"] * 20.0
        net_worth_score = max(0.0, min(100.0, answers["net_worth"] / 10_000))
        capacity = 0.25 * age_score + 0.25 * horizon_score + 0.25 * income_score + 0.25 * net_worth_score

        # Willingness: 0-100
        loss_score = answers["loss_reaction"] * 20.0
        exp_score = answers["experience"] * 20.0
        obj_map = {"preservation": 0.0, "income": 25.0, "balanced": 50.0,
                   "growth": 75.0, "speculation": 100.0}
        obj_score = obj_map[objective]
        willingness = (loss_score + exp_score + obj_score) / 3.0

        score = 0.5 * capacity + 0.5 * willingness
        score = max(0.0, min(100.0, score))

        if score < 40:
            tolerance = RiskTolerance.CONSERVATIVE
        elif score < 70:
            tolerance = RiskTolerance.MODERATE
        else:
            tolerance = RiskTolerance.AGGRESSIVE

        return RiskProfile(
            tolerance=tolerance,
            score=score,
            capacity_score=capacity,
            willingness_score=willingness,
        )

    # ------------------------------------------------------------------
    # Goal-based investing
    # ------------------------------------------------------------------

    def required_return(self, goal: Goal) -> float:
        """Annual return needed to reach goal target from current savings + contributions.

        Returns -1.0 if the goal is infeasible (target unreachable even at 200% return).
        """
        if goal.current_savings >= goal.target_amount:
            return 0.0
        if goal.years <= 0:
            return -1.0

        def gap(rate):
            return _future_value(rate, goal.years, goal.current_savings,
                                 goal.monthly_contribution) - goal.target_amount

        try:
            return brentq(gap, -0.5, 2.0, xtol=1e-10)
        except ValueError:
            return -1.0

    def monthly_contribution_needed(self, goal: Goal, annual_rate: float) -> float:
        """Monthly contribution required to reach goal at given annual return.

        Uses annual compounding for the lump-sum growth and an annual annuity
        factor for contributions (consistent with the _fv reference formula).
        """
        if goal.years <= 0:
            return 0.0
        fv_current = goal.current_savings * (1.0 + annual_rate) ** goal.years
        remaining = goal.target_amount - fv_current
        if remaining <= 0:
            return 0.0
        if abs(annual_rate) < 1e-12:
            return remaining / (goal.years * 12.0)
        growth = (1.0 + annual_rate) ** goal.years
        annuity_factor = (growth - 1.0) / annual_rate
        return remaining / (12.0 * annuity_factor)

    def optimize_allocation(
        self, target_return: float, tolerance: RiskTolerance
    ) -> dict[str, float]:
        """Mean-variance optimal weights for target return under risk constraints.

        Uses analytical Lagrange-multiplier solution for the equality-constrained
        problem, with an active-set pass for the equity cap and non-negativity.
        """
        n = len(ASSET_CLASSES)
        max_eq = MAX_EQUITY[tolerance]
        eq_idx = [ASSET_CLASSES.index(a) for a in EQUITY_ASSETS]

        def solve_subset(free_idx, A, b):
            """Solve equality-constrained QP on a subset of assets."""
            if not free_idx:
                return None
            S_inv = COVARIANCE[np.ix_(free_idx, free_idx)]
            try:
                S_inv = np.linalg.inv(S_inv)
            except np.linalg.LinAlgError:
                return None
            A_sub = A[:, free_idx]
            M = A_sub @ S_inv @ A_sub.T
            try:
                w_sub = S_inv @ A_sub.T @ np.linalg.solve(M, b)
            except np.linalg.LinAlgError:
                return None
            return w_sub

        # Build constraint matrix: sum(w)=1, w'mu=target_return
        A_base = np.vstack([np.ones(n), EXPECTED_RETURNS])
        b_base = np.array([1.0, target_return])

        # Start with all assets free
        free = list(range(n))
        w = None

        for _ in range(n + 1):
            w_sub = solve_subset(free, A_base, b_base)
            if w_sub is None:
                break
            w_full = np.zeros(n)
            w_full[free] = w_sub

            # Check equity cap
            eq_sum = sum(w_full[i] for i in eq_idx)
            if eq_sum > max_eq + 1e-9:
                # Add equity cap as equality constraint
                eq_row = np.zeros(n)
                for i in eq_idx:
                    eq_row[i] = 1.0
                A_cap = np.vstack([A_base, eq_row])
                b_cap = np.array([1.0, target_return, max_eq])
                w_sub2 = solve_subset(free, A_cap, b_cap)
                if w_sub2 is not None:
                    w_full = np.zeros(n)
                    w_full[free] = w_sub2
                else:
                    break

            # Check non-negativity
            neg_in_free = [i for i, fi in enumerate(free) if w_full[fi] < -1e-12]
            if not neg_in_free:
                w = w_full
                break
            # Remove most negative from free set
            worst = min(neg_in_free, key=lambda i: w_full[free[i]])
            free.pop(worst)

        if w is None or not np.isfinite(w).all() or np.any(w < -1e-9):
            w = np.full(n, 1.0 / n)

        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum()

        return {asset: float(w[i]) for i, asset in enumerate(ASSET_CLASSES)}

    def glide_path(self, goal: Goal, equity_start: float = 0.60) -> list[tuple[int, float]]:
        """Linear derisking path from equity_start to 10% equity over goal horizon."""
        equity_end = 0.10
        path = []
        for year in range(goal.years + 1):
            frac = year / max(goal.years, 1)
            weight = equity_start + (equity_end - equity_start) * frac
            path.append((year, weight))
        return path

    # ------------------------------------------------------------------
    # Tax-loss harvesting
    # ------------------------------------------------------------------

    def find_tax_loss_opportunities(
        self,
        lots: list[TaxLot],
        prices: dict[str, float],
        today: date,
    ) -> list[TaxLossOpportunity]:
        """Identify lots with unrealized losses, flagging wash-sale violations."""
        opportunities = []
        for lot in lots:
            current_price = prices.get(lot.asset, lot.current_price)
            unrealized = lot.shares * (current_price - lot.cost_basis)
            if unrealized >= 0:
                continue
            days_held = (today - lot.purchase_date).days
            wash_sale = days_held < self.wash_sale_window_days
            opportunities.append(TaxLossOpportunity(
                asset=lot.asset,
                unrealized_loss=unrealized,
                tax_savings=abs(unrealized) * self.tax_rate,
                wash_sale_blocked=wash_sale,
            ))
        return opportunities

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(
        self, holdings: dict[str, float], target: dict[str, float]
    ) -> list[RebalanceTrade]:
        """Generate trades to move holdings to target weights."""
        total = sum(holdings.values())
        if total <= 0:
            return []
        trades = []
        all_assets = set(holdings) | set(target)
        for asset in sorted(all_assets):
            current = holdings.get(asset, 0.0)
            target_value = target.get(asset, 0.0) * total
            trade = target_value - current
            if abs(trade) > 1e-9:
                trades.append(RebalanceTrade(asset=asset, trade=trade))
        return trades

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        answers: dict,
        goals: list[Goal],
        holdings: Optional[dict[str, float]] = None,
    ) -> InvestmentPlan:
        """Generate a complete personalized investment plan."""
        start = time.perf_counter()

        profile = self.assess_risk_profile(answers)

        goal_plans = []
        for goal in goals:
            rr = self.required_return(goal)
            feasible = 0.0 <= rr <= 0.15  # 15% annual return is unrealistic
            target_ret = min(rr, 0.10) if feasible else 0.05
            alloc = self.optimize_allocation(target_ret, profile.tolerance)
            port_ret = float(np.dot(
                np.array([alloc[a] for a in ASSET_CLASSES]), EXPECTED_RETURNS
            ))
            port_vol = float(np.sqrt(
                np.array([alloc[a] for a in ASSET_CLASSES])
                @ COVARIANCE
                @ np.array([alloc[a] for a in ASSET_CLASSES])
            ))
            pmt = self.monthly_contribution_needed(goal, port_ret)
            goal_plans.append(GoalPlan(
                goal=goal,
                required_return=rr,
                feasible=feasible,
                expected_return=port_ret,
                volatility=port_vol,
                allocation=alloc,
                monthly_contribution_needed=pmt,
            ))

        # Blended metrics: weight by goal target amount
        total_target = sum(g.target_amount for g in goals) or 1.0
        blended_ret = sum(
            gp.expected_return * gp.goal.target_amount for gp in goal_plans
        ) / total_target
        blended_vol = sum(
            gp.volatility * gp.goal.target_amount for gp in goal_plans
        ) / total_target
        blended_weights: dict[str, float] = {}
        for asset in ASSET_CLASSES:
            blended_weights[asset] = sum(
                gp.allocation.get(asset, 0.0) * gp.goal.target_amount
                for gp in goal_plans
            ) / total_target

        # Rebalancing trades
        rebalancing_trades: list[RebalanceTrade] = []
        if holdings:
            rebalancing_trades = self.rebalance(holdings, blended_weights)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return InvestmentPlan(
            risk_profile=profile,
            goal_plans=goal_plans,
            blended_expected_return=blended_ret,
            blended_volatility=blended_vol,
            blended_weights=blended_weights,
            rebalancing_trades=rebalancing_trades,
            generation_time_ms=elapsed_ms,
        )


def _future_value(annual_rate: float, years: int, present_value: float,
                  monthly_contribution: float) -> float:
    """Future value of lump sum plus monthly contributions."""
    if abs(annual_rate) < 1e-12:
        return present_value + monthly_contribution * 12.0 * years
    growth = (1.0 + annual_rate) ** years
    return present_value * growth + monthly_contribution * 12.0 * (growth - 1.0) / annual_rate
