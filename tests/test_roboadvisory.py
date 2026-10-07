"""Tests for robo-advisory engine.

Covers the three pillars of a CFA Institute-style robo-advisor:
risk profiling, goal-based investing (mean-variance optimization),
and tax-loss harvesting — plus rebalancing and plan-generation latency.
"""

import math
from datetime import date, timedelta

import pytest

from roboadvisory import (
    Goal,
    InvestmentPlan,
    RebalanceTrade,
    RiskTolerance,
    RoboAdvisoryEngine,
    TaxLot,
)


def _answers(**overrides):
    """Canonical moderate-risk questionnaire answers, overridable per test."""
    base = {
        "age": 45,
        "horizon_years": 15,
        "income_stability": 3,
        "net_worth": 250_000,
        "loss_reaction": 3,
        "experience": 3,
        "objective": "balanced",
    }
    base.update(overrides)
    return base


def _fv(annual_rate, years, present_value, monthly_contribution):
    """Independent future-value reference (lump sum + monthly annuity)."""
    if abs(annual_rate) < 1e-12:
        return present_value + monthly_contribution * 12.0 * years
    growth = (1.0 + annual_rate) ** years
    return present_value * growth + monthly_contribution * 12.0 * (growth - 1.0) / annual_rate


# ---------------------------------------------------------------------------
# Risk profiling
# ---------------------------------------------------------------------------


class TestRiskProfiling:
    def test_conservative_profile(self):
        engine = RoboAdvisoryEngine()
        profile = engine.assess_risk_profile(
            _answers(age=70, horizon_years=3, income_stability=2, net_worth=100_000,
                     loss_reaction=1, experience=1, objective="preservation")
        )
        assert profile.tolerance is RiskTolerance.CONSERVATIVE
        assert profile.score < 40

    def test_aggressive_profile(self):
        engine = RoboAdvisoryEngine()
        profile = engine.assess_risk_profile(
            _answers(age=30, horizon_years=20, income_stability=4, net_worth=500_000,
                     loss_reaction=5, experience=5, objective="growth")
        )
        assert profile.tolerance is RiskTolerance.AGGRESSIVE
        assert profile.score >= 70

    def test_moderate_profile(self):
        engine = RoboAdvisoryEngine()
        profile = engine.assess_risk_profile(_answers())
        assert profile.tolerance is RiskTolerance.MODERATE
        assert 40 <= profile.score < 70

    def test_score_within_bounds(self):
        engine = RoboAdvisoryEngine()
        for age, horizon, loss_reaction in [(25, 30, 5), (80, 1, 1), (50, 10, 3)]:
            profile = engine.assess_risk_profile(
                _answers(age=age, horizon_years=horizon, loss_reaction=loss_reaction)
            )
            assert 0.0 <= profile.score <= 100.0

    def test_missing_answers_raise(self):
        engine = RoboAdvisoryEngine()
        with pytest.raises(ValueError, match="age"):
            engine.assess_risk_profile({"horizon_years": 10})

    def test_unknown_objective_raises(self):
        engine = RoboAdvisoryEngine()
        with pytest.raises(ValueError, match="objective"):
            engine.assess_risk_profile(_answers(objective="speculate"))

    def test_capacity_and_willingness_split(self):
        engine = RoboAdvisoryEngine()
        profile = engine.assess_risk_profile(_answers())
        assert 0.0 <= profile.capacity_score <= 100.0
        assert 0.0 <= profile.willingness_score <= 100.0
        assert profile.score == pytest.approx(
            0.5 * profile.capacity_score + 0.5 * profile.willingness_score
        )


# ---------------------------------------------------------------------------
# Goal-based investing
# ---------------------------------------------------------------------------


class TestGoalBasedInvesting:
    def test_required_return_positive_when_underfunded(self):
        engine = RoboAdvisoryEngine()
        goal = Goal(name="House", target_amount=500_000, years=20,
                    current_savings=50_000, monthly_contribution=500)
        rr = engine.required_return(goal)
        assert 0.0 < rr < 1.0

    def test_required_return_zero_when_fully_funded(self):
        engine = RoboAdvisoryEngine()
        goal = Goal(name="House", target_amount=500_000, years=20,
                    current_savings=600_000, monthly_contribution=0)
        assert engine.required_return(goal) == 0.0

    def test_required_return_matches_future_value(self):
        engine = RoboAdvisoryEngine()
        goal = Goal(name="House", target_amount=500_000, years=20,
                    current_savings=50_000, monthly_contribution=500)
        rr = engine.required_return(goal)
        assert _fv(rr, goal.years, goal.current_savings, goal.monthly_contribution) == pytest.approx(
            goal.target_amount, rel=1e-6
        )

    def test_infeasible_goal_flagged(self):
        engine = RoboAdvisoryEngine()
        goal = Goal(name="Yacht", target_amount=10_000_000, years=1,
                    current_savings=0, monthly_contribution=0)
        plan = engine.generate_plan(_answers(), [goal])
        assert plan.goal_plans[0].feasible is False

    def test_monthly_contribution_needed_reaches_target(self):
        engine = RoboAdvisoryEngine()
        goal = Goal(name="College", target_amount=100_000, years=10,
                    current_savings=0, monthly_contribution=0)
        pmt = engine.monthly_contribution_needed(goal, 0.06)
        assert pmt > 0
        assert _fv(0.06, goal.years, 0.0, pmt) == pytest.approx(goal.target_amount, rel=1e-9)

    def test_allocation_weights_sum_to_one(self):
        engine = RoboAdvisoryEngine()
        alloc = engine.optimize_allocation(target_return=0.05, tolerance=RiskTolerance.MODERATE)
        assert math.isclose(sum(alloc.values()), 1.0, abs_tol=1e-6)
        assert all(w >= 0.0 for w in alloc.values())

    def test_allocation_respects_risk_profile(self):
        engine = RoboAdvisoryEngine()
        conservative = engine.optimize_allocation(0.05, RiskTolerance.CONSERVATIVE)
        aggressive = engine.optimize_allocation(0.05, RiskTolerance.AGGRESSIVE)
        eq = ("US Equity", "Intl Equity", "EM Equity", "REITs")
        cons_equity = sum(conservative[a] for a in eq)
        agg_equity = sum(aggressive[a] for a in eq)
        assert cons_equity < agg_equity
        assert conservative["Bonds"] > aggressive["Bonds"]

    def test_glide_path_derisks_monotonically(self):
        engine = RoboAdvisoryEngine()
        goal = Goal(name="Retirement", target_amount=1_000_000, years=10)
        path = engine.glide_path(goal, equity_start=0.60)
        assert path[0][1] == pytest.approx(0.60)
        assert path[-1][1] == pytest.approx(0.10)
        weights = [w for _, w in path]
        assert all(a >= b for a, b in zip(weights, weights[1:]))

    def test_plan_includes_all_goals(self):
        engine = RoboAdvisoryEngine()
        goals = [
            Goal(name="Emergency", target_amount=50_000, years=3),
            Goal(name="House", target_amount=400_000, years=10),
            Goal(name="Retirement", target_amount=1_500_000, years=25),
        ]
        plan = engine.generate_plan(_answers(), goals)
        assert len(plan.goal_plans) == 3
        assert {gp.goal.name for gp in plan.goal_plans} == {"Emergency", "House", "Retirement"}


# ---------------------------------------------------------------------------
# Tax-loss harvesting
# ---------------------------------------------------------------------------


class TestTaxLossHarvesting:
    def test_identifies_unrealized_losses(self):
        engine = RoboAdvisoryEngine(tax_rate=0.25)
        lots = [TaxLot(asset="US Equity", shares=100, cost_basis=120.0,
                       current_price=100.0, purchase_date=date(2024, 1, 15))]
        opps = engine.find_tax_loss_opportunities(lots, {}, date(2026, 1, 1))
        assert len(opps) == 1
        assert opps[0].unrealized_loss == pytest.approx(-2000.0)
        assert opps[0].wash_sale_blocked is False

    def test_ignores_gains(self):
        engine = RoboAdvisoryEngine()
        lots = [TaxLot(asset="US Equity", shares=100, cost_basis=80.0,
                       current_price=100.0, purchase_date=date(2024, 1, 15))]
        assert engine.find_tax_loss_opportunities(lots, {}, date(2026, 1, 1)) == []

    def test_wash_sale_blocks_recent_purchase(self):
        engine = RoboAdvisoryEngine()
        today = date(2026, 1, 31)
        recent = TaxLot(asset="US Equity", shares=100, cost_basis=120.0,
                       current_price=100.0, purchase_date=today - timedelta(days=10))
        old = TaxLot(asset="Bonds", shares=50, cost_basis=110.0,
                     current_price=100.0, purchase_date=today - timedelta(days=45))
        opps = engine.find_tax_loss_opportunities([recent, old], {}, today)
        by_asset = {o.asset: o for o in opps}
        assert by_asset["US Equity"].wash_sale_blocked is True
        assert by_asset["Bonds"].wash_sale_blocked is False

    def test_tax_savings_equal_loss_times_rate(self):
        engine = RoboAdvisoryEngine(tax_rate=0.25)
        lots = [TaxLot(asset="US Equity", shares=100, cost_basis=120.0,
                       current_price=100.0, purchase_date=date(2024, 1, 15))]
        opps = engine.find_tax_loss_opportunities(lots, {}, date(2026, 1, 1))
        assert opps[0].tax_savings == pytest.approx(2000.0 * 0.25)

    def test_current_price_override_from_prices_map(self):
        engine = RoboAdvisoryEngine()
        lots = [TaxLot(asset="US Equity", shares=10, cost_basis=100.0,
                       current_price=100.0, purchase_date=date(2024, 1, 15))]
        opps = engine.find_tax_loss_opportunities(
            lots, {"US Equity": 90.0}, date(2026, 1, 1)
        )
        assert opps[0].unrealized_loss == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------


class TestRebalancing:
    def test_trades_sum_to_zero(self):
        engine = RoboAdvisoryEngine()
        holdings = {"US Equity": 6_000.0, "Bonds": 4_000.0}
        target = {"US Equity": 0.5, "Bonds": 0.3, "Cash": 0.2}
        trades = engine.rebalance(holdings, target)
        assert sum(t.trade for t in trades) == pytest.approx(0.0, abs=1e-9)

    def test_trades_move_toward_target(self):
        engine = RoboAdvisoryEngine()
        holdings = {"US Equity": 6_000.0, "Bonds": 4_000.0}
        target = {"US Equity": 0.5, "Bonds": 0.3, "Cash": 0.2}
        trades = {t.asset: t for t in engine.rebalance(holdings, target)}
        assert trades["US Equity"].trade == pytest.approx(-1_000.0)
        assert trades["Cash"].trade == pytest.approx(2_000.0)
        assert all(isinstance(t, RebalanceTrade) for t in trades.values())


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


class TestPlanGeneration:
    def test_plan_generation_under_100ms(self):
        engine = RoboAdvisoryEngine()
        goals = [
            Goal(name="Emergency", target_amount=50_000, years=3, current_savings=10_000),
            Goal(name="House", target_amount=400_000, years=10, current_savings=50_000),
            Goal(name="Retirement", target_amount=1_500_000, years=25,
                 current_savings=100_000, monthly_contribution=1_000),
        ]
        plan = engine.generate_plan(_answers(), goals)
        assert isinstance(plan, InvestmentPlan)
        assert plan.generation_time_ms < 100.0

    def test_blended_metrics_consistent(self):
        engine = RoboAdvisoryEngine()
        goals = [
            Goal(name="A", target_amount=100_000, years=5),
            Goal(name="B", target_amount=900_000, years=20),
        ]
        plan = engine.generate_plan(_answers(), goals)
        returns = [gp.expected_return for gp in plan.goal_plans]
        assert plan.blended_expected_return == pytest.approx(
            sum(returns) / len(returns), abs=1e-9
        )
        assert plan.blended_volatility >= 0.0
        assert math.isclose(sum(plan.blended_weights.values()), 1.0, abs_tol=1e-6)

    def test_plan_with_holdings_produces_rebalancing_trades(self):
        engine = RoboAdvisoryEngine()
        goals = [Goal(name="Retirement", target_amount=1_000_000, years=20,
                      current_savings=100_000)]
        holdings = {"US Equity": 80_000.0, "Bonds": 20_000.0}
        plan = engine.generate_plan(_answers(), goals, holdings=holdings)
        assert len(plan.rebalancing_trades) > 0
        assert sum(t.trade for t in plan.rebalancing_trades) == pytest.approx(0.0, abs=1e-9)
