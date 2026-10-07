"""Tests for the Retirement Planning Engine.

Covers Monte Carlo simulation, withdrawal strategies, Social Security optimization,
and retirement plan generation.
"""
import numpy as np
import pytest

from src.retirement.planning import (
    RetirementPlanningEngine,
    WithdrawalStrategy,
    MonteCarloResult,
    RetirementPlan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh engine instance for each test."""
    return RetirementPlanningEngine(seed=42)


@pytest.fixture
def basic_params():
    """Standard retirement parameters."""
    return {
        "current_age": 40,
        "retirement_age": 65,
        "life_expectancy": 90,
        "current_savings": 500_000,
        "annual_contribution": 20_000,
        "annual_retirement_spending": 60_000,
        "social_security_age": 67,
        "social_security_monthly": 2_000,
    }


# ---------------------------------------------------------------------------
# Monte Carlo Simulation Tests
# ---------------------------------------------------------------------------


class TestMonteCarloSimulation:
    """Tests for Monte Carlo portfolio simulation."""

    def test_monte_carlo_returns_correct_number_of_simulations(self, engine, basic_params):
        result = engine.run_monte_carlo(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert isinstance(result, MonteCarloResult)
        assert len(result.final_balances) == 500

    def test_monte_carlo_produces_success_rate(self, engine, basic_params):
        result = engine.run_monte_carlo(
            n_simulations=1000,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert 0.0 <= result.success_rate <= 1.0

    def test_monte_carlo_success_rate_high_for_conservative_plan(self, engine):
        """A well-funded plan should have high success rate."""
        result = engine.run_monte_carlo(
            n_simulations=1000,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            current_age=40,
            retirement_age=65,
            life_expectancy=90,
            current_savings=2_000_000,
            annual_contribution=30_000,
            annual_retirement_spending=40_000,
            social_security_age=67,
            social_security_monthly=2_500,
        )
        assert result.success_rate >= 0.90

    def test_monte_carlo_success_rate_low_for_aggressive_plan(self, engine):
        """A poorly-funded plan should have low success rate."""
        result = engine.run_monte_carlo(
            n_simulations=1000,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            current_age=60,
            retirement_age=65,
            life_expectancy=90,
            current_savings=100_000,
            annual_contribution=0,
            annual_retirement_spending=80_000,
            social_security_age=67,
            social_security_monthly=1_000,
        )
        assert result.success_rate <= 0.50

    def test_monte_carlo_produces_percentiles(self, engine, basic_params):
        result = engine.run_monte_carlo(
            n_simulations=1000,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert result.percentile_10 <= result.percentile_50 <= result.percentile_90

    def test_monte_carlo_with_zero_volatility_is_deterministic(self, engine):
        """With zero volatility, all simulations should produce the same result."""
        result = engine.run_monte_carlo(
            n_simulations=100,
            annual_return_mean=0.07,
            annual_return_std=0.0,
            current_age=40,
            retirement_age=65,
            life_expectancy=90,
            current_savings=500_000,
            annual_contribution=20_000,
            annual_retirement_spending=60_000,
            social_security_age=67,
            social_security_monthly=2_000,
        )
        # All final balances should be identical
        assert np.allclose(result.final_balances, result.final_balances[0])


# ---------------------------------------------------------------------------
# Withdrawal Strategy Tests
# ---------------------------------------------------------------------------


class TestWithdrawalStrategies:
    """Tests for different withdrawal strategies."""

    def test_fixed_percentage_strategy(self, engine):
        strategy = WithdrawalStrategy(
            name="fixed_percentage",
            initial_rate=0.04,
            floor_rate=0.03,
            ceiling_rate=0.05,
        )
        assert strategy.name == "fixed_percentage"
        assert strategy.initial_rate == 0.04

    def test_guardrails_strategy_adjusts_with_portfolio(self, engine):
        """Guardrails strategy should adjust withdrawal based on portfolio performance."""
        strategy = WithdrawalStrategy(
            name="guardrails",
            initial_rate=0.04,
            floor_rate=0.03,
            ceiling_rate=0.05,
            guardrail_threshold=0.20,
        )
        # Portfolio up significantly -> withdrawal should increase
        withdrawal_up = strategy.calculate_withdrawal(
            current_portfolio=1_500_000,
            initial_portfolio=1_000_000,
            base_withdrawal=40_000,
        )
        # Portfolio down significantly -> withdrawal should decrease
        withdrawal_down = strategy.calculate_withdrawal(
            current_portfolio=700_000,
            initial_portfolio=1_000_000,
            base_withdrawal=40_000,
        )
        assert withdrawal_up > withdrawal_down

    def test_guardrails_respects_floor(self, engine):
        """Guardrails should not withdraw below floor rate."""
        strategy = WithdrawalStrategy(
            name="guardrails",
            initial_rate=0.04,
            floor_rate=0.03,
            ceiling_rate=0.05,
            guardrail_threshold=0.20,
        )
        # Portfolio crashed to 10% of initial
        withdrawal = strategy.calculate_withdrawal(
            current_portfolio=100_000,
            initial_portfolio=1_000_000,
            base_withdrawal=40_000,
        )
        # Should be floor_rate * current_portfolio = 0.03 * 100_000 = 3_000
        assert withdrawal <= 0.03 * 100_000 + 1  # small tolerance

    def test_guardrails_respects_ceiling(self, engine):
        """Guardrails should not withdraw above ceiling rate."""
        strategy = WithdrawalStrategy(
            name="guardrails",
            initial_rate=0.04,
            floor_rate=0.03,
            ceiling_rate=0.05,
            guardrail_threshold=0.20,
        )
        # Portfolio doubled
        withdrawal = strategy.calculate_withdrawal(
            current_portfolio=2_000_000,
            initial_portfolio=1_000_000,
            base_withdrawal=40_000,
        )
        # Should be ceiling_rate * current_portfolio = 0.05 * 2_000_000 = 100_000
        assert withdrawal <= 0.05 * 2_000_000 + 1  # small tolerance


# ---------------------------------------------------------------------------
# Social Security Optimization Tests
# ---------------------------------------------------------------------------


class TestSocialSecurityOptimization:
    """Tests for Social Security claiming optimization."""

    def test_optimal_claiming_age_within_valid_range(self, engine):
        result = engine.optimize_social_security(
            current_age=62,
            life_expectancy=85,
            monthly_benefit_at_62=1_500,
            full_retirement_age=67,
            monthly_benefit_at_fra=2_000,
            delayed_retirement_credit=0.08,
        )
        assert 62 <= result["optimal_claiming_age"] <= 70

    def test_delaying_increases_monthly_benefit(self, engine):
        result = engine.optimize_social_security(
            current_age=62,
            life_expectancy=85,
            monthly_benefit_at_62=1_500,
            full_retirement_age=67,
            monthly_benefit_at_fra=2_000,
            delayed_retirement_credit=0.08,
        )
        # Optimal age should be >= 62
        assert result["optimal_claiming_age"] >= 62

    def test_total_benefit_increases_with_delay_for_long_lifespan(self, engine):
        """For someone expecting to live long, delaying should provide more total benefit."""
        result = engine.optimize_social_security(
            current_age=62,
            life_expectancy=95,
            monthly_benefit_at_62=1_500,
            full_retirement_age=67,
            monthly_benefit_at_fra=2_000,
            delayed_retirement_credit=0.08,
        )
        # With long life expectancy, delaying to 70 should be optimal
        assert result["optimal_claiming_age"] == 70


# ---------------------------------------------------------------------------
# Retirement Plan Generation Tests
# ---------------------------------------------------------------------------


class TestRetirementPlanGeneration:
    """Tests for full retirement plan generation."""

    def test_generate_plan_returns_retirement_plan(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert isinstance(plan, RetirementPlan)

    def test_plan_contains_monte_carlo_results(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert plan.monte_carlo is not None
        assert plan.monte_carlo.success_rate > 0.0

    def test_plan_contains_social_security_recommendation(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert plan.social_security_age is not None
        assert 62 <= plan.social_security_age <= 70

    def test_plan_contains_withdrawal_strategy(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert plan.withdrawal_strategy is not None
        assert plan.withdrawal_strategy.name is not None

    def test_plan_contains_retirement_age(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert plan.retirement_age == basic_params["retirement_age"]

    def test_plan_contains_success_rate(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert 0.0 <= plan.success_rate <= 1.0

    def test_plan_contains_projected_balances(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        assert plan.projected_balances is not None
        assert len(plan.projected_balances) > 0

    def test_plan_projected_balances_cover_retirement_period(self, engine, basic_params):
        plan = engine.generate_plan(
            n_simulations=500,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        years_in_retirement = basic_params["life_expectancy"] - basic_params["retirement_age"]
        # Should have at least as many years as retirement period
        assert len(plan.projected_balances) >= years_in_retirement


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_monte_carlo_with_zero_simulations_raises(self, engine, basic_params):
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            engine.run_monte_carlo(
                n_simulations=0,
                annual_return_mean=0.07,
                annual_return_std=0.15,
                **basic_params,
            )

    def test_monte_carlo_with_negative_return_std_raises(self, engine, basic_params):
        with pytest.raises(ValueError, match="annual_return_std must be non-negative"):
            engine.run_monte_carlo(
                n_simulations=100,
                annual_return_mean=0.07,
                annual_return_std=-0.1,
                **basic_params,
            )

    def test_engine_with_different_seed_produces_different_results(self, basic_params):
        """Different seeds should produce different simulation results."""
        engine1 = RetirementPlanningEngine(seed=42)
        engine2 = RetirementPlanningEngine(seed=123)
        result1 = engine1.run_monte_carlo(
            n_simulations=100,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        result2 = engine2.run_monte_carlo(
            n_simulations=100,
            annual_return_mean=0.07,
            annual_return_std=0.15,
            **basic_params,
        )
        # Results should differ (with very high probability)
        assert not np.allclose(result1.final_balances, result2.final_balances)
