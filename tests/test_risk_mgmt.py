"""Tests for risk management engine — VaR, CVaR, stress testing, scenario analysis."""

import numpy as np
import pytest

from risk.management import RiskManagementEngine, StressScenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return RiskManagementEngine(confidence_level=0.95, time_horizon=1)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, 252)


@pytest.fixture
def sample_portfolio():
    return {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}


# ---------------------------------------------------------------------------
# VaR tests
# ---------------------------------------------------------------------------


class TestVaR:
    def test_historical_var_positive(self, engine, sample_returns):
        var = engine.historical_var(sample_returns)
        assert var > 0

    def test_parametric_var_positive(self, engine, sample_returns):
        var = engine.parametric_var(sample_returns)
        assert var > 0

    def test_monte_carlo_var_positive(self, engine, sample_returns):
        var = engine.monte_carlo_var(sample_returns, n_sims=1000)
        assert var > 0

    def test_var_increases_with_confidence(self, sample_returns):
        e95 = RiskManagementEngine(confidence_level=0.95)
        e99 = RiskManagementEngine(confidence_level=0.99)
        assert e99.historical_var(sample_returns) >= e95.historical_var(sample_returns)

    def test_var_scales_with_horizon(self, sample_returns):
        e1 = RiskManagementEngine(confidence_level=0.95, time_horizon=1)
        e10 = RiskManagementEngine(confidence_level=0.95, time_horizon=10)
        assert e10.historical_var(sample_returns) > e1.historical_var(sample_returns)


# ---------------------------------------------------------------------------
# CVaR tests
# ---------------------------------------------------------------------------


class TestCVaR:
    def test_historical_cvar_positive(self, engine, sample_returns):
        cvar = engine.historical_cvar(sample_returns)
        assert cvar > 0

    def test_cvar_greater_than_var(self, engine, sample_returns):
        var = engine.historical_var(sample_returns)
        cvar = engine.historical_cvar(sample_returns)
        assert cvar >= var

    def test_parametric_cvar_positive(self, engine, sample_returns):
        cvar = engine.parametric_cvar(sample_returns)
        assert cvar > 0


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------


class TestStressTesting:
    def test_stress_test_returns_loss(self, engine, sample_portfolio):
        result = engine.stress_test(sample_portfolio, shock=-0.10)
        assert result["portfolio_loss"] > 0

    def test_stress_test_with_correlation(self, engine):
        portfolio = {"AAPL": 0.5, "MSFT": 0.5}
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        result = engine.stress_test(portfolio, shock=-0.15, correlation=corr)
        assert result["portfolio_loss"] > 0

    def test_stress_test_severe_shock(self, engine, sample_portfolio):
        mild = engine.stress_test(sample_portfolio, shock=-0.05)
        severe = engine.stress_test(sample_portfolio, shock=-0.30)
        assert severe["portfolio_loss"] > mild["portfolio_loss"]


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------


class TestScenarioAnalysis:
    def test_scenario_analysis_returns_results(self, engine, sample_portfolio):
        scenarios = [
            StressScenario(name="recession", shocks={"AAPL": -0.20, "GOOGL": -0.15, "MSFT": -0.10}),
            StressScenario(name="recovery", shocks={"AAPL": 0.10, "GOOGL": 0.08, "MSFT": 0.12}),
        ]
        results = engine.scenario_analysis(sample_portfolio, scenarios)
        assert len(results) == 2
        assert results[0]["portfolio_loss"] > 0
        assert results[1]["portfolio_loss"] < 0

    def test_scenario_analysis_ranks_by_loss(self, engine, sample_portfolio):
        scenarios = [
            StressScenario(name="mild", shocks={"AAPL": -0.05, "GOOGL": -0.05, "MSFT": -0.05}),
            StressScenario(name="severe", shocks={"AAPL": -0.30, "GOOGL": -0.30, "MSFT": -0.30}),
        ]
        results = engine.scenario_analysis(sample_portfolio, scenarios)
        losses = [r["portfolio_loss"] for r in results]
        assert losses == sorted(losses, reverse=True)


# ---------------------------------------------------------------------------
# Capital adequacy
# ---------------------------------------------------------------------------


class TestCapitalAdequacy:
    def test_capital_adequacy_ratio(self, engine):
        ratio = engine.capital_adequacy_ratio(capital=1000000, rwa=800000)
        assert ratio == pytest.approx(1.25)

    def test_capital_adequacy_below_minimum(self, engine):
        ratio = engine.capital_adequacy_ratio(capital=50000, rwa=800000)
        assert ratio < 0.08

    def test_capital_adequacy_above_minimum(self, engine):
        ratio = engine.capital_adequacy_ratio(capital=2000000, rwa=800000)
        assert ratio > 0.08


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_returns_raises(self, engine):
        with pytest.raises(ValueError):
            engine.historical_var(np.array([]))

    def test_single_asset_portfolio(self, engine):
        result = engine.stress_test({"AAPL": 1.0}, shock=-0.10)
        assert result["portfolio_loss"] > 0

    def test_zero_shock_no_loss(self, engine, sample_portfolio):
        result = engine.stress_test(sample_portfolio, shock=0.0)
        assert result["portfolio_loss"] == pytest.approx(0.0)
