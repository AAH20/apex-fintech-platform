"""Tests for the Wealth Management Engine."""
import pytest
import numpy as np

from src.wealth.management import WealthManagementEngine, ClientProfile, RiskTolerance


@pytest.fixture
def engine():
    """Create a fresh engine instance for each test."""
    return WealthManagementEngine()


@pytest.fixture
def conservative_client():
    return ClientProfile(
        age=65,
        annual_income=80_000,
        net_worth=1_500_000,
        risk_tolerance=RiskTolerance.CONSERVATIVE,
        investment_horizon_years=10,
        liquidity_needs=50_000,
    )


@pytest.fixture
def moderate_client():
    return ClientProfile(
        age=40,
        annual_income=150_000,
        net_worth=500_000,
        risk_tolerance=RiskTolerance.MODERATE,
        investment_horizon_years=20,
        liquidity_needs=30_000,
    )


@pytest.fixture
def aggressive_client():
    return ClientProfile(
        age=30,
        annual_income=200_000,
        net_worth=200_000,
        risk_tolerance=RiskTolerance.AGGRESSIVE,
        investment_horizon_years=30,
        liquidity_needs=10_000,
    )


@pytest.fixture
def sample_returns():
    """Sample annual returns for 5 asset classes over 10 years."""
    return np.array([
        [0.07, 0.12, -0.05, 0.15, 0.08, 0.10, -0.02, 0.13, 0.09, 0.11],   # US Equity
        [0.05, 0.08, 0.03, 0.09, 0.06, 0.07, 0.04, 0.08, 0.05, 0.06],   # Intl Equity
        [0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03],   # Bonds
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],   # Cash
        [0.06, 0.10, -0.08, 0.18, 0.07, 0.12, -0.04, 0.15, 0.08, 0.10], # REITs
    ])


@pytest.fixture
def sample_covariance():
    """Sample covariance matrix for 5 asset classes."""
    return np.array([
        [0.0225, 0.0120, 0.0015, 0.0001, 0.0150],
        [0.0120, 0.0289, 0.0018, 0.0001, 0.0180],
        [0.0015, 0.0018, 0.0009, 0.0002, 0.0020],
        [0.0001, 0.0001, 0.0002, 0.0001, 0.0001],
        [0.0150, 0.0180, 0.0020, 0.0001, 0.0400],
    ])


class TestClientProfiling:
    """Test client profiling functionality."""

    def test_risk_score_conservative(self, engine, conservative_client):
        profile = engine.assess_client_profile(conservative_client)
        assert profile["risk_score"] <= 30

    def test_risk_score_aggressive(self, engine, aggressive_client):
        profile = engine.assess_client_profile(aggressive_client)
        assert profile["risk_score"] >= 70

    def test_risk_score_moderate(self, engine, moderate_client):
        profile = engine.assess_client_profile(moderate_client)
        assert 30 < profile["risk_score"] < 70

    def test_horizon_adjusts_risk(self, engine):
        young = ClientProfile(age=25, annual_income=100_000, net_worth=50_000,
                             risk_tolerance=RiskTolerance.MODERATE,
                             investment_horizon_years=35, liquidity_needs=5_000)
        old = ClientProfile(age=60, annual_income=100_000, net_worth=50_000,
                           risk_tolerance=RiskTolerance.MODERATE,
                           investment_horizon_years=5, liquidity_needs=5_000)
        young_profile = engine.assess_client_profile(young)
        old_profile = engine.assess_client_profile(old)
        assert young_profile["risk_score"] > old_profile["risk_score"]

    def test_liquidity_needs_factor(self, engine):
        high_liq = ClientProfile(age=40, annual_income=150_000, net_worth=500_000,
                                risk_tolerance=RiskTolerance.MODERATE,
                                investment_horizon_years=20, liquidity_needs=200_000)
        low_liq = ClientProfile(age=40, annual_income=150_000, net_worth=500_000,
                               risk_tolerance=RiskTolerance.MODERATE,
                               investment_horizon_years=20, liquidity_needs=5_000)
        high_profile = engine.assess_client_profile(high_liq)
        low_profile = engine.assess_client_profile(low_liq)
        assert high_profile["risk_score"] < low_profile["risk_score"]


class TestPortfolioConstruction:
    """Test portfolio construction functionality."""

    def test_asset_allocation_conservative(self, engine, conservative_client):
        allocation = engine.construct_portfolio(conservative_client)
        assert allocation["us_equity"] <= 0.30
        assert allocation["bonds"] >= 0.40
        assert allocation["cash"] >= 0.10

    def test_asset_allocation_aggressive(self, engine, aggressive_client):
        allocation = engine.construct_portfolio(aggressive_client)
        assert allocation["us_equity"] >= 0.40
        assert allocation["intl_equity"] >= 0.15
        assert allocation["bonds"] <= 0.20

    def test_allocation_sums_to_one(self, engine, moderate_client):
        allocation = engine.construct_portfolio(moderate_client)
        assert abs(sum(allocation.values()) - 1.0) < 1e-6

    def test_allocation_all_non_negative(self, engine, moderate_client):
        allocation = engine.construct_portfolio(moderate_client)
        assert all(v >= 0 for v in allocation.values())

    def test_mean_variance_optimization(self, engine, sample_returns, sample_covariance):
        result = engine.optimize_portfolio(sample_returns, sample_covariance,
                                           target_return=0.08)
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        assert all(w >= 0 for w in result["weights"])

    def test_efficient_frontier(self, engine, sample_returns, sample_covariance):
        frontier = engine.compute_efficient_frontier(sample_returns, sample_covariance,
                                                     n_points=5)
        assert len(frontier) == 5
        for point in frontier:
            assert "return" in point
            assert "volatility" in point
            assert "sharpe" in point


class TestTaxOptimization:
    """Test tax optimization functionality."""

    def test_tax_loss_harvesting_basic(self, engine):
        holdings = [
            {"symbol": "AAPL", "shares": 100, "cost_basis": 150.0, "current_price": 130.0},
            {"symbol": "GOOGL", "shares": 50, "cost_basis": 2000.0, "current_price": 2200.0},
        ]
        result = engine.identify_tax_losses(holdings)
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["unrealized_loss"] == -2000.0

    def test_tax_loss_harvesting_multiple(self, engine):
        holdings = [
            {"symbol": "AAPL", "shares": 100, "cost_basis": 150.0, "current_price": 130.0},
            {"symbol": "MSFT", "shares": 200, "cost_basis": 300.0, "current_price": 250.0},
            {"symbol": "GOOGL", "shares": 50, "cost_basis": 2000.0, "current_price": 2200.0},
        ]
        result = engine.identify_tax_losses(holdings)
        assert len(result) == 2
        symbols = {r["symbol"] for r in result}
        assert symbols == {"AAPL", "MSFT"}

    def test_asset_location_recommendation(self, engine, moderate_client):
        result = engine.recommend_asset_location(moderate_client)
        assert "taxable" in result
        assert "tax_deferred" in result
        assert "tax_free" in result
        assert len(result["taxable"]) > 0
        assert len(result["tax_deferred"]) > 0

    def test_tax_efficient_rebalancing(self, engine):
        current = {"us_equity": 0.70, "bonds": 0.20, "cash": 0.10}
        target = {"us_equity": 0.60, "bonds": 0.30, "cash": 0.10}
        result = engine.tax_efficient_rebalance(current, target)
        assert "trades" in result
        assert "estimated_tax_impact" in result
        assert isinstance(result["trades"], list)


class TestWealthPlanGeneration:
    """Test end-to-end wealth plan generation."""

    def test_generate_plan_returns_complete_structure(self, engine, moderate_client):
        plan = engine.generate_wealth_plan(moderate_client)
        assert "client_profile" in plan
        assert "portfolio" in plan
        assert "tax_strategy" in plan
        assert "projections" in plan
        assert "recommendations" in plan

    def test_plan_projections_include_retirement(self, engine, moderate_client):
        plan = engine.generate_wealth_plan(moderate_client)
        projections = plan["projections"]
        assert "retirement_age_value" in projections
        assert projections["retirement_age_value"] > 0

    def test_plan_recommendations_non_empty(self, engine, moderate_client):
        plan = engine.generate_wealth_plan(moderate_client)
        assert len(plan["recommendations"]) > 0

    def test_plan_includes_tax_loss_opportunities(self, engine, moderate_client):
        holdings = [
            {"symbol": "VTI", "shares": 500, "cost_basis": 200.0, "current_price": 180.0},
        ]
        plan = engine.generate_wealth_plan(moderate_client, holdings=holdings)
        tax_strategy = plan["tax_strategy"]
        assert "tax_loss_harvesting" in tax_strategy
        assert len(tax_strategy["tax_loss_harvesting"]) > 0


class TestRetirementProjection:
    """Test retirement projection calculations."""

    def test_future_value_calculation(self, engine):
        fv = engine.future_value(present_value=100_000, annual_return=0.07,
                                 years=20, annual_contribution=10_000)
        assert fv > 100_000
        assert fv > 500_000  # Should be substantial after 20 years

    def test_inflation_adjusted_projection(self, engine):
        nominal = engine.future_value(present_value=100_000, annual_return=0.07,
                                      years=20, annual_contribution=0)
        real = engine.future_value(present_value=100_000, annual_return=0.07,
                                   years=20, annual_contribution=0,
                                   inflation_rate=0.03)
        assert real < nominal

    def test_safe_withdrawal_rate(self, engine):
        withdrawal = engine.safe_withdrawal_amount(portfolio_value=1_000_000,
                                                   withdrawal_rate=0.04)
        assert withdrawal == 40_000


class TestRiskMetrics:
    """Test risk metric calculations."""

    def test_portfolio_volatility(self, engine, sample_returns, sample_covariance):
        weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
        vol = engine.portfolio_volatility(weights, sample_covariance)
        assert vol > 0
        assert vol < 0.5  # Reasonable range

    def test_sharpe_ratio(self, engine, sample_returns, sample_covariance):
        weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
        returns = np.mean(sample_returns, axis=1)
        sharpe = engine.sharpe_ratio(weights, returns, sample_covariance,
                                     risk_free_rate=0.02)
        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_max_drawdown(self, engine):
        values = np.array([100, 110, 105, 120, 90, 95, 130])
        mdd = engine.max_drawdown(values)
        assert mdd > 0
        assert mdd <= 1.0
