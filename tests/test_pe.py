"""
TDD tests for Private Equity Analytics Engine.
Covers: waterfall analysis, carried interest, IRR, MOIC, edge cases.
"""
import pytest
import numpy as np

from src.pe.analytics import PEAnalyticsEngine


@pytest.fixture
def engine():
    """Default PE engine: 2% mgmt fee, 20% carry, 8% hurdle, full catch-up."""
    return PEAnalyticsEngine(
        committed_capital=100_000_000,
        management_fee=0.02,
        carried_interest=0.20,
        hurdle_rate=0.08,
        catch_up=1.0,
    )


class TestWaterfall:
    def test_simple_waterfall(self, engine):
        """Simple waterfall: return of capital + preferred + 80/20 split."""
        result = engine.waterfall_distribution(
            total_proceeds=15_000_000,
            invested_capital=10_000_000,
            preferred_return=800_000,
        )
        assert result.lp_return_of_capital == pytest.approx(10_000_000)
        assert result.lp_preferred_return == pytest.approx(800_000)
        assert result.lp_total + result.gp_carry == pytest.approx(15_000_000)

    def test_hurdle_rate(self, engine):
        """Hurdle rate: preferred return paid before carry."""
        result = engine.waterfall_distribution(
            total_proceeds=10_800_000,
            invested_capital=10_000_000,
            preferred_return=800_000,
        )
        assert result.lp_preferred_return == pytest.approx(800_000)
        assert result.gp_carry == pytest.approx(0.0)

    def test_catch_up(self, engine):
        """Catch-up: GP gets 100% until they reach carry% of total profit."""
        result = engine.waterfall_distribution(
            total_proceeds=20_000_000,
            invested_capital=10_000_000,
            preferred_return=800_000,
        )
        # Total profit = 10M, GP target = 20% of 10M = 2M
        # After capital + preferred: 20M - 10M - 800k = 9.2M remaining
        # Catch-up = min(9.2M, 2M) = 2M
        # Remaining = 9.2M - 2M = 7.2M, split 80/20
        # GP total = 2M + 20% of 7.2M = 2M + 1.44M = 3.44M
        assert result.gp_carry == pytest.approx(3_440_000, rel=1e-4)

    def test_carried_interest(self, engine):
        """Carried interest: GP's share of profits."""
        carry = engine.calculate_carried_interest(
            total_proceeds=15_000_000,
            invested_capital=10_000_000,
            preferred_return=800_000,
        )
        assert carry == pytest.approx(1_640_000, rel=1e-4)


class TestIRR:
    def test_irr_calculation(self, engine):
        """IRR for -1000 now, +1100 in 1 year = 10%."""
        cash_flows = np.array([-1000, 1100])
        irr = engine.compute_irr(cash_flows)
        assert irr == pytest.approx(0.10, rel=1e-4)


class TestMOIC:
    def test_moic_calculation(self, engine):
        """MOIC = Total Distributions / Invested Capital."""
        moic = engine.compute_moic(
            total_distributions=25_000_000,
            invested_capital=10_000_000,
        )
        assert moic == pytest.approx(2.5)


class TestEdgeCases:
    def test_zero_loss(self, engine):
        """Zero loss: proceeds = invested capital."""
        result = engine.waterfall_distribution(
            total_proceeds=10_000_000,
            invested_capital=10_000_000,
            preferred_return=0.0,
        )
        assert result.lp_total == pytest.approx(10_000_000)
        assert result.gp_carry == pytest.approx(0.0)

    def test_total_loss(self, engine):
        """Total loss: proceeds = 0."""
        result = engine.waterfall_distribution(
            total_proceeds=0,
            invested_capital=10_000_000,
            preferred_return=0.0,
        )
        assert result.lp_total == pytest.approx(0.0)
        assert result.gp_carry == pytest.approx(0.0)
