"""Tests for the Real Estate Analytics Engine."""
import pytest
import pandas as pd
import numpy as np

from src.realestate.analytics import RealEstateEngine


@pytest.fixture
def engine():
    """Create a fresh engine instance for each test."""
    return RealEstateEngine()


@pytest.fixture
def sample_property():
    """Sample property data for valuation tests."""
    return {
        "address": "123 Main St",
        "city": "New York",
        "property_type": "multifamily",
        "units": 50,
        "sqft": 45000,
        "year_built": 2010,
        "gross_potential_rent": 1_200_000,
        "vacancy_rate": 0.05,
        "operating_expenses": 400_000,
        "cap_rate": 0.055,
    }


@pytest.fixture
def sample_reit_data():
    """Sample REIT financial data."""
    return {
        "ticker": "O",
        "price": 55.00,
        "shares_outstanding": 850_000_000,
        "net_income": 1_200_000_000,
        "depreciation": 800_000_000,
        "gains_on_sales": 50_000_000,
        "ffo": 1_950_000_000,
        "total_debt": 12_000_000_000,
        "cash": 500_000_000,
        "total_assets": 25_000_000_000,
        "total_liabilities": 14_000_000_000,
        "annual_dividend": 3.12,
        "property_count": 1200,
        "portfolio_noi": 1_800_000_000,
    }


@pytest.fixture
def sample_cash_flows():
    """Sample DCF cash flows (5 years + terminal)."""
    return [-5_000_000, 400_000, 420_000, 440_000, 460_000, 480_000]


class TestPropertyValuation:
    """Test property valuation methods."""

    def test_income_approach_valuation(self, engine, sample_property):
        """Value = NOI / Cap Rate."""
        noi = sample_property["gross_potential_rent"] * (1 - sample_property["vacancy_rate"]) - sample_property["operating_expenses"]
        value = engine.value_property_income_approach(noi, sample_property["cap_rate"])
        expected_noi = 1_200_000 * 0.95 - 400_000
        assert value == pytest.approx(expected_noi / 0.055, rel=0.01)

    def test_income_approach_zero_cap_rate_raises(self, engine):
        """Zero cap rate should raise ValueError."""
        with pytest.raises(ValueError, match="Cap rate must be positive"):
            engine.value_property_income_approach(100_000, 0)

    def test_income_approach_negative_noi(self, engine):
        """Negative NOI should still compute (distressed asset)."""
        value = engine.value_property_income_approach(-50_000, 0.08)
        assert value == pytest.approx(-625_000, rel=0.01)

    def test_dcf_valuation_basic(self, engine):
        """DCF with explicit terminal value."""
        cash_flows = [-1_000_000, 100_000, 110_000, 120_000, 130_000, 1_500_000]
        npv = engine.value_property_dcf(cash_flows, discount_rate=0.10)
        assert npv > 0
        assert isinstance(npv, float)

    def test_dcf_valuation_with_terminal_cap_rate(self, engine):
        """DCF using terminal cap rate on final year NOI."""
        cash_flows = [-2_000_000, 200_000, 210_000, 220_000, 230_000, 240_000]
        npv = engine.value_property_dcf(
            cash_flows, discount_rate=0.09, terminal_cap_rate=0.06
        )
        assert npv > 0

    def test_dcf_empty_cash_flows_raises(self, engine):
        """Empty cash flows should raise ValueError."""
        with pytest.raises(ValueError, match="Cash flows cannot be empty"):
            engine.value_property_dcf([], discount_rate=0.10)

    def test_dcf_negative_discount_rate_raises(self, engine):
        """Negative discount rate should raise ValueError."""
        with pytest.raises(ValueError, match="Discount rate must be non-negative"):
            engine.value_property_dcf([-100, 50, 60], discount_rate=-0.05)


class TestCapRateAnalysis:
    """Test cap rate calculations."""

    def test_going_in_cap_rate(self, engine):
        """Going-in cap rate = NOI / Purchase Price."""
        noi = 500_000
        price = 8_000_000
        cap = engine.calculate_going_in_cap_rate(noi, price)
        assert cap == pytest.approx(0.0625, rel=0.01)

    def test_exit_cap_rate(self, engine):
        """Exit cap rate = Year N NOI / Exit Price."""
        noi = 600_000
        exit_price = 10_000_000
        cap = engine.calculate_exit_cap_rate(noi, exit_price)
        assert cap == pytest.approx(0.06, rel=0.01)

    def test_implied_cap_rate_from_market_cap(self, engine):
        """Implied cap rate = NOI / (Market Cap + Debt)."""
        noi = 1_000_000
        market_cap = 15_000_000
        debt = 5_000_000
        cap = engine.calculate_implied_cap_rate(noi, market_cap, debt)
        assert cap == pytest.approx(0.05, rel=0.01)

    def test_cap_rate_zero_price_raises(self, engine):
        """Zero price should raise ValueError."""
        with pytest.raises(ValueError, match="Price must be positive"):
            engine.calculate_going_in_cap_rate(100_000, 0)

    def test_cap_rate_spread(self, engine):
        """Cap rate spread = Exit Cap - Going-in Cap."""
        spread = engine.calculate_cap_rate_spread(0.065, 0.055)
        assert spread == pytest.approx(0.01, rel=0.01)


class TestREITAnalysis:
    """Test REIT-specific analytics."""

    def test_calculate_ffo(self, engine):
        """FFO = Net Income + Depreciation - Gains on Sales."""
        ffo = engine.calculate_ffo(
            net_income=1_200_000_000,
            depreciation=800_000_000,
            gains_on_sales=50_000_000,
        )
        assert ffo == pytest.approx(1_950_000_000, rel=0.01)

    def test_calculate_affo(self, engine):
        """AFFO = FFO - Recurring Capex - Straight-line Rent."""
        affo = engine.calculate_affo(
            ffo=1_950_000_000,
            recurring_capex=100_000_000,
            straight_line_rent=30_000_000,
        )
        assert affo == pytest.approx(1_820_000_000, rel=0.01)

    def test_calculate_nav(self, engine):
        """NAV = Total Assets - Total Liabilities."""
        nav = engine.calculate_nav(
            total_assets=25_000_000_000,
            total_liabilities=14_000_000_000,
        )
        assert nav == pytest.approx(11_000_000_000, rel=0.01)

    def test_reit_metrics(self, engine, sample_reit_data):
        """Full REIT analysis returns key metrics."""
        metrics = engine.analyze_reit(**sample_reit_data)
        assert "ffo_per_share" in metrics
        assert "affo" in metrics
        assert "nav_per_share" in metrics
        assert "dividend_yield" in metrics
        assert "payout_ratio" in metrics
        assert "implied_cap_rate" in metrics
        assert "debt_to_ebitda" in metrics
        assert metrics["ffo_per_share"] > 0
        assert metrics["dividend_yield"] > 0

    def test_reit_dividend_yield(self, engine):
        """Dividend Yield = Annual Dividend / Price."""
        div_yield = engine.calculate_dividend_yield(3.12, 55.00)
        assert div_yield == pytest.approx(0.0567, rel=0.01)

    def test_reit_payout_ratio(self, engine):
        """Payout Ratio = Dividend / FFO per Share."""
        payout = engine.calculate_payout_ratio(3.12, 2.30)
        assert payout == pytest.approx(1.356, rel=0.01)

    def test_reit_price_to_ffo(self, engine):
        """P/FFO = Price / FFO per Share."""
        ratio = engine.calculate_price_to_ffo(55.00, 2.30)
        assert ratio == pytest.approx(23.91, rel=0.01)

    def test_reit_debt_to_assets(self, engine):
        """Debt / Total Assets."""
        ratio = engine.calculate_debt_to_assets(12_000_000_000, 25_000_000_000)
        assert ratio == pytest.approx(0.48, rel=0.01)


class TestInvestmentMetrics:
    """Test investment return metrics."""

    def test_calculate_irr(self, engine):
        """IRR of a simple cash flow stream."""
        cash_flows = [-1_000_000, 300_000, 300_000, 300_000, 300_000]
        irr = engine.calculate_irr(cash_flows)
        assert irr > 0
        assert irr < 1

    def test_calculate_equity_multiple(self, engine):
        """Equity Multiple = Total Distributions / Total Invested."""
        cash_flows = [-1_000_000, 200_000, 200_000, 200_000, 200_000, 200_000, 200_000]
        em = engine.calculate_equity_multiple(cash_flows)
        assert em == pytest.approx(1.2, rel=0.01)

    def test_calculate_loan_constant(self, engine):
        """Mortgage constant = Annual Debt Service / Loan Amount."""
        # 5% rate, 30-year amortization
        constant = engine.calculate_loan_constant(0.05, 30)
        assert constant > 0
        assert constant < 0.1

    def test_calculate_ltv(self, engine):
        """LTV = Loan Amount / Property Value."""
        ltv = engine.calculate_ltv(6_000_000, 8_000_000)
        assert ltv == pytest.approx(0.75, rel=0.01)

    def test_calculate_dscr(self, engine):
        """DSCR = NOI / Debt Service."""
        dscr = engine.calculate_dscr(500_000, 350_000)
        assert dscr == pytest.approx(1.428, rel=0.01)

    def test_calculate_dscr_insufficient(self, engine):
        """DSCR < 1 indicates insufficient coverage."""
        dscr = engine.calculate_dscr(300_000, 400_000)
        assert dscr < 1


class TestPortfolioAnalysis:
    """Test portfolio-level analytics."""

    def test_portfolio_summary(self, engine):
        """Portfolio summary aggregates multiple properties."""
        properties = [
            {"noi": 500_000, "value": 8_000_000, "type": "multifamily"},
            {"noi": 300_000, "value": 5_000_000, "type": "office"},
            {"noi": 200_000, "value": 3_000_000, "type": "retail"},
        ]
        summary = engine.portfolio_summary(properties)
        assert summary["total_value"] == pytest.approx(16_000_000, rel=0.01)
        assert summary["total_noi"] == pytest.approx(1_000_000, rel=0.01)
        assert summary["weighted_avg_cap_rate"] == pytest.approx(0.0625, rel=0.01)
        assert summary["property_count"] == 3

    def test_portfolio_by_type(self, engine):
        """Group portfolio metrics by property type."""
        properties = [
            {"noi": 500_000, "value": 8_000_000, "type": "multifamily"},
            {"noi": 300_000, "value": 5_000_000, "type": "office"},
            {"noi": 200_000, "value": 3_000_000, "type": "multifamily"},
        ]
        by_type = engine.portfolio_by_type(properties)
        assert "multifamily" in by_type
        assert "office" in by_type
        assert by_type["multifamily"]["count"] == 2
        assert by_type["office"]["count"] == 1


class TestDataFrameIntegration:
    """Test pandas DataFrame integration."""

    def test_properties_to_dataframe(self, engine):
        """Convert property list to DataFrame."""
        properties = [
            {"address": "123 Main", "noi": 500_000, "value": 8_000_000, "type": "multifamily"},
            {"address": "456 Oak", "noi": 300_000, "value": 5_000_000, "type": "office"},
        ]
        df = engine.properties_to_dataframe(properties)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "cap_rate" in df.columns
        assert df["cap_rate"].iloc[0] == pytest.approx(0.0625, rel=0.01)

    def test_reit_to_dataframe(self, engine):
        """Convert REIT data to DataFrame."""
        reits = [
            {"ticker": "O", "price": 55.0, "ffo": 1_950_000_000, "dividend": 3.12, "shares_outstanding": 850_000_000},
            {"ticker": "SPG", "price": 120.0, "ffo": 2_500_000_000, "dividend": 8.00, "shares_outstanding": 300_000_000},
        ]
        df = engine.reits_to_dataframe(reits)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "dividend_yield" in df.columns
        assert "payout_ratio" in df.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_noi_calculation_with_zero_vacancy(self, engine):
        """NOI with zero vacancy rate."""
        noi = engine.calculate_noi(1_000_000, 0.0, 300_000)
        assert noi == pytest.approx(700_000, rel=0.01)

    def test_noi_calculation_with_full_vacancy(self, engine):
        """NOI with 100% vacancy."""
        noi = engine.calculate_noi(1_000_000, 1.0, 300_000)
        assert noi == pytest.approx(-300_000, rel=0.01)

    def test_noi_calculation_zero_gpr(self, engine):
        """NOI with zero gross potential rent."""
        noi = engine.calculate_noi(0, 0.05, 100_000)
        assert noi == pytest.approx(-100_000, rel=0.01)

    def test_irr_all_negative_cash_flows(self, engine):
        """IRR with all negative cash flows should return NaN or raise."""
        cash_flows = [-100, -50, -30]
        irr = engine.calculate_irr(cash_flows)
        assert np.isnan(irr) or irr < 0

    def test_equity_multiple_zero_investment(self, engine):
        """Equity multiple with zero initial investment."""
        cash_flows = [0, 100, 200, 300]
        em = engine.calculate_equity_multiple(cash_flows)
        assert np.isinf(em) or em > 0
