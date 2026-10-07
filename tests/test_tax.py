"""Tests for tax optimization engine — HNW tax strategies.

Covers tax-loss harvesting, estate planning, and charitable giving.
"""

import pytest

from tax.optimization import (
    AssetLot,
    CharitableGivingStrategy,
    EstatePlan,
    TaxBracket,
    TaxOptimizationEngine,
)


# ---------------------------------------------------------------------------
# Tax Brackets
# ---------------------------------------------------------------------------


class TestTaxBrackets:
    """Tests for tax bracket calculations."""

    def test_single_filer_2024_brackets(self):
        """2024 single filer brackets produce correct marginal rates."""
        brackets = TaxBracket.single_2024()
        assert len(brackets) == 7
        assert brackets[0].rate == pytest.approx(0.10)
        assert brackets[0].lower == 0
        assert brackets[0].upper == pytest.approx(11_600)
        assert brackets[-1].rate == pytest.approx(0.37)
        assert brackets[-1].lower == pytest.approx(609_350)

    def test_married_filing_jointly_2024_brackets(self):
        """2024 MFJ brackets are wider than single."""
        brackets = TaxBracket.married_filing_jointly_2024()
        assert len(brackets) == 7
        assert brackets[0].rate == pytest.approx(0.10)
        assert brackets[0].upper == pytest.approx(23_200)
        assert brackets[-1].rate == pytest.approx(0.37)

    def test_tax_owed_progressive(self):
        """Tax is computed progressively across brackets."""
        brackets = TaxBracket.single_2024()
        # $50,000 income: 10% on first 11,600 + 12% on next 35,550 + 22% on remainder
        tax = TaxBracket.compute_tax(50_000, brackets)
        expected = 0.10 * 11_600 + 0.12 * (47_150 - 11_600) + 0.22 * (50_000 - 47_150)
        assert tax == pytest.approx(expected)

    def test_tax_owed_zero_income(self):
        """Zero income produces zero tax."""
        brackets = TaxBracket.single_2024()
        assert TaxBracket.compute_tax(0, brackets) == 0.0

    def test_tax_owed_top_bracket(self):
        """Income in top bracket taxed correctly."""
        brackets = TaxBracket.single_2024()
        tax = TaxBracket.compute_tax(700_000, brackets)
        assert tax > 0
        # Verify it's progressive (not flat 37%)
        assert tax < 0.37 * 700_000


# ---------------------------------------------------------------------------
# Tax-Loss Harvesting
# ---------------------------------------------------------------------------


class TestTaxLossHarvesting:
    """Tests for tax-loss harvesting strategies."""

    def test_identify_loss_lots(self):
        """Engine identifies lots with unrealized losses."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        lots = [
            AssetLot("AAPL", 100, 150.0, 180.0),  # $3,000 gain
            AssetLot("TSLA", 50, 300.0, 250.0),   # $2,500 loss
            AssetLot("MSFT", 200, 200.0, 210.0),  # $2,000 gain
            AssetLot("NVDA", 30, 500.0, 400.0),   # $3,000 loss
        ]
        loss_lots = engine.identify_loss_lots(lots)
        assert len(loss_lots) == 2
        symbols = {lot.symbol for lot in loss_lots}
        assert symbols == {"TSLA", "NVDA"}

    def test_calculate_tax_savings_from_harvest(self):
        """Harvesting losses saves taxes at ordinary income rate."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        lots = [
            AssetLot("TSLA", 100, 300.0, 250.0),  # $5,000 loss
        ]
        savings = engine.calculate_harvest_savings(lots, ordinary_income_rate=0.32)
        assert savings == pytest.approx(5_000 * 0.32)

    def test_wash_sale_detection(self):
        """Wash sale rule detected when repurchasing within 30 days."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        # Sold TSLA at loss, bought back 15 days later
        engine.record_sale("TSLA", 100, 250.0, 300.0, days_ago=15)
        is_wash = engine.is_wash_sale("TSLA", days_after_sale=15)
        assert is_wash is True

    def test_no_wash_sale_after_30_days(self):
        """No wash sale if repurchased after 30 days."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        engine.record_sale("TSLA", 100, 250.0, 300.0, days_ago=31)
        is_wash = engine.is_wash_sale("TSLA", days_after_sale=31)
        assert is_wash is False

    def test_net_gains_losses(self):
        """Net capital gains computed correctly."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        lots = [
            AssetLot("AAPL", 100, 150.0, 180.0),  # $3,000 gain
            AssetLot("TSLA", 50, 300.0, 250.0),   # $2,500 loss
        ]
        net = engine.net_capital_gains(lots)
        assert net == pytest.approx(500.0)

    def test_harvest_recommendation(self):
        """Engine recommends harvesting when losses exceed threshold."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        lots = [
            AssetLot("TSLA", 100, 300.0, 200.0),  # $10,000 loss
        ]
        rec = engine.recommend_harvest(lots, min_loss_threshold=1_000)
        assert rec.should_harvest is True
        assert rec.estimated_savings > 0
        assert rec.lots_to_harvest[0].symbol == "TSLA"


# ---------------------------------------------------------------------------
# Estate Planning
# ---------------------------------------------------------------------------


class TestEstatePlanning:
    """Tests for estate planning and estate tax calculations."""

    def test_estate_tax_below_exemption(self):
        """No estate tax when estate is below exemption."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        plan = EstatePlan(gross_estate=5_000_000, exemption=13_610_000)
        tax = engine.compute_estate_tax(plan)
        assert tax == 0.0

    def test_estate_tax_above_exemption(self):
        """Estate tax computed at 40% on amount above exemption."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        plan = EstatePlan(gross_estate=20_000_000, exemption=13_610_000)
        tax = engine.compute_estate_tax(plan)
        assert tax == pytest.approx((20_000_000 - 13_610_000) * 0.40)

    def test_annual_gift_exclusion(self):
        """Annual gift exclusion reduces taxable estate."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        plan = EstatePlan(
            gross_estate=15_000_000,
            exemption=13_610_000,
            annual_gifts=18_000 * 10,  # $18k/year for 10 years
        )
        tax = engine.compute_estate_tax(plan)
        taxable = 15_000_000 - 13_610_000 - 180_000
        assert tax == pytest.approx(max(0, taxable) * 0.40)

    def test_marital_deduction(self):
        """Unlimited marital deduction eliminates tax for surviving spouse."""
        engine = TaxOptimizationEngine(filing_status="married_jointly", state_ca=False)
        plan = EstatePlan(
            gross_estate=50_000_000,
            exemption=27_220_000,
            marital_deduction=True,
        )
        tax = engine.compute_estate_tax(plan)
        assert tax == 0.0

    def test_estate_tax_with_charitable_deduction(self):
        """Charitable deduction reduces taxable estate."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        plan = EstatePlan(
            gross_estate=20_000_000,
            exemption=13_610_000,
            charitable_deduction=2_000_000,
        )
        tax = engine.compute_estate_tax(plan)
        taxable = 20_000_000 - 13_610_000 - 2_000_000
        assert tax == pytest.approx(taxable * 0.40)


# ---------------------------------------------------------------------------
# Charitable Giving
# ---------------------------------------------------------------------------


class TestCharitableGiving:
    """Tests for charitable giving optimization."""

    def test_cash_donation_deduction(self):
        """Cash donations deductible up to 60% of AGI."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        strategy = CharitableGivingStrategy(
            agi=500_000,
            cash_donations=50_000,
        )
        deduction = engine.compute_charitable_deduction(strategy)
        assert deduction == pytest.approx(50_000)

    def test_appreciated_asset_donation(self):
        """Donating appreciated assets avoids capital gains tax."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        strategy = CharitableGivingStrategy(
            agi=500_000,
            appreciated_asset_donations=100_000,
            asset_cost_basis=20_000,
        )
        deduction = engine.compute_charitable_deduction(strategy)
        # Deduction is at fair market value
        assert deduction == pytest.approx(100_000)

    def test_appreciated_asset_tax_savings(self):
        """Donating appreciated assets saves capital gains tax."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        strategy = CharitableGivingStrategy(
            agi=500_000,
            appreciated_asset_donations=100_000,
            asset_cost_basis=20_000,
        )
        savings = engine.compute_charitable_tax_savings(strategy, ltcg_rate=0.20)
        # Savings = income tax deduction + avoided capital gains tax
        # Income tax: $100,000 deduction * 35% marginal rate = $35,000
        # Avoided LTCG: $80,000 gain * 20% = $16,000
        # Total: $51,000
        assert savings == pytest.approx(100_000 * 0.35 + 80_000 * 0.20)

    def test_donor_advised_fund_recommendation(self):
        """DAF recommended for lumpy income years."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        strategy = CharitableGivingStrategy(
            agi=2_000_000,  # High income year
            intended_annual_giving=50_000,
        )
        rec = engine.recommend_charitable_strategy(strategy)
        assert rec["use_donor_advised_fund"] is True
        assert rec["bunching_recommended"] is True

    def test_charitable_deduction_capped(self):
        """Cash deduction capped at 60% of AGI."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        strategy = CharitableGivingStrategy(
            agi=100_000,
            cash_donations=80_000,  # Exceeds 60% cap
        )
        deduction = engine.compute_charitable_deduction(strategy)
        assert deduction == pytest.approx(60_000)


# ---------------------------------------------------------------------------
# Tax Optimization Engine
# ---------------------------------------------------------------------------


class TestTaxOptimizationEngine:
    """Tests for the main optimization engine."""

    def test_engine_initialization(self):
        """Engine initializes with correct parameters."""
        engine = TaxOptimizationEngine(
            filing_status="single",
            state_ca=True,
            ordinary_income=500_000,
        )
        assert engine.filing_status == "single"
        assert engine.state_ca is True
        assert engine.ordinary_income == 500_000

    def test_optimize_returns_strategies(self):
        """Engine returns optimization strategies."""
        engine = TaxOptimizationEngine(
            filing_status="single",
            state_ca=True,
            ordinary_income=500_000,
        )
        result = engine.optimize()
        assert isinstance(result, dict)
        assert "tax_loss_harvesting" in result
        assert "estate_planning" in result
        assert "charitable_giving" in result

    def test_roth_conversion_analysis(self):
        """Roth conversion analysis computes tax cost."""
        engine = TaxOptimizationEngine(
            filing_status="single",
            state_ca=True,
            ordinary_income=200_000,
        )
        result = engine.analyze_roth_conversion(conversion_amount=50_000)
        assert result["conversion_amount"] == 50_000
        assert result["additional_tax"] > 0
        assert result["marginal_rate"] > 0

    def test_income_shifting_recommendation(self):
        """Engine recommends income shifting for high earners."""
        engine = TaxOptimizationEngine(
            filing_status="married_jointly",
            state_ca=False,
            ordinary_income=800_000,
        )
        result = engine.recommend_income_shifting()
        assert result["recommended"] is True
        assert result["potential_savings"] > 0

    def test_tax_bracket_optimization(self):
        """Engine identifies bracket optimization opportunities."""
        engine = TaxOptimizationEngine(
            filing_status="single",
            state_ca=True,
            ordinary_income=110_000,  # In 24% bracket
        )
        result = engine.optimize_bracket_position(target_bracket=0.22)
        assert result["current_bracket"] == pytest.approx(0.24)
        assert result["recommendation"] is not None

    def test_capital_gains_tax_computation(self):
        """Long-term capital gains taxed at preferential rates."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        tax = engine.compute_capital_gains_tax(
            gains=100_000,
            holding_period_years=2,
            ordinary_income=200_000,
        )
        assert tax > 0
        # LTCG rate should be 15% or 20%, not ordinary rates
        assert tax < 0.25 * 100_000

    def test_short_term_capital_gains_as_ordinary(self):
        """Short-term gains taxed as ordinary income."""
        engine = TaxOptimizationEngine(filing_status="single", state_ca=True)
        st_tax = engine.compute_capital_gains_tax(
            gains=100_000,
            holding_period_years=0.5,
            ordinary_income=200_000,
        )
        ltcg_tax = engine.compute_capital_gains_tax(
            gains=100_000,
            holding_period_years=2,
            ordinary_income=200_000,
        )
        assert st_tax > ltcg_tax

    def test_comprehensive_optimization(self):
        """Full optimization produces actionable recommendations."""
        engine = TaxOptimizationEngine(
            filing_status="married_jointly",
            state_ca=True,
            ordinary_income=1_000_000,
        )
        lots = [
            AssetLot("AAPL", 100, 150.0, 200.0),   # $5,000 gain
            AssetLot("TSLA", 200, 300.0, 250.0),   # $10,000 loss
            AssetLot("MSFT", 50, 200.0, 250.0),    # $2,500 gain
        ]
        estate = EstatePlan(
            gross_estate=30_000_000,
            exemption=27_220_000,
        )
        charitable = CharitableGivingStrategy(
            agi=1_000_000,
            cash_donations=100_000,
            appreciated_asset_donations=50_000,
            asset_cost_basis=10_000,
        )
        result = engine.comprehensive_optimize(lots, estate, charitable)
        assert "total_tax_savings" in result
        assert "strategies" in result
        assert result["total_tax_savings"] > 0
        assert len(result["strategies"]) >= 3

    def test_numpy_portfolio_tax_optimization(self):
        """Numpy-based portfolio tax optimization works correctly."""
        engine = TaxOptimizationEngine(
            filing_status="single",
            state_ca=True,
            ordinary_income=500_000,
        )
        lots = [
            AssetLot("AAPL", 100, 150.0, 200.0),   # $5,000 gain
            AssetLot("TSLA", 200, 300.0, 250.0),   # $10,000 loss
            AssetLot("MSFT", 50, 200.0, 250.0),    # $2,500 gain
        ]
        result = engine.optimize_portfolio_tax(lots)
        assert result["tax_drag"] > 0
        assert result["total_harvestable_loss"] == pytest.approx(10_000)
        assert "TSLA" in result["loss_lots"]
        assert result["portfolio_value"] == pytest.approx(82_500)
        assert len(result["recommendations"]) > 0
