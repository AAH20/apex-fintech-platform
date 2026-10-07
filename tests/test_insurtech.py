"""Tests for Insurtech analytics engine."""
import math

import numpy as np
import pytest

from src.insurtech.analytics import (
    InsurtechEngine,
    MortalityTable,
    Policy,
    ReserveResult,
    RiskPoolResult,
)


# ---------------------------------------------------------------------------
# Mortality Table Tests
# ---------------------------------------------------------------------------


class TestMortalityTable:
    """Test mortality table construction and queries."""

    def test_mortality_table_basic_construction(self):
        ages = np.array([0, 1, 2, 3, 4])
        q_x = np.array([0.01, 0.005, 0.003, 0.002, 0.001])
        mt = MortalityTable(ages=ages, q_x=q_x)
        assert mt.ages is not None
        assert mt.q_x is not None
        assert len(mt.ages) == 5

    def test_mortality_table_q_x_bounds(self):
        ages = np.array([0, 1, 2])
        q_x = np.array([0.5, 0.3, 0.1])
        mt = MortalityTable(ages=ages, q_x=q_x)
        assert np.all(mt.q_x >= 0)
        assert np.all(mt.q_x <= 1)

    def test_mortality_table_invalid_q_x_raises(self):
        ages = np.array([0, 1, 2])
        q_x = np.array([0.5, 1.5, 0.1])
        with pytest.raises(ValueError, match="q_x must be between 0 and 1"):
            MortalityTable(ages=ages, q_x=q_x)

    def test_mortality_table_negative_q_x_raises(self):
        ages = np.array([0, 1, 2])
        q_x = np.array([0.5, -0.1, 0.1])
        with pytest.raises(ValueError, match="q_x must be between 0 and 1"):
            MortalityTable(ages=ages, q_x=q_x)

    def test_mortality_table_length_mismatch_raises(self):
        ages = np.array([0, 1, 2])
        q_x = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="ages and q_x must have same length"):
            MortalityTable(ages=ages, q_x=q_x)

    def test_mortality_table_survival_probability(self):
        ages = np.array([0, 1, 2, 3])
        q_x = np.array([0.1, 0.1, 0.1, 0.1])
        mt = MortalityTable(ages=ages, q_x=q_x)
        # p_x = 1 - q_x = 0.9 for each year
        # 2-year survival from age 0: 0.9 * 0.9 = 0.81
        p2 = mt.survival_probability(age=0, years=2)
        assert math.isclose(p2, 0.81, rel_tol=1e-9)

    def test_mortality_table_one_year_survival(self):
        ages = np.array([0, 1, 2])
        q_x = np.array([0.05, 0.03, 0.02])
        mt = MortalityTable(ages=ages, q_x=q_x)
        p1 = mt.survival_probability(age=0, years=1)
        assert math.isclose(p1, 0.95, rel_tol=1e-9)

    def test_mortality_table_death_probability(self):
        ages = np.array([0, 1, 2])
        q_x = np.array([0.1, 0.2, 0.3])
        mt = MortalityTable(ages=ages, q_x=q_x)
        # Probability of dying within 2 years from age 0:
        # q_0 + p_0 * q_1 = 0.1 + 0.9 * 0.2 = 0.28
        d2 = mt.death_probability(age=0, years=2)
        assert math.isclose(d2, 0.28, rel_tol=1e-9)

    def test_mortality_table_interpolates_age(self):
        ages = np.array([0, 10, 20])
        q_x = np.array([0.01, 0.02, 0.03])
        mt = MortalityTable(ages=ages, q_x=q_x)
        # Interpolated q at age 5 should be 0.015
        q5 = mt.q_at_age(5)
        assert math.isclose(q5, 0.015, rel_tol=1e-9)

    def test_mortality_table_q_at_exact_age(self):
        ages = np.array([0, 10, 20])
        q_x = np.array([0.01, 0.02, 0.03])
        mt = MortalityTable(ages=ages, q_x=q_x)
        assert math.isclose(mt.q_at_age(10), 0.02, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Premium Calculation Tests
# ---------------------------------------------------------------------------


class TestPremiumCalculation:
    """Test premium calculation methods."""

    def setup_method(self):
        ages = np.array([25, 26, 27, 28, 29, 30])
        q_x = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        self.mt = MortalityTable(ages=ages, q_x=q_x)
        self.engine = InsurtechEngine(mortality_table=self.mt)

    def test_net_premium_term_life(self):
        """Net premium for term life = sum insured * q_x * v (discount factor)."""
        policy = Policy(
            sum_insured=100_000,
            age=25,
            term_years=1,
            mortality_table=self.mt,
        )
        premium = self.engine.calculate_net_premium(policy)
        # q_25 = 0.001, v = 1/1.05 ≈ 0.95238
        # net_premium = 100000 * 0.001 * 0.95238 ≈ 95.24
        expected = 100_000 * 0.001 / 1.05
        assert math.isclose(premium, expected, rel_tol=1e-6)

    def test_gross_premium_includes_loading(self):
        """Gross premium = net premium * (1 + loading_factor)."""
        policy = Policy(
            sum_insured=100_000,
            age=25,
            term_years=1,
            mortality_table=self.mt,
        )
        net = self.engine.calculate_net_premium(policy)
        gross = self.engine.calculate_gross_premium(policy, loading_factor=0.20)
        assert math.isclose(gross, net * 1.20, rel_tol=1e-9)

    def test_gross_premium_zero_loading(self):
        """Gross premium equals net premium when loading is zero."""
        policy = Policy(
            sum_insured=50_000,
            age=25,
            term_years=1,
            mortality_table=self.mt,
        )
        net = self.engine.calculate_net_premium(policy)
        gross = self.engine.calculate_gross_premium(policy, loading_factor=0.0)
        assert math.isclose(gross, net, rel_tol=1e-9)

    def test_premium_higher_for_older_age(self):
        """Older age should produce higher premium."""
        policy_young = Policy(
            sum_insured=100_000,
            age=25,
            term_years=1,
            mortality_table=self.mt,
        )
        policy_old = Policy(
            sum_insured=100_000,
            age=29,
            term_years=1,
            mortality_table=self.mt,
        )
        # With same q_x, premiums should be equal
        p_young = self.engine.calculate_net_premium(policy_young)
        p_old = self.engine.calculate_net_premium(policy_old)
        assert math.isclose(p_young, p_old, rel_tol=1e-9)

    def test_premium_scales_with_sum_insured(self):
        """Premium should scale linearly with sum insured."""
        policy_100k = Policy(
            sum_insured=100_000,
            age=25,
            term_years=1,
            mortality_table=self.mt,
        )
        policy_200k = Policy(
            sum_insured=200_000,
            age=25,
            term_years=1,
            mortality_table=self.mt,
        )
        p_100k = self.engine.calculate_net_premium(policy_100k)
        p_200k = self.engine.calculate_net_premium(policy_200k)
        assert math.isclose(p_200k, 2 * p_100k, rel_tol=1e-9)

    def test_multi_year_term_premium(self):
        """Multi-year term premium sums discounted expected claims."""
        policy = Policy(
            sum_insured=100_000,
            age=25,
            term_years=3,
            mortality_table=self.mt,
        )
        premium = self.engine.calculate_net_premium(policy)
        # Year 1: 100000 * q_25 * v^1
        # Year 2: 100000 * (1-q_25) * q_26 * v^2
        # Year 3: 100000 * (1-q_25) * (1-q_26) * q_27 * v^3
        q = 0.001
        v = 1 / 1.05
        expected = (
            100_000 * q * v
            + 100_000 * (1 - q) * q * v**2
            + 100_000 * (1 - q) ** 2 * q * v**3
        )
        assert math.isclose(premium, expected, rel_tol=1e-6)

    def test_premium_with_zero_mortality(self):
        """Zero mortality should produce zero net premium."""
        ages = np.array([25, 26, 27])
        q_x = np.array([0.0, 0.0, 0.0])
        mt = MortalityTable(ages=ages, q_x=q_x)
        engine = InsurtechEngine(mortality_table=mt)
        policy = Policy(
            sum_insured=100_000,
            age=25,
            term_years=1,
            mortality_table=mt,
        )
        premium = engine.calculate_net_premium(policy)
        assert premium == 0.0


# ---------------------------------------------------------------------------
# Reserving Tests
# ---------------------------------------------------------------------------


class TestReserving:
    """Test reserve calculation methods."""

    def setup_method(self):
        ages = np.array([25, 26, 27, 28, 29, 30])
        q_x = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        self.mt = MortalityTable(ages=ages, q_x=q_x)
        self.engine = InsurtechEngine(mortality_table=self.mt)

    def test_chain_ladder_basic(self):
        """Chain ladder: project ultimate from loss triangle."""
        # Simple 3x3 triangle
        triangle = np.array([
            [100, 120, 140],
            [110, 130, 0],
            [120, 0, 0],
        ])
        result = self.engine.chain_ladder(triangle)
        assert isinstance(result, ReserveResult)
        assert result.ultimate is not None
        assert result.reserve is not None
        assert result.reserve >= 0

    def test_chain_ladder_development_factors(self):
        """Chain ladder development factors should be > 1 for increasing losses."""
        triangle = np.array([
            [100, 120, 140],
            [110, 130, 0],
            [120, 0, 0],
        ])
        result = self.engine.chain_ladder(triangle)
        # Age-to-age factors: 120/100=1.2, 140/120≈1.167, 130/110≈1.182
        assert len(result.development_factors) == 2
        assert result.development_factors[0] > 1.0
        assert result.development_factors[1] > 1.0

    def test_chain_ladder_ultimate_greater_than_latest(self):
        """Ultimate should exceed latest diagonal for increasing triangle."""
        triangle = np.array([
            [100, 120, 140],
            [110, 130, 0],
            [120, 0, 0],
        ])
        result = self.engine.chain_ladder(triangle)
        latest_diagonal = np.array([140, 130, 120])
        assert np.all(result.ultimate >= latest_diagonal)

    def test_bornhuetter_ferguson_basic(self):
        """Bornhuetter-Ferguson: blend of chain ladder and expected losses."""
        triangle = np.array([
            [100, 120, 140],
            [110, 130, 0],
            [120, 0, 0],
        ])
        expected_losses = np.array([150, 145, 140])
        result = self.engine.bornhuetter_ferguson(triangle, expected_losses)
        assert isinstance(result, ReserveResult)
        assert result.reserve >= 0

    def test_bornhuetter_ferguson_weights(self):
        """BF with weight=1 should equal chain ladder result."""
        triangle = np.array([
            [100, 120, 140],
            [110, 130, 0],
            [120, 0, 0],
        ])
        expected_losses = np.array([150, 145, 140])
        result_full = self.engine.bornhuetter_ferguson(
            triangle, expected_losses, weight=1.0
        )
        result_cl = self.engine.chain_ladder(triangle)
        assert math.isclose(result_full.reserve, result_cl.reserve, rel_tol=1e-6)

    def test_bornhuetter_ferguson_zero_weight(self):
        """BF with weight=0 should use only expected losses."""
        triangle = np.array([
            [100, 120, 140],
            [110, 130, 0],
            [120, 0, 0],
        ])
        expected_losses = np.array([150, 145, 140])
        result = self.engine.bornhuetter_ferguson(
            triangle, expected_losses, weight=0.0
        )
        # Reserve = expected - paid (latest diagonal)
        latest = np.array([140, 130, 120])
        expected_reserve = np.sum(expected_losses - latest)
        assert math.isclose(result.reserve, expected_reserve, rel_tol=1e-6)

    def test_loss_ratio_method(self):
        """Loss ratio method: reserve = expected_loss_ratio * earned_premium - paid."""
        earned_premium = np.array([1000, 1200, 1400])
        paid_losses = np.array([500, 600, 0])
        expected_loss_ratio = 0.70
        result = self.engine.loss_ratio_method(
            earned_premium, paid_losses, expected_loss_ratio
        )
        assert isinstance(result, ReserveResult)
        # Expected ultimate = 0.70 * earned_premium
        expected_ultimate = expected_loss_ratio * earned_premium
        assert np.allclose(result.ultimate, expected_ultimate)

    def test_loss_ratio_reserve_calculation(self):
        """Reserve = ultimate - paid."""
        earned_premium = np.array([1000, 1200, 1400])
        paid_losses = np.array([500, 600, 0])
        expected_loss_ratio = 0.70
        result = self.engine.loss_ratio_method(
            earned_premium, paid_losses, expected_loss_ratio
        )
        expected_reserve = np.sum(
            expected_loss_ratio * earned_premium - paid_losses
        )
        assert math.isclose(result.reserve, expected_reserve, rel_tol=1e-6)

    def test_reserve_result_has_triangle(self):
        """ReserveResult should store the projected triangle."""
        triangle = np.array([
            [100, 120, 140],
            [110, 130, 0],
            [120, 0, 0],
        ])
        result = self.engine.chain_ladder(triangle)
        assert result.triangle is not None
        assert result.triangle.shape == triangle.shape


# ---------------------------------------------------------------------------
# Risk Pooling Tests
# ---------------------------------------------------------------------------


class TestRiskPooling:
    """Test risk pooling and diversification."""

    def setup_method(self):
        ages = np.array([25, 26, 27])
        q_x = np.array([0.001, 0.001, 0.001])
        mt = MortalityTable(ages=ages, q_x=q_x)
        self.engine = InsurtechEngine(mortality_table=mt)

    def test_risk_pool_variance_reduction(self):
        """Pooling should reduce variance compared to individual risk."""
        individual_variances = np.array([100, 100, 100, 100])
        result = self.engine.risk_pool(individual_variances)
        assert isinstance(result, RiskPoolResult)
        # Pooled variance should be less than sum of individual variances
        assert result.pooled_variance < np.sum(individual_variances)

    def test_risk_pool_perfect_correlation(self):
        """With perfect correlation, pooled variance = individual variance."""
        individual_variances = np.array([100, 100, 100])
        result = self.engine.risk_pool(
            individual_variances, correlation=1.0
        )
        # With perfect correlation, variance of average = σ² (no diversification)
        assert math.isclose(
            result.pooled_variance, 100, rel_tol=1e-9
        )

    def test_risk_pool_zero_correlation(self):
        """With zero correlation, pooled variance = avg_variance / n."""
        individual_variances = np.array([100, 100, 100, 100])
        result = self.engine.risk_pool(
            individual_variances, correlation=0.0
        )
        # Var(average) = σ²/n = 100/4 = 25
        expected = 100 / 4
        assert math.isclose(result.pooled_variance, expected, rel_tol=1e-9)

    def test_risk_pool_diversification_benefit(self):
        """Diversification benefit = total_individual - n * pooled."""
        individual_variances = np.array([100, 100, 100])
        result = self.engine.risk_pool(
            individual_variances, correlation=0.0
        )
        n = len(individual_variances)
        expected_benefit = np.sum(individual_variances) - n * result.pooled_variance
        assert math.isclose(
            result.diversification_benefit, expected_benefit, rel_tol=1e-9
        )

    def test_risk_pool_coefficient_of_variation(self):
        """CV should decrease with pooling."""
        individual_variances = np.array([100, 100, 100, 100])
        result = self.engine.risk_pool(
            individual_variances, correlation=0.0
        )
        # CV = sqrt(pooled_variance) / mean
        # With zero correlation and equal variances, CV = sqrt(100/4) / 100 = 0.05
        assert result.coefficient_of_variation < 1.0

    def test_risk_pool_single_risk(self):
        """Single risk pool should have zero diversification benefit."""
        individual_variances = np.array([100])
        result = self.engine.risk_pool(individual_variances)
        assert math.isclose(result.pooled_variance, 100, rel_tol=1e-9)
        assert math.isclose(result.diversification_benefit, 0, rel_tol=1e-9)

    def test_risk_pool_large_pool_converges(self):
        """Large pool with zero correlation approaches zero variance."""
        n = 1000
        individual_variances = np.ones(n) * 100
        result = self.engine.risk_pool(
            individual_variances, correlation=0.0
        )
        # Pooled variance = 100 / 1000 = 0.1
        assert math.isclose(result.pooled_variance, 0.1, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Engine Integration Tests
# ---------------------------------------------------------------------------


class TestInsurtechEngine:
    """Test engine initialization and integration."""

    def test_engine_initializes_with_mortality_table(self):
        ages = np.array([25, 26, 27])
        q_x = np.array([0.001, 0.001, 0.001])
        mt = MortalityTable(ages=ages, q_x=q_x)
        engine = InsurtechEngine(mortality_table=mt)
        assert engine.mortality_table is not None

    def test_engine_default_interest_rate(self):
        ages = np.array([25, 26, 27])
        q_x = np.array([0.001, 0.001, 0.001])
        mt = MortalityTable(ages=ages, q_x=q_x)
        engine = InsurtechEngine(mortality_table=mt)
        assert engine.interest_rate == 0.05

    def test_engine_custom_interest_rate(self):
        ages = np.array([25, 26, 27])
        q_x = np.array([0.001, 0.001, 0.001])
        mt = MortalityTable(ages=ages, q_x=q_x)
        engine = InsurtechEngine(mortality_table=mt, interest_rate=0.03)
        assert engine.interest_rate == 0.03

    def test_engine_premium_uses_interest_rate(self):
        """Higher interest rate should produce lower premium (more discounting)."""
        ages = np.array([25, 26, 27])
        q_x = np.array([0.001, 0.001, 0.001])
        mt = MortalityTable(ages=ages, q_x=q_x)
        engine_low = InsurtechEngine(mortality_table=mt, interest_rate=0.03)
        engine_high = InsurtechEngine(mortality_table=mt, interest_rate=0.08)
        policy = Policy(
            sum_insured=100_000,
            age=25,
            term_years=1,
            mortality_table=mt,
        )
        p_low = engine_low.calculate_net_premium(policy)
        p_high = engine_high.calculate_net_premium(policy)
        assert p_low > p_high

    def test_engine_full_pricing_workflow(self):
        """End-to-end: mortality table -> policy -> premium -> reserve."""
        ages = np.array([25, 26, 27, 28, 29, 30])
        q_x = np.array([0.001, 0.0012, 0.0015, 0.0018, 0.002, 0.0025])
        mt = MortalityTable(ages=ages, q_x=q_x)
        engine = InsurtechEngine(mortality_table=mt, interest_rate=0.05)

        # Price a policy
        policy = Policy(
            sum_insured=250_000,
            age=25,
            term_years=5,
            mortality_table=mt,
        )
        net = engine.calculate_net_premium(policy)
        gross = engine.calculate_gross_premium(policy, loading_factor=0.25)
        assert net > 0
        assert gross > net

        # Calculate reserves
        triangle = np.array([
            [5000, 6000, 7000, 7500, 8000],
            [5500, 6500, 7200, 7800, 0],
            [5200, 6200, 7100, 0, 0],
            [5800, 6800, 0, 0, 0],
            [6000, 0, 0, 0, 0],
        ])
        reserve_result = engine.chain_ladder(triangle)
        assert reserve_result.reserve > 0

        # Risk pooling
        variances = np.array([1e6, 1.2e6, 0.8e6, 1.1e6, 0.9e6])
        pool_result = engine.risk_pool(variances, correlation=0.3)
        assert pool_result.pooled_variance < np.sum(variances)
