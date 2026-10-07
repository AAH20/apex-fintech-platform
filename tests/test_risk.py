"""Tests for risk aggregation engine with dependence uncertainty."""

import time

import numpy as np
import pytest

from risk.aggregation import RiskAggregationEngine


@pytest.fixture
def engine():
    return RiskAggregationEngine(confidence_level=0.95, n_samples=5000)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Core bound tests
# ---------------------------------------------------------------------------


class TestVarBounds:
    def test_independent_case(self, engine, rng):
        """Independent risks: lower bound equals independent VaR."""
        n_risks, n_samples = 5, 5000
        samples = rng.normal(0, 1, (n_risks, n_samples))
        lower, upper = engine.compute_var_bounds(samples)
        # Lower bound should match the independent VaR
        independent_sum = samples.sum(axis=0)
        independent_var = float(np.percentile(independent_sum, 95))
        assert lower == pytest.approx(independent_var, rel=0.01)
        # Upper bound should be strictly greater (diversification benefit)
        assert upper > lower

    def test_perfectly_correlated_case(self, engine, rng):
        """Perfectly correlated risks: upper bound equals sum of individual VaRs."""
        n_risks, n_samples = 5, 5000
        base = rng.normal(0, 1, n_samples)
        samples = np.tile(base, (n_risks, 1))
        lower, upper = engine.compute_var_bounds(samples)
        # Comonotonic sum = n * base, so VaR = n * VaR(base)
        expected_var = n_risks * float(np.percentile(base, 95))
        assert upper == pytest.approx(expected_var, rel=0.01)

    def test_arbitrary_dependence_case(self, engine, rng):
        """Arbitrary dependence: lower bound <= upper bound always."""
        n_risks, n_samples = 10, 3000
        samples = rng.standard_t(df=3, size=(n_risks, n_samples))
        lower, upper = engine.compute_var_bounds(samples)
        assert lower <= upper

    def test_var_bounds_ordering(self, engine, rng):
        """Lower VaR bound must not exceed upper VaR bound."""
        samples = rng.normal(0, 1, (8, 2000))
        lower, upper = engine.compute_var_bounds(samples)
        assert lower <= upper


class TestCvarBounds:
    def test_cvar_bounds_ordering(self, engine, rng):
        """Lower CVaR bound must not exceed upper CVaR bound."""
        samples = rng.normal(0, 1, (6, 3000))
        lower, upper = engine.compute_cvar_bounds(samples)
        assert lower <= upper

    def test_cvar_bounds_bracket_independent(self, engine, rng):
        """Independent CVaR should equal the lower CVaR bound."""
        n_risks, n_samples = 5, 5000
        samples = rng.normal(0, 1, (n_risks, n_samples))
        lower, upper = engine.compute_cvar_bounds(samples)
        independent_sum = samples.sum(axis=0)
        var = float(np.percentile(independent_sum, 95))
        independent_cvar = float(np.mean(independent_sum[independent_sum >= var]))
        assert lower == pytest.approx(independent_cvar, rel=0.01)
        assert upper > lower


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_risk(self, engine, rng):
        """Single risk: bounds equal the risk's own VaR."""
        samples = rng.normal(0, 1, (1, 5000))
        lower, upper = engine.compute_var_bounds(samples)
        expected = float(np.percentile(samples[0], 95))
        assert lower == pytest.approx(expected, rel=0.01)
        assert upper == pytest.approx(expected, rel=0.01)

    def test_zero_risk(self, engine):
        """Zero risk (all zeros): bounds are zero."""
        samples = np.zeros((5, 1000))
        lower, upper = engine.compute_var_bounds(samples)
        assert lower == 0.0
        assert upper == 0.0

    def test_negative_risk(self, engine, rng):
        """Negative risk (profits): bounds handle correctly."""
        samples = rng.normal(-1, 0.5, (4, 3000))
        lower, upper = engine.compute_var_bounds(samples)
        assert lower <= upper
        # VaR should be negative (profit) for high confidence
        assert upper < 0


# ---------------------------------------------------------------------------
# Rearrangement algorithm
# ---------------------------------------------------------------------------


class TestRearrangementAlgorithm:
    def test_output_shapes(self, engine, rng):
        """Rearrangement returns arrays of correct shape."""
        samples = rng.normal(0, 1, (5, 2000))
        lower_sum, upper_sum = engine.rearrangement_algorithm(samples)
        assert lower_sum.shape == (2000,)
        assert upper_sum.shape == (2000,)

    def test_upper_sum_heavier_tail(self, engine, rng):
        """Comonotonic sum has heavier upper tail than independent sum."""
        samples = rng.normal(0, 1, (4, 5000))
        lower_sum, upper_sum = engine.rearrangement_algorithm(samples)
        # The 95th percentile of the comonotonic sum should be larger
        assert np.percentile(upper_sum, 95) > np.percentile(lower_sum, 95)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_100_risks_under_2_seconds(self, engine, rng):
        """100-risk portfolio computes VaR bounds in < 2 seconds."""
        samples = rng.normal(0, 1, (100, 5000))
        start = time.perf_counter()
        engine.compute_var_bounds(samples)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
