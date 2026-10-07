"""Tests for cardinality-constrained portfolio optimizer (TDD)."""
import time

import numpy as np

from portfolio.optimizer import CardinalityConstrainedOptimizer


def _make_problem(n=10, k=3, seed=42):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.20, n)
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + np.eye(n) * 0.01
    return mu, Sigma


class TestCardinalityConstrainedOptimizer:
    """10 tests for the cardinality-constrained optimizer."""

    def test_basic_optimization(self):
        """Basic optimization returns valid weights."""
        mu, Sigma = _make_problem(n=10, k=3)
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=3)
        opt.optimize()
        w = opt.get_weights()
        assert w is not None
        assert len(w) == 10
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= -1e-10)

    def test_cardinality_constraint(self):
        """Number of non-zero weights never exceeds k."""
        mu, Sigma = _make_problem(n=20, k=5)
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=5)
        opt.optimize()
        w = opt.get_weights()
        assert np.count_nonzero(w > 1e-8) <= 5

    def test_sector_limits(self):
        """Sector constraints are respected."""
        mu, Sigma = _make_problem(n=12, k=4)
        sectors = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        sector_limits = {0: 0.4, 1: 0.3, 2: 0.5, 3: 0.2}
        opt = CardinalityConstrainedOptimizer(
            mu, Sigma, k=4, sectors=sectors, sector_limits=sector_limits
        )
        opt.optimize()
        w = opt.get_weights()
        for sec, limit in sector_limits.items():
            assert w[sectors == sec].sum() <= limit + 1e-6

    def test_single_asset(self):
        """n=1 edge case."""
        mu = np.array([0.1])
        Sigma = np.array([[0.04]])
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=1)
        opt.optimize()
        w = opt.get_weights()
        assert np.isclose(w[0], 1.0)

    def test_zero_risk(self):
        """Near-zero covariance edge case."""
        mu = np.array([0.1, 0.15, 0.12])
        Sigma = np.eye(3) * 1e-10
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=2)
        opt.optimize()
        w = opt.get_weights()
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert np.count_nonzero(w > 1e-8) <= 2

    def test_negative_returns(self):
        """Negative expected returns."""
        mu = np.array([-0.05, -0.02, -0.10])
        A = np.array([[1.0, 0.5], [0.5, 1.0], [0.3, 0.7]])
        Sigma = A @ A.T + np.eye(3) * 0.01
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=2)
        opt.optimize()
        w = opt.get_weights()
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= -1e-10)

    def test_weights_sum_to_one(self):
        """Weights always sum to 1 (fully invested)."""
        mu, Sigma = _make_problem(n=15, k=4)
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=4)
        opt.optimize()
        w = opt.get_weights()
        assert np.isclose(w.sum(), 1.0, atol=1e-6)

    def test_no_short_selling(self):
        """All weights are non-negative (long-only)."""
        mu, Sigma = _make_problem(n=10, k=3)
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=3)
        opt.optimize()
        w = opt.get_weights()
        assert np.all(w >= -1e-10)

    def test_k_equals_n(self):
        """When k >= n, all assets can be selected."""
        mu, Sigma = _make_problem(n=5, k=5)
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=5)
        opt.optimize()
        w = opt.get_weights()
        assert np.count_nonzero(w > 1e-8) <= 5

    def test_performance_500_assets(self):
        """500-asset universe with k=50 solves in <5 seconds."""
        rng = np.random.default_rng(42)
        n, k = 500, 50
        mu = rng.uniform(0.05, 0.20, n)
        A = rng.standard_normal((n, n))
        Sigma = A @ A.T / n + np.eye(n) * 0.01
        opt = CardinalityConstrainedOptimizer(mu, Sigma, k=k)
        start = time.time()
        opt.optimize()
        elapsed = time.time() - start
        w = opt.get_weights()
        assert elapsed < 5.0, f"Took {elapsed:.2f}s, must be <5s"
        assert np.count_nonzero(w > 1e-8) <= k
