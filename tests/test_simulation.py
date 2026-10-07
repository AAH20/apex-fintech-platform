"""Tests for Monte Carlo simulation engine.

Covers GBM simulation, European option pricing, American option pricing,
convergence, and edge cases.
"""
import numpy as np

from simulation import MonteCarloEngine


class TestGBMSimulation:
    """Tests for GBM path simulation."""

    def test_gbm_path_shape(self):
        """GBM returns correct path shape."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        paths = engine.simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, T=1.0, n_paths=5)
        assert paths.shape == (5, 101)

    def test_gbm_starts_at_s0(self):
        """All GBM paths start at s0."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        paths = engine.simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, T=1.0, n_paths=10)
        assert np.allclose(paths[:, 0], 100.0)

    def test_gbm_positive(self):
        """GBM paths are always positive."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        paths = engine.simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, T=1.0, n_paths=50)
        assert np.all(paths > 0)

    def test_gbm_reproducible_with_seed(self):
        """Same seed produces identical paths."""
        engine1 = MonteCarloEngine(dt=0.01, seed=123)
        engine2 = MonteCarloEngine(dt=0.01, seed=123)
        paths1 = engine1.simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, T=1.0, n_paths=3)
        paths2 = engine2.simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, T=1.0, n_paths=3)
        assert np.allclose(paths1, paths2)


class TestEuropeanOptionPricing:
    """Tests for European option pricing."""

    def test_european_call_price(self):
        """Monte Carlo European call price is reasonable."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        result = engine.price_european_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=50000, option_type="call"
        )
        # Black-Scholes price for ATM call ~ 10.45
        assert 9.0 < result.price < 12.0

    def test_european_put_price(self):
        """Monte Carlo European put price is reasonable."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        result = engine.price_european_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=50000, option_type="put"
        )
        # Black-Scholes price for ATM put ~ 5.57
        assert 4.0 < result.price < 7.0

    def test_call_put_parity(self):
        """Put-call parity: C - P = S0 - K*exp(-rT)."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        s0, k, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        call = engine.price_european_option(
            s0=s0, k=k, r=r, sigma=sigma, T=T, n_paths=50000, option_type="call"
        )
        put = engine.price_european_option(
            s0=s0, k=k, r=r, sigma=sigma, T=T, n_paths=50000, option_type="put"
        )
        lhs = call.price - put.price
        rhs = s0 - k * np.exp(-r * T)
        assert abs(lhs - rhs) < 0.5


class TestAmericanOptionPricing:
    """Tests for American option pricing."""

    def test_american_call_price(self):
        """American call price is reasonable and >= European call."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        american = engine.price_american_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=20000, option_type="call"
        )
        european = engine.price_european_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=20000, option_type="call"
        )
        # American call on non-dividend stock equals European call
        assert abs(american.price - european.price) < 1.0
        assert american.price > 0


class TestConvergence:
    """Tests for Monte Carlo convergence."""

    def test_more_paths_lower_std_error(self):
        """More paths should give lower standard error."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        result_low = engine.price_european_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=1000, option_type="call"
        )
        result_high = engine.price_european_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, n_paths=50000, option_type="call"
        )
        assert result_high.std_error < result_low.std_error


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_vol(self):
        """With zero vol, option price equals discounted intrinsic value."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        s0, k, r, T = 100.0, 100.0, 0.05, 1.0
        result = engine.price_european_option(
            s0=s0, k=k, r=r, sigma=0.0, T=T, n_paths=1000, option_type="call"
        )
        expected = max(s0 - k * np.exp(-r * T), 0.0)
        assert abs(result.price - expected) < 0.01

    def test_zero_time(self):
        """With zero time to maturity, option price equals intrinsic value."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        s0, k, r, sigma = 100.0, 100.0, 0.05, 0.2
        result = engine.price_european_option(
            s0=s0, k=k, r=r, sigma=sigma, T=0.0, n_paths=1000, option_type="call"
        )
        assert abs(result.price - max(s0 - k, 0.0)) < 0.01

    def test_deep_itm_call(self):
        """Deep in-the-money call is worth approximately S0 - K*exp(-rT)."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        s0, k, r, sigma, T = 100.0, 50.0, 0.05, 0.2, 1.0
        result = engine.price_european_option(
            s0=s0, k=k, r=r, sigma=sigma, T=T, n_paths=20000, option_type="call"
        )
        intrinsic = s0 - k * np.exp(-r * T)
        assert abs(result.price - intrinsic) < 1.0

    def test_deep_otm_call(self):
        """Deep out-of-the-money call is worth approximately zero."""
        engine = MonteCarloEngine(dt=0.01, seed=42)
        result = engine.price_european_option(
            s0=100.0, k=200.0, r=0.05, sigma=0.2, T=0.25, n_paths=20000, option_type="call"
        )
        assert result.price < 1.0
