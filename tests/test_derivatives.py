"""Tests for the Derivatives Pricing Engine."""
import time
import pytest
import numpy as np
from scipy.stats import norm

from src.derivatives.pricing import DerivativesPricingEngine


@pytest.fixture
def engine():
    """Create a fresh pricing engine."""
    return DerivativesPricingEngine()


class TestBlackScholes:
    """Test Black-Scholes closed-form pricing."""

    def test_call_price_known_value(self, engine):
        """BS call for S=100, K=100, r=0.05, sigma=0.2, T=1 should be ~10.4506."""
        price = engine.black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert price == pytest.approx(10.4506, rel=1e-3)

    def test_put_price_known_value(self, engine):
        """BS put for S=100, K=100, r=0.05, sigma=0.2, T=1 should be ~5.5735."""
        price = engine.black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "put")
        assert price == pytest.approx(5.5735, rel=1e-3)

    def test_put_call_parity(self, engine):
        """C - P = S - K*exp(-rT)."""
        S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.5
        call = engine.black_scholes_price(S, K, T, r, sigma, "call")
        put = engine.black_scholes_price(S, K, T, r, sigma, "put")
        assert call - put == pytest.approx(S - K * np.exp(-r * T), rel=1e-10)

    def test_deep_itm_call(self, engine):
        """Deep ITM call should be close to S - K*exp(-rT)."""
        S, K, r, sigma, T = 200.0, 100.0, 0.05, 0.2, 1.0
        price = engine.black_scholes_price(S, K, T, r, sigma, "call")
        intrinsic = S - K * np.exp(-r * T)
        assert price >= intrinsic
        assert price < intrinsic + 1.0

    def test_deep_otm_call(self, engine):
        """Deep OTM call should be near zero."""
        price = engine.black_scholes_price(50.0, 200.0, 0.1, 0.05, 0.2, "call")
        assert price < 0.01

    def test_zero_vol_call(self, engine):
        """Zero vol call is max(S*exp(-rT) - K*exp(-rT), 0) for r>0."""
        S, K, r, T = 100.0, 100.0, 0.05, 1.0
        price = engine.black_scholes_price(S, K, T, r, 0.0, "call")
        expected = max(S - K * np.exp(-r * T), 0.0)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_invalid_option_type_raises(self, engine):
        with pytest.raises(ValueError):
            engine.black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "invalid")


class TestGreeks:
    """Test Greeks computation."""

    def test_delta_call(self, engine):
        """Delta for ATM call should be ~0.5-0.65."""
        delta = engine.delta(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert 0.5 < delta < 0.65

    def test_delta_put(self, engine):
        """Delta for ATM put should be ~-0.5 to -0.35."""
        delta = engine.delta(100.0, 100.0, 1.0, 0.05, 0.2, "put")
        assert -0.5 < delta < -0.35

    def test_delta_call_put_relationship(self, engine):
        """Delta_call - Delta_put = 1."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        d_call = engine.delta(S, K, T, r, sigma, "call")
        d_put = engine.delta(S, K, T, r, sigma, "put")
        assert d_call - d_put == pytest.approx(1.0, rel=1e-10)

    def test_gamma(self, engine):
        """Gamma should be positive and reasonable for ATM."""
        gamma = engine.gamma(100.0, 100.0, 1.0, 0.05, 0.2)
        assert gamma > 0
        assert gamma < 0.1

    def test_gamma_symmetric(self, engine):
        """Gamma is the same for call and put."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        g_call = engine.gamma(S, K, T, r, sigma, "call")
        g_put = engine.gamma(S, K, T, r, sigma, "put")
        assert g_call == pytest.approx(g_put, rel=1e-10)

    def test_vega(self, engine):
        """Vega should be positive and reasonable."""
        vega = engine.vega(100.0, 100.0, 1.0, 0.05, 0.2)
        assert vega > 0
        assert vega < 50

    def test_theta_call(self, engine):
        """Theta for call should generally be negative (time decay)."""
        theta = engine.theta(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert theta < 0

    def test_rho_call(self, engine):
        """Rho for call should be positive."""
        rho = engine.rho(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert rho > 0

    def test_rho_put(self, engine):
        """Rho for put should be negative."""
        rho = engine.rho(100.0, 100.0, 1.0, 0.05, 0.2, "put")
        assert rho < 0

    def test_all_greeks_return_dict(self, engine):
        """all_greeks should return all five Greeks."""
        greeks = engine.all_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks


class TestMonteCarlo:
    """Test Monte Carlo simulation pricing."""

    def test_mc_call_converges_to_bs(self, engine):
        """MC call price should converge to BS price."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        bs_price = engine.black_scholes_price(S, K, T, r, sigma, "call")
        mc_price = engine.monte_carlo_price(S, K, T, r, sigma, "call", n_paths=100_000, seed=42)
        assert mc_price == pytest.approx(bs_price, rel=0.02)

    def test_mc_put_converges_to_bs(self, engine):
        """MC put price should converge to BS price."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        bs_price = engine.black_scholes_price(S, K, T, r, sigma, "put")
        mc_price = engine.monte_carlo_price(S, K, T, r, sigma, "put", n_paths=100_000, seed=42)
        assert mc_price == pytest.approx(bs_price, rel=0.02)

    def test_mc_with_dividend_yield(self, engine):
        """MC should support dividend yield."""
        S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0
        price = engine.monte_carlo_price(S, K, T, r, sigma, "call", q=q, n_paths=50_000, seed=42)
        assert price > 0
        assert price < S

    def test_mc_standard_error_decreases(self, engine):
        """More paths should give smaller standard error."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        _, se_low = engine.monte_carlo_price(S, K, T, r, sigma, "call", n_paths=10_000, seed=42, return_std_err=True)
        _, se_high = engine.monte_carlo_price(S, K, T, r, sigma, "call", n_paths=100_000, seed=42, return_std_err=True)
        assert se_high < se_low


class TestImpliedVolatility:
    """Test implied volatility calculation."""

    def test_implied_vol_round_trip(self, engine):
        """IV of a BS price should recover the original vol."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.25, 1.0
        call_price = engine.black_scholes_price(S, K, T, r, sigma, "call")
        iv = engine.implied_volatility(call_price, S, K, T, r, "call")
        assert iv == pytest.approx(sigma, rel=1e-4)

    def test_implied_vol_put(self, engine):
        """IV for put should also round-trip."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.3, 0.5
        put_price = engine.black_scholes_price(S, K, T, r, sigma, "put")
        iv = engine.implied_volatility(put_price, S, K, T, r, "put")
        assert iv == pytest.approx(sigma, rel=1e-4)

    def test_implied_vol_no_solution(self, engine):
        """Price below intrinsic should raise."""
        with pytest.raises(ValueError):
            engine.implied_volatility(0.01, 100.0, 100.0, 1.0, 0.05, "call")


class TestExoticOptions:
    """Test exotic option pricing."""

    def test_asian_call(self, engine):
        """Asian call should be cheaper than vanilla call (lower variance)."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        vanilla = engine.black_scholes_price(S, K, T, r, sigma, "call")
        asian = engine.asian_option_price(S, K, T, r, sigma, "call", n_paths=50_000, n_steps=50, seed=42)
        assert asian < vanilla
        assert asian > 0

    def test_barrier_call_knockout(self, engine):
        """Up-and-out call should be cheaper than vanilla."""
        S, K, r, sigma, T, barrier = 100.0, 100.0, 0.05, 0.2, 1.0, 150.0
        vanilla = engine.black_scholes_price(S, K, T, r, sigma, "call")
        barrier_price = engine.barrier_option_price(S, K, T, r, sigma, "call", barrier, "up-and-out", n_paths=50_000, seed=42)
        assert barrier_price < vanilla
        assert barrier_price > 0

    def test_digital_call(self, engine):
        """Digital call pays 1 if S_T > K."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        price = engine.digital_option_price(S, K, T, r, sigma, "call")
        # Digital call = exp(-rT) * N(d2)
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        expected = np.exp(-r * T) * norm.cdf(d2)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_digital_call_bounds(self, engine):
        """Digital call price should be between 0 and exp(-rT)."""
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        price = engine.digital_option_price(S, K, T, r, sigma, "call")
        assert 0 < price < np.exp(-r * T)


class TestPerformance:
    """Test that pricing is fast enough."""

    def test_bs_pricing_under_10ms(self, engine):
        """Black-Scholes should price in under 10ms."""
        start = time.perf_counter()
        for _ in range(100):
            engine.black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.010

    def test_greeks_under_10ms(self, engine):
        """Greeks computation should be under 10ms."""
        start = time.perf_counter()
        for _ in range(100):
            engine.all_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.010


class TestInputValidation:
    """Test input validation."""

    def test_negative_spot_raises(self, engine):
        with pytest.raises(ValueError):
            engine.black_scholes_price(-100.0, 100.0, 1.0, 0.05, 0.2, "call")

    def test_negative_strike_raises(self, engine):
        with pytest.raises(ValueError):
            engine.black_scholes_price(100.0, -100.0, 1.0, 0.05, 0.2, "call")

    def test_negative_time_raises(self, engine):
        with pytest.raises(ValueError):
            engine.black_scholes_price(100.0, 100.0, -1.0, 0.05, 0.2, "call")

    def test_negative_vol_raises(self, engine):
        with pytest.raises(ValueError):
            engine.black_scholes_price(100.0, 100.0, 1.0, 0.05, -0.2, "call")
