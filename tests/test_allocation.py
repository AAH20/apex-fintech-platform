"""Tests for asset allocation engine — strategic, tactical, factor, Black-Litterman, risk parity."""
import numpy as np
import pytest

from portfolio import AssetAllocationEngine


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_returns():
    """Annualized expected returns for 4 assets."""
    return np.array([0.08, 0.10, 0.06, 0.12])


@pytest.fixture
def simple_cov():
    """Simple 4x4 covariance matrix (annualized)."""
    return np.array([
        [0.04, 0.01, 0.005, 0.015],
        [0.01, 0.09, 0.01, 0.02],
        [0.005, 0.01, 0.0225, 0.008],
        [0.015, 0.02, 0.008, 0.16],
    ])


@pytest.fixture
def engine():
    return AssetAllocationEngine(risk_free_rate=0.02)


# ---------------------------------------------------------------------------
# Strategic allocation (mean-variance)
# ---------------------------------------------------------------------------

class TestStrategaticAllocation:
    """Tests for strategic (mean-variance) allocation."""

    def test_strategic_allocation_sums_to_one(self, engine, simple_returns, simple_cov):
        w = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=2.0)
        assert np.isclose(np.sum(w), 1.0, atol=1e-6)

    def test_strategic_allocation_no_short_selling(self, engine, simple_returns, simple_cov):
        w = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=2.0)
        assert np.all(w >= -1e-10)

    def test_strategic_allocation_higher_risk_aversion_more_conservative(
        self, engine, simple_returns, simple_cov
    ):
        """Higher risk aversion → lower portfolio volatility."""
        w_low = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=0.5)
        w_high = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=10.0)
        vol_low = np.sqrt(w_low @ simple_cov @ w_low)
        vol_high = np.sqrt(w_high @ simple_cov @ w_high)
        assert vol_high < vol_low

    def test_strategic_allocation_equal_inputs_gives_equal_weights(self, engine):
        """Identical assets → equal weights."""
        n = 3
        mu = np.array([0.1, 0.1, 0.1])
        cov = np.array([
            [0.04, 0.01, 0.01],
            [0.01, 0.04, 0.01],
            [0.01, 0.01, 0.04],
        ])
        w = engine.strategic_allocation(mu, cov, risk_aversion=2.0)
        assert np.allclose(w, 1.0 / n, atol=1e-4)

    def test_strategic_allocation_higher_return_gets_more_weight(self, engine):
        """Asset with higher return (same risk) gets more weight."""
        mu = np.array([0.05, 0.15])
        cov = np.array([
            [0.04, 0.0],
            [0.0, 0.04],
        ])
        w = engine.strategic_allocation(mu, cov, risk_aversion=1.0)
        assert w[1] > w[0]

    def test_strategic_allocation_rejects_mismatched_dimensions(self, engine):
        mu = np.array([0.08, 0.10])
        cov = np.eye(3)
        with pytest.raises(ValueError):
            engine.strategic_allocation(mu, cov, risk_aversion=2.0)


# ---------------------------------------------------------------------------
# Tactical allocation
# ---------------------------------------------------------------------------

class TestTacticalAllocation:
    """Tests for tactical allocation (tilts from strategic benchmark)."""

    def test_tactical_allocation_sums_to_one(self, engine, simple_returns, simple_cov):
        strategic = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=2.0)
        tilts = np.array([0.05, -0.03, 0.02, -0.04])
        w = engine.tactical_allocation(strategic, tilts, simple_returns, simple_cov)
        assert np.isclose(np.sum(w), 1.0, atol=1e-6)

    def test_tactical_allocation_positive_tilt_increases_weight(
        self, engine, simple_returns, simple_cov
    ):
        strategic = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=2.0)
        tilts = np.array([0.1, 0.0, 0.0, 0.0])
        w = engine.tactical_allocation(strategic, tilts, simple_returns, simple_cov)
        assert w[0] > strategic[0]

    def test_tactical_allocation_negative_tilt_decreases_weight(
        self, engine, simple_returns, simple_cov
    ):
        strategic = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=2.0)
        tilts = np.array([-0.1, 0.0, 0.0, 0.0])
        w = engine.tactical_allocation(strategic, tilts, simple_returns, simple_cov)
        assert w[0] < strategic[0]

    def test_tactical_allocation_zero_tilts_returns_strategic(
        self, engine, simple_returns, simple_cov
    ):
        strategic = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=2.0)
        tilts = np.zeros(4)
        w = engine.tactical_allocation(strategic, tilts, simple_returns, simple_cov)
        assert np.allclose(w, strategic, atol=1e-6)

    def test_tactical_allocation_respects_max_deviation(
        self, engine, simple_returns, simple_cov
    ):
        """Tilts are capped by max_deviation parameter."""
        strategic = engine.strategic_allocation(simple_returns, simple_cov, risk_aversion=2.0)
        tilts = np.array([0.5, 0.0, 0.0, 0.0])
        max_dev = 0.1
        w = engine.tactical_allocation(
            strategic, tilts, simple_returns, simple_cov, max_deviation=max_dev
        )
        assert abs(w[0] - strategic[0]) <= max_dev + 1e-6

    def test_tactical_allocation_rejects_mismatched_tilts(self, engine):
        strategic = np.array([0.25, 0.25, 0.25, 0.25])
        tilts = np.array([0.1, -0.1])
        with pytest.raises(ValueError):
            engine.tactical_allocation(strategic, tilts, np.zeros(4), np.eye(4))


# ---------------------------------------------------------------------------
# Factor investing
# ---------------------------------------------------------------------------

class TestFactorInvesting:
    """Tests for factor-based allocation."""

    def test_factor_allocation_sums_to_one(self, engine):
        factor_exposures = np.array([
            [1.0, 0.5, 0.2],
            [0.8, 1.0, 0.3],
            [0.3, 0.2, 1.0],
            [0.6, 0.4, 0.8],
        ])
        factor_returns = np.array([0.06, 0.04, 0.03])
        w = engine.factor_allocation(factor_exposures, factor_returns)
        assert np.isclose(np.sum(w), 1.0, atol=1e-6)

    def test_factor_allocation_higher_exposure_gets_more_weight(self, engine):
        """Asset with higher factor exposure gets more weight."""
        factor_exposures = np.array([
            [1.0, 0.0],
            [0.5, 0.0],
        ])
        factor_returns = np.array([0.08, 0.02])
        w = engine.factor_allocation(factor_exposures, factor_returns)
        assert w[0] > w[1]

    def test_factor_allocation_rejects_mismatched_dimensions(self, engine):
        factor_exposures = np.array([[1.0, 0.5], [0.8, 1.0]])
        factor_returns = np.array([0.06, 0.04, 0.03])
        with pytest.raises(ValueError):
            engine.factor_allocation(factor_exposures, factor_returns)

    def test_factor_allocation_no_short_selling(self, engine):
        factor_exposures = np.array([
            [1.0, 0.5, 0.2],
            [0.8, 1.0, 0.3],
            [0.3, 0.2, 1.0],
        ])
        factor_returns = np.array([0.06, 0.04, 0.03])
        w = engine.factor_allocation(factor_exposures, factor_returns)
        assert np.all(w >= -1e-10)


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

class TestBlackLitterman:
    """Tests for Black-Litterman model."""

    def test_black_litterman_sums_to_one(self, engine, simple_cov):
        market_weights = np.array([0.4, 0.3, 0.2, 0.1])
        views = np.array([0.12, 0.05])
        view_assets = np.array([0, 2])
        w = engine.black_litterman(simple_cov, market_weights, views, view_assets)
        assert np.isclose(np.sum(w), 1.0, atol=1e-6)

    def test_black_litterman_view_increases_weight(self, engine, simple_cov):
        """A positive view on an asset increases its weight."""
        market_weights = np.array([0.25, 0.25, 0.25, 0.25])
        views = np.array([0.15])
        view_assets = np.array([0])
        w = engine.black_litterman(simple_cov, market_weights, views, view_assets)
        assert w[0] > market_weights[0]

    def test_black_litterman_no_views_returns_market(self, engine, simple_cov):
        """With no views, BL returns market weights."""
        market_weights = np.array([0.4, 0.3, 0.2, 0.1])
        w = engine.black_litterman(
            simple_cov, market_weights, np.array([]), np.array([], dtype=int)
        )
        assert np.allclose(w, market_weights, atol=1e-6)

    def test_black_litterman_rejects_mismatched_views(self, engine, simple_cov):
        market_weights = np.array([0.25, 0.25, 0.25, 0.25])
        views = np.array([0.1, 0.05])
        view_assets = np.array([0])
        with pytest.raises(ValueError):
            engine.black_litterman(simple_cov, market_weights, views, view_assets)


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------

class TestRiskParity:
    """Tests for risk parity allocation."""

    def test_risk_parity_sums_to_one(self, engine, simple_cov):
        w = engine.risk_parity(simple_cov)
        assert np.isclose(np.sum(w), 1.0, atol=1e-6)

    def test_risk_parity_equal_risk_contribution(self, engine, simple_cov):
        """All assets contribute equally to portfolio risk."""
        w = engine.risk_parity(simple_cov)
        n = len(w)
        port_var = w @ simple_cov @ w
        # Marginal risk contribution: (Cov @ w)_i
        marginal = simple_cov @ w
        # Percentage risk contribution: w_i * (Cov @ w)_i / port_var
        risk_contrib = w * marginal / port_var
        assert np.allclose(risk_contrib, 1.0 / n, atol=1e-4)

    def test_risk_parity_no_short_selling(self, engine, simple_cov):
        w = engine.risk_parity(simple_cov)
        assert np.all(w >= -1e-10)

    def test_risk_parity_rejects_non_symmetric_cov(self, engine):
        cov = np.array([
            [0.04, 0.01],
            [0.02, 0.09],
        ])
        with pytest.raises(ValueError):
            engine.risk_parity(cov)
