"""
TDD tests for Commodities Analytics Engine.

Covers futures curves, roll yield, contango/backwardation,
and commodity pricing models.

References:
    - CFA Institute, "Commodatives" (Reading 48)
    - Hull, J. "Options, Futures, and Other Derivatives"
"""

import numpy as np
import pytest

from commodities import CommoditiesEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Default engine: spot=100, r=5%, storage=2%, convenience=0%."""
    return CommoditiesEngine(
        spot_price=100.0,
        risk_free_rate=0.05,
        storage_cost=0.02,
        convenience_yield=0.0,
    )


@pytest.fixture
def backwardation_engine():
    """Engine in backwardation: convenience yield > r + storage."""
    return CommoditiesEngine(
        spot_price=100.0,
        risk_free_rate=0.03,
        storage_cost=0.01,
        convenience_yield=0.08,
    )


# ---------------------------------------------------------------------------
# Test 1: Futures price — cost-of-carry model
# ---------------------------------------------------------------------------


class TestFuturesPrice:
    def test_futures_price_at_zero_maturity_equals_spot(self, engine):
        """As T→0, futures price converges to spot."""
        price = engine.futures_price(time_to_maturity=0.0)
        assert price == pytest.approx(100.0)

    def test_futures_price_increases_with_maturity_in_contango(self, engine):
        """With r + storage > convenience yield, futures rise with maturity."""
        p1 = engine.futures_price(time_to_maturity=0.25)
        p2 = engine.futures_price(time_to_maturity=1.0)
        assert p2 > p1

    def test_futures_price_known_value(self):
        """F = S * exp((r + u - y) * T) — verify exact computation."""
        eng = CommoditiesEngine(
            spot_price=50.0,
            risk_free_rate=0.04,
            storage_cost=0.01,
            convenience_yield=0.0,
        )
        # F = 50 * exp(0.05 * 0.5) = 50 * exp(0.025)
        expected = 50.0 * np.exp(0.025)
        assert eng.futures_price(time_to_maturity=0.5) == pytest.approx(expected)

    def test_futures_price_decreases_in_backwardation(self, backwardation_engine):
        """When convenience yield > r + storage, futures fall with maturity."""
        p1 = backwardation_engine.futures_price(time_to_maturity=0.25)
        p2 = backwardation_engine.futures_price(time_to_maturity=1.0)
        assert p2 < p1

    def test_negative_maturity_raises(self, engine):
        """Negative time-to-maturity is invalid."""
        with pytest.raises(ValueError, match="time_to_maturity must be non-negative"):
            engine.futures_price(time_to_maturity=-0.1)


# ---------------------------------------------------------------------------
# Test 2: Futures curve construction
# ---------------------------------------------------------------------------


class TestFuturesCurve:
    def test_curve_returns_array_of_same_length(self, engine):
        """Curve has one price per maturity."""
        maturities = np.array([0.25, 0.5, 0.75, 1.0])
        curve = engine.futures_curve(maturities)
        assert isinstance(curve, np.ndarray)
        assert len(curve) == len(maturities)

    def test_curve_is_monotonically_increasing_in_contango(self, engine):
        """In contango, longer maturities have higher prices."""
        maturities = np.array([0.1, 0.3, 0.6, 1.0])
        curve = engine.futures_curve(maturities)
        assert np.all(np.diff(curve) > 0)

    def test_curve_is_monotonically_decreasing_in_backwardation(
        self, backwardation_engine
    ):
        """In backwardation, longer maturities have lower prices."""
        maturities = np.array([0.1, 0.3, 0.6, 1.0])
        curve = backwardation_engine.futures_curve(maturities)
        assert np.all(np.diff(curve) < 0)

    def test_curve_endpoint_matches_single_futures_price(self, engine):
        """Curve at T=1.0 matches direct futures_price(1.0)."""
        maturities = np.array([0.5, 1.0])
        curve = engine.futures_curve(maturities)
        assert curve[-1] == pytest.approx(engine.futures_price(1.0))

    def test_empty_maturities_returns_empty_array(self, engine):
        """Empty input yields empty curve."""
        curve = engine.futures_curve(np.array([]))
        assert len(curve) == 0


# ---------------------------------------------------------------------------
# Test 3: Roll yield
# ---------------------------------------------------------------------------


class TestRollYield:
    def test_roll_yield_positive_in_contango(self):
        """In contango (far > near), rolling forward earns positive roll yield."""
        eng = CommoditiesEngine(spot_price=100.0)
        # Near month = 98, Far month = 100 → buy low, sell high
        ry = eng.roll_yield(near_price=98.0, far_price=100.0)
        assert ry > 0

    def test_roll_yield_negative_in_backwardation(self):
        """In backwardation (far < near), rolling forward gives negative roll yield."""
        eng = CommoditiesEngine(spot_price=100.0)
        # Near month = 102, Far month = 100 → buy high, sell low
        ry = eng.roll_yield(near_price=102.0, far_price=100.0)
        assert ry < 0

    def test_roll_yield_zero_when_prices_equal(self):
        """No roll yield when near = far."""
        eng = CommoditiesEngine(spot_price=100.0)
        assert eng.roll_yield(near_price=100.0, far_price=100.0) == pytest.approx(0.0)

    def test_roll_yield_known_value(self):
        """Roll yield = (far - near) / near."""
        eng = CommoditiesEngine(spot_price=100.0)
        ry = eng.roll_yield(near_price=90.0, far_price=99.0)
        assert ry == pytest.approx((99.0 - 90.0) / 90.0)

    def test_annualized_roll_yield_known_value(self):
        """Annualized roll yield = (1 + roll_yield)^(1/T) - 1."""
        eng = CommoditiesEngine(spot_price=100.0)
        ry = eng.roll_yield(near_price=95.0, far_price=100.0)
        T = 0.25  # 3 months
        annualized = eng.annualized_roll_yield(near_price=95.0, far_price=100.0, time_between=T)
        expected = (1.0 + ry) ** (1.0 / T) - 1.0
        assert annualized == pytest.approx(expected)

    def test_annualized_roll_yield_zero_time_raises(self):
        """Zero time-between-contracts is invalid."""
        eng = CommoditiesEngine(spot_price=100.0)
        with pytest.raises(ValueError, match="time_between must be positive"):
            eng.annualized_roll_yield(near_price=95.0, far_price=100.0, time_between=0.0)


# ---------------------------------------------------------------------------
# Test 4: Contango / Backwardation detection
# ---------------------------------------------------------------------------


class TestContangoBackwardation:
    def test_contango_detected_when_far_greater_than_near(self, engine):
        """Contango: futures curve slopes upward."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = engine.futures_curve(maturities)
        assert engine.is_contango(maturities, curve) is True

    def test_backwardation_detected_when_far_less_than_near(
        self, backwardation_engine
    ):
        """Backwardation: futures curve slopes downward."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = backwardation_engine.futures_curve(maturities)
        assert backwardation_engine.is_backwardation(maturities, curve) is True

    def test_contango_false_in_backwardation(self, backwardation_engine):
        """Contango flag is False when market is in backwardation."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = backwardation_engine.futures_curve(maturities)
        assert backwardation_engine.is_contango(maturities, curve) is False

    def test_backwardation_false_in_contango(self, engine):
        """Backwardation flag is False when market is in contango."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = engine.futures_curve(maturities)
        assert engine.is_backwardation(maturities, curve) is False

    def test_flat_curve_is_neither_contango_nor_backwardation(self):
        """Flat curve (all same price) is neither contango nor backwardation."""
        eng = CommoditiesEngine(spot_price=100.0)
        maturities = np.array([0.25, 0.5, 1.0])
        flat_curve = np.array([100.0, 100.0, 100.0])
        assert eng.is_contango(maturities, flat_curve) is False
        assert eng.is_backwardation(maturities, flat_curve) is False

    def test_curve_slope_positive_in_contango(self, engine):
        """Curve slope is positive in contango."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = engine.futures_curve(maturities)
        slope = engine.curve_slope(maturities, curve)
        assert slope > 0

    def test_curve_slope_negative_in_backwardation(self, backwardation_engine):
        """Curve slope is negative in backwardation."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = backwardation_engine.futures_curve(maturities)
        slope = backwardation_engine.curve_slope(maturities, curve)
        assert slope < 0


# ---------------------------------------------------------------------------
# Test 5: Basis analysis
# ---------------------------------------------------------------------------


class TestBasis:
    def test_basis_is_futures_minus_spot(self, engine):
        """Basis = Futures - Spot."""
        f = engine.futures_price(time_to_maturity=0.5)
        basis = engine.basis(futures_price=f)
        assert basis == pytest.approx(f - 100.0)

    def test_basis_positive_in_contango(self, engine):
        """Basis is positive in contango (futures > spot)."""
        f = engine.futures_price(time_to_maturity=1.0)
        assert engine.basis(futures_price=f) > 0

    def test_basis_negative_in_backwardation(self, backwardation_engine):
        """Basis is negative in backwardation (futures < spot)."""
        f = backwardation_engine.futures_price(time_to_maturity=1.0)
        assert backwardation_engine.basis(futures_price=f) < 0

    def test_annualized_basis(self, engine):
        """Annualized basis = (F - S) / (S * T)."""
        f = engine.futures_price(time_to_maturity=0.5)
        T = 0.5
        ann_basis = engine.annualized_basis(futures_price=f, time_to_maturity=T)
        expected = (f - 100.0) / (100.0 * T)
        assert ann_basis == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 6: Implied convenience yield
# ---------------------------------------------------------------------------


class TestImpliedConvenienceYield:
    def test_implied_convenience_yield_known_value(self):
        """Back out convenience yield from observed futures price."""
        eng = CommoditiesEngine(
            spot_price=100.0,
            risk_free_rate=0.05,
            storage_cost=0.02,
            convenience_yield=0.03,
        )
        T = 0.5
        market_f = eng.futures_price(time_to_maturity=T)
        implied_y = eng.implied_convenience_yield(
            market_futures_price=market_f, time_to_maturity=T
        )
        assert implied_y == pytest.approx(0.03, rel=1e-6)

    def test_implied_convenience_yield_zero_time_raises(self, engine):
        """Zero maturity is invalid for implied yield."""
        with pytest.raises(ValueError, match="time_to_maturity must be positive"):
            engine.implied_convenience_yield(
                market_futures_price=100.0, time_to_maturity=0.0
            )


# ---------------------------------------------------------------------------
# Test 7: Curve interpolation
# ---------------------------------------------------------------------------


class TestCurveInterpolation:
    def test_interpolate_curve_returns_price_at_target(self, engine):
        """Linear interpolation on the curve gives a price at any T."""
        maturities = np.array([0.25, 0.5, 0.75, 1.0])
        curve = engine.futures_curve(maturities)
        target_T = 0.6
        price = engine.interpolate_curve(maturities, curve, target_T)
        # Should be between the prices at T=0.5 and T=0.75
        p50 = engine.futures_price(0.5)
        p75 = engine.futures_price(0.75)
        assert p50 < price < p75

    def test_interpolate_curve_exact_at_node(self, engine):
        """Interpolation at a known maturity returns the exact price."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = engine.futures_curve(maturities)
        price = engine.interpolate_curve(maturities, curve, 0.5)
        assert price == pytest.approx(curve[1])

    def test_interpolate_curve_out_of_range_raises(self, engine):
        """Extrapolation beyond curve range is invalid."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = engine.futures_curve(maturities)
        with pytest.raises(ValueError, match="target maturity .* out of range"):
            engine.interpolate_curve(maturities, curve, 1.5)


# ---------------------------------------------------------------------------
# Test 8: Roll yield series
# ---------------------------------------------------------------------------


class TestRollYieldSeries:
    def test_roll_yield_series_length(self, engine):
        """Roll yield series has one element per contract pair."""
        near_prices = np.array([100.0, 99.0, 98.0, 97.0])
        far_prices = np.array([99.0, 98.0, 97.0, 96.0])
        ry_series = engine.roll_yield_series(near_prices, far_prices)
        assert len(ry_series) == len(near_prices)

    def test_roll_yield_series_values(self, engine):
        """Each element is (far[i] - near[i]) / near[i]."""
        near_prices = np.array([100.0, 90.0])
        far_prices = np.array([110.0, 99.0])
        ry_series = engine.roll_yield_series(near_prices, far_prices)
        expected_0 = (110.0 - 100.0) / 100.0
        expected_1 = (99.0 - 90.0) / 90.0
        assert ry_series[0] == pytest.approx(expected_0)
        assert ry_series[1] == pytest.approx(expected_1)

    def test_roll_yield_series_mismatched_lengths_raises(self, engine):
        """Mismatched array lengths are invalid."""
        with pytest.raises(ValueError, match="near_prices and far_prices must have same length"):
            engine.roll_yield_series(
                near_prices=np.array([100.0, 99.0]),
                far_prices=np.array([99.0]),
            )


# ---------------------------------------------------------------------------
# Test 9: Summary statistics
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    def test_curve_summary_returns_dict(self, engine):
        """Summary returns a dict with expected keys."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = engine.futures_curve(maturities)
        summary = engine.curve_summary(maturities, curve)
        assert isinstance(summary, dict)
        assert "slope" in summary
        assert "is_contango" in summary
        assert "is_backwardation" in summary
        assert "min_price" in summary
        assert "max_price" in summary

    def test_curve_summary_contango(self, engine):
        """Summary correctly identifies contango."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = engine.futures_curve(maturities)
        summary = engine.curve_summary(maturities, curve)
        assert summary["is_contango"] is True
        assert summary["is_backwardation"] is False

    def test_curve_summary_backwardation(self, backwardation_engine):
        """Summary correctly identifies backwardation."""
        maturities = np.array([0.25, 0.5, 1.0])
        curve = backwardation_engine.futures_curve(maturities)
        summary = backwardation_engine.curve_summary(maturities, curve)
        assert summary["is_contango"] is False
        assert summary["is_backwardation"] is True


# ---------------------------------------------------------------------------
# Test 10: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_spot_raises(self):
        """Negative spot price is invalid."""
        with pytest.raises(ValueError, match="spot_price must be positive"):
            CommoditiesEngine(spot_price=-10.0)

    def test_negative_risk_free_rate_raises(self):
        """Negative risk-free rate is invalid."""
        with pytest.raises(ValueError, match="risk_free_rate must be non-negative"):
            CommoditiesEngine(spot_price=100.0, risk_free_rate=-0.01)

    def test_negative_storage_cost_raises(self):
        """Negative storage cost is invalid."""
        with pytest.raises(ValueError, match="storage_cost must be non-negative"):
            CommoditiesEngine(spot_price=100.0, storage_cost=-0.01)

    def test_negative_convenience_yield_raises(self):
        """Negative convenience yield is invalid."""
        with pytest.raises(ValueError, match="convenience_yield must be non-negative"):
            CommoditiesEngine(spot_price=100.0, convenience_yield=-0.01)
