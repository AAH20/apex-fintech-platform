"""Tests for fixed income analytics engine.

Covers: bond pricing, yield curve construction, duration/convexity,
credit spread analysis, and DV01.

References:
    Fabozzi, F. (2016). Fixed Income Analysis, 3rd ed. Wiley.
    Tuckman, B. & Serrat, A. (2012). Fixed Income Securities, 3rd ed. Wiley.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fixedincome.analytics import FixedIncomeEngine


# ---------------------------------------------------------------------------
# Bond pricing
# ---------------------------------------------------------------------------


class TestBondPricing:
    """Tests for bond price calculation."""

    def test_zero_coupon_bond_price(self):
        """Zero-coupon bond price = face / (1 + y)^T."""
        engine = FixedIncomeEngine()
        price = engine.price_zero_coupon(face=1000.0, ytm=0.05, maturity=5.0)
        expected = 1000.0 / (1.05**5)
        assert price == pytest.approx(expected, rel=1e-10)

    def test_coupon_bond_price_at_par(self):
        """Bond priced at par when coupon rate equals YTM."""
        engine = FixedIncomeEngine()
        price = engine.price_coupon_bond(
            face=1000.0, coupon_rate=0.05, ytm=0.05, maturity=10.0, freq=2
        )
        assert price == pytest.approx(1000.0, abs=1e-6)

    def test_coupon_bond_price_premium(self):
        """Bond trades at premium when coupon > YTM."""
        engine = FixedIncomeEngine()
        price = engine.price_coupon_bond(
            face=1000.0, coupon_rate=0.08, ytm=0.05, maturity=10.0, freq=2
        )
        assert price > 1000.0

    def test_coupon_bond_price_discount(self):
        """Bond trades at discount when coupon < YTM."""
        engine = FixedIncomeEngine()
        price = engine.price_coupon_bond(
            face=1000.0, coupon_rate=0.03, ytm=0.06, maturity=10.0, freq=2
        )
        assert price < 1000.0

    def test_coupon_bond_price_semi_annual(self):
        """Semi-annual coupon bond pricing matches manual calculation."""
        engine = FixedIncomeEngine()
        price = engine.price_coupon_bond(
            face=1000.0, coupon_rate=0.06, ytm=0.05, maturity=5.0, freq=2
        )
        # Manual: 30 * annuity(2.5%, 10) + 1000 / 1.025^10
        periods = 10
        c = 30.0  # dollar coupon per period = 1000 * 0.06 / 2
        y = 0.025
        expected = c * (1 - (1 + y) ** (-periods)) / y + 1000.0 / (1 + y) ** periods
        assert price == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Yield to maturity
# ---------------------------------------------------------------------------


class TestYieldToMaturity:
    """Tests for YTM calculation."""

    def test_ytm_at_par(self):
        """YTM equals coupon rate when bond is priced at par."""
        engine = FixedIncomeEngine()
        ytm = engine.yield_to_maturity(
            price=1000.0, face=1000.0, coupon_rate=0.05, maturity=10.0, freq=2
        )
        assert ytm == pytest.approx(0.05, abs=1e-6)

    def test_ytm_premium_bond(self):
        """YTM < coupon rate for premium bond."""
        engine = FixedIncomeEngine()
        ytm = engine.yield_to_maturity(
            price=1100.0, face=1000.0, coupon_rate=0.06, maturity=10.0, freq=2
        )
        assert ytm < 0.06

    def test_ytm_discount_bond(self):
        """YTM > coupon rate for discount bond."""
        engine = FixedIncomeEngine()
        ytm = engine.yield_to_maturity(
            price=900.0, face=1000.0, coupon_rate=0.04, maturity=10.0, freq=2
        )
        assert ytm > 0.04


# ---------------------------------------------------------------------------
# Yield curve
# ---------------------------------------------------------------------------


class TestYieldCurve:
    """Tests for yield curve construction and interpolation."""

    def test_discount_factor(self):
        """Discount factor = 1 / (1 + spot)^T."""
        engine = FixedIncomeEngine()
        df = engine.discount_factor(spot_rate=0.05, maturity=2.0)
        assert df == pytest.approx(1.0 / (1.05**2), rel=1e-10)

    def test_spot_rate_from_discount_factor(self):
        """Spot rate = (1/df)^(1/T) - 1."""
        engine = FixedIncomeEngine()
        df = 0.9
        spot = engine.spot_rate_from_df(df, maturity=3.0)
        assert spot == pytest.approx((1.0 / 0.9) ** (1.0 / 3.0) - 1.0, rel=1e-10)

    def test_forward_rate(self):
        """Forward rate from two spot rates."""
        engine = FixedIncomeEngine()
        # 1y spot = 3%, 2y spot = 4% -> 1y1y forward ≈ 5.01%
        fwd = engine.forward_rate(spot_short=0.03, t_short=1.0, spot_long=0.04, t_long=2.0)
        expected = ((1.04**2) / (1.03**1)) - 1.0
        assert fwd == pytest.approx(expected, rel=1e-10)

    def test_yield_curve_interpolation(self):
        """Linear interpolation of spot rates."""
        engine = FixedIncomeEngine()
        maturities = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        spot_rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045])
        curve = engine.build_yield_curve(maturities, spot_rates)
        # Interpolate at 4 years (between 3y and 5y)
        rate_4y = curve.spot_rate(4.0)
        assert rate_4y == pytest.approx(0.0325, abs=1e-6)

    def test_yield_curve_extrapolation_flat(self):
        """Flat extrapolation beyond the curve endpoints."""
        engine = FixedIncomeEngine()
        maturities = np.array([1.0, 2.0, 3.0])
        spot_rates = np.array([0.02, 0.025, 0.03])
        curve = engine.build_yield_curve(maturities, spot_rates)
        assert curve.spot_rate(0.5) == pytest.approx(0.02, abs=1e-6)
        assert curve.spot_rate(5.0) == pytest.approx(0.03, abs=1e-6)


# ---------------------------------------------------------------------------
# Duration and convexity
# ---------------------------------------------------------------------------


class TestDurationConvexity:
    """Tests for duration and convexity calculations."""

    def test_macaulay_duration_zero_coupon(self):
        """Macaulay duration of zero-coupon bond equals its maturity."""
        engine = FixedIncomeEngine()
        dur = engine.macaulay_duration(
            face=1000.0, coupon_rate=0.0, ytm=0.05, maturity=7.0, freq=2
        )
        assert dur == pytest.approx(7.0, abs=1e-6)

    def test_modified_duration(self):
        """Modified duration = Macaulay duration / (1 + y/freq)."""
        engine = FixedIncomeEngine()
        mac_dur = engine.macaulay_duration(
            face=1000.0, coupon_rate=0.05, ytm=0.05, maturity=10.0, freq=2
        )
        mod_dur = engine.modified_duration(
            face=1000.0, coupon_rate=0.05, ytm=0.05, maturity=10.0, freq=2
        )
        assert mod_dur == pytest.approx(mac_dur / (1.0 + 0.05 / 2.0), rel=1e-10)

    def test_convexity_positive(self):
        """Convexity is always positive for standard bonds."""
        engine = FixedIncomeEngine()
        conv = engine.convexity(
            face=1000.0, coupon_rate=0.05, ytm=0.05, maturity=10.0, freq=2
        )
        assert conv > 0.0

    def test_dv01(self):
        """DV01 = modified duration * price * 0.0001."""
        engine = FixedIncomeEngine()
        price = engine.price_coupon_bond(
            face=1000.0, coupon_rate=0.05, ytm=0.05, maturity=10.0, freq=2
        )
        mod_dur = engine.modified_duration(
            face=1000.0, coupon_rate=0.05, ytm=0.05, maturity=10.0, freq=2
        )
        dv01 = engine.dv01(
            face=1000.0, coupon_rate=0.05, ytm=0.05, maturity=10.0, freq=2
        )
        assert dv01 == pytest.approx(mod_dur * price * 0.0001, rel=1e-10)

    def test_price_change_approximation(self):
        """Duration + convexity approximation for price change."""
        engine = FixedIncomeEngine()
        face, coupon, ytm, mat, freq = 1000.0, 0.05, 0.05, 10.0, 2
        price = engine.price_coupon_bond(face, coupon, ytm, mat, freq)
        mod_dur = engine.modified_duration(face, coupon, ytm, mat, freq)
        conv = engine.convexity(face, coupon, ytm, mat, freq)

        dy = 0.01  # 100bp increase
        # Approximate price change
        dp_approx = (-mod_dur * dy + 0.5 * conv * dy**2) * price
        price_approx = price + dp_approx

        # Actual price at new yield
        price_actual = engine.price_coupon_bond(face, coupon, ytm + dy, mat, freq)

        # Approximation should be close (within 1% of actual)
        assert abs(price_approx - price_actual) / price_actual < 0.01


# ---------------------------------------------------------------------------
# Credit spread
# ---------------------------------------------------------------------------


class TestCreditSpread:
    """Tests for credit spread analysis."""

    def test_credit_spread_basic(self):
        """Credit spread = corporate yield - risk-free yield."""
        engine = FixedIncomeEngine()
        spread = engine.credit_spread(corporate_yield=0.07, risk_free_yield=0.04)
        assert spread == pytest.approx(0.03, abs=1e-10)

    def test_z_spread(self):
        """Z-spread: constant spread added to spot curve to match price."""
        engine = FixedIncomeEngine()
        # Build a simple flat curve at 3%
        maturities = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        spot_rates = np.array([0.03] * 6)
        curve = engine.build_yield_curve(maturities, spot_rates)

        # Price a 5y bond with 5% coupon at z-spread of 0
        price_no_spread = engine.price_bond_with_spread(
            face=1000.0,
            coupon_rate=0.05,
            maturity=5.0,
            freq=2,
            yield_curve=curve,
            z_spread=0.0,
        )
        # Should be close to par (slight difference due to semi-annual vs annual)
        assert price_no_spread > 900.0

        # With positive z-spread, price should be lower
        price_with_spread = engine.price_bond_with_spread(
            face=1000.0,
            coupon_rate=0.05,
            maturity=5.0,
            freq=2,
            yield_curve=curve,
            z_spread=0.02,
        )
        assert price_with_spread < price_no_spread

    def test_credit_spread_duration(self):
        """Credit spread duration measures sensitivity to spread changes."""
        engine = FixedIncomeEngine()
        maturities = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        spot_rates = np.array([0.03] * 6)
        curve = engine.build_yield_curve(maturities, spot_rates)

        csd = engine.credit_spread_duration(
            face=1000.0,
            coupon_rate=0.05,
            maturity=5.0,
            freq=2,
            yield_curve=curve,
            z_spread=0.02,
        )
        assert csd > 0.0

    def test_price_with_credit_spread(self):
        """Bond price with credit spread < price without spread."""
        engine = FixedIncomeEngine()
        maturities = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        spot_rates = np.array([0.03] * 6)
        curve = engine.build_yield_curve(maturities, spot_rates)

        price_no_spread = engine.price_bond_with_spread(
            face=1000.0,
            coupon_rate=0.05,
            maturity=5.0,
            freq=2,
            yield_curve=curve,
            z_spread=0.0,
        )
        price_with_spread = engine.price_bond_with_spread(
            face=1000.0,
            coupon_rate=0.05,
            maturity=5.0,
            freq=2,
            yield_curve=curve,
            z_spread=0.03,
        )
        assert price_with_spread < price_no_spread


# ---------------------------------------------------------------------------
# Portfolio analytics
# ---------------------------------------------------------------------------


class TestPortfolioAnalytics:
    """Tests for portfolio-level risk metrics."""

    def test_portfolio_duration(self):
        """Portfolio duration is weighted average of bond durations."""
        engine = FixedIncomeEngine()
        # Two bonds: 5y and 10y, both at par
        price_5y = engine.price_coupon_bond(1000.0, 0.05, 0.05, 5.0, 2)
        price_10y = engine.price_coupon_bond(1000.0, 0.05, 0.05, 10.0, 2)
        dur_5y = engine.modified_duration(1000.0, 0.05, 0.05, 5.0, 2)
        dur_10y = engine.modified_duration(1000.0, 0.05, 0.05, 10.0, 2)

        # Equal weights
        weights = np.array([0.5, 0.5])
        portfolio_dur = engine.portfolio_duration(
            prices=np.array([price_5y, price_10y]),
            durations=np.array([dur_5y, dur_10y]),
            weights=weights,
        )
        expected = 0.5 * dur_5y + 0.5 * dur_10y
        assert portfolio_dur == pytest.approx(expected, rel=1e-10)

    def test_portfolio_convexity(self):
        """Portfolio convexity is weighted average of bond convexities."""
        engine = FixedIncomeEngine()
        price_5y = engine.price_coupon_bond(1000.0, 0.05, 0.05, 5.0, 2)
        price_10y = engine.price_coupon_bond(1000.0, 0.05, 0.05, 10.0, 2)
        conv_5y = engine.convexity(1000.0, 0.05, 0.05, 5.0, 2)
        conv_10y = engine.convexity(1000.0, 0.05, 0.05, 10.0, 2)

        weights = np.array([0.5, 0.5])
        portfolio_conv = engine.portfolio_convexity(
            prices=np.array([price_5y, price_10y]),
            convexities=np.array([conv_5y, conv_10y]),
            weights=weights,
        )
        expected = 0.5 * conv_5y + 0.5 * conv_10y
        assert portfolio_conv == pytest.approx(expected, rel=1e-10)
