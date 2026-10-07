"""Fixed income analytics engine.

Provides bond pricing, yield curve construction, duration/convexity,
credit spread analysis, and portfolio risk metrics.

References:
    Fabozzi, F. (2016). Fixed Income Analysis, 3rd ed. Wiley.
    Tuckman, B. & Serrat, A. (2012). Fixed Income Securities, 3rd ed. Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class YieldCurve:
    """Yield curve with linear interpolation and flat extrapolation."""

    maturities: np.ndarray
    spot_rates: np.ndarray

    def spot_rate(self, maturity: float) -> float:
        """Get spot rate at given maturity via linear interpolation."""
        return float(np.interp(maturity, self.maturities, self.spot_rates))

    def discount_factor(self, maturity: float) -> float:
        """Get discount factor at given maturity."""
        rate = self.spot_rate(maturity)
        return 1.0 / (1.0 + rate) ** maturity

    def forward_rate(self, t1: float, t2: float) -> float:
        """Get forward rate between t1 and t2."""
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (df1 / df2) ** (1.0 / (t2 - t1)) - 1.0


class FixedIncomeEngine:
    """Fixed income analytics engine for bond pricing and risk metrics."""

    # -----------------------------------------------------------------------
    # Bond pricing
    # -----------------------------------------------------------------------

    def price_zero_coupon(self, face: float, ytm: float, maturity: float) -> float:
        """Price a zero-coupon bond.

        Args:
            face: Face value (par) of the bond.
            ytm: Yield to maturity (annual, continuously compounded or simple).
            maturity: Time to maturity in years.

        Returns:
            Present value of the zero-coupon bond.
        """
        return face / (1.0 + ytm) ** maturity

    def price_coupon_bond(
        self,
        face: float,
        coupon_rate: float,
        ytm: float,
        maturity: float,
        freq: int = 2,
    ) -> float:
        """Price a coupon-paying bond.

        Args:
            face: Face value of the bond.
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%).
            ytm: Yield to maturity (annual).
            maturity: Time to maturity in years.
            freq: Coupon payments per year (default 2 = semi-annual).

        Returns:
            Present value of the coupon bond.
        """
        periods = int(maturity * freq)
        coupon = face * coupon_rate / freq
        y_per_period = ytm / freq

        # PV of coupons
        pv_coupons = coupon * (1.0 - (1.0 + y_per_period) ** (-periods)) / y_per_period
        # PV of face
        pv_face = face / (1.0 + y_per_period) ** periods

        return pv_coupons + pv_face

    # -----------------------------------------------------------------------
    # Yield to maturity
    # -----------------------------------------------------------------------

    def yield_to_maturity(
        self,
        price: float,
        face: float,
        coupon_rate: float,
        maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculate yield to maturity using Brent's method.

        Args:
            price: Current market price of the bond.
            face: Face value of the bond.
            coupon_rate: Annual coupon rate.
            maturity: Time to maturity in years.
            freq: Coupon payments per year.

        Returns:
            Yield to maturity (annual).
        """
        # Define the pricing error function
        def price_error(y: float) -> float:
            return self.price_coupon_bond(face, coupon_rate, y, maturity, freq) - price

        # Use Brent's method to find the root
        # Bracket: try a wide range
        try:
            ytm = brentq(price_error, -0.5, 1.0, xtol=1e-12, maxiter=200)
        except ValueError:
            # If the standard bracket fails, try a wider range
            ytm = brentq(price_error, -0.99, 10.0, xtol=1e-12, maxiter=200)

        return float(ytm)

    # -----------------------------------------------------------------------
    # Yield curve
    # -----------------------------------------------------------------------

    def discount_factor(self, spot_rate: float, maturity: float) -> float:
        """Calculate discount factor from spot rate.

        Args:
            spot_rate: Annual spot rate.
            maturity: Time to maturity in years.

        Returns:
            Discount factor.
        """
        return 1.0 / (1.0 + spot_rate) ** maturity

    def spot_rate_from_df(self, df: float, maturity: float) -> float:
        """Calculate spot rate from discount factor.

        Args:
            df: Discount factor.
            maturity: Time to maturity in years.

        Returns:
            Annual spot rate.
        """
        return (1.0 / df) ** (1.0 / maturity) - 1.0

    def forward_rate(
        self, spot_short: float, t_short: float, spot_long: float, t_long: float
    ) -> float:
        """Calculate forward rate between two maturities.

        Args:
            spot_short: Spot rate at the shorter maturity.
            t_short: Shorter maturity in years.
            spot_long: Spot rate at the longer maturity.
            t_long: Longer maturity in years.

        Returns:
            Forward rate between t_short and t_long.
        """
        df_short = self.discount_factor(spot_short, t_short)
        df_long = self.discount_factor(spot_long, t_long)
        return (df_short / df_long) ** (1.0 / (t_long - t_short)) - 1.0

    def build_yield_curve(
        self, maturities: np.ndarray, spot_rates: np.ndarray
    ) -> YieldCurve:
        """Build a yield curve from market data.

        Args:
            maturities: Array of maturities in years.
            spot_rates: Array of spot rates corresponding to maturities.

        Returns:
            YieldCurve object with interpolation.
        """
        return YieldCurve(maturities=np.array(maturities), spot_rates=np.array(spot_rates))

    # -----------------------------------------------------------------------
    # Duration and convexity
    # -----------------------------------------------------------------------

    def macaulay_duration(
        self,
        face: float,
        coupon_rate: float,
        ytm: float,
        maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculate Macaulay duration.

        Args:
            face: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            maturity: Time to maturity in years.
            freq: Coupon payments per year.

        Returns:
            Macaulay duration in years.
        """
        periods = int(maturity * freq)
        coupon = face * coupon_rate / freq
        y_per_period = ytm / freq

        # Weighted average time to cash flows
        weighted_time = 0.0
        for t in range(1, periods + 1):
            time_in_years = t / freq
            cf = coupon if t < periods else coupon + face
            pv_cf = cf / (1.0 + y_per_period) ** t
            weighted_time += time_in_years * pv_cf

        price = self.price_coupon_bond(face, coupon_rate, ytm, maturity, freq)
        return weighted_time / price

    def modified_duration(
        self,
        face: float,
        coupon_rate: float,
        ytm: float,
        maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculate modified duration.

        Args:
            face: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            maturity: Time to maturity in years.
            freq: Coupon payments per year.

        Returns:
            Modified duration.
        """
        mac_dur = self.macaulay_duration(face, coupon_rate, ytm, maturity, freq)
        return mac_dur / (1.0 + ytm / freq)

    def convexity(
        self,
        face: float,
        coupon_rate: float,
        ytm: float,
        maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculate convexity.

        Args:
            face: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            maturity: Time to maturity in years.
            freq: Coupon payments per year.

        Returns:
            Convexity measure.
        """
        periods = int(maturity * freq)
        coupon = face * coupon_rate / freq
        y_per_period = ytm / freq

        # Convexity = (1/P) * sum[ t*(t+1) * CF_t / (1+y)^(t+2) ] / freq^2
        convexity_sum = 0.0
        for t in range(1, periods + 1):
            cf = coupon if t < periods else coupon + face
            pv_cf = cf / (1.0 + y_per_period) ** (t + 2)
            convexity_sum += t * (t + 1) * pv_cf

        price = self.price_coupon_bond(face, coupon_rate, ytm, maturity, freq)
        return convexity_sum / (price * freq**2)

    def dv01(
        self,
        face: float,
        coupon_rate: float,
        ytm: float,
        maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculate DV01 (dollar value of 1 basis point).

        Args:
            face: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            maturity: Time to maturity in years.
            freq: Coupon payments per year.

        Returns:
            DV01 = modified_duration * price * 0.0001.
        """
        price = self.price_coupon_bond(face, coupon_rate, ytm, maturity, freq)
        mod_dur = self.modified_duration(face, coupon_rate, ytm, maturity, freq)
        return mod_dur * price * 0.0001

    # -----------------------------------------------------------------------
    # Credit spread
    # -----------------------------------------------------------------------

    def credit_spread(self, corporate_yield: float, risk_free_yield: float) -> float:
        """Calculate credit spread.

        Args:
            corporate_yield: Yield on the corporate bond.
            risk_free_yield: Risk-free yield (e.g., Treasury).

        Returns:
            Credit spread (corporate_yield - risk_free_yield).
        """
        return corporate_yield - risk_free_yield

    def price_bond_with_spread(
        self,
        face: float,
        coupon_rate: float,
        maturity: float,
        freq: int,
        yield_curve: YieldCurve,
        z_spread: float,
    ) -> float:
        """Price a bond using a yield curve and z-spread.

        Args:
            face: Face value of the bond.
            coupon_rate: Annual coupon rate.
            maturity: Time to maturity in years.
            freq: Coupon payments per year.
            yield_curve: YieldCurve object.
            z_spread: Zero-volatility spread to add to the spot curve.

        Returns:
            Bond price.
        """
        periods = int(maturity * freq)
        coupon = face * coupon_rate / freq

        price = 0.0
        for t in range(1, periods + 1):
            time_in_years = t / freq
            spot = yield_curve.spot_rate(time_in_years)
            y = spot + z_spread
            df = 1.0 / (1.0 + y / freq) ** t
            cf = coupon if t < periods else coupon + face
            price += cf * df

        return price

    def credit_spread_duration(
        self,
        face: float,
        coupon_rate: float,
        maturity: float,
        freq: int,
        yield_curve: YieldCurve,
        z_spread: float,
    ) -> float:
        """Calculate credit spread duration.

        Measures sensitivity of bond price to changes in credit spread.

        Args:
            face: Face value of the bond.
            coupon_rate: Annual coupon rate.
            maturity: Time to maturity in years.
            freq: Coupon payments per year.
            yield_curve: YieldCurve object.
            z_spread: Current z-spread.

        Returns:
            Credit spread duration.
        """
        # Use numerical differentiation
        dy = 0.0001  # 1 bp
        price_up = self.price_bond_with_spread(
            face, coupon_rate, maturity, freq, yield_curve, z_spread + dy
        )
        price_down = self.price_bond_with_spread(
            face, coupon_rate, maturity, freq, yield_curve, z_spread - dy
        )
        price = self.price_bond_with_spread(
            face, coupon_rate, maturity, freq, yield_curve, z_spread
        )

        # CSD = -(1/P) * dP/dy
        return -(price_up - price_down) / (2.0 * dy * price)

    # -----------------------------------------------------------------------
    # Portfolio analytics
    # -----------------------------------------------------------------------

    def portfolio_duration(
        self, prices: np.ndarray, durations: np.ndarray, weights: np.ndarray
    ) -> float:
        """Calculate portfolio duration as weighted average.

        Args:
            prices: Array of bond prices.
            durations: Array of bond durations.
            weights: Array of portfolio weights (must sum to 1).

        Returns:
            Portfolio duration.
        """
        return float(np.sum(weights * durations))

    def portfolio_convexity(
        self, prices: np.ndarray, convexities: np.ndarray, weights: np.ndarray
    ) -> float:
        """Calculate portfolio convexity as weighted average.

        Args:
            prices: Array of bond prices.
            convexities: Array of bond convexities.
            weights: Array of portfolio weights (must sum to 1).

        Returns:
            Portfolio convexity.
        """
        return float(np.sum(weights * convexities))
