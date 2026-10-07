"""Tests for trading analytics engine — market impact, liquidity, TCA."""
import time

import numpy as np
import pytest

from trading import TradingAnalyticsEngine


# ---------------------------------------------------------------------------
# Market impact models
# ---------------------------------------------------------------------------


class TestSqrtImpact:
    """Tests for the square-root market impact model."""

    def test_sqrt_impact_basic(self):
        """Basic sqrt impact: Y * sigma * sqrt(Q/V)."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        impact = engine.sqrt_impact(quantity=1000, daily_volume=100_000, volatility=0.02)
        # 0.5 * 0.02 * sqrt(1000/100000) = 0.5 * 0.02 * 0.1 = 0.001
        assert impact == pytest.approx(0.001)

    def test_sqrt_impact_scales_with_sqrt_quantity(self):
        """Impact scales with sqrt(Q): 4x quantity -> 2x impact."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        impact1 = engine.sqrt_impact(quantity=1000, daily_volume=100_000, volatility=0.02)
        impact2 = engine.sqrt_impact(quantity=4000, daily_volume=100_000, volatility=0.02)
        assert impact2 == pytest.approx(2 * impact1)

    def test_sqrt_impact_scales_with_volatility(self):
        """Impact scales linearly with volatility."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        impact1 = engine.sqrt_impact(quantity=1000, daily_volume=100_000, volatility=0.01)
        impact2 = engine.sqrt_impact(quantity=1000, daily_volume=100_000, volatility=0.02)
        assert impact2 == pytest.approx(2 * impact1)

    def test_sqrt_impact_decreases_with_volume(self):
        """Impact decreases with sqrt of volume: 4x volume -> 0.5x impact."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        impact1 = engine.sqrt_impact(quantity=1000, daily_volume=100_000, volatility=0.02)
        impact2 = engine.sqrt_impact(quantity=1000, daily_volume=400_000, volatility=0.02)
        assert impact2 == pytest.approx(0.5 * impact1)

    def test_sqrt_impact_custom_coefficient(self):
        """Custom Y coefficient scales impact proportionally."""
        engine_default = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        engine_high = TradingAnalyticsEngine(sqrt_coefficient=1.0)
        impact_default = engine_default.sqrt_impact(1000, 100_000, 0.02)
        impact_high = engine_high.sqrt_impact(1000, 100_000, 0.02)
        assert impact_high == pytest.approx(2 * impact_default)


class TestLinearImpact:
    """Tests for the linear market impact model."""

    def test_linear_impact_basic(self):
        """Basic linear impact: coefficient * sigma * (Q/V)."""
        engine = TradingAnalyticsEngine(linear_coefficient=0.1)
        impact = engine.linear_impact(quantity=1000, daily_volume=100_000, volatility=0.02)
        # 0.1 * 0.02 * (1000/100000) = 0.1 * 0.02 * 0.01 = 0.00002
        assert impact == pytest.approx(0.00002)

    def test_linear_impact_scales_with_quantity(self):
        """Linear impact scales linearly with quantity."""
        engine = TradingAnalyticsEngine(linear_coefficient=0.1)
        impact1 = engine.linear_impact(quantity=1000, daily_volume=100_000, volatility=0.02)
        impact2 = engine.linear_impact(quantity=2000, daily_volume=100_000, volatility=0.02)
        assert impact2 == pytest.approx(2 * impact1)


class TestImpactDecomposition:
    """Tests for temporary/permanent impact decomposition."""

    def test_decomposition_sums_to_total(self):
        """Temporary + permanent = total impact."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        result = engine.decompose_impact(
            quantity=1000, daily_volume=100_000, volatility=0.02, decay=0.5
        )
        assert result.temporary + result.permanent == pytest.approx(result.total)

    def test_decomposition_with_decay(self):
        """With decay=0.3, temporary is 30% of total."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        result = engine.decompose_impact(
            quantity=1000, daily_volume=100_000, volatility=0.02, decay=0.3
        )
        total = 0.5 * 0.02 * np.sqrt(1000 / 100_000)
        assert result.total == pytest.approx(total)
        assert result.temporary == pytest.approx(total * 0.3)
        assert result.permanent == pytest.approx(total * 0.7)

    def test_decomposition_zero_decay_all_permanent(self):
        """With decay=0, all impact is permanent."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        result = engine.decompose_impact(
            quantity=1000, daily_volume=100_000, volatility=0.02, decay=0.0
        )
        assert result.temporary == pytest.approx(0.0)
        assert result.permanent == pytest.approx(result.total)


# ---------------------------------------------------------------------------
# Liquidity analysis
# ---------------------------------------------------------------------------


class TestSpreadAnalysis:
    """Tests for bid-ask spread analysis."""

    def test_spread_basic(self):
        """Spread = ask - bid, spread_bps = spread/mid * 10000."""
        engine = TradingAnalyticsEngine()
        result = engine.spread_metrics(bid=99.5, ask=100.5, mid=100.0)
        assert result.spread == pytest.approx(1.0)
        assert result.spread_bps == pytest.approx(100.0)

    def test_spread_tight(self):
        """Tight spread of 1 cent on $100 stock = 1 bp."""
        engine = TradingAnalyticsEngine()
        result = engine.spread_metrics(bid=99.99, ask=100.00, mid=99.995)
        assert result.spread == pytest.approx(0.01)
        assert result.spread_bps == pytest.approx(1.0, abs=0.1)


class TestOrderBookImbalance:
    """Tests for order book imbalance."""

    def test_imbalance_basic(self):
        """Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        engine = TradingAnalyticsEngine()
        bid_vols = np.array([100, 200, 300])
        ask_vols = np.array([150, 250, 350])
        imbalance = engine.book_imbalance(bid_vols, ask_vols)
        # (600 - 750) / (600 + 750) = -150/1350
        assert imbalance == pytest.approx(-150 / 1350, abs=0.001)

    def test_imbalance_balanced(self):
        """Balanced book -> imbalance = 0."""
        engine = TradingAnalyticsEngine()
        bid_vols = np.array([100, 200, 300])
        ask_vols = np.array([100, 200, 300])
        imbalance = engine.book_imbalance(bid_vols, ask_vols)
        assert imbalance == pytest.approx(0.0)

    def test_imbalance_bid_heavy(self):
        """More bid volume -> positive imbalance."""
        engine = TradingAnalyticsEngine()
        bid_vols = np.array([500, 500])
        ask_vols = np.array([100, 100])
        imbalance = engine.book_imbalance(bid_vols, ask_vols)
        assert imbalance > 0


class TestAmihudIlliquidity:
    """Tests for Amihud illiquidity measure."""

    def test_amihud_basic(self):
        """Amihud = mean(|return| / volume)."""
        engine = TradingAnalyticsEngine()
        returns = np.array([0.01, -0.02, 0.015])
        volumes = np.array([1000, 2000, 1500])
        amihud = engine.amihud_illiquidity(returns, volumes)
        # |r|/v = [0.00001, 0.00001, 0.00001] -> mean = 0.00001
        assert amihud == pytest.approx(0.00001)

    def test_amihud_higher_for_illiquid(self):
        """Same returns with lower volume -> higher Amihud."""
        engine = TradingAnalyticsEngine()
        returns = np.array([0.01, 0.02])
        amihud_high = engine.amihud_illiquidity(returns, np.array([100, 100]))
        amihud_low = engine.amihud_illiquidity(returns, np.array([10000, 10000]))
        assert amihud_high > amihud_low


class TestKyleLambda:
    """Tests for Kyle's lambda estimation."""

    def test_kyle_lambda_perfect_linear(self):
        """Perfect linear relationship: delta_price = 0.001 * order_flow."""
        engine = TradingAnalyticsEngine()
        price_changes = np.array([0.1, 0.2, 0.3, 0.4])
        order_flow = np.array([100, 200, 300, 400])
        lam = engine.kyle_lambda(price_changes, order_flow)
        assert lam == pytest.approx(0.001)

    def test_kyle_lambda_positive(self):
        """Positive order flow -> positive price impact -> positive lambda."""
        engine = TradingAnalyticsEngine()
        price_changes = np.array([0.0, 0.1, 0.2, 0.3])
        order_flow = np.array([0, 100, 200, 300])
        lam = engine.kyle_lambda(price_changes, order_flow)
        assert lam > 0


# ---------------------------------------------------------------------------
# Transaction cost analysis
# ---------------------------------------------------------------------------


class TestVWAPSlippage:
    """Tests for VWAP slippage computation."""

    def test_vwap_slippage_buy(self):
        """Buy slippage = (VWAP - arrival) / arrival * 10000."""
        engine = TradingAnalyticsEngine()
        prices = np.array([100.0, 101.0, 102.0])
        volumes = np.array([1000, 2000, 3000])
        slippage = engine.vwap_slippage(prices, volumes, arrival_price=100.0, side="buy")
        # VWAP = 608000/6000 = 101.333...
        # slippage = (101.333 - 100) / 100 * 10000 = 133.33 bps
        assert slippage == pytest.approx(133.33, abs=0.1)

    def test_vwap_slippage_sell(self):
        """Sell slippage = (arrival - VWAP) / arrival * 10000."""
        engine = TradingAnalyticsEngine()
        prices = np.array([99.0, 98.0, 97.0])
        volumes = np.array([1000, 2000, 3000])
        slippage = engine.vwap_slippage(prices, volumes, arrival_price=100.0, side="sell")
        # VWAP = (99*1000 + 98*2000 + 97*3000) / 6000 = 586000/6000 = 97.6667
        # slippage = (100 - 97.6667) / 100 * 10000 = 233.33 bps
        assert slippage == pytest.approx(233.33, abs=0.1)

    def test_vwap_slippage_no_slippage(self):
        """VWAP = arrival -> zero slippage."""
        engine = TradingAnalyticsEngine()
        prices = np.array([100.0, 100.0, 100.0])
        volumes = np.array([1000, 2000, 3000])
        slippage = engine.vwap_slippage(prices, volumes, arrival_price=100.0, side="buy")
        assert slippage == pytest.approx(0.0)


class TestImplementationShortfall:
    """Tests for implementation shortfall."""

    def test_is_buy(self):
        """Buy IS = (avg_exec - arrival) / arrival * 10000."""
        engine = TradingAnalyticsEngine()
        is_bps = engine.implementation_shortfall(
            arrival_price=100.0,
            execution_prices=np.array([101.0, 102.0]),
            quantities=np.array([500, 500]),
            side="buy",
        )
        # avg = 101.5, IS = (101.5 - 100) / 100 * 10000 = 150 bps
        assert is_bps == pytest.approx(150.0)

    def test_is_sell(self):
        """Sell IS = (arrival - avg_exec) / arrival * 10000."""
        engine = TradingAnalyticsEngine()
        is_bps = engine.implementation_shortfall(
            arrival_price=100.0,
            execution_prices=np.array([99.0, 98.0]),
            quantities=np.array([500, 500]),
            side="sell",
        )
        # avg = 98.5, IS = (100 - 98.5) / 100 * 10000 = 150 bps
        assert is_bps == pytest.approx(150.0)

    def test_is_zero_when_at_arrival(self):
        """IS = 0 when execution at arrival price."""
        engine = TradingAnalyticsEngine()
        is_bps = engine.implementation_shortfall(
            arrival_price=100.0,
            execution_prices=np.array([100.0, 100.0]),
            quantities=np.array([500, 500]),
            side="buy",
        )
        assert is_bps == pytest.approx(0.0)


class TestTotalCostAnalysis:
    """Tests for full TCA decomposition."""

    def test_total_cost_basic(self):
        """Total cost = IS + commission."""
        engine = TradingAnalyticsEngine()
        result = engine.total_cost(
            arrival_price=100.0,
            execution_prices=np.array([101.0, 102.0]),
            quantities=np.array([500, 500]),
            commission=10.0,
            side="buy",
        )
        # IS = 150 bps, commission = 10 / (100*1000) * 10000 = 1 bp
        # total_cost_bps = 151, total_cost = 151/10000 * 100000 = 1510
        assert result.implementation_shortfall_bps == pytest.approx(150.0)
        assert result.commission_bps == pytest.approx(1.0)
        assert result.total_cost_bps == pytest.approx(151.0)
        assert result.total_cost == pytest.approx(1510.0)

    def test_total_cost_with_market_impact(self):
        """With volume/volatility, market impact is estimated via sqrt model."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        result = engine.total_cost(
            arrival_price=100.0,
            execution_prices=np.array([101.0, 102.0]),
            quantities=np.array([500, 500]),
            commission=10.0,
            side="buy",
            daily_volume=100_000,
            volatility=0.02,
        )
        # market_impact = 0.5 * 0.02 * sqrt(1000/100000) * 10000 = 10 bps
        # timing_cost = IS - market_impact = 150 - 10 = 140 bps
        assert result.market_impact_bps == pytest.approx(10.0)
        assert result.timing_cost_bps == pytest.approx(140.0)

    def test_total_cost_sell_side(self):
        """Sell side TCA works correctly."""
        engine = TradingAnalyticsEngine()
        result = engine.total_cost(
            arrival_price=100.0,
            execution_prices=np.array([99.0, 98.0]),
            quantities=np.array([500, 500]),
            commission=10.0,
            side="sell",
        )
        # IS = 150 bps (sell), commission = 1 bp
        assert result.implementation_shortfall_bps == pytest.approx(150.0)
        assert result.total_cost_bps == pytest.approx(151.0)


# ---------------------------------------------------------------------------
# Participation rate and convenience methods
# ---------------------------------------------------------------------------


class TestParticipationRate:
    """Tests for participation rate computation."""

    def test_participation_rate_basic(self):
        """Participation = Q / V."""
        engine = TradingAnalyticsEngine()
        rate = engine.participation_rate(quantity=1000, daily_volume=100_000)
        assert rate == pytest.approx(0.01)

    def test_participation_rate_high(self):
        """High participation rate."""
        engine = TradingAnalyticsEngine()
        rate = engine.participation_rate(quantity=50_000, daily_volume=100_000)
        assert rate == pytest.approx(0.5)


class TestExpectedImpactBps:
    """Tests for expected impact in bps."""

    def test_expected_impact_bps(self):
        """Expected impact in bps = sqrt_impact * 10000."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        impact_bps = engine.expected_impact_bps(quantity=1000, daily_volume=100_000, volatility=0.02)
        # 0.001 * 10000 = 10 bps
        assert impact_bps == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and validation tests."""

    def test_zero_volume_raises(self):
        """Zero daily volume raises ValueError."""
        engine = TradingAnalyticsEngine()
        with pytest.raises(ValueError, match="daily_volume must be positive"):
            engine.sqrt_impact(quantity=1000, daily_volume=0, volatility=0.02)

    def test_zero_volatility_zero_impact(self):
        """Zero volatility -> zero impact."""
        engine = TradingAnalyticsEngine()
        impact = engine.sqrt_impact(quantity=1000, daily_volume=100_000, volatility=0.0)
        assert impact == pytest.approx(0.0)

    def test_negative_quantity_raises(self):
        """Negative quantity raises ValueError."""
        engine = TradingAnalyticsEngine()
        with pytest.raises(ValueError, match="quantity must be non-negative"):
            engine.sqrt_impact(quantity=-1000, daily_volume=100_000, volatility=0.02)

    def test_invalid_side_raises(self):
        """Invalid side raises ValueError."""
        engine = TradingAnalyticsEngine()
        with pytest.raises(ValueError, match="side must be"):
            engine.vwap_slippage(
                prices=np.array([100.0]),
                volumes=np.array([1000]),
                arrival_price=100.0,
                side="invalid",
            )

    def test_empty_arrays_raise(self):
        """Empty price/volume arrays raise ValueError."""
        engine = TradingAnalyticsEngine()
        with pytest.raises(ValueError, match="must not be empty"):
            engine.vwap_slippage(
                prices=np.array([]),
                volumes=np.array([]),
                arrival_price=100.0,
                side="buy",
            )

    def test_mismatched_array_lengths_raise(self):
        """Mismatched price/volume array lengths raise ValueError."""
        engine = TradingAnalyticsEngine()
        with pytest.raises(ValueError, match="same shape"):
            engine.vwap_slippage(
                prices=np.array([100.0, 101.0]),
                volumes=np.array([1000]),
                arrival_price=100.0,
                side="buy",
            )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance tests."""

    def test_sqrt_impact_1000_under_100ms(self):
        """1000 sqrt impact computations in < 100ms."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        t0 = time.perf_counter()
        for i in range(1000):
            engine.sqrt_impact(
                quantity=1000 + i,
                daily_volume=100_000 + i * 10,
                volatility=0.02,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"1000 sqrt impacts took {elapsed_ms:.1f}ms"

    def test_total_cost_100_under_100ms(self):
        """100 TCA computations in < 100ms."""
        engine = TradingAnalyticsEngine(sqrt_coefficient=0.5)
        prices = np.array([100.0, 101.0, 102.0])
        volumes = np.array([1000, 2000, 3000])
        t0 = time.perf_counter()
        for _ in range(100):
            engine.total_cost(
                arrival_price=100.0,
                execution_prices=prices,
                quantities=volumes,
                commission=10.0,
                side="buy",
                daily_volume=100_000,
                volatility=0.02,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"100 TCA computations took {elapsed_ms:.1f}ms"
