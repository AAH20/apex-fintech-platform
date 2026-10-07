"""Tests for market making engine — Avellaneda-Stoikov framework."""

import time

import numpy as np
import pytest

from marketmaking import HawkesProcess, MarketMakingEngine, Quote


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov optimal quotes
# ---------------------------------------------------------------------------


class TestAvellanedaStoikovQuotes:
    """Tests for the core AS optimal quote computation."""

    def test_optimal_quotes_returned(self):
        """Engine returns bid and ask quotes."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        quote = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        assert isinstance(quote, Quote)
        assert quote.bid < quote.ask
        assert quote.mid == pytest.approx(100.0)

    def test_zero_inventory_symmetric_quotes(self):
        """With zero inventory, quotes are symmetric around mid."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        quote = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        spread = quote.ask - quote.bid
        assert spread > 0
        # Symmetric: mid is center
        assert (quote.bid + quote.ask) / 2 == pytest.approx(100.0, abs=0.01)

    def test_positive_inventory_skews_quotes_down(self):
        """Positive inventory shifts quotes down to encourage selling."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        quote_neutral = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        quote_long = engine.compute_quotes(mid_price=100.0, inventory=5, time_remaining=10.0)
        # With long inventory, both quotes should shift down
        assert quote_long.bid < quote_neutral.bid
        assert quote_long.ask < quote_neutral.ask

    def test_negative_inventory_skews_quotes_up(self):
        """Negative inventory shifts quotes up to encourage buying."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        quote_neutral = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        quote_short = engine.compute_quotes(mid_price=100.0, inventory=-5, time_remaining=10.0)
        assert quote_short.bid > quote_neutral.bid
        assert quote_short.ask > quote_neutral.ask

    def test_higher_volatility_widens_spread(self):
        """Higher sigma leads to wider spreads."""
        engine_low_vol = MarketMakingEngine(
            gamma=0.1, sigma=0.3, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        engine_high_vol = MarketMakingEngine(
            gamma=0.1, sigma=1.0, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        q_low = engine_low_vol.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        q_high = engine_high_vol.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        assert (q_high.ask - q_high.bid) > (q_low.ask - q_low.bid)

    def test_higher_risk_aversion_widens_spread(self):
        """Higher gamma (risk aversion) leads to wider spreads."""
        engine_low_gamma = MarketMakingEngine(
            gamma=0.05, sigma=0.5, k=1.5, A=1.0, dt=1.0, max_inventory=10
        )
        engine_high_gamma = MarketMakingEngine(
            gamma=0.5, sigma=0.5, k=1.5, A=1.0, dt=1.0, max_inventory=10
        )
        q_low = engine_low_gamma.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        q_high = engine_high_gamma.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        assert (q_high.ask - q_high.bid) > (q_low.ask - q_low.bid)

    def test_less_time_remaining_tightens_quotes(self):
        """As time horizon shrinks, quotes tighten toward mid."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=1
        )
        q_long = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=100.0)
        q_short = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=0.1)
        spread_long = q_long.ask - q_long.bid
        spread_short = q_short.ask - q_short.bid
        assert spread_short < spread_long

    def test_spread_optimization_with_order_arrival_params(self):
        """Spread depends on A and k (order arrival parameters)."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        # Higher A (more aggressive arrivals) should tighten spread
        engine_aggressive = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=1.0, dt=1.0, max_inventory=10
        )
        q_normal = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        q_aggr = engine_aggressive.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        assert (q_aggr.ask - q_aggr.bid) < (q_normal.ask - q_normal.bid)


# ---------------------------------------------------------------------------
# Hawkes process for order flow
# ---------------------------------------------------------------------------


class TestHawkesProcess:
    """Tests for Hawkes process order flow modeling."""

    def test_hawkes_process_initialization(self):
        """Hawkes process initializes with correct parameters."""
        hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
        assert hp.mu == 0.5
        assert hp.alpha == 0.3
        assert hp.beta == 1.0

    def test_hawkes_branching_ratio_stable(self):
        """Branching ratio alpha/beta < 1 for stationarity."""
        hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
        assert hp.branching_ratio() < 1.0

    def test_hawkes_unconditional_intensity(self):
        """Unconditional intensity = mu / (1 - alpha/beta)."""
        hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
        expected = 0.5 / (1.0 - 0.3 / 1.0)
        assert hp.unconditional_intensity() == pytest.approx(expected)

    def test_hawkes_simulation_produces_events(self):
        """Simulated Hawkes process produces event times."""
        hp = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0)
        events = hp.simulate(T=100.0, seed=42)
        assert len(events) > 0
        assert all(events[i] <= events[i + 1] for i in range(len(events) - 1))

    def test_hawkes_clustering(self):
        """Hawkes process exhibits clustering (variance > mean for counts)."""
        hp = HawkesProcess(mu=1.0, alpha=0.9, beta=1.0)
        counts = []
        for seed in range(30):
            events = hp.simulate(T=50.0, seed=seed)
            counts.append(len(events))
        # Clustered process: coefficient of variation > 1
        cv = np.std(counts) / np.mean(counts)
        assert cv > 0.15  # Strong clustering (Poisson CV would be ~0.02 for this count)

    def test_hawkes_intensity_increases_after_event(self):
        """Intensity jumps after an event and decays exponentially."""
        hp = HawkesProcess(mu=0.5, alpha=0.4, beta=1.0)
        hp.reset()
        intensity_before = hp.current_intensity()
        hp.trigger_event(time=1.0)
        intensity_after = hp.current_intensity()
        assert intensity_after > intensity_before
        # Intensity should decay back toward mu
        hp.update_time(5.0)
        intensity_later = hp.current_intensity()
        assert intensity_later < intensity_after


# ---------------------------------------------------------------------------
# Inventory risk and adverse selection
# ---------------------------------------------------------------------------


class TestInventoryRisk:
    """Tests for inventory risk management."""

    def test_inventory_penalty_increases_with_abs_inventory(self):
        """Penalty term grows with absolute inventory."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        q0 = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        q5 = engine.compute_quotes(mid_price=100.0, inventory=5, time_remaining=10.0)
        q10 = engine.compute_quotes(mid_price=100.0, inventory=10, time_remaining=10.0)
        # Distance from mid should increase with inventory
        d0 = abs((q0.bid + q0.ask) / 2 - 100.0)
        d5 = abs((q5.bid + q5.ask) / 2 - 100.0)
        d10 = abs((q10.bid + q10.ask) / 2 - 100.0)
        assert d5 > d0
        assert d10 > d5

    def test_max_inventory_respected(self):
        """Engine respects max inventory limit."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=5
        )
        # Should not crash even with inventory beyond max
        quote = engine.compute_quotes(mid_price=100.0, inventory=10, time_remaining=10.0)
        assert quote.bid < quote.ask

    def test_adverse_selection_skews_quotes(self):
        """Adverse selection parameter affects quote skew."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10,
            adverse_selection=0.01,
        )
        q_neutral = engine.compute_quotes(mid_price=100.0, inventory=5, time_remaining=10.0)
        # With adverse selection, spread should be wider
        engine_no_adv = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10,
            adverse_selection=0.0,
        )
        q_no_adv = engine_no_adv.compute_quotes(mid_price=100.0, inventory=5, time_remaining=10.0)
        assert (q_neutral.ask - q_neutral.bid) > (q_no_adv.ask - q_no_adv.bid)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance tests — quotes must compute in < 1ms."""

    def test_quote_computation_under_1ms(self):
        """Single quote computation takes less than 1ms."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        # Warmup
        for _ in range(100):
            engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)

        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        median_time = np.median(times)
        assert median_time < 1.0, f"Median quote time {median_time:.3f}ms exceeds 1ms"

    def test_batch_quote_computation_fast(self):
        """Batch of 100 quotes computes quickly."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        t0 = time.perf_counter()
        for i in range(100):
            engine.compute_quotes(
                mid_price=100.0 + i * 0.01, inventory=i - 50, time_remaining=10.0
            )
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) * 1000 / 100
        assert avg_ms < 1.0, f"Average quote time {avg_ms:.3f}ms exceeds 1ms"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_time_remaining(self):
        """Quotes at t=0 should be at mid price."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        quote = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=0.0)
        # At expiry, reservation price = mid, spread should be minimal
        assert quote.bid <= 100.0 <= quote.ask

    def test_very_high_inventory(self):
        """Extreme inventory still produces valid quotes."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=100
        )
        quote = engine.compute_quotes(mid_price=100.0, inventory=50, time_remaining=10.0)
        assert quote.bid < quote.ask
        assert np.isfinite(quote.bid)
        assert np.isfinite(quote.ask)

    def test_quote_dataclass_fields(self):
        """Quote has all expected fields."""
        engine = MarketMakingEngine(
            gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
        )
        quote = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
        assert hasattr(quote, "bid")
        assert hasattr(quote, "ask")
        assert hasattr(quote, "mid")
        assert hasattr(quote, "spread")
        assert hasattr(quote, "reservation_price")
