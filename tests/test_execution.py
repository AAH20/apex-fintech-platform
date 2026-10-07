"""Tests for optimal execution engine (Almgren-Chriss + transient impact)."""
import time
import numpy as np
import pytest

from src.execution.optimal import OptimalExecutionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(**overrides):
    """Create an engine with sensible defaults, allowing overrides."""
    defaults = dict(
        total_shares=10_000,
        time_horizon=1.0,
        risk_aversion=1e-6,
        volatility=0.2,
        permanent_impact=0.01,
        temporary_impact=0.05,
        decay_rate=5.0,
        num_slices=100,
    )
    defaults.update(overrides)
    return OptimalExecutionEngine(**defaults)


# ---------------------------------------------------------------------------
# Trajectory shape & boundary conditions
# ---------------------------------------------------------------------------

class TestTrajectoryShape:
    def test_trading_rate_integrates_to_total_shares(self):
        engine = make_engine()
        traj = engine.compute_trajectory()
        dt = engine.time_horizon / (engine.num_slices - 1)
        trading_rate = -np.diff(traj) / dt
        assert np.isclose(np.sum(trading_rate) * dt, engine.total_shares, rtol=1e-6)

    def test_trajectory_length_matches_num_slices(self):
        engine = make_engine(num_slices=50)
        traj = engine.compute_trajectory()
        assert len(traj) == 50

    def test_trajectory_starts_at_total_shares(self):
        engine = make_engine()
        traj = engine.compute_trajectory()
        assert np.isclose(traj[0], engine.total_shares, rtol=1e-6)

    def test_trajectory_ends_at_zero(self):
        engine = make_engine()
        traj = engine.compute_trajectory()
        assert np.isclose(traj[-1], 0.0, atol=1e-6)

    def test_trajectory_is_monotonically_decreasing(self):
        engine = make_engine()
        traj = engine.compute_trajectory()
        assert np.all(np.diff(traj) <= 1e-10)

    def test_trajectory_values_are_non_negative(self):
        engine = make_engine()
        traj = engine.compute_trajectory()
        assert np.all(traj >= -1e-10)


# ---------------------------------------------------------------------------
# Risk aversion effects
# ---------------------------------------------------------------------------

class TestRiskAversion:
    def test_zero_risk_aversion_gives_linear_trajectory(self):
        engine = make_engine(risk_aversion=0.0)
        traj = engine.compute_trajectory()
        # Linear: X * (1 - t/T)
        t = np.linspace(0, 1, engine.num_slices)
        expected = engine.total_shares * (1 - t)
        np.testing.assert_allclose(traj, expected, rtol=1e-4)

    def test_higher_risk_aversion_front_loads(self):
        low_risk = make_engine(risk_aversion=1e-8)
        high_risk = make_engine(risk_aversion=1e-4)
        traj_low = low_risk.compute_trajectory()
        traj_high = high_risk.compute_trajectory()
        # Higher risk aversion → more shares traded early → lower remaining shares early
        assert traj_high[1] < traj_low[1]
        assert traj_high[len(traj_high) // 2] < traj_low[len(traj_low) // 2]


# ---------------------------------------------------------------------------
# Cost decomposition
# ---------------------------------------------------------------------------

class TestCostDecomposition:
    def test_expected_cost_is_positive(self):
        engine = make_engine()
        assert engine.expected_cost() > 0

    def test_expected_cost_increases_with_volatility(self):
        low_vol = make_engine(volatility=0.1)
        high_vol = make_engine(volatility=0.5)
        assert high_vol.expected_cost() > low_vol.expected_cost()

    def test_expected_cost_increases_with_risk_aversion(self):
        low_risk = make_engine(risk_aversion=1e-8)
        high_risk = make_engine(risk_aversion=1e-4)
        assert high_risk.expected_cost() > low_risk.expected_cost()

    def test_cost_decomposition_sums_to_total(self):
        engine = make_engine()
        total = engine.expected_cost()
        impact = engine.expected_impact()
        risk = engine.expected_risk()
        assert np.isclose(total, impact + risk, rtol=1e-6)


# ---------------------------------------------------------------------------
# Transient impact (Obizhaeva-Wang)
# ---------------------------------------------------------------------------

class TestTransientImpact:
    def test_higher_decay_rate_reduces_impact(self):
        slow_decay = make_engine(decay_rate=1.0)
        fast_decay = make_engine(decay_rate=20.0)
        # Faster decay → less residual impact → lower cost
        assert fast_decay.expected_cost() < slow_decay.expected_cost()

    def test_trajectory_with_transient_impact_differs_from_permanent_only(self):
        # Use parameters where temporary impact has a noticeable effect
        engine_both = make_engine(
            permanent_impact=0.01, temporary_impact=0.5, risk_aversion=1.0
        )
        engine_perm_only = make_engine(
            permanent_impact=0.01, temporary_impact=0.0, risk_aversion=1.0
        )
        traj_both = engine_both.compute_trajectory()
        traj_perm = engine_perm_only.compute_trajectory()
        # Trajectories should differ when temporary impact is present
        assert not np.allclose(traj_both, traj_perm, rtol=1e-6)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_total_shares_raises(self):
        with pytest.raises(ValueError):
            make_engine(total_shares=-100)

    def test_zero_time_horizon_raises(self):
        with pytest.raises(ValueError):
            make_engine(time_horizon=0)

    def test_negative_volatility_raises(self):
        with pytest.raises(ValueError):
            make_engine(volatility=-0.1)

    def test_negative_risk_aversion_raises(self):
        with pytest.raises(ValueError):
            make_engine(risk_aversion=-1e-6)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_computes_10k_shares_under_100ms(self):
        engine = make_engine(total_shares=10_000, num_slices=200)
        start = time.perf_counter()
        engine.compute_trajectory()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Took {elapsed_ms:.1f}ms, must be <100ms"
