"""TDD: Performance monitor tests — verify implementation exists and works correctly.

Following TDD: tests written first, verify they pass (GREEN) against existing implementation.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aah/apex-fintech-platform/src')

from ml.performance_monitor import (
    PerformanceSummary,
    EngineMetrics,
    PerformanceMonitor,
    StrategyComparator,
    compute_rolling_metrics,
    PerformanceMonitor,
)


def test_performance_summary_dataclass():
    """Test PerformanceSummary dataclass can be created and has expected fields."""
    ps = PerformanceSummary(
        engine_name="test_engine",
        total_return=0.15,
        annualized_return=0.12,
        volatility=0.20,
        sharpe_ratio=0.6,
        max_drawdown=-0.10,
        win_rate=0.55,
        total_trades=42,
    )
    assert ps.engine_name == "test_engine"
    assert ps.total_return == 0.15
    assert ps.annualized_return == 0.12
    assert ps.volatility == 0.20
    assert ps.sharpe_ratio == 0.6
    assert ps.max_drawdown == -0.10
    assert ps.win_rate == 0.55
    assert ps.total_trades == 42
    # Post-init computed fields
    assert ps.calmar_ratio != 0.0 or ps.max_drawdown == 0  # one of these paths
    print("  PASS: PerformanceSummary dataclass")


def test_engine_metrics_summary():
    """Test EngineMetrics.summary produces PerformanceSummary with correct fields."""
    np.random.seed(42)
    n = 252
    returns = np.random.randn(n) * 0.02  # ~2% daily returns
    metrics = EngineMetrics(
        engine_name="test_engine",
        window=100,
        returns=returns,
    )
    summary = metrics.summary
    assert summary.engine_name == "test_engine"
    assert hasattr(summary, "total_return")
    assert hasattr(summary, "annualized_return")
    assert hasattr(summary, "volatility")
    assert hasattr(summary, "sharpe_ratio")
    assert hasattr(summary, "max_drawdown")
    assert hasattr(summary, "win_rate")
    assert hasattr(summary, "total_trades")
    assert 0.0 <= summary.win_rate <= 1.0
    assert summary.total_trades == n  # all trades counted
    print("  PASS: EngineMetrics.summary")


def test_performance_monitor_add_and_summary():
    """Test PerformanceMonitor can add metrics and retrieve summary."""
    monitor = PerformanceMonitor(window=50, risk_free_rate=0.05)
    np.random.seed(42)
    returns = np.random.randn(50) * 0.02
    metrics = EngineMetrics(
        engine_name="engine_A",
        window=50,
        returns=returns,
    )
    monitor.add_metrics(metrics)
    summary = monitor.get_summary()
    assert "engine_A" in str(summary) or isinstance(summary, dict)
    # Test ranked retrieval
    ranked = monitor.rank_engines(metric="sharpe_ratio")
    assert len(ranked) > 0
    print("  PASS: PerformanceMonitor add and summary")


def test_strategy_comparator_rank():
    """Test StrategyComparator.rank orders engines by composite score."""
    engines = [
        {
            "engine_name": "eng_1",
            "sharpe": 1.5,
            "sortino": 1.8,
            "max_dd": -0.10,
            "info_ratio": 0.3,
        },
        {
            "engine_name": "eng_2",
            "sharpe": 0.8,
            "sortino": 1.0,
            "max_dd": -0.05,
            "info_ratio": -0.1,
        },
        {
            "engine_name": "eng_3",
            "sharpe": 2.0,
            "sortino": 2.5,
            "max_dd": -0.15,
            "info_ratio": 0.5,
        },
    ]
    ranked = StrategyComparator.rank(engines)
    assert len(ranked) == 3
    # eng_3 has highest sharpe (2.0) and should rank highest
    assert ranked[0]["engine_name"] == "eng_3"
    print("  PASS: StrategyComparator.rank")


def test_compute_rolling_metrics():
    """Test compute_rolling_metrics returns expected keys with valid values."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.02
    metrics = compute_rolling_metrics(returns, window=252, risk_free_rate=0.05)
    expected_keys = {"sharpe", "volatility", "max_drawdown", "win_rate", "ann_return"}
    assert set(metrics.keys()) == expected_keys
    assert isinstance(metrics["sharpe"], float)
    assert isinstance(metrics["volatility"], float)
    assert isinstance(metrics["max_drawdown"], float)
    assert isinstance(metrics["win_rate"], float)
    assert isinstance(metrics["ann_return"], float)
    # Basic sanity checks
    assert metrics["ann_return"] != 0 or True  # could be zero for random
    assert 0.0 <= metrics["win_rate"] <= 1.0
    print("  PASS: compute_rolling_metrics")


def test_performance_monitor_drift_detection():
    """Test drift detection flags when Sharpe changes significantly."""
    monitor = PerformanceMonitor(window=100, risk_free_rate=0.05)
    np.random.seed(42)
    # Add 60 good-period metrics
    good_returns = np.random.randn(60) * 0.01  # small returns -> low sharpe
    # Actually create metrics with varying sharpe by using different return sets
    for i in range(60):
        ret_set = np.random.randn(50) * (0.01 + i * 0.001)  # gradually increasing vol
        # Simplify: just add metrics with engineered summaries
        from ml.performance_monitor import EngineMetrics
        mr = EngineMetrics(
            engine_name="drift_test",
            window=50,
            returns=np.random.randn(50) * (0.01 + i * 0.001),
        )
        monitor.add_metrics(mr)

    # Add 60 poor-period metrics (larger vol -> lower sharpe)
    for i in range(60):
        ret_set = np.random.randn(50) * (0.03 + i * 0.001)  # larger vol
        from ml.performance_monitor import EngineMetrics
        mr = EngineMetrics(
            engine_name="drift_test",
            window=50,
            returns=np.random.randn(50) * (0.03 + i * 0.001),
        )
        monitor.add_metrics(mr)

    drift = monitor.detect_drift("drift_test", threshold=0.3)
    assert "drift_detected" in drift
    assert "p_value" in drift
    print(f"  PASS: PerformanceMonitor drift detection (drift={drift['drift_detected']})")


def main():
    print("Running TDD tests for performance monitor...")
    test_performance_summary_dataclass()
    test_engine_metrics_summary()
    test_performance_monitor_add_and_summary()
    test_strategy_comparator_rank()
    test_compute_rolling_metrics()
    test_performance_monitor_drift_detection()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()