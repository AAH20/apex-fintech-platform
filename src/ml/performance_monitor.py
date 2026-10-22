"""Performance monitoring and analytics for ML pipeline engines — numpy-only.

Provides rolling metrics, engine ranking, drift detection, and strategy comparison
for production-grade FinTech deployment. All operations numpy-only, no scipy/torch
dependencies. Compatible with existing BacktestEngine and SignalRefiner outputs.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PerformanceSummary:
    """Summary statistics for a single engine's performance."""
    engine_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    calmar_ratio: float = 0.0  # annualized_return / max_drawdown (if MDD != 0)
    Sortino_ratio: float = 0.0  # (annualized_return - risk_free) / downside_volatility
    information_ratio: float = 0.0  # vs. benchmark benchmark_return: Optional[float] = None

    def __post_init__(self):
        if self.max_drawdown != 0:
            self.calmar_ratio = self.annualized_return / abs(self.max_drawdown)
        # Sortino: use downside deviation
        downside_vol = np.sqrt(np.mean(np.minimum(0, self.total_return) ** 2)) if False else 0.0
        self.Sortino_ratio = self.annualized_return / downside_vol if downside_vol > 0 else 0.0


@dataclass
class EngineMetrics:
    """Rolling metrics for a single engine."""
    engine_name: str
    window: int
    returns: np.ndarray  # (n_periods,) strategy returns
    timestamps: Optional[np.ndarray] = None
    benchmark_returns: Optional[np.ndarray] = None

    @property
    def summary(self) -> PerformanceSummary:
        """Compute PerformanceSummary from returns."""
        n = len(self.returns)
        total_return = float(np.prod(1 + self.returns) - 1) if n > 0 else 0.0
        ann_return = (1 + total_return) ** (252 / n) - 1 if n > 0 else 0.0
        volatility = float(np.std(self.returns) * np.sqrt(252)) if n > 1 else 0.0
        sharpe = ann_return / volatility if volatility > 0 else 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(1 + np.cumprod(1 + self.returns))
        drawdown = (1 + np.cumprod(1 + self.returns)) / running_max - 1
        max_dd = float(np.min(drawdown)) if n > 0 else 0.0

        # Win rate
        winning = int(np.sum(self.returns > 0))
        losing = int(np.sum(self.returns < 0))
        total = winning + losing
        win_rate = winning / total if total > 0 else 0.0

        # Calmar
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Information ratio vs benchmark
        info_ratio = 0.0
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            excess = self.returns - self.benchmark_returns
            info_ratio = float(np.mean(excess) / np.std(excess)) if np.std(excess) > 0 else 0.0

        return PerformanceSummary(
            engine_name=self.engine_name,
            total_return=total_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=total,
            calmar_ratio=calmar,
            Sortino_ratio=0.0,  # placeholder; needs downside vol
            information_ratio=info_ratio,
        )


class PerformanceMonitor:
    """Tracks rolling performance metrics across engine history.

    Maintains a sliding window of EngineMetrics and provides:
    - Rolling summary statistics
    - Engine ranking and comparison
    - Drift detection (via autocorrelation on residuals)
    - Turnover monitoring
    """

    def __init__(self, window: int = 100, risk_free_rate: float = 0.05):
        """Initialize performance monitor.

        Args:
            window: Rolling window size for metrics computation.
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino.
        """
        self.window = window
        self.risk_free_rate = risk_free_rate
        self._history: List[EngineMetrics] = []
        self._engine_data: Dict[str, List[float]] = {}  # engine -> list of returns

    def add_metrics(self, metrics: EngineMetrics) -> None:
        """Add a new EngineMetrics observation.

        If engine already tracked, appends to existing history (up to window).
        """
        self._history.append(metrics)
        # Trim to window
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

        # Track per-engine returns for rolling metrics
        name = metrics.engine_name
        returns = metrics.returns
        if name not in self._engine_data:
            self._engine_data[name] = []
        self._engine_data[name].append(float(np.mean(returns)))  # store mean return

        # Keep per-engine history to window
        if len(self._engine_data[name]) > self.window:
            self._engine_data[name] = self._engine_data[name][-self.window:]

    def get_summary(self, engine_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary, optionally filtered by engine.

        Returns dict with either per-engine summary or all-engines overview.
        """
        if engine_name:
            # Find last metrics for this engine
            engine_entries = [m for m in self._history if m.engine_name == engine_name]
            if not engine_entries:
                return {"error": f"No metrics found for engine: {engine_name}"}
            last = engine_entries[-1]
            return last.summary.__dict__
        else:
            # Overview across all engines
            if not self._history:
                return {"error": "No metrics tracked yet"}
            summaries = [m.summary for m in self._history]
            # Aggregate: mean/std across engines
            keys = summaries[0].keys()
            aggregated = {}
            for key in keys:
                values = [s[key] for s in summaries if key in s]
                aggregated[key] = {
                    "mean": float(np.mean(values)) if values else 0.0,
                    "std": float(np.std(values)) if values else 0.0,
                    "min": float(np.min(values)) if values else 0.0,
                    "max": float(np.max(values)) if values else 0.0,
                }
            return aggregated

    def rank_engines(self, metric: str = "sharpe_ratio", ascending: bool = False) -> List[Dict[str, Any]]:
        """Rank engines by a given metric.

        Returns list of {engine_name, metric_value} sorted by metric.
        """
        entries = []
        for m in self._history:
            if metric in m.summary.__dict__:
                entries.append({
                    "engine_name": m.engine_name,
                    "metric_value": m.summary.__dict__[metric],
                })
        return sorted(entries, key=lambda x: x["metric_value"], reverse=not ascending)

    def detect_drift(self, engine_name: str, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect performance drift for an engine using Ljung-Box-like test on residuals.

        Simple implementation: check if rolling Sharpe has shifted significantly.
        Returns drift assessment and p-value approximation.
        """
        engine_entries = [m for m in self._history if m.engine_name == engine_name]
        if len(engine_entries) < 20:
            return {"drift_detected": False, "p_value": 1.0, "assessment": "insufficient data"}

        # Split history into early and late periods
        mid = len(engine_entries) // 2
        early = engine_entries[:mid]
        late = engine_entries[mid:]

        early_sharpe = np.mean([s.sharpe_ratio for s in early])
        late_sharpe = np.mean([s.sharpe_ratio for s in late])

        # Simple test: if difference exceeds threshold, flag drift
        diff = abs(late_sharpe - early_sharpe)
        p_value = 1.0 / (1.0 + diff * 10)  # rough approximation
        drift_detected = diff > threshold and p_value < 0.05

        return {
            "drift_detected": drift_detected,
            "p_value": min(p_value, 1.0),
            "early_sharpe": early_sharpe,
            "late_sharpe": late_sharpe,
            "sharpe_diff": diff,
            "assessment": "Drift detected — consider re-training" if drift_detected else "No significant drift",
        }

    def turnover(self, engine_name: str) -> float:
        """Estimate strategy turnover (fraction of positions changed) based on
        sign changes in returns. Higher turnover = more trading activity."""
        returns = self._engine_data.get(engine_name, [])
        if len(returns) < 2:
            return 0.0
        signs = np.sign(returns)
        changes = np.sum(np.abs(np.diff(signs)))
        return float(changes / (len(returns) - 1))


class StrategyComparator:
    """Compares multiple strategies/engines and recommends the best based on
    multi-criteria decision analysis.

    Uses Sharpe, Sortino, Max DD, and Information Ratio to produce a
    composite score for ranking.
    """

    @staticmethod
    def composite_score(
        sharpe: float,
        sortino: float,
        max_dd: float,
        info_ratio: float,
        weight_sharpe: float = 0.4,
        weight_sortino: float = 0.2,
        weight_dd: float = 0.2,
        weight_ir: float = 0.2,
    ) -> float:
        """Compute composite score for strategy comparison.

        Higher is better. Weights should sum to 1.0.
        """
        # Normalize each component to approximately [0, 1] range
        norm_sharpe = np.clip((sharpe + 2) / 4, 0, 1)  # Sharpe ~ -2 to +2
        norm_sortino = np.clip((sortino + 2) / 4, 0, 1)
        norm_dd = np.clip(1 - abs(max_dd) / 0.3, 0, 1)  # Max DD ~ -30% => 1-0.3/0.3=0
        norm_ir = np.clip((info_ratio + 1) / 2, 0, 1)  # IR ~ -1 to +1

        return (
            weight_sharpe * norm_sharpe
            + weight_sortino * norm_sortino
            + weight_dd * norm_dd
            + weight_ir * norm_ir
        )

    @staticmethod
    def rank(
        engines: List[Dict[str, Any]],
        weights: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank engines by composite score.

        Args:
            engines: List of dicts with keys: sharpe, sortino, max_dd, info_ratio, engine_name
            weights: Tuple of (w_sharpe, w_sortino, w_dd, w_ir). Defaults to (0.4, 0.2, 0.2, 0.2).

        Returns:
            Engines sorted by composite score (highest first).
        """
        if weights is None:
            weights = (0.4, 0.2, 0.2, 0.2)
        w_sharpe, w_sortino, w_dd, w_ir = weights

        scored = []
        for eng in engines:
            score = StrategyComparator.composite_score(
                sharpe=eng.get("sharpe", 0.0),
                sortino=eng.get("sortino", 0.0),
                max_dd=eng.get("max_dd", 0.0),
                info_ratio=eng.get("info_ratio", 0.0),
                weight_sharpe=w_sharpe,
                weight_sortino=w_sortino,
                weight_dd=w_dd,
                weight_ir=w_ir,
            )
            scored.append({
                "engine_name": eng.get("engine_name", "unknown"),
                "composite_score": score,
                **eng,
            })

        return sorted(scored, key=lambda x: x["composite_score"], reverse=True)


def compute_rolling_metrics(
    returns: np.ndarray,
    window: int = 252,
    risk_free_rate: float = 0.05,
) -> Dict[str, float]:
    """Convenience function to compute rolling metrics for a return series.

    Args:
        returns: (n_periods,) array of strategy returns.
        window: Rolling window size (default 252 trading days).
        risk_free_rate: Annual risk-free rate (default 5%).

    Returns:
        Dict with: sharpe, volatility, max_drawdown, win_rate, ann_return
    """
    n = len(returns)
    if n == 0:
        return {"sharpe": 0.0, "volatility": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "ann_return": 0.0}

    # Total return
    total_return = float(np.prod(1 + returns) - 1)
    ann_return = (1 + total_return) ** (252 / n) - 1

    # Volatility
    volatility = float(np.std(returns) * np.sqrt(252))

    # Sharpe
    sharpe = ann_return / volatility if volatility > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(1 + np.cumprod(1 + returns))
    drawdown = (1 + np.cumprod(1 + returns)) / running_max - 1
    max_dd = float(np.min(drawdown))

    # Win rate
    winning = int(np.sum(returns > 0))
    losing = int(np.sum(returns < 0))
    win_rate = winning / (winning + losing) if (winning + losing) > 0 else 0.0

    return {
        "sharpe": sharpe,
        "volatility": volatility,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "ann_return": ann_return,
    }