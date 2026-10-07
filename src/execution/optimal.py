"""Optimal execution engine using Almgren-Chriss framework with transient impact.

References:
    Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio transactions."
        Journal of Risk, 3(2), 5-40.
    Obizhaeva, A. & Wang, J. (2013). "Optimal trading strategy and supply/demand dynamics."
        Journal of Financial Markets, 16(1), 1-32.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class OptimalExecutionEngine:
    """Compute optimal execution trajectory for a single order.

    Uses the Almgren-Chriss closed-form solution with linear permanent and
    temporary price impact, extended with Obizhaeva-Wang transient impact decay.

    Parameters
    ----------
    total_shares : float
        Total number of shares to execute (X).
    time_horizon : float
        Execution time horizon in years (T).
    risk_aversion : float
        Risk aversion parameter (λ). Higher → more front-loaded.
    volatility : float
        Annualized volatility of the stock (σ).
    permanent_impact : float
        Permanent impact coefficient (γ). Price moves γ·v per unit traded.
    temporary_impact : float
        Temporary impact coefficient (η). Cost of trading v shares.
    decay_rate : float
        Transient impact decay rate (κ). Higher → faster decay of temporary impact.
    num_slices : int
        Number of discrete time slices for the trajectory.
    """

    def __init__(
        self,
        total_shares: float = 10_000,
        time_horizon: float = 1.0,
        risk_aversion: float = 1e-6,
        volatility: float = 0.2,
        permanent_impact: float = 0.01,
        temporary_impact: float = 0.05,
        decay_rate: float = 5.0,
        num_slices: int = 100,
    ) -> None:
        if total_shares <= 0:
            raise ValueError("total_shares must be positive")
        if time_horizon <= 0:
            raise ValueError("time_horizon must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")
        if risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if permanent_impact < 0:
            raise ValueError("permanent_impact must be non-negative")
        if temporary_impact < 0:
            raise ValueError("temporary_impact must be non-negative")
        if decay_rate <= 0:
            raise ValueError("decay_rate must be positive")
        if num_slices < 2:
            raise ValueError("num_slices must be at least 2")

        self.total_shares = float(total_shares)
        self.time_horizon = float(time_horizon)
        self.risk_aversion = float(risk_aversion)
        self.volatility = float(volatility)
        self.permanent_impact = float(permanent_impact)
        self.temporary_impact = float(temporary_impact)
        self.decay_rate = float(decay_rate)
        self.num_slices = int(num_slices)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _effective_temporary_impact(self) -> float:
        """Compute effective temporary impact accounting for transient decay.

        The OW transient kernel κ·e^(-κτ) reduces the effective impact over
        the horizon. We approximate the average effective impact as:
            η_eff = η · (1 - e^(-κT)) / (κT)
        """
        kappa_t = self.decay_rate * self.time_horizon
        # Avoid division by zero for very small κT
        if kappa_t < 1e-10:
            return self.temporary_impact
        decay_factor = (1.0 - np.exp(-kappa_t)) / kappa_t
        return self.temporary_impact * decay_factor

    def _kappa(self) -> float:
        """Compute the AC curvature parameter κ = sqrt(λ·σ² / η_eff)."""
        eta_eff = self._effective_temporary_impact()
        if eta_eff < 1e-15:
            # No impact → infinite κ → linear trajectory
            return 0.0
        return np.sqrt(self.risk_aversion * self.volatility**2 / eta_eff)

    def compute_trajectory(self) -> NDArray[np.float64]:
        """Compute the optimal execution trajectory.

        Returns the remaining shares x(t) at each time slice.
        x(0) = X, x(T) = 0.

        Uses the AC closed-form: x*(t) = X · sinh(κ(T-t)) / sinh(κT)
        """
        t = np.linspace(0.0, self.time_horizon, self.num_slices)
        kappa = self._kappa()

        if kappa < 1e-10:
            # Linear trajectory (no risk aversion or no impact)
            return self.total_shares * (1.0 - t / self.time_horizon)

        sinh_kt = np.sinh(kappa * self.time_horizon)
        if sinh_kt < 1e-15:
            return self.total_shares * (1.0 - t / self.time_horizon)

        remaining = self.total_shares * np.sinh(kappa * (self.time_horizon - t)) / sinh_kt
        # Enforce exact boundary conditions
        remaining[0] = self.total_shares
        remaining[-1] = 0.0
        return remaining

    # ------------------------------------------------------------------
    # Cost decomposition
    # ------------------------------------------------------------------

    def expected_impact(self) -> float:
        """Expected total price impact cost (permanent + temporary).

        Permanent impact cost: γ · X² / 2
        Temporary impact cost: η_eff · ∫ v(t)² dt  (approximated)
        """
        # Permanent impact: γ · X² / 2
        perm_cost = 0.5 * self.permanent_impact * self.total_shares**2

        # Temporary impact: η_eff · ∫ v² dt
        # For AC trajectory, ∫ v² dt = X² · κ / sinh(κT) · cosh(κT) ... 
        # Simplified: use numerical integration
        traj = self.compute_trajectory()
        dt = self.time_horizon / (self.num_slices - 1)
        v = -np.diff(traj) / dt  # trading rate
        eta_eff = self._effective_temporary_impact()
        temp_cost = eta_eff * np.sum(v**2) * dt

        return perm_cost + temp_cost

    def expected_risk(self) -> float:
        """Expected risk cost: λ · σ² · ∫ x(t)² dt"""
        traj = self.compute_trajectory()
        dt = self.time_horizon / (self.num_slices - 1)
        return self.risk_aversion * self.volatility**2 * np.sum(traj**2) * dt

    def expected_cost(self) -> float:
        """Total expected cost = impact cost + risk cost."""
        return self.expected_impact() + self.expected_risk()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def trading_schedule(self) -> NDArray[np.float64]:
        """Return the number of shares to trade at each time slice."""
        traj = self.compute_trajectory()
        schedule = -np.diff(traj)
        schedule = np.append(schedule, traj[-1])  # last slice trades remaining
        return schedule

    def summary(self) -> dict:
        """Return a summary of the execution plan."""
        return {
            "total_shares": self.total_shares,
            "time_horizon": self.time_horizon,
            "risk_aversion": self.risk_aversion,
            "volatility": self.volatility,
            "permanent_impact": self.permanent_impact,
            "temporary_impact": self.temporary_impact,
            "decay_rate": self.decay_rate,
            "num_slices": self.num_slices,
            "expected_cost": self.expected_cost(),
            "expected_impact": self.expected_impact(),
            "expected_risk": self.expected_risk(),
            "kappa": self._kappa(),
        }
