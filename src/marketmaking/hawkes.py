"""Hawkes process for modeling order flow clustering.

A Hawkes process is a self-exciting point process where the arrival of
an event increases the likelihood of future events. This is used to
model the clustering behavior observed in financial order flow.

The intensity is:
    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

where:
    mu = baseline intensity
    alpha = excitation (jump in intensity after an event)
    beta = decay rate of excitation

References:
    Hawkes, A.G. (1971). Spectra of some self-exciting and mutually
    exciting point processes. Biometrika, 58(1), 83-90.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HawkesProcess:
    """Self-exciting Hawkes process for order flow modeling.

    Attributes:
        mu: Baseline intensity (events per unit time).
        alpha: Excitation parameter (intensity jump per event).
        beta: Decay rate of excitation.
    """

    mu: float
    alpha: float
    beta: float

    # Internal state
    _event_times: list[float] = field(default_factory=list, init=False)
    _current_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.beta <= 0:
            raise ValueError("beta must be positive")

    def branching_ratio(self) -> float:
        """Compute the branching ratio (alpha / beta).

        Must be < 1 for a stationary process.
        """
        return self.alpha / self.beta

    def unconditional_intensity(self) -> float:
        """Compute the unconditional (stationary) intensity.

        E[lambda] = mu / (1 - alpha/beta)
        """
        br = self.branching_ratio()
        if br >= 1.0:
            return float("inf")
        return self.mu / (1.0 - br)

    def reset(self) -> None:
        """Reset the process state."""
        self._event_times.clear()
        self._current_time = 0.0

    def current_intensity(self) -> float:
        """Compute the current intensity lambda(t).

        lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))
        """
        if not self._event_times:
            return self.mu

        t = self._current_time
        excitation = sum(math.exp(-self.beta * (t - ti)) for ti in self._event_times if ti <= t)
        return self.mu + self.alpha * excitation

    def trigger_event(self, time: float) -> None:
        """Record an event at the given time.

        Args:
            time: Time of the event.
        """
        if time < self._current_time:
            raise ValueError("Event time must be >= current time")
        self._current_time = time
        self._event_times.append(time)

    def update_time(self, time: float) -> None:
        """Update the current time without triggering an event.

        Args:
            time: New current time.
        """
        if time < self._current_time:
            raise ValueError("Time must be >= current time")
        self._current_time = time

    def simulate(self, T: float, seed: int | None = None) -> np.ndarray:
        """Simulate the Hawkes process using Ogata's thinning algorithm.

        Args:
            T: Time horizon for simulation.
            seed: Random seed for reproducibility.

        Returns:
            Array of event times.
        """
        if T <= 0:
            raise ValueError("T must be positive")

        rng = np.random.default_rng(seed)
        events: list[float] = []
        t = 0.0

        while True:
            # Current intensity
            if not events:
                lambda_t = self.mu
            else:
                lambda_t = self.mu + self.alpha * sum(
                    math.exp(-self.beta * (t - ti)) for ti in events
                )

            # Generate candidate time
            u = rng.exponential(1.0 / lambda_t)
            t_candidate = t + u

            if t_candidate > T:
                break

            # Thinning: accept with probability lambda(t_candidate) / lambda_t
            if not events:
                lambda_candidate = self.mu
            else:
                lambda_candidate = self.mu + self.alpha * sum(
                    math.exp(-self.beta * (t_candidate - ti)) for ti in events
                )

            if rng.random() < lambda_candidate / lambda_t:
                events.append(t_candidate)

            t = t_candidate

        return np.array(events)

    def fit(self, event_times: np.ndarray, max_iter: int = 100) -> None:
        """Fit Hawkes process parameters using maximum likelihood.

        Uses EM algorithm for Hawkes process parameter estimation.

        Args:
            event_times: Observed event times.
            max_iter: Maximum number of EM iterations.
        """
        if len(event_times) < 2:
            raise ValueError("Need at least 2 events to fit")

        T = event_times[-1] - event_times[0]
        n = len(event_times)

        # Initialize parameters
        self.mu = n / T
        self.alpha = 0.1
        self.beta = 1.0

        for _ in range(max_iter):
            # E-step: compute probabilities
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(i):
                    dt = event_times[i] - event_times[j]
                    if dt > 0:
                        p[i, j] = self.alpha * math.exp(-self.beta * dt)

            # Normalize
            row_sums = p.sum(axis=1)
            for i in range(n):
                if row_sums[i] > 0:
                    p[i] /= row_sums[i]

            # M-step: update parameters
            self.mu = (n - p.sum()) / T
            self.alpha = p.sum() / n
            # Update beta using weighted average
            weights = p.sum()
            if weights > 0:
                dt_sum = 0.0
                for i in range(n):
                    for j in range(i):
                        dt = event_times[i] - event_times[j]
                        if dt > 0:
                            dt_sum += p[i, j] * dt
                if dt_sum > 0:
                    self.beta = weights / dt_sum
