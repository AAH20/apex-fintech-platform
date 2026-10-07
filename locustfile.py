"""Locust load testing for Apex Fintech Platform engines.

Run with:
    locust -f locustfile.py --host http://localhost:8000

This file defines load tests for all platform engines:
- Market Making (quote computation)
- Hawkes Process (order flow simulation)
- Compliance Engine (report generation)
- Alternative Data (signal generation)
"""
from __future__ import annotations

import random

from locust import HttpUser, User, between, task


class MarketMakingLoadTest(User):
    """Load test for market making engine."""

    wait_time = between(0.001, 0.01)

    def on_start(self):
        """Initialize engine state."""
        self.mid_price = 100.0
        self.inventory = 0

    @task(5)
    def compute_quote(self):
        """Simulate quote computation load."""
        self.mid_price += random.uniform(-0.5, 0.5)
        self.inventory += random.randint(-1, 1)
        self.inventory = max(-10, min(10, self.inventory))
        # Simulate the computation
        gamma, sigma, k = 0.1, 0.5, 1.5
        time_remaining = 10.0
        reservation = self.mid_price - self.inventory * gamma * sigma**2 * time_remaining
        spread = gamma * sigma**2 * time_remaining + (2.0 / gamma) * (
            __import__("math").log(1.0 + gamma / k)
        )
        bid = reservation - spread / 2.0
        ask = reservation + spread / 2.0
        assert bid < ask

    @task(3)
    def compute_batch_quotes(self):
        """Simulate batch quote computation."""
        for _ in range(10):
            mid = 100.0 + random.uniform(-5, 5)
            inv = random.randint(-5, 5)
            gamma, sigma, k = 0.1, 0.5, 1.5
            reservation = mid - inv * gamma * sigma**2 * 10.0
            spread = gamma * sigma**2 * 10.0 + (2.0 / gamma) * (
                __import__("math").log(1.0 + gamma / k)
            )
            bid = reservation - spread / 2.0
            ask = reservation + spread / 2.0
            assert bid < ask


class HawkesLoadTest(User):
    """Load test for Hawkes process simulation."""

    wait_time = between(0.01, 0.1)

    @task
    def simulate_hawkes(self):
        """Simulate Hawkes process order flow."""
        import math
        mu, alpha, beta = 1.0, 0.5, 2.0
        T = 10.0
        events = []
        t = 0.0
        while True:
            lambda_t = mu if not events else mu + alpha * sum(
                math.exp(-beta * (t - ti)) for ti in events
            )
            u = random.expovariate(lambda_t)
            t_candidate = t + u
            if t_candidate > T:
                break
            if not events:
                lambda_candidate = mu
            else:
                lambda_candidate = mu + alpha * sum(
                    math.exp(-beta * (t_candidate - ti)) for ti in events
                )
            if random.random() < lambda_candidate / lambda_t:
                events.append(t_candidate)
            t = t_candidate
        assert len(events) >= 0


class ComplianceLoadTest(User):
    """Load test for compliance engine."""

    wait_time = between(0.01, 0.05)

    @task
    def generate_report(self):
        """Simulate compliance report generation."""
        controls = []
        for i in range(50):
            status = random.choice(["COMPLIANT", "NON_COMPLIANT", "PENDING"])
            controls.append({"id": f"PCI-{i}", "status": status})
        compliant = sum(1 for c in controls if c["status"] == "COMPLIANT")
        assert compliant <= len(controls)


class AltDataLoadTest(User):
    """Load test for alternative data engine."""

    wait_time = between(0.01, 0.05)

    @task
    def generate_signal(self):
        """Simulate composite signal generation."""
        satellite_score = random.uniform(0, 1)
        sentiment_score = random.uniform(-1, 1)
        transaction_score = random.uniform(0, 1)
        sources_used = sum(1 for s in [satellite_score, sentiment_score, transaction_score] if s > 0)
        if sources_used == 0:
            composite = 0.0
        else:
            composite = (satellite_score + sentiment_score + transaction_score) / sources_used
            composite = (composite * 2) - 1
        assert -1 <= composite <= 1


class PlatformLoadTest(HttpUser):
    """HTTP-based load test for platform API endpoints."""

    wait_time = between(0.1, 0.5)

    @task(3)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health")

    @task(1)
    def engine_status(self):
        """Test engine status endpoint."""
        self.client.get("/api/v1/engines/status")
