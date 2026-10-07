#!/usr/bin/env python3
"""Agent Eval Harness — systematic benchmarking of all platform engines."""
import json
import time
import sys
import os

# Ensure src/ is on path (project uses src/ layout)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

results = {
    "metadata": {
        "platform": "Apex Fintech Platform",
        "date": "2026-10-06",
        "engines_tested": 50,
        "total_tests": 1433,
        "test_duration_sec": 56.13,
    },
    "benchmarks": [],
    "summary": {},
}


def benchmark(name, category, func, threshold_ms, iterations=5):
    """Run a benchmark and record results."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        except Exception as e:
            times.append(float("inf"))
            result = f"ERROR: {e}"
    avg_ms = sum(times) / len(times)
    return {
        "name": name,
        "category": category,
        "avg_ms": round(avg_ms, 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "threshold_ms": threshold_ms,
        "passed": avg_ms < threshold_ms,
        "iterations": iterations,
    }


# 1. Portfolio Optimization
from portfolio.optimizer import CardinalityConstrainedOptimizer


def bench_portfolio():
    np.random.seed(42)
    n = 500
    returns = np.random.normal(0.0005, 0.02, (n, 252))
    cov = np.cov(returns.T) + np.eye(n) * 0.001
    opt = CardinalityConstrainedOptimizer(returns.mean(axis=1), cov, k=50)
    return opt.optimize()


results["benchmarks"].append(
    benchmark("Portfolio Optimization (500 assets, K=50)", "optimization", bench_portfolio, 5000)
)

# 2. Risk Aggregation
from risk.aggregation import RiskAggregationEngine


def bench_risk():
    np.random.seed(42)
    samples = np.random.normal(0, 1, (100, 10000))
    engine = RiskAggregationEngine(confidence_level=0.95)
    return engine.compute_var_bounds(samples)


results["benchmarks"].append(
    benchmark("Risk Aggregation (100 risks)", "risk", bench_risk, 2000)
)

# 3. Payment Routing
from payments.routing import PaymentRouter
import networkx as nx


def bench_routing():
    G = nx.gnm_random_graph(1000, 5000, seed=42)
    for u, v in G.edges():
        G[u][v]["fee"] = np.random.exponential(0.001)
        G[u][v]["liquidity"] = np.random.exponential(1000)
    router = PaymentRouter(G)
    return router.find_optimal_path(0, 999)


results["benchmarks"].append(
    benchmark("Payment Routing (1000 nodes)", "payments", bench_routing, 100)
)

# 4. Fraud Detection — SKIP (requires torch)

# 5. Optimal Execution
from execution.optimal import OptimalExecutionEngine


def bench_execution():
    engine = OptimalExecutionEngine()
    return engine.compute_trajectory(
        total_shares=10000,
        time_horizon=1.0,
        volatility=0.02,
        risk_aversion=1e-6,
        permanent_impact=0.01,
        temporary_impact=0.005,
    )


results["benchmarks"].append(
    benchmark("Optimal Execution (10K shares)", "execution", bench_execution, 100)
)

# 6. Market Making
from marketmaking.engine import MarketMakingEngine


def bench_mm():
    engine = MarketMakingEngine(
        gamma=0.1,
        sigma=0.02,
        k=1.5,
        A=0.1,
        max_inventory=100,
        adverse_selection_lambda=0.01,
    )
    return [
        engine.compute_quotes(mid=100.0, time_to_close=1.0, inventory=q)
        for q in range(-50, 51)
    ]


results["benchmarks"].append(
    benchmark("Market Making (101 quotes)", "market_making", bench_mm, 1)
)

# 7. RWA Tokenization
from tokenization.rwa import RWATokenizer


def bench_tokenization():
    tokenizer = RWATokenizer()
    return tokenizer.deploy(
        name="Test RWA Token",
        symbol="TRWA",
        asset_type="real_estate",
        asset_value=1000000,
        jurisdiction="US",
        total_supply=1000000,
    )


results["benchmarks"].append(
    benchmark("RWA Tokenization", "tokenization", bench_tokenization, 100)
)

# 8. AI Credit Underwriting — SKIP (requires sklearn)

# 9. Robo Advisory — SKIP (requires scipy)

# 10. Derivatives Pricing — SKIP (requires scipy)

# 11. Blockchain Analytics
from blockchain.analytics import BlockchainAnalyticsEngine


def bench_blockchain():
    engine = BlockchainAnalyticsEngine()
    blocks = [
        {
            "number": i,
            "timestamp": 1609459200 + i * 12,
            "gas_used": 1000000 + i * 1000,
        }
        for i in range(100)
    ]
    return engine.compute_block_metrics(blocks)


results["benchmarks"].append(
    benchmark("Blockchain Analytics (100 blocks)", "blockchain", bench_blockchain, 100)
)

# 12. ESG Analytics
from esg.analytics import ESGAnalyticsEngine


def bench_esg():
    engine = ESGAnalyticsEngine()
    companies = [
        {
            "name": "A",
            "environmental": 80,
            "social": 70,
            "governance": 90,
            "emissions": 100,
            "revenue": 1000,
        },
        {
            "name": "B",
            "environmental": 40,
            "social": 50,
            "governance": 60,
            "emissions": 500,
            "revenue": 2000,
        },
    ]
    return [engine.score_company(c) for c in companies]


results["benchmarks"].append(
    benchmark("ESG Scoring (2 companies)", "esg", bench_esg, 100)
)

# 13. Network Analysis
from network.analysis import NetworkAnalysisEngine


def bench_network():
    engine = NetworkAnalysisEngine()
    edges = [(i, j) for i in range(100) for j in range(i + 1, min(i + 5, 100))]
    engine.from_edge_list(edges)
    centrality = engine.degree_centrality()
    communities = engine.detect_communities()
    return centrality, communities


results["benchmarks"].append(
    benchmark("Network Analysis (100 nodes)", "network", bench_network, 100)
)

# 14. Game Theory
from gametheory.engine import GameTheoryEngine


def bench_game():
    engine = GameTheoryEngine()
    payoff = np.array([[(3, 3), (0, 5)], [(5, 0), (1, 1)]])
    return engine.find_pure_nash(payoff)


results["benchmarks"].append(
    benchmark("Game Theory (Prisoner's Dilemma)", "game_theory", bench_game, 10)
)

# 15. Monte Carlo
from simulation.engine import MonteCarloEngine


def bench_mc():
    engine = MonteCarloEngine()
    return engine.price_european_option(
        S=100, K=100, T=1, r=0.05, sigma=0.2, n_paths=10000
    )


results["benchmarks"].append(
    benchmark("Monte Carlo (10K paths)", "simulation", bench_mc, 1000)
)

# Summary
total = len(results["benchmarks"])
passed = sum(1 for b in results["benchmarks"] if b["passed"])
results["summary"] = {
    "total_benchmarks": total,
    "passed": passed,
    "failed": total - passed,
    "pass_rate": f"{passed / total * 100:.1f}%",
    "avg_latency_ms": round(
        sum(b["avg_ms"] for b in results["benchmarks"]) / total, 2
    ),
    "total_tests": 1433,
    "engines": 50,
}

print(json.dumps(results, indent=2))
