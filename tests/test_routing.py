"""Tests for the payment routing optimization engine."""

import networkx as nx
import pytest

from payments import PaymentRouter


def _build_graph():
    """Build a sample payment channel network."""
    g = nx.DiGraph()
    edges = [
        ("A", "B", 1.0, 100.0),
        ("B", "C", 2.0, 50.0),
        ("A", "C", 5.0, 200.0),
        ("C", "D", 1.0, 80.0),
        ("B", "D", 4.0, 30.0),
    ]
    for u, v, fee, liq in edges:
        g.add_edge(u, v, fee=fee, liquidity=liq)
    return g


class TestSimplePath:
    def test_find_optimal_path_simple(self):
        router = PaymentRouter(_build_graph())
        path = router.find_optimal_path("A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_get_path_cost(self):
        router = PaymentRouter(_build_graph())
        path = ["A", "B", "C", "D"]
        assert router.get_path_cost(path) == pytest.approx(4.0)


class TestNoPath:
    def test_no_path_raises(self):
        g = _build_graph()
        g.add_node("Z")
        router = PaymentRouter(g)
        with pytest.raises(nx.NetworkXNoPath):
            router.find_optimal_path("A", "Z")


class TestFeeConstraint:
    def test_fee_constraint_filters_expensive_path(self):
        router = PaymentRouter(_build_graph())
        path = router.find_optimal_path("A", "D", max_fee=4.0)
        assert router.get_path_cost(path) <= 4.0

    def test_fee_constraint_no_valid_path(self):
        router = PaymentRouter(_build_graph())
        with pytest.raises(nx.NetworkXNoPath):
            router.find_optimal_path("A", "D", max_fee=0.5)


class TestLiquidityConstraint:
    def test_liquidity_constraint(self):
        router = PaymentRouter(_build_graph())
        path = router.find_optimal_path("A", "D", min_liquidity=40.0)
        assert all(
            router.graph[u][v]["liquidity"] >= 40.0
            for u, v in zip(path, path[1:])
        )

    def test_liquidity_constraint_no_valid_path(self):
        router = PaymentRouter(_build_graph())
        with pytest.raises(nx.NetworkXNoPath):
            router.find_optimal_path("A", "D", min_liquidity=500.0)


class TestEdgeCases:
    def test_single_node_path(self):
        g = nx.DiGraph()
        g.add_node("A")
        router = PaymentRouter(g)
        assert router.find_optimal_path("A", "A") == ["A"]

    def test_disconnected_graph(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", fee=1.0, liquidity=10.0)
        g.add_edge("C", "D", fee=1.0, liquidity=10.0)
        router = PaymentRouter(g)
        with pytest.raises(nx.NetworkXNoPath):
            router.find_optimal_path("A", "C")

    def test_negative_fee_edge(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", fee=-1.0, liquidity=10.0)
        g.add_edge("B", "C", fee=2.0, liquidity=10.0)
        g.add_edge("A", "C", fee=5.0, liquidity=10.0)
        router = PaymentRouter(g)
        path = router.find_optimal_path("A", "C")
        assert path == ["A", "B", "C"]
        assert router.get_path_cost(path) == pytest.approx(1.0)


class TestGetAllPaths:
    def test_get_all_paths(self):
        router = PaymentRouter(_build_graph())
        paths = router.get_all_paths("A", "D")
        assert isinstance(paths, list)
        assert len(paths) >= 1
        assert all(isinstance(p, list) for p in paths)

    def test_get_all_paths_with_fee_constraint(self):
        router = PaymentRouter(_build_graph())
        paths = router.get_all_paths("A", "D", max_fee=4.0)
        for p in paths:
            for u, v in zip(p, p[1:]):
                assert router.graph[u][v]["fee"] <= 4.0
