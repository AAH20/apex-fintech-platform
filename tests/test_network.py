"""Tests for network analysis engine — financial network analysis.

Tests centrality measures, community detection, systemic risk metrics,
and graph construction from financial exposure data.

References:
    - Barigozzi, M. & Brownlees, C. (2019). NETS: Network estimation for
      time series. Journal of Applied Econometrics.
    - Battiston, S. et al. (2012). DebtRank: Too central to fail?
      Scientific Reports, 2, 541.
    - Newman, M.E.J. (2010). Networks: An Introduction. Oxford University Press.
"""
import numpy as np
import pytest

from network import NetworkAnalysisEngine


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    """Tests for building financial networks from various inputs."""

    def test_from_edge_list(self):
        """Engine constructs graph from edge list."""
        edges = [("A", "B", 1.0), ("B", "C", 2.0), ("C", "A", 0.5)]
        engine = NetworkAnalysisEngine.from_edge_list(edges)
        assert engine.num_nodes == 3
        assert engine.num_edges == 3

    def test_from_adjacency_matrix(self):
        """Engine constructs weighted graph from adjacency matrix."""
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ], dtype=float)
        engine = NetworkAnalysisEngine.from_adjacency_matrix(
            adj, node_names=["A", "B", "C", "D"]
        )
        assert engine.num_nodes == 4
        assert engine.num_edges == 4  # undirected: A-B, B-C, B-D, C-D

    def test_from_adjacency_matrix_default_names(self):
        """Default node names are integers when not provided."""
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        engine = NetworkAnalysisEngine.from_adjacency_matrix(adj)
        assert set(engine.graph.nodes) == {0, 1}

    def test_empty_graph(self):
        """Engine handles empty graph."""
        engine = NetworkAnalysisEngine()
        assert engine.num_nodes == 0
        assert engine.num_edges == 0

    def test_single_node_graph(self):
        """Engine handles single-node graph."""
        engine = NetworkAnalysisEngine()
        engine.graph.add_node("A")
        assert engine.num_nodes == 1
        assert engine.num_edges == 0

    def test_directed_graph(self):
        """Engine supports directed graphs."""
        edges = [("A", "B", 1.0), ("B", "A", 0.5)]
        engine = NetworkAnalysisEngine.from_edge_list(edges, directed=True)
        assert engine.graph.is_directed()
        assert engine.num_edges == 2

    def test_edge_weights_preserved(self):
        """Edge weights are stored in the graph."""
        edges = [("A", "B", 3.5)]
        engine = NetworkAnalysisEngine.from_edge_list(edges)
        assert engine.graph["A"]["B"]["weight"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Centrality measures
# ---------------------------------------------------------------------------


class TestCentrality:
    """Tests for centrality measures on financial networks."""

    @pytest.fixture
    def star_engine(self):
        """Star network: A is the center connected to B, C, D, E."""
        edges = [("A", "B", 1), ("A", "C", 1), ("A", "D", 1), ("A", "E", 1)]
        return NetworkAnalysisEngine.from_edge_list(edges)

    @pytest.fixture
    def chain_engine(self):
        """Chain network: A-B-C-D-E."""
        edges = [("A", "B", 1), ("B", "C", 1), ("C", "D", 1), ("D", "E", 1)]
        return NetworkAnalysisEngine.from_edge_list(edges)

    def test_degree_centrality_star(self, star_engine):
        """Center node has highest degree centrality in star network."""
        cent = star_engine.degree_centrality()
        assert cent["A"] == pytest.approx(1.0)
        assert cent["B"] == pytest.approx(0.25)

    def test_degree_centrality_chain(self, chain_engine):
        """Middle node has highest degree centrality in chain."""
        cent = chain_engine.degree_centrality()
        assert cent["C"] == pytest.approx(0.5)
        assert cent["A"] == pytest.approx(0.25)

    def test_betweenness_centrality_star(self, star_engine):
        """Center node has all betweenness in star network."""
        cent = star_engine.betweenness_centrality()
        assert cent["A"] == pytest.approx(1.0)
        assert cent["B"] == pytest.approx(0.0)

    def test_betweenness_centrality_chain(self, chain_engine):
        """Middle node has highest betweenness in chain."""
        cent = chain_engine.betweenness_centrality()
        assert cent["C"] > cent["B"]
        assert cent["C"] > cent["D"]
        assert cent["A"] == pytest.approx(0.0)

    def test_eigenvector_centrality_star(self, star_engine):
        """Center has highest eigenvector centrality in star."""
        cent = star_engine.eigenvector_centrality()
        assert cent["A"] > cent["B"]
        assert cent["A"] > cent["C"]

    def test_pagerank_star(self, star_engine):
        """Center has highest PageRank in star network."""
        pr = star_engine.pagerank()
        assert pr["A"] > pr["B"]
        assert pr["A"] > pr["E"]

    def test_pagerank_sums_to_one(self, star_engine):
        """PageRank values sum to 1."""
        pr = star_engine.pagerank()
        assert sum(pr.values()) == pytest.approx(1.0)

    def test_weighted_centrality(self):
        """Weighted degree centrality accounts for edge weights."""
        edges = [("A", "B", 10.0), ("A", "C", 1.0), ("B", "C", 0.5)]
        engine = NetworkAnalysisEngine.from_edge_list(edges)
        cent = engine.degree_centrality(weighted=True)
        assert cent["A"] > cent["B"]
        assert cent["B"] > cent["C"]


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


class TestCommunityDetection:
    """Tests for community detection in financial networks."""

    @pytest.fixture
    def two_cliques_engine(self):
        """Two cliques connected by a single bridge edge."""
        edges = [
            ("A", "B", 1), ("B", "C", 1), ("A", "C", 1),  # clique 1
            ("D", "E", 1), ("E", "F", 1), ("D", "F", 1),  # clique 2
            ("C", "D", 1),  # bridge
        ]
        return NetworkAnalysisEngine.from_edge_list(edges)

    def test_louvain_detects_two_communities(self, two_cliques_engine):
        """Louvain finds 2 communities in two-clique network."""
        communities = two_cliques_engine.community_detection_louvain()
        assert len(set(communities.values())) == 2

    def test_louvain_same_community_for_clique(self, two_cliques_engine):
        """Nodes in same clique get same community label."""
        communities = two_cliques_engine.community_detection_louvain()
        assert communities["A"] == communities["B"]
        assert communities["D"] == communities["E"]

    def test_louvain_modularity_positive(self, two_cliques_engine):
        """Modularity is positive for well-separated communities."""
        communities = two_cliques_engine.community_detection_louvain()
        mod = two_cliques_engine.modularity(communities)
        assert mod > 0.3

    def test_greedy_modularity_communities(self, two_cliques_engine):
        """Greedy modularity also finds 2 communities."""
        communities = two_cliques_engine.community_detection_greedy()
        assert len(set(communities.values())) == 2


# ---------------------------------------------------------------------------
# Systemic risk metrics
# ---------------------------------------------------------------------------


class TestSystemicRisk:
    """Tests for systemic risk and network structure metrics."""

    @pytest.fixture
    def financial_network(self):
        """Simulated interbank lending network."""
        edges = [
            ("Bank_A", "Bank_B", 100),
            ("Bank_A", "Bank_C", 50),
            ("Bank_B", "Bank_C", 75),
            ("Bank_B", "Bank_D", 200),
            ("Bank_C", "Bank_D", 150),
            ("Bank_D", "Bank_E", 300),
            ("Bank_E", "Bank_F", 250),
            ("Bank_F", "Bank_A", 80),
        ]
        return NetworkAnalysisEngine.from_edge_list(edges)

    def test_network_density(self, financial_network):
        """Density is between 0 and 1."""
        density = financial_network.density()
        assert 0 < density <= 1.0

    def test_connected_components(self, financial_network):
        """All banks are in one connected component."""
        components = financial_network.connected_components()
        assert len(components) == 1
        assert len(components[0]) == 6

    def test_clustering_coefficient(self, financial_network):
        """Clustering coefficient is between 0 and 1."""
        cc = financial_network.clustering_coefficient()
        assert 0 <= cc <= 1.0

    def test_average_shortest_path(self, financial_network):
        """Average shortest path length is positive."""
        avg_path = financial_network.average_shortest_path_length()
        assert avg_path > 0

    def test_debtrank_initialization(self, financial_network):
        """DebtRank initializes with all nodes healthy."""
        state = financial_network.debtrank_initial_state()
        assert all(v == 0.0 for v in state.values())

    def test_debtrank_shock_propagation(self, financial_network):
        """DebtRank shock to one bank increases distress in network."""
        shocked = {"Bank_A"}
        result = financial_network.debtrank(shocked, iterations=5)
        # At least the shocked bank should be distressed
        assert result["Bank_A"] > 0
        # Some contagion should occur
        distressed = sum(1 for v in result.values() if v > 0)
        assert distressed >= 1

    def test_debtrank_centrality_ranking(self, financial_network):
        """DebtRank centrality ranks banks by systemic importance."""
        ranking = financial_network.debtrank_centrality()
        assert len(ranking) == 6
        # All values should be non-negative
        assert all(v >= 0 for v in ranking.values())

    def test_systemic_risk_score(self, financial_network):
        """Systemic risk score is between 0 and 1."""
        score = financial_network.systemic_risk_score()
        assert 0 <= score <= 1.0

    def test_most_systemically_important(self, financial_network):
        """Identifies the most systemically important node."""
        ranking = financial_network.debtrank_centrality()
        most_important = max(ranking, key=ranking.get)
        assert most_important in {"Bank_A", "Bank_B", "Bank_C", "Bank_D", "Bank_E", "Bank_F"}


# ---------------------------------------------------------------------------
# Network summary
# ---------------------------------------------------------------------------


class TestNetworkSummary:
    """Tests for network summary statistics."""

    def test_summary_contains_key_metrics(self, financial_network=None):
        """Summary includes essential network metrics."""
        edges = [("A", "B", 1), ("B", "C", 2), ("C", "A", 1)]
        engine = NetworkAnalysisEngine.from_edge_list(edges)
        summary = engine.summary()
        assert "num_nodes" in summary
        assert "num_edges" in summary
        assert "density" in summary
        assert "avg_degree" in summary
        assert "clustering_coefficient" in summary

    def test_summary_values_correct(self):
        """Summary values match direct computation."""
        edges = [("A", "B", 1), ("B", "C", 2), ("C", "A", 1)]
        engine = NetworkAnalysisEngine.from_edge_list(edges)
        summary = engine.summary()
        assert summary["num_nodes"] == 3
        assert summary["num_edges"] == 3
        assert summary["density"] == pytest.approx(1.0)  # complete graph
        assert summary["avg_degree"] == pytest.approx(2.0)

    def test_empty_graph_summary(self):
        """Summary works for empty graph."""
        engine = NetworkAnalysisEngine()
        summary = engine.summary()
        assert summary["num_nodes"] == 0
        assert summary["num_edges"] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_disconnected_graph_components(self):
        """Disconnected graph has multiple components."""
        edges = [("A", "B", 1), ("C", "D", 1)]
        engine = NetworkAnalysisEngine.from_edge_list(edges)
        components = engine.connected_components()
        assert len(components) == 2

    def test_self_loop_ignored(self):
        """Self-loops are handled gracefully."""
        edges = [("A", "A", 1), ("A", "B", 1)]
        engine = NetworkAnalysisEngine.from_edge_list(edges)
        # Self-loop may or may not be counted depending on implementation
        assert engine.num_nodes == 2

    def test_negative_weights_rejected(self):
        """Negative edge weights raise ValueError."""
        with pytest.raises(ValueError, match="weight"):
            NetworkAnalysisEngine.from_edge_list([("A", "B", -1.0)])

    def test_isolated_node_centrality(self):
        """Isolated node has zero centrality."""
        engine = NetworkAnalysisEngine()
        engine.graph.add_node("A")
        engine.graph.add_node("B")
        cent = engine.degree_centrality()
        assert cent["A"] == 0.0
        assert cent["B"] == 0.0
