"""Network analysis engine for financial networks.

Implements centrality measures, community detection, and systemic risk
metrics for analyzing financial networks (interbank lending, payment
flows, counterparty exposure networks).

References:
    - Barigozzi, M. & Brownlees, C. (2019). NETS: Network estimation
      for time series. Journal of Applied Econometrics.
    - Battiston, S. et al. (2012). DebtRank: Too central to fail?
      Scientific Reports, 2, 541.
    - Newman, M.E.J. (2010). Networks: An Introduction. Oxford
      University Press.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


class NetworkAnalysisEngine:
    """Engine for analyzing financial networks.

    Supports construction from edge lists and adjacency matrices,
    centrality measures, community detection, and systemic risk metrics.
    """

    def __init__(self) -> None:
        """Initialize with an empty undirected graph."""
        self.graph: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @classmethod
    def from_edge_list(
        cls,
        edges: list[tuple[str, str, float]],
        directed: bool = False,
    ) -> NetworkAnalysisEngine:
        """Build network from a list of (source, target, weight) edges.

        Args:
            edges: List of (source, target, weight) tuples.
            directed: If True, create a directed graph.

        Returns:
            NetworkAnalysisEngine instance.

        Raises:
            ValueError: If any edge weight is negative.
        """
        for _, _, w in edges:
            if w < 0:
                raise ValueError(f"Edge weight must be non-negative, got {w}")

        engine = cls()
        engine.graph = nx.DiGraph() if directed else nx.Graph()
        engine.graph.add_weighted_edges_from(edges)
        return engine

    @classmethod
    def from_adjacency_matrix(
        cls,
        matrix: np.ndarray,
        node_names: list[str] | None = None,
    ) -> NetworkAnalysisEngine:
        """Build network from a weighted adjacency matrix.

        Args:
            matrix: NxN numpy array of edge weights.
            node_names: Optional list of node labels.

        Returns:
            NetworkAnalysisEngine instance.
        """
        engine = cls()
        n = matrix.shape[0]
        if node_names is None:
            node_names = list(range(n))
        engine.graph = nx.from_numpy_array(matrix, create_using=nx.Graph)
        mapping = {i: node_names[i] for i in range(n)}
        engine.graph = nx.relabel_nodes(engine.graph, mapping)
        return engine

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of edges in the network."""
        return self.graph.number_of_edges()

    # ------------------------------------------------------------------
    # Centrality measures
    # ------------------------------------------------------------------

    def degree_centrality(self, weighted: bool = False) -> dict[str, float]:
        """Compute degree centrality for all nodes.

        Args:
            weighted: If True, use strength (sum of edge weights).

        Returns:
            Dict mapping node -> centrality value.
        """
        if self.num_nodes == 0:
            return {}
        if weighted:
            return dict(self.graph.degree(weight="weight"))
        return nx.degree_centrality(self.graph)

    def betweenness_centrality(self) -> dict[str, float]:
        """Compute betweenness centrality for all nodes."""
        if self.num_nodes == 0:
            return {}
        return nx.betweenness_centrality(self.graph, weight="weight")

    def eigenvector_centrality(self) -> dict[str, float]:
        """Compute eigenvector centrality for all nodes."""
        if self.num_nodes == 0:
            return {}
        try:
            return nx.eigenvector_centrality(self.graph, weight="weight", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            # Fall back to degree centrality if convergence fails
            return nx.degree_centrality(self.graph)

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Compute PageRank for all nodes.

        Args:
            alpha: Damping parameter (default 0.85).

        Returns:
            Dict mapping node -> PageRank value.
        """
        if self.num_nodes == 0:
            return {}
        return nx.pagerank(self.graph, alpha=alpha, weight="weight")

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def community_detection_louvain(self) -> dict[str, int]:
        """Detect communities using Louvain method.

        Returns:
            Dict mapping node -> community label.
        """
        if self.num_nodes == 0:
            return {}
        communities = nx.community.louvain_communities(self.graph, weight="weight")
        result: dict[str, int] = {}
        for label, community in enumerate(communities):
            for node in community:
                result[node] = label
        return result

    def community_detection_greedy(self) -> dict[str, int]:
        """Detect communities using greedy modularity maximization.

        Returns:
            Dict mapping node -> community label.
        """
        if self.num_nodes == 0:
            return {}
        communities = nx.community.greedy_modularity_communities(
            self.graph, weight="weight"
        )
        result: dict[str, int] = {}
        for label, community in enumerate(communities):
            for node in community:
                result[node] = label
        return result

    def modularity(self, communities: dict[str, int]) -> float:
        """Compute modularity of a community partition.

        Args:
            communities: Dict mapping node -> community label.

        Returns:
            Modularity value in [-0.5, 1].
        """
        if self.num_nodes == 0:
            return 0.0
        # Convert to list of sets format expected by networkx
        groups: dict[int, set] = {}
        for node, label in communities.items():
            groups.setdefault(label, set()).add(node)
        return nx.community.modularity(self.graph, list(groups.values()), weight="weight")

    # ------------------------------------------------------------------
    # Systemic risk metrics
    # ------------------------------------------------------------------

    def debtrank_initial_state(self) -> dict[str, float]:
        """Initialize DebtRank state (all nodes healthy).

        Returns:
            Dict mapping node -> 0.0 (no distress).
        """
        return {node: 0.0 for node in self.graph.nodes}

    def debtrank(
        self,
        shocked_nodes: set[str],
        iterations: int = 10,
        equity: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute DebtRank distress propagation.

        Implements the Battiston et al. (2012) DebtRank algorithm:
        distress propagates through the network via relative exposure.

        Args:
            shocked_nodes: Set of initially distressed nodes.
            iterations: Number of propagation rounds.
            equity: Optional dict of node equity values. If None,
                all nodes have equal equity of 1.0.

        Returns:
            Dict mapping node -> distress level in [0, 1].
        """
        if self.num_nodes == 0:
            return {}

        if equity is None:
            equity = {node: 1.0 for node in self.graph.nodes}

        # Initialize: shocked nodes have distress = 1
        state: dict[str, float] = {
            node: 1.0 if node in shocked_nodes else 0.0
            for node in self.graph.nodes
        }

        # Propagate distress
        for _ in range(iterations):
            new_state = dict(state)
            for node in self.graph.nodes:
                if node in shocked_nodes:
                    continue
                incoming_distress = 0.0
                for neighbor in self.graph.predecessors(node) if self.graph.is_directed() else self.graph.neighbors(node):
                    if state[neighbor] > 0:
                        # Relative exposure = weight / equity
                        weight = self.graph[neighbor][node].get("weight", 1.0)
                        rel_exposure = weight / max(equity.get(neighbor, 1.0), 1e-10)
                        incoming_distress += state[neighbor] * rel_exposure
                new_state[node] = min(incoming_distress, 1.0)
            state = new_state

        return state

    def debtrank_centrality(self) -> dict[str, float]:
        """Compute DebtRank centrality for each node.

        Measures how much distress each node would cause if it were
        the initial shock.

        Returns:
            Dict mapping node -> systemic importance score.
        """
        if self.num_nodes == 0:
            return {}
        centrality: dict[str, float] = {}
        for node in self.graph.nodes:
            result = self.debtrank({node}, iterations=5)
            # Centrality = total distress caused in the network
            centrality[node] = sum(result.values())
        return centrality

    def systemic_risk_score(self) -> float:
        """Compute overall systemic risk score for the network.

        Combines network density, average clustering, and concentration
        of centrality into a single risk score.

        Returns:
            Risk score in [0, 1].
        """
        if self.num_nodes == 0:
            return 0.0

        # Factor 1: Network density (denser = more interconnected = riskier)
        density = self.density()

        # Factor 2: Clustering coefficient (higher = more contagion risk)
        clustering = self.clustering_coefficient()

        # Factor 3: Centrality concentration (Gini-like)
        cent = self.degree_centrality()
        if cent:
            values = sorted(cent.values())
            n = len(values)
            if n > 1 and sum(values) > 0:
                # Normalized Gini coefficient
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (
                    n * np.sum(values)
                )
            else:
                gini = 0.0
        else:
            gini = 0.0

        # Weighted combination
        score = 0.4 * density + 0.3 * clustering + 0.3 * max(0.0, gini)
        return min(max(score, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Network structure metrics
    # ------------------------------------------------------------------

    def density(self) -> float:
        """Network density (ratio of actual to possible edges)."""
        if self.num_nodes <= 1:
            return 0.0
        return nx.density(self.graph)

    def connected_components(self) -> list[set[str]]:
        """List of connected components."""
        return list(nx.connected_components(self.graph))

    def clustering_coefficient(self) -> float:
        """Average clustering coefficient."""
        if self.num_nodes == 0:
            return 0.0
        return nx.average_clustering(self.graph, weight="weight")

    def average_shortest_path_length(self) -> float:
        """Average shortest path length across all pairs."""
        if self.num_nodes <= 1:
            return 0.0
        if not nx.is_connected(self.graph):
            # Use largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            return nx.average_shortest_path_length(subgraph, weight="weight")
        return nx.average_shortest_path_length(self.graph, weight="weight")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics for the network.

        Returns:
            Dict with key network metrics.
        """
        avg_degree = (
            2 * self.num_edges / self.num_nodes if self.num_nodes > 0 else 0.0
        )
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.density(),
            "avg_degree": avg_degree,
            "clustering_coefficient": self.clustering_coefficient(),
            "num_components": len(self.connected_components()),
        }
