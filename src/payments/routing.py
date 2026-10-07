"""Payment routing optimization engine."""

from __future__ import annotations

import networkx as nx


class PaymentRouter:
    """Find optimal payment paths in a channel network.

    The graph is a ``nx.DiGraph`` where each edge has:
      - ``fee``: transaction fee (float)
      - ``liquidity``: available liquidity (float)
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    def _filtered_graph(
        self,
        max_fee: float | None = None,
        min_liquidity: float | None = None,
    ) -> nx.DiGraph:
        """Return subgraph containing only edges satisfying constraints."""
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nodes())
        for u, v, data in self.graph.edges(data=True):
            fee = data.get("fee", 0.0)
            liq = data.get("liquidity", float("inf"))
            if max_fee is not None and fee > max_fee:
                continue
            if min_liquidity is not None and liq < min_liquidity:
                continue
            g.add_edge(u, v, **data)
        return g

    def find_optimal_path(
        self,
        source,
        target,
        max_fee: float | None = None,
        min_liquidity: float | None = None,
    ) -> list:
        """Return the minimum-fee path from source to target.

        Uses Dijkstra for non-negative fees, Bellman-Ford when negative
        fees are present. Raises ``nx.NetworkXNoPath`` if no valid path.
        """
        g = self._filtered_graph(max_fee, min_liquidity)
        has_negative = any(
            d.get("fee", 0.0) < 0.0 for _, _, d in g.edges(data=True)
        )
        method = "bellman-ford" if has_negative else "dijkstra"
        return nx.shortest_path(g, source, target, weight="fee", method=method)

    def get_path_cost(self, path: list) -> float:
        """Return the total fee along *path*."""
        return sum(
            self.graph[u][v].get("fee", 0.0) for u, v in zip(path, path[1:])
        )

    def get_all_paths(
        self,
        source,
        target,
        max_fee: float | None = None,
        min_liquidity: float | None = None,
    ) -> list:
        """Return all simple paths from source to target satisfying constraints."""
        g = self._filtered_graph(max_fee, min_liquidity)
        try:
            return list(nx.all_simple_paths(g, source, target))
        except nx.NodeNotFound:
            return []
