"""GNN-based fraud detection engine.

Implements:
- FraudDetector: GAT-based node classifier for fraud scoring
- FraudRingDetector: Dense subgraph detection for fraud ring identification
- ConceptDriftHandler: Adaptive thresholding for evolving fraud patterns

References:
- Hooi et al. (2016) FRAUDAR: Bounding Graph Fraud in the Face of Camouflage
- Liu et al. (2022) HA-GNN: Heterogeneous Graph Neural Network for Fraud Detection
"""
from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class FraudDetector(nn.Module):
    """GAT-based fraud detector for transaction graphs.

    Uses Graph Attention Networks to learn structural patterns indicative
    of fraud rings, with attention mechanisms providing camouflage resistance.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: in_channels -> hidden_channels * heads
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))

        # Last layer: no concat, output hidden_channels
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        else:
            # Single layer case: project down
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Edge feature processing — flexible input dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(max(in_channels, 1), hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        self._last_attention = None
        self.attention = None

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass returning fraud probability scores for each node."""
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None

        # Process edge features if available
        if edge_attr is not None and edge_attr.numel() > 0:
            # Ensure edge_attr has compatible dimensions
            if edge_attr.size(1) != self.edge_mlp[0].in_features:
                # Project to expected dimension
                edge_attr = F.pad(edge_attr, (0, self.edge_mlp[0].in_features - edge_attr.size(1)))
            _ = self.edge_mlp(edge_attr)  # Edge features processed for future use

        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=None)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global graph representation for context
        if hasattr(data, "batch") and data.batch is not None:
            global_repr = global_mean_pool(x, data.batch)
            # Expand back to node level
            global_per_node = global_repr[data.batch]
        else:
            global_mean = x.mean(dim=0, keepdim=True)
            global_per_node = global_mean.expand(x.size(0), -1)

        # Combine local and global
        combined = torch.cat([x, global_per_node], dim=-1)

        # Classify
        logits = self.classifier(combined).squeeze(-1)
        scores = torch.sigmoid(logits)

        return scores

    def score_transaction(self, data: Data, node_idx: int) -> float:
        """Score a single transaction (node) for fraud probability."""
        self.eval()
        with torch.no_grad():
            scores = self(data)
        return scores[node_idx].item()

    def detect_fraud_rings(self, data: Data, threshold: float = 0.5) -> list[list[int]]:
        """Detect fraud rings as dense subgraphs of high-scoring nodes.

        Uses connected components on the subgraph induced by nodes
        scoring above threshold, filtered by density.
        """
        self.eval()
        with torch.no_grad():
            scores = self(data)

        # Get high-scoring nodes
        high_nodes = set((scores >= threshold).nonzero(as_tuple=True)[0].tolist())
        if not high_nodes:
            return []

        # Build adjacency for high-scoring nodes
        edge_index = data.edge_index
        adj: dict[int, set[int]] = {n: set() for n in high_nodes}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in high_nodes and dst in high_nodes:
                adj[src].add(dst)
                adj[dst].add(src)

        # Find connected components (candidate rings)
        visited: set[int] = set()
        rings: list[list[int]] = []

        for node in high_nodes:
            if node in visited:
                continue
            # BFS
            component: list[int] = []
            queue = deque([node])
            visited.add(node)
            while queue:
                curr = queue.popleft()
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Filter by density: fraud rings are dense subgraphs
            if len(component) >= 2:
                n = len(component)
                max_edges = n * (n - 1)
                actual_edges = sum(len(adj[c]) for c in component) // 2
                density = actual_edges / max_edges if max_edges > 0 else 0
                if density >= 0.3:  # Minimum density threshold
                    rings.append(sorted(component))

        return rings

    def feature_importance(self, data: Data) -> dict[str, float]:
        """Compute feature importance via gradient-based attribution."""
        self.eval()
        x = data.x.clone().requires_grad_(True)
        data_copy = Data(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
        )
        scores = self(data_copy)
        # Sum of scores as target for attribution
        scores.sum().backward()

        importance = x.grad.abs().mean(dim=0)
        # Normalize
        total = importance.sum()
        if total > 0:
            importance = importance / total

        feature_names = ["amount", "frequency", "recency"]
        result = {}
        for i, name in enumerate(feature_names[: len(importance)]):
            result[name] = importance[i].item()
        # Generic names for remaining features
        for i in range(len(feature_names), len(importance)):
            result[f"feature_{i}"] = importance[i].item()
        return result


class FraudRingDetector:
    """Detects fraud rings using dense subgraph analysis.

    Based on FRAUDAR (Hooi et al., 2016): fraud rings form dense
    subgraphs that are computationally expensive to camouflage.
    """

    def __init__(self, threshold: float = 0.5, min_density: float = 0.3, min_ring_size: int = 2):
        self.threshold = threshold
        self.min_density = min_density
        self.min_ring_size = min_ring_size

    def find_rings(self, data: Data, scores: torch.Tensor) -> list[list[int]]:
        """Find fraud rings as dense subgraphs among high-scoring nodes."""
        high_nodes = set((scores >= self.threshold).nonzero(as_tuple=True)[0].tolist())
        if not high_nodes:
            return []

        # Build adjacency
        edge_index = data.edge_index
        adj: dict[int, set[int]] = {n: set() for n in high_nodes}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in high_nodes and dst in high_nodes:
                adj[src].add(dst)
                adj[dst].add(src)

        # Find connected components with density filter
        visited: set[int] = set()
        rings: list[list[int]] = []

        for node in high_nodes:
            if node in visited:
                continue
            component: list[int] = []
            queue = deque([node])
            visited.add(node)
            while queue:
                curr = queue.popleft()
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) >= self.min_ring_size:
                n = len(component)
                max_edges = n * (n - 1)
                actual_edges = sum(len(adj[c]) for c in component) // 2
                density = actual_edges / max_edges if max_edges > 0 else 0
                if density >= self.min_density:
                    rings.append(sorted(component))

        return rings

    def compute_ring_score(self, data: Data, ring: list[int]) -> float:
        """Compute a fraud score for a ring based on density and connectivity."""
        if len(ring) < 2:
            return 0.0

        edge_index = data.edge_index
        ring_set = set(ring)
        internal_edges = 0
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in ring_set and dst in ring_set:
                internal_edges += 1

        n = len(ring)
        max_edges = n * (n - 1)
        density = internal_edges / max_edges if max_edges > 0 else 0
        return density


class ConceptDriftHandler:
    """Handles concept drift in fraud patterns using adaptive thresholds.

    Monitors score distribution and adjusts detection thresholds
    when significant distribution shifts are detected.
    """

    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.scores_window: deque[float] = deque(maxlen=window_size)
        self.baseline_mean: float | None = None
        self.baseline_std: float | None = None
        self._threshold: float = 0.5
        self._baseline_established: bool = False

    def update(self, score: float) -> None:
        """Update the drift handler with a new fraud score."""
        self.scores_window.append(score)

        # Establish baseline from first half of window
        if not self._baseline_established and len(self.scores_window) >= self.window_size // 2:
            self._update_baseline()
            self._baseline_established = True

        # Auto-adjust threshold based on drift state
        self._auto_adjust_threshold()

    def _update_baseline(self) -> None:
        """Update baseline statistics from first half of window."""
        first_half = list(self.scores_window)[: self.window_size // 2]
        if not first_half:
            return
        self.baseline_mean = sum(first_half) / len(first_half)
        variance = sum((s - self.baseline_mean) ** 2 for s in first_half) / len(first_half)
        self.baseline_std = math.sqrt(variance) if variance > 0 else 0.1

    def drift_detected(self) -> bool:
        """Check if concept drift has been detected.

        Uses a simple z-score test on recent scores vs baseline.
        """
        if not self._baseline_established or self.baseline_mean is None or self.baseline_std is None:
            return False
        if len(self.scores_window) < self.window_size // 2:
            return False

        # Compare second half against baseline from first half
        second_half = list(self.scores_window)[self.window_size // 2 :]
        if not second_half:
            return False

        recent_mean = sum(second_half) / len(second_half)
        z_score = abs(recent_mean - self.baseline_mean) / (self.baseline_std + 1e-8)
        return z_score > self.sensitivity

    def get_threshold(self) -> float:
        """Get the current adaptive threshold."""
        return self._threshold

    def adjust_threshold(self, new_threshold: float) -> None:
        """Manually adjust the detection threshold."""
        self._threshold = max(0.0, min(1.0, new_threshold))

    def _auto_adjust_threshold(self) -> None:
        """Auto-adjust threshold based on drift state."""
        if self.drift_detected():
            # Lower threshold to catch more potential fraud during drift
            self._threshold = max(0.1, self._threshold * 0.8)
        else:
            # Gradually restore threshold toward default
            self._threshold = min(0.5, self._threshold * 1.05)

    def reset(self) -> None:
        """Reset the drift handler state."""
        self.scores_window.clear()
        self.baseline_mean = None
        self.baseline_std = None
        self._threshold = 0.5
        self._baseline_established = False
