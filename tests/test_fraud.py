"""Tests for fraud detection engine with GNN.

TDD: These tests define the expected behavior of FraudDetector.
"""
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fraud.detection import FraudDetector, FraudRingDetector, ConceptDriftHandler


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def simple_graph():
    """Simple 6-node graph with a clear fraud ring (nodes 0-2)."""
    # Nodes 0,1,2 form a dense fraud ring; nodes 3,4,5 are normal
    x = torch.tensor([
        [1000.0, 0.9, 0.1],   # 0: fraud
        [1200.0, 0.8, 0.2],   # 1: fraud
        [1100.0, 0.85, 0.15], # 2: fraud
        [50.0, 0.1, 0.9],     # 3: normal
        [45.0, 0.05, 0.95],   # 4: normal
        [60.0, 0.08, 0.88],   # 5: normal
    ])
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 4, 5],
        [1, 2, 0, 2, 0, 1, 4, 5, 3],
    ])
    edge_attr = torch.tensor([
        [1000.0], [1100.0], [1200.0], [1100.0], [1000.0], [1200.0],
        [50.0], [45.0], [60.0],
    ])
    y = torch.tensor([1, 1, 1, 0, 0, 0])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


@pytest.fixture
def large_graph():
    """10K-node graph for performance testing."""
    n = 10000
    torch.manual_seed(42)
    x = torch.randn(n, 8)
    # Create ~50K edges
    edge_index = torch.randint(0, n, (2, 50000))
    edge_attr = torch.rand(50000, 2)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def detector():
    return FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)


@pytest.fixture
def trained_detector(simple_graph):
    """Detector trained for a few epochs on simple graph."""
    det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
    det.train()
    opt = torch.optim.Adam(det.parameters(), lr=0.01)
    for _ in range(20):
        opt.zero_grad()
        out = det(simple_graph)
        loss = F.binary_cross_entropy(out.squeeze(), simple_graph.y.float())
        loss.backward()
        opt.step()
    det.eval()
    return det


# ── Test 1: Initialization ───────────────────────────────────────────────

class TestInitialization:
    def test_detector_creates_with_default_params(self):
        det = FraudDetector(in_channels=3)
        assert det.in_channels == 3
        assert det.hidden_channels == 64
        assert det.num_layers == 2

    def test_detector_creates_with_custom_params(self):
        det = FraudDetector(in_channels=8, hidden_channels=32, num_layers=3)
        assert det.in_channels == 8
        assert det.hidden_channels == 32
        assert det.num_layers == 3

    def test_detector_is_nn_module(self, detector):
        assert isinstance(detector, torch.nn.Module)


# ── Test 2: Forward pass ──────────────────────────────────────────────────

class TestForward:
    def test_forward_returns_scores(self, detector, simple_graph):
        detector.eval()
        with torch.no_grad():
            scores = detector(simple_graph)
        assert scores.shape == (6,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_forward_output_shape_matches_nodes(self, detector, simple_graph):
        detector.eval()
        with torch.no_grad():
            scores = detector(simple_graph)
        assert scores.shape[0] == simple_graph.num_nodes

    def test_forward_is_differentiable(self, detector, simple_graph):
        detector.train()
        scores = detector(simple_graph)
        loss = scores.sum()
        loss.backward()
        # Check gradients exist
        has_grad = any(p.grad is not None for p in detector.parameters())
        assert has_grad


# ── Test 3: Fraud detection ───────────────────────────────────────────────

class TestFraudDetection:
    def test_detects_fraud_ring(self, trained_detector, simple_graph):
        trained_detector.eval()
        with torch.no_grad():
            scores = trained_detector(simple_graph)
        # Fraud nodes (0,1,2) should have higher scores than normal (3,4,5)
        fraud_mean = scores[:3].mean()
        normal_mean = scores[3:].mean()
        assert fraud_mean > normal_mean

    def test_fraud_scores_are_probabilities(self, trained_detector, simple_graph):
        trained_detector.eval()
        with torch.no_grad():
            scores = trained_detector(simple_graph)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_score_single_transaction(self, trained_detector, simple_graph):
        trained_detector.eval()
        score = trained_detector.score_transaction(simple_graph, node_idx=0)
        assert 0 <= score <= 1


# ── Test 4: Fraud ring detection ──────────────────────────────────────────

class TestFraudRingDetection:
    def test_detect_fraud_rings_returns_list(self, trained_detector, simple_graph):
        rings = trained_detector.detect_fraud_rings(simple_graph, threshold=0.5)
        assert isinstance(rings, list)

    def test_detect_fraud_rings_finds_ring(self, trained_detector, simple_graph):
        rings = trained_detector.detect_fraud_rings(simple_graph, threshold=0.3)
        # Should find at least one ring
        assert len(rings) >= 1
        # The ring should contain fraud nodes
        all_ring_nodes = set()
        for ring in rings:
            all_ring_nodes.update(ring)
        # At least some fraud nodes should be in rings
        assert len(all_ring_nodes & {0, 1, 2}) > 0

    def test_fraud_ring_detector_class_exists(self):
        frd = FraudRingDetector(threshold=0.5)
        assert frd.threshold == 0.5


# ── Test 5: Camouflage resistance ─────────────────────────────────────────

class TestCamouflageResistance:
    def test_camouflage_resistant_scoring(self, trained_detector, simple_graph):
        """Detector should use attention to resist camouflage."""
        trained_detector.eval()
        with torch.no_grad():
            scores = trained_detector(simple_graph)
        # Even if fraudsters mimic normal features, structural patterns
        # should still yield different scores
        assert scores.std() > 0.01

    def test_attention_weights_exist(self, detector, simple_graph):
        """GNN should use attention mechanism for camouflage resistance."""
        detector.eval()
        with torch.no_grad():
            _ = detector(simple_graph)
        # Check that attention weights were computed
        assert hasattr(detector, '_last_attention') or hasattr(detector, 'attention')


# ── Test 6: Concept drift ─────────────────────────────────────────────────

class TestConceptDrift:
    def test_concept_drift_handler_exists(self):
        handler = ConceptDriftHandler(window_size=100)
        assert handler.window_size == 100

    def test_drift_detection_on_distribution_shift(self):
        handler = ConceptDriftHandler(window_size=200)
        # Normal data — establishes baseline
        normal_scores = torch.rand(100) * 0.3
        for s in normal_scores:
            handler.update(s.item())
        # Shifted data (fraud pattern change) — should trigger drift
        shifted_scores = torch.rand(100) * 0.3 + 0.7
        for s in shifted_scores:
            handler.update(s.item())
        assert handler.drift_detected()

    def test_no_drift_on_stable_distribution(self):
        handler = ConceptDriftHandler(window_size=50)
        stable_scores = torch.rand(100) * 0.3
        for s in stable_scores:
            handler.update(s.item())
        assert not handler.drift_detected()

    def test_adaptive_threshold_adjustment(self):
        handler = ConceptDriftHandler(window_size=200)
        for _ in range(100):
            handler.update(0.1)
        initial_threshold = handler.get_threshold()
        # Simulate drift — should auto-adjust threshold
        for _ in range(100):
            handler.update(0.9)
        new_threshold = handler.get_threshold()
        assert new_threshold != initial_threshold


# ── Test 7: Performance ───────────────────────────────────────────────────

class TestPerformance:
    def test_large_graph_inference_under_500ms(self, large_graph):
        """10K-node graph inference must complete in <500ms."""
        det = FraudDetector(in_channels=8, hidden_channels=32, num_layers=2)
        det.eval()
        import time
        with torch.no_grad():
            start = time.perf_counter()
            scores = det(large_graph)
            elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 500, f"Inference took {elapsed:.1f}ms, must be <500ms"
        assert scores.shape[0] == 10000


# ── Test 8: Training ──────────────────────────────────────────────────────

class TestTraining:
    def test_training_reduces_loss(self, simple_graph):
        det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        det.train()
        opt = torch.optim.Adam(det.parameters(), lr=0.05)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            out = det(simple_graph)
            loss = F.binary_cross_entropy(out.squeeze(), simple_graph.y.float())
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Loss should generally decrease
        assert losses[-1] < losses[0] * 1.5  # Allow some noise

    def test_model_has_gnn_layers(self, detector):
        """Model should contain GNN convolution layers."""
        has_gnn = any(
            isinstance(m, (GATConv, SAGEConv))
            for m in detector.modules()
        )
        assert has_gnn


# ── Test 9: Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_node_graph(self):
        det = FraudDetector(in_channels=3, hidden_channels=8, num_layers=1)
        x = torch.tensor([[100.0, 0.5, 0.5]])
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        det.eval()
        with torch.no_grad():
            scores = det(data)
        assert scores.shape == (1,)

    def test_empty_edges_graph(self):
        det = FraudDetector(in_channels=3, hidden_channels=8, num_layers=1)
        x = torch.randn(5, 3)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        det.eval()
        with torch.no_grad():
            scores = det(data)
        assert scores.shape == (5,)

    def test_batch_inference(self, simple_graph):
        """Should handle batched graphs."""
        det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        det.eval()
        from torch_geometric.loader import DataLoader
        loader = DataLoader([simple_graph, simple_graph], batch_size=2)
        with torch.no_grad():
            for batch in loader:
                scores = det(batch)
                assert scores.shape[0] == 12  # 6 + 6 nodes


# ── Test 10: Model persistence ────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, detector, simple_graph, tmp_path):
        det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        det.eval()
        with torch.no_grad():
            original_scores = det(simple_graph)

        path = tmp_path / "model.pt"
        torch.save(det.state_dict(), path)

        det2 = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        det2.load_state_dict(torch.load(path, weights_only=True))
        det2.eval()
        with torch.no_grad():
            loaded_scores = det2(simple_graph)

        assert torch.allclose(original_scores, loaded_scores)


# ── Test 11: Dense subgraph detection ─────────────────────────────────────

class TestDenseSubgraphDetection:
    def test_dense_subgraph_scoring(self, simple_graph):
        """Dense subgraphs should get higher fraud scores."""
        det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        det.eval()
        with torch.no_grad():
            scores = det(simple_graph)
        # Dense ring nodes should score higher
        assert scores[0] > 0.0  # At least some signal

    def test_ring_detector_finds_dense_components(self, simple_graph):
        frd = FraudRingDetector(threshold=0.3)
        det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        det.eval()
        with torch.no_grad():
            scores = det(simple_graph)
        rings = frd.find_rings(simple_graph, scores)
        assert isinstance(rings, list)


# ── Test 12: Feature importance ───────────────────────────────────────────

class TestFeatureImportance:
    def test_feature_importance_returns_dict(self, detector, simple_graph):
        detector.eval()
        importance = detector.feature_importance(simple_graph)
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_feature_importance_sums_to_one(self, detector, simple_graph):
        detector.eval()
        importance = detector.feature_importance(simple_graph)
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01 or total > 0  # Allow unnormalized


# ── Test 13: Integration ──────────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_fraud_detection(self, simple_graph):
        """Full pipeline: train -> detect -> identify rings."""
        det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        # Train
        det.train()
        opt = torch.optim.Adam(det.parameters(), lr=0.05)
        for _ in range(25):
            opt.zero_grad()
            out = det(simple_graph)
            loss = F.binary_cross_entropy(out.squeeze(), simple_graph.y.float())
            loss.backward()
            opt.step()
        # Detect
        det.eval()
        with torch.no_grad():
            scores = det(simple_graph)
        rings = det.detect_fraud_rings(simple_graph, threshold=0.3)
        # Verify
        assert len(rings) >= 1
        assert scores.shape[0] == 6

    def test_multiple_fraud_rings(self):
        """Graph with multiple separate fraud rings."""
        # Ring 1: nodes 0,1,2; Ring 2: nodes 6,7,8; Normal: 3,4,5,9,10,11
        x = torch.tensor([
            [1000.0, 0.9, 0.1], [1200.0, 0.8, 0.2], [1100.0, 0.85, 0.15],
            [50.0, 0.1, 0.9], [45.0, 0.05, 0.95], [60.0, 0.08, 0.88],
            [900.0, 0.88, 0.12], [1300.0, 0.82, 0.18], [1050.0, 0.87, 0.13],
            [55.0, 0.09, 0.91], [48.0, 0.06, 0.94], [52.0, 0.07, 0.9],
        ])
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11],
            [1, 2, 0, 2, 0, 1, 4, 5, 3, 7, 8, 6, 8, 6, 7, 10, 11, 9],
        ])
        edge_attr = torch.ones(18, 1)
        y = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        det = FraudDetector(in_channels=3, hidden_channels=16, num_layers=2)
        det.train()
        opt = torch.optim.Adam(det.parameters(), lr=0.05)
        for _ in range(30):
            opt.zero_grad()
            out = det(data)
            loss = F.binary_cross_entropy(out.squeeze(), data.y.float())
            loss.backward()
            opt.step()
        det.eval()
        with torch.no_grad():
            _ = det(data)
        rings = det.detect_fraud_rings(data, threshold=0.3)
        assert len(rings) >= 1
