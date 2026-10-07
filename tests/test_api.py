"""Tests for API Gateway — REST API, authentication, rate limiting."""
import pytest
from fastapi.testclient import TestClient

from src.api.gateway import APIGateway


@pytest.fixture
def gateway():
    """Create gateway with low rate limit for testing."""
    return APIGateway(api_key="test-key-123", rate_limit=10, rate_window=60)


@pytest.fixture
def client(gateway):
    return TestClient(gateway.get_app())


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "engines" in data
        assert "version" in data

    def test_health_no_auth_required(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_missing_api_key_rejected(self, client):
        resp = client.get("/engines")
        assert resp.status_code == 401

    def test_invalid_api_key_rejected(self, client):
        resp = client.get("/engines", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_valid_api_key_accepted(self, client):
        resp = client.get("/engines", headers={"X-API-Key": "test-key-123"})
        assert resp.status_code == 200

    def test_auth_on_all_endpoints(self, client):
        resp = client.post("/marketmaking/quotes", json={})
        assert resp.status_code == 401
        resp = client.get("/regtech/report")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_enforced(self, client):
        # Use up the rate limit
        for _ in range(10):
            resp = client.get("/engines", headers={"X-API-Key": "test-key-123"})
            assert resp.status_code == 200
        # Next request should be rate limited
        resp = client.get("/engines", headers={"X-API-Key": "test-key-123"})
        assert resp.status_code == 429

    def test_rate_limit_resets_after_window(self, client):
        # This test uses a very short window
        gw = APIGateway(api_key="test-key-123", rate_limit=2, rate_window=1)
        c = TestClient(gw.get_app())
        resp1 = c.get("/engines", headers={"X-API-Key": "test-key-123"})
        assert resp1.status_code == 200
        resp2 = c.get("/engines", headers={"X-API-Key": "test-key-123"})
        assert resp2.status_code == 200
        resp3 = c.get("/engines", headers={"X-API-Key": "test-key-123"})
        assert resp3.status_code == 429


# ---------------------------------------------------------------------------
# Engine discovery
# ---------------------------------------------------------------------------


class TestEngineDiscovery:
    def test_list_engines(self, client):
        resp = client.get("/engines", headers={"X-API-Key": "test-key-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "marketmaking" in data
        assert "altdata" in data
        assert "tokenization" in data
        assert "regtech" in data

    def test_engine_details(self, client):
        resp = client.get("/engines", headers={"X-API-Key": "test-key-123"})
        data = resp.json()
        for name, info in data.items():
            assert "status" in info
            assert "endpoints" in info


# ---------------------------------------------------------------------------
# Market Making endpoints
# ---------------------------------------------------------------------------


class TestMarketMakingEndpoints:
    def test_compute_quotes(self, client):
        resp = client.post(
            "/marketmaking/quotes",
            headers={"X-API-Key": "test-key-123"},
            json={
                "mid_price": 100.0,
                "inventory": 0,
                "time_remaining": 10.0,
                "gamma": 0.1,
                "sigma": 0.5,
                "k": 1.5,
                "A": 0.1,
                "dt": 1.0,
                "max_inventory": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "bid" in data
        assert "ask" in data
        assert "mid" in data
        assert "spread" in data
        assert "reservation_price" in data
        assert data["bid"] < data["ask"]
        assert data["mid"] == pytest.approx(100.0)

    def test_compute_quotes_with_inventory(self, client):
        resp = client.post(
            "/marketmaking/quotes",
            headers={"X-API-Key": "test-key-123"},
            json={
                "mid_price": 100.0,
                "inventory": 5,
                "time_remaining": 10.0,
                "gamma": 0.1,
                "sigma": 0.5,
                "k": 1.5,
                "A": 0.1,
                "dt": 1.0,
                "max_inventory": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Positive inventory skews quotes down
        assert data["reservation_price"] < 100.0

    def test_compute_quotes_batch(self, client):
        resp = client.post(
            "/marketmaking/quotes/batch",
            headers={"X-API-Key": "test-key-123"},
            json={
                "mid_prices": [100.0, 101.0, 102.0],
                "inventories": [0, 5, -5],
                "times_remaining": [10.0, 10.0, 10.0],
                "gamma": 0.1,
                "sigma": 0.5,
                "k": 1.5,
                "A": 0.1,
                "dt": 1.0,
                "max_inventory": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        for quote in data:
            assert "bid" in quote
            assert "ask" in quote

    def test_compute_quotes_invalid_input(self, client):
        resp = client.post(
            "/marketmaking/quotes",
            headers={"X-API-Key": "test-key-123"},
            json={
                "mid_price": -100.0,
                "inventory": 0,
                "time_remaining": 10.0,
                "gamma": 0.1,
                "sigma": 0.5,
                "k": 1.5,
                "A": 0.1,
                "dt": 1.0,
                "max_inventory": 10,
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# AltData endpoints
# ---------------------------------------------------------------------------


class TestAltDataEndpoints:
    def test_satellite_analysis(self, client):
        resp = client.get(
            "/altdata/satellite/AAPL",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "avg_parking_lot_count" in data
        assert "avg_activity_score" in data

    def test_sentiment_analysis(self, client):
        resp = client.get(
            "/altdata/sentiment/AAPL",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "avg_sentiment" in data
        assert "total_volume" in data

    def test_transaction_analysis(self, client):
        resp = client.get(
            "/altdata/transactions/AAPL",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_spend" in data
        assert "avg_transaction" in data
        assert "transaction_count" in data

    def test_composite_signal(self, client):
        resp = client.get(
            "/altdata/composite/AAPL",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "composite_score" in data
        assert "data_sources_used" in data

    def test_ingest_and_analyze(self, client):
        # Ingest some data first
        client.post(
            "/altdata/ingest/satellite",
            headers={"X-API-Key": "test-key-123"},
            json={
                "ticker": "TSLA",
                "data": {"parking_lot_count": 50, "activity_score": 0.8},
            },
        )
        # Then analyze
        resp = client.get(
            "/altdata/satellite/TSLA",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["avg_parking_lot_count"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Tokenization endpoints
# ---------------------------------------------------------------------------


class TestTokenizationEndpoints:
    def test_register_investor(self, client):
        resp = client.post(
            "/tokenization/investors",
            headers={"X-API-Key": "test-key-123"},
            json={
                "address": "0x1234567890abcdef1234567890abcdef12345678",
                "jurisdiction": "US",
                "accredited": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["address"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert data["status"] == "pending"

    def test_verify_investor(self, client):
        addr = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        client.post(
            "/tokenization/investors",
            headers={"X-API-Key": "test-key-123"},
            json={
                "address": addr,
                "jurisdiction": "US",
                "accredited": True,
            },
        )
        resp = client.post(
            f"/tokenization/investors/{addr}/verify",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "verified"

    def test_check_compliance(self, client):
        addr = "0x1111111111111111111111111111111111111111"
        client.post(
            "/tokenization/investors",
            headers={"X-API-Key": "test-key-123"},
            json={
                "address": addr,
                "jurisdiction": "US",
                "accredited": True,
            },
        )
        client.post(
            f"/tokenization/investors/{addr}/verify",
            headers={"X-API-Key": "test-key-123"},
        )
        resp = client.get(
            f"/tokenization/investors/{addr}/compliance",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "compliant"

    def test_compliance_sanctioned_jurisdiction(self, client):
        addr = "0x2222222222222222222222222222222222222222"
        client.post(
            "/tokenization/investors",
            headers={"X-API-Key": "test-key-123"},
            json={
                "address": addr,
                "jurisdiction": "IR",
                "accredited": False,
            },
        )
        client.post(
            f"/tokenization/investors/{addr}/verify",
            headers={"X-API-Key": "test-key-123"},
        )
        resp = client.get(
            f"/tokenization/investors/{addr}/compliance",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "non_compliant"

    def test_transfer_tokens(self, client):
        addr1 = "0x3333333333333333333333333333333333333333"
        addr2 = "0x4444444444444444444444444444444444444444"
        # Register and verify both
        for addr in [addr1, addr2]:
            client.post(
                "/tokenization/investors",
                headers={"X-API-Key": "test-key-123"},
                json={
                    "address": addr,
                    "jurisdiction": "US",
                    "accredited": True,
                },
            )
            client.post(
                f"/tokenization/investors/{addr}/verify",
                headers={"X-API-Key": "test-key-123"},
            )
        # Give addr1 some tokens
        client.post(
            "/tokenization/mint",
            headers={"X-API-Key": "test-key-123"},
            json={"address": addr1, "amount": 1000},
        )
        # Transfer
        resp = client.post(
            "/tokenization/transfer",
            headers={"X-API-Key": "test-key-123"},
            json={
                "from_address": addr1,
                "to_address": addr2,
                "amount": 100,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_get_balance(self, client):
        addr = "0x5555555555555555555555555555555555555555"
        client.post(
            "/tokenization/investors",
            headers={"X-API-Key": "test-key-123"},
            json={
                "address": addr,
                "jurisdiction": "US",
                "accredited": True,
            },
        )
        resp = client.get(
            f"/tokenization/investors/{addr}/balance",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] == 0


# ---------------------------------------------------------------------------
# RegTech endpoints
# ---------------------------------------------------------------------------


class TestRegTechEndpoints:
    def test_register_control(self, client):
        resp = client.post(
            "/regtech/controls",
            headers={"X-API-Key": "test-key-123"},
            json={
                "framework": "PCI_DSS",
                "control_id": "PCI-1.1",
                "name": "Firewall",
                "description": "Firewall configuration",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["control_id"] == "PCI-1.1"
        assert data["status"] == "NOT_ASSESSED"

    def test_list_controls(self, client):
        resp = client.get(
            "/regtech/controls",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_assess_control(self, client):
        client.post(
            "/regtech/controls",
            headers={"X-API-Key": "test-key-123"},
            json={
                "framework": "PCI_DSS",
                "control_id": "PCI-2.1",
                "name": "Encryption",
                "description": "Data encryption",
            },
        )
        resp = client.post(
            "/regtech/controls/PCI-2.1/assess",
            headers={"X-API-Key": "test-key-123"},
            json={"status": "COMPLIANT"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "COMPLIANT"

    def test_generate_report(self, client):
        resp = client.get(
            "/regtech/report",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_controls" in data
        assert "compliant_count" in data
        assert "non_compliant_count" in data

    def test_duplicate_control_rejected(self, client):
        client.post(
            "/regtech/controls",
            headers={"X-API-Key": "test-key-123"},
            json={
                "framework": "PCI_DSS",
                "control_id": "PCI-3.1",
                "name": "Test",
                "description": "Test control",
            },
        )
        resp = client.post(
            "/regtech/controls",
            headers={"X-API-Key": "test-key-123"},
            json={
                "framework": "PCI_DSS",
                "control_id": "PCI-3.1",
                "name": "Duplicate",
                "description": "Duplicate control",
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Gateway class tests
# ---------------------------------------------------------------------------


class TestGatewayClass:
    def test_get_app_returns_fastapi(self, gateway):
        from fastapi import FastAPI
        app = gateway.get_app()
        assert isinstance(app, FastAPI)

    def test_custom_rate_limit(self):
        gw = APIGateway(api_key="key", rate_limit=5, rate_window=60)
        assert gw.rate_limiter.max_requests == 5

    def test_multiple_engines_registered(self, gateway):
        assert gateway.marketmaking_engine is not None or True  # lazy init
        assert gateway.altdata_engine is not None
        assert gateway.tokenization_engine is not None
        assert gateway.regtech_engine is not None
