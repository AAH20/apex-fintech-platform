"""Tests for the Alternative Data Engine."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.altdata.engine import AlternativeDataEngine


@pytest.fixture
def engine():
    """Create a fresh engine instance for each test."""
    return AlternativeDataEngine(api_key="test_key")


@pytest.fixture
def sample_satellite_data():
    return [
        {"date": "2024-01-01", "ticker": "AAPL", "parking_lot_count": 150, "activity_score": 0.85},
        {"date": "2024-01-02", "ticker": "AAPL", "parking_lot_count": 160, "activity_score": 0.90},
        {"date": "2024-01-03", "ticker": "AAPL", "parking_lot_count": 145, "activity_score": 0.80},
    ]


@pytest.fixture
def sample_sentiment_data():
    return [
        {"date": "2024-01-01", "ticker": "AAPL", "source": "twitter", "sentiment": 0.75, "volume": 1000},
        {"date": "2024-01-02", "ticker": "AAPL", "source": "twitter", "sentiment": 0.60, "volume": 1200},
        {"date": "2024-01-03", "ticker": "AAPL", "source": "reddit", "sentiment": 0.80, "volume": 800},
    ]


@pytest.fixture
def sample_transaction_data():
    return [
        {"date": "2024-01-01", "ticker": "AAPL", "merchant": "Apple Store", "amount": 1500.00, "category": "electronics"},
        {"date": "2024-01-02", "ticker": "AAPL", "merchant": "Apple Store", "amount": 2000.00, "category": "electronics"},
        {"date": "2024-01-03", "ticker": "AAPL", "merchant": "Best Buy", "amount": 800.00, "category": "electronics"},
    ]


class TestAlternativeDataEngineInit:
    """Test engine initialization."""

    def test_engine_initializes_with_api_key(self, engine):
        assert engine.api_key == "test_key"
        assert engine.base_url == "https://api.altdata.example.com/v1"

    def test_engine_initializes_without_api_key(self):
        eng = AlternativeDataEngine()
        assert eng.api_key is None

    def test_engine_has_empty_data_store(self, engine):
        assert engine.data_store == {}


class TestSatelliteData:
    """Test satellite data ingestion and analysis."""

    @patch("src.altdata.engine.requests.get")
    def test_fetch_satellite_data_success(self, mock_get, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"date": "2024-01-01", "ticker": "AAPL", "parking_lot_count": 150}]}
        mock_get.return_value = mock_response

        result = engine.fetch_satellite_data("AAPL", "2024-01-01", "2024-01-31")
        assert result is not None
        assert "data" in result

    @patch("src.altdata.engine.requests.get")
    def test_fetch_satellite_data_failure(self, mock_get, engine):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            engine.fetch_satellite_data("INVALID", "2024-01-01", "2024-01-31")

    def test_ingest_satellite_data(self, engine, sample_satellite_data):
        engine.ingest_satellite_data(sample_satellite_data)
        assert "satellite" in engine.data_store
        assert len(engine.data_store["satellite"]) == 3

    def test_analyze_satellite_trends(self, engine, sample_satellite_data):
        engine.ingest_satellite_data(sample_satellite_data)
        trends = engine.analyze_satellite_trends("AAPL")
        assert "avg_parking_lot_count" in trends
        assert "avg_activity_score" in trends
        assert trends["avg_parking_lot_count"] == pytest.approx(151.67, rel=0.01)


class TestSocialSentiment:
    """Test social sentiment data ingestion and analysis."""

    @patch("src.altdata.engine.requests.get")
    def test_fetch_sentiment_data_success(self, mock_get, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"date": "2024-01-01", "ticker": "AAPL", "sentiment": 0.75}]}
        mock_get.return_value = mock_response

        result = engine.fetch_sentiment_data("AAPL", "2024-01-01", "2024-01-31")
        assert result is not None

    def test_ingest_sentiment_data(self, engine, sample_sentiment_data):
        engine.ingest_sentiment_data(sample_sentiment_data)
        assert "sentiment" in engine.data_store
        assert len(engine.data_store["sentiment"]) == 3

    def test_analyze_sentiment_trends(self, engine, sample_sentiment_data):
        engine.ingest_sentiment_data(sample_sentiment_data)
        trends = engine.analyze_sentiment_trends("AAPL")
        assert "avg_sentiment" in trends
        assert "total_volume" in trends
        assert trends["avg_sentiment"] == pytest.approx(0.7167, rel=0.01)
        assert trends["total_volume"] == 3000


class TestCreditCardTransactions:
    """Test credit card transaction data ingestion and analysis."""

    @patch("src.altdata.engine.requests.get")
    def test_fetch_transaction_data_success(self, mock_get, engine):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"date": "2024-01-01", "ticker": "AAPL", "amount": 1500.00}]}
        mock_get.return_value = mock_response

        result = engine.fetch_transaction_data("AAPL", "2024-01-01", "2024-01-31")
        assert result is not None

    def test_ingest_transaction_data(self, engine, sample_transaction_data):
        engine.ingest_transaction_data(sample_transaction_data)
        assert "transactions" in engine.data_store
        assert len(engine.data_store["transactions"]) == 3

    def test_analyze_transaction_trends(self, engine, sample_transaction_data):
        engine.ingest_transaction_data(sample_transaction_data)
        trends = engine.analyze_transaction_trends("AAPL")
        assert "total_spend" in trends
        assert "avg_transaction" in trends
        assert "transaction_count" in trends
        assert trends["total_spend"] == pytest.approx(4300.00, rel=0.01)
        assert trends["transaction_count"] == 3


class TestCompositeSignals:
    """Test composite signal generation from multiple data sources."""

    def test_generate_composite_signal(self, engine, sample_satellite_data, sample_sentiment_data, sample_transaction_data):
        engine.ingest_satellite_data(sample_satellite_data)
        engine.ingest_sentiment_data(sample_sentiment_data)
        engine.ingest_transaction_data(sample_transaction_data)

        signal = engine.generate_composite_signal("AAPL")
        assert "composite_score" in signal
        assert "satellite_component" in signal
        assert "sentiment_component" in signal
        assert "transaction_component" in signal
        assert -1 <= signal["composite_score"] <= 1

    def test_composite_signal_missing_data(self, engine):
        signal = engine.generate_composite_signal("AAPL")
        assert signal["composite_score"] == 0.0
        assert signal["data_sources_used"] == 0


class TestDataExport:
    """Test data export functionality."""

    def test_export_to_dataframe(self, engine, sample_satellite_data):
        engine.ingest_satellite_data(sample_satellite_data)
        df = engine.export_to_dataframe("satellite")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "ticker" in df.columns

    def test_export_empty_raises(self, engine):
        with pytest.raises(ValueError):
            engine.export_to_dataframe("nonexistent")

    def test_get_summary(self, engine, sample_satellite_data, sample_sentiment_data):
        engine.ingest_satellite_data(sample_satellite_data)
        engine.ingest_sentiment_data(sample_sentiment_data)
        summary = engine.get_summary()
        assert "satellite" in summary
        assert "sentiment" in summary
        assert summary["satellite"]["count"] == 3
        assert summary["sentiment"]["count"] == 3
