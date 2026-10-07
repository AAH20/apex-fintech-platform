"""Alternative Data Engine for investment research.

Aggregates and analyzes alternative data sources including satellite imagery,
social sentiment, and credit card transactions.
"""
import requests
import pandas as pd
from typing import Optional


class AlternativeDataEngine:
    """Engine for aggregating and analyzing alternative data sources."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.altdata.example.com/v1"
        self.data_store = {}

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make authenticated API request."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.get(f"{self.base_url}/{endpoint}", headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
        return response.json()

    # --- Satellite Data ---
    def fetch_satellite_data(self, ticker: str, start_date: str, end_date: str) -> dict:
        """Fetch satellite imagery data for a ticker."""
        return self._make_request("satellite", {
            "ticker": ticker, "start_date": start_date, "end_date": end_date
        })

    def ingest_satellite_data(self, data: list):
        """Ingest satellite data records."""
        self.data_store.setdefault("satellite", []).extend(data)

    def analyze_satellite_trends(self, ticker: str) -> dict:
        """Analyze satellite data trends for a ticker."""
        records = [r for r in self.data_store.get("satellite", []) if r["ticker"] == ticker]
        if not records:
            return {"avg_parking_lot_count": 0.0, "avg_activity_score": 0.0}
        df = pd.DataFrame(records)
        return {
            "avg_parking_lot_count": float(df["parking_lot_count"].mean()),
            "avg_activity_score": float(df["activity_score"].mean()),
        }

    # --- Social Sentiment ---
    def fetch_sentiment_data(self, ticker: str, start_date: str, end_date: str) -> dict:
        """Fetch social sentiment data for a ticker."""
        return self._make_request("sentiment", {
            "ticker": ticker, "start_date": start_date, "end_date": end_date
        })

    def ingest_sentiment_data(self, data: list):
        """Ingest social sentiment records."""
        self.data_store.setdefault("sentiment", []).extend(data)

    def analyze_sentiment_trends(self, ticker: str) -> dict:
        """Analyze sentiment trends for a ticker."""
        records = [r for r in self.data_store.get("sentiment", []) if r["ticker"] == ticker]
        if not records:
            return {"avg_sentiment": 0.0, "total_volume": 0}
        df = pd.DataFrame(records)
        return {
            "avg_sentiment": float(df["sentiment"].mean()),
            "total_volume": int(df["volume"].sum()),
        }

    # --- Credit Card Transactions ---
    def fetch_transaction_data(self, ticker: str, start_date: str, end_date: str) -> dict:
        """Fetch credit card transaction data for a ticker."""
        return self._make_request("transactions", {
            "ticker": ticker, "start_date": start_date, "end_date": end_date
        })

    def ingest_transaction_data(self, data: list):
        """Ingest credit card transaction records."""
        self.data_store.setdefault("transactions", []).extend(data)

    def analyze_transaction_trends(self, ticker: str) -> dict:
        """Analyze transaction trends for a ticker."""
        records = [r for r in self.data_store.get("transactions", []) if r["ticker"] == ticker]
        if not records:
            return {"total_spend": 0.0, "avg_transaction": 0.0, "transaction_count": 0}
        df = pd.DataFrame(records)
        return {
            "total_spend": float(df["amount"].sum()),
            "avg_transaction": float(df["amount"].mean()),
            "transaction_count": int(len(df)),
        }

    # --- Composite Signals ---
    def generate_composite_signal(self, ticker: str) -> dict:
        """Generate composite signal from all available data sources."""
        satellite = self.analyze_satellite_trends(ticker)
        sentiment = self.analyze_sentiment_trends(ticker)
        transactions = self.analyze_transaction_trends(ticker)

        sources_used = 0
        components = {}

        if satellite["avg_activity_score"] > 0:
            components["satellite_component"] = satellite["avg_activity_score"]
            sources_used += 1
        else:
            components["satellite_component"] = 0.0

        if sentiment["avg_sentiment"] > 0:
            components["sentiment_component"] = sentiment["avg_sentiment"]
            sources_used += 1
        else:
            components["sentiment_component"] = 0.0

        if transactions["total_spend"] > 0:
            # Normalize transaction component to 0-1 scale
            components["transaction_component"] = min(transactions["total_spend"] / 10000.0, 1.0)
            sources_used += 1
        else:
            components["transaction_component"] = 0.0

        if sources_used == 0:
            composite = 0.0
        else:
            composite = sum(components.values()) / sources_used
            # Scale to -1 to 1 range
            composite = (composite * 2) - 1

        return {
            "composite_score": composite,
            "data_sources_used": sources_used,
            **components,
        }

    # --- Export & Summary ---
    def export_to_dataframe(self, source: str) -> pd.DataFrame:
        """Export ingested data as a pandas DataFrame."""
        if source not in self.data_store:
            raise ValueError(f"No data found for source: {source}")
        return pd.DataFrame(self.data_store[source])

    def get_summary(self) -> dict:
        """Get summary of all ingested data."""
        summary = {}
        for source, records in self.data_store.items():
            summary[source] = {"count": len(records)}
        return summary
