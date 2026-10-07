"""Tests for the Behavioral Finance Analytics Engine."""
import numpy as np
import pytest

from src.behavioral.analytics import BehavioralFinanceEngine


@pytest.fixture
def engine():
    """Create a fresh engine instance for each test."""
    return BehavioralFinanceEngine()


# ─── Loss Aversion ───────────────────────────────────────────────────────────

class TestLossAversion:
    def test_loss_aversion_detected(self, engine):
        """Investor holds losers too long and sells winners too quickly."""
        trades = [
            {"symbol": "AAPL", "action": "SELL", "pnl": 500, "holding_days": 3},
            {"symbol": "GOOG", "action": "SELL", "pnl": 800, "holding_days": 5},
            {"symbol": "TSLA", "action": "HOLD", "pnl": -2000, "holding_days": 120},
            {"symbol": "META", "action": "HOLD", "pnl": -1500, "holding_days": 90},
        ]
        result = engine.detect_loss_aversion(trades)
        assert result["detected"] is True
        assert result["score"] > 0.5

    def test_no_loss_aversion(self, engine):
        """Balanced selling behavior — no loss aversion."""
        trades = [
            {"symbol": "AAPL", "action": "SELL", "pnl": 500, "holding_days": 30},
            {"symbol": "GOOG", "action": "SELL", "pnl": -300, "holding_days": 45},
            {"symbol": "TSLA", "action": "SELL", "pnl": 800, "holding_days": 60},
            {"symbol": "META", "action": "SELL", "pnl": -200, "holding_days": 25},
        ]
        result = engine.detect_loss_aversion(trades)
        assert result["detected"] is False

    def test_loss_aversion_empty_trades(self, engine):
        """Empty trade list should not raise."""
        result = engine.detect_loss_aversion([])
        assert result["detected"] is False
        assert result["score"] == 0.0


# ─── Overconfidence ─────────────────────────────────────────────────────────

class TestOverconfidence:
    def test_overconfidence_high_turnover(self, engine):
        """Excessive trading frequency indicates overconfidence."""
        trades = [{"symbol": f"STK{i}", "action": "BUY"} for i in range(50)]
        result = engine.detect_overconfidence(trades, portfolio_value=100_000)
        assert result["detected"] is True
        assert result["score"] > 0.5

    def test_overconfidence_low_turnover(self, engine):
        """Moderate trading — no overconfidence."""
        trades = [{"symbol": f"STK{i}", "action": "BUY"} for i in range(5)]
        result = engine.detect_overconfidence(trades, portfolio_value=100_000)
        assert result["detected"] is False

    def test_overconfidence_concentrated_positions(self, engine):
        """High concentration in few positions signals overconfidence."""
        holdings = [
            {"symbol": "AAPL", "weight": 0.60},
            {"symbol": "GOOG", "weight": 0.30},
            {"symbol": "TSLA", "weight": 0.10},
        ]
        result = engine.detect_overconfidence([], portfolio_value=100_000, holdings=holdings)
        assert result["detected"] is True


# ─── Anchoring Bias ──────────────────────────────────────────────────────────

class TestAnchoringBias:
    def test_anchoring_on_entry_price(self, engine):
        """Investor fixates on entry price as reference point."""
        decisions = [
            {"symbol": "AAPL", "entry_price": 150, "current_price": 180, "action": "HOLD", "rationale": "waiting to break even"},
            {"symbol": "GOOG", "entry_price": 2800, "current_price": 3200, "action": "HOLD", "rationale": "waiting to break even"},
        ]
        result = engine.detect_anchoring(decisions)
        assert result["detected"] is True
        assert result["score"] > 0.5

    def test_no_anchoring(self, engine):
        """Decisions based on fundamentals, not entry price."""
        decisions = [
            {"symbol": "AAPL", "entry_price": 150, "current_price": 180, "action": "SELL", "rationale": "overvalued on DCF"},
            {"symbol": "GOOG", "entry_price": 2800, "current_price": 3200, "action": "SELL", "rationale": "sector rotation"},
        ]
        result = engine.detect_anchoring(decisions)
        assert result["detected"] is False


# ─── Herding Bias ────────────────────────────────────────────────────────────

class TestHerdingBias:
    def test_herding_detected(self, engine):
        """Investor follows crowd — buys what others buy."""
        social_signals = [
            {"symbol": "AAPL", "social_volume": 5000, "sentiment": 0.9},
            {"symbol": "AAPL", "social_volume": 8000, "sentiment": 0.85},
            {"symbol": "AAPL", "social_volume": 12000, "sentiment": 0.95},
        ]
        own_trades = [
            {"symbol": "AAPL", "action": "BUY"},
            {"symbol": "AAPL", "action": "BUY"},
        ]
        result = engine.detect_herding(own_trades, social_signals)
        assert result["detected"] is True

    def test_no_herding(self, engine):
        """Independent decisions — low social correlation."""
        social_signals = [
            {"symbol": "AAPL", "social_volume": 100, "sentiment": 0.2},
        ]
        own_trades = [
            {"symbol": "AAPL", "action": "SELL"},
        ]
        result = engine.detect_herding(own_trades, social_signals)
        assert result["detected"] is False


# ─── Disposition Effect ─────────────────────────────────────────────────────

class TestDispositionEffect:
    def test_disposition_effect_detected(self, engine):
        """Selling winners too early, holding losers too long."""
        trades = [
            {"symbol": "AAPL", "action": "SELL", "pnl_pct": 0.05, "holding_days": 5},
            {"symbol": "GOOG", "action": "SELL", "pnl_pct": 0.08, "holding_days": 7},
            {"symbol": "TSLA", "action": "HOLD", "pnl_pct": -0.15, "holding_days": 200},
            {"symbol": "META", "action": "HOLD", "pnl_pct": -0.20, "holding_days": 180},
        ]
        result = engine.detect_disposition_effect(trades)
        assert result["detected"] is True
        assert result["score"] > 0.5

    def test_no_disposition_effect(self, engine):
        """No systematic difference in holding periods."""
        trades = [
            {"symbol": "AAPL", "action": "SELL", "pnl_pct": 0.05, "holding_days": 60},
            {"symbol": "GOOG", "action": "SELL", "pnl_pct": -0.10, "holding_days": 45},
        ]
        result = engine.detect_disposition_effect(trades)
        assert result["detected"] is False


# ─── Mental Accounting ──────────────────────────────────────────────────────

class TestMentalAccounting:
    def test_mental_accounting_detected(self, engine):
        """Treating money differently based on source/label."""
        accounts = [
            {"name": "retirement", "balance": 500_000, "risk_profile": "conservative"},
            {"name": "gambling", "balance": 50_000, "risk_profile": "aggressive"},
            {"name": "house_downpayment", "balance": 200_000, "risk_profile": "conservative"},
        ]
        result = engine.detect_mental_accounting(accounts)
        assert result["detected"] is True
        assert result["score"] > 0.3

    def test_no_mental_accounting(self, engine):
        """Uniform risk treatment across accounts."""
        accounts = [
            {"name": "retirement", "balance": 500_000, "risk_profile": "moderate"},
            {"name": "trading", "balance": 50_000, "risk_profile": "moderate"},
        ]
        result = engine.detect_mental_accounting(accounts)
        assert result["detected"] is False


# ─── Recency Bias ───────────────────────────────────────────────────────────

class TestRecencyBias:
    def test_recency_bias_detected(self, engine):
        """Overweighting recent returns in decision-making."""
        historical_returns = np.array([0.02, 0.01, -0.01, 0.03, 0.015, -0.005, 0.025])
        recent_returns = np.array([0.08, 0.12, 0.15])
        result = engine.detect_recency_bias(historical_returns, recent_returns)
        assert result["detected"] is True
        assert result["score"] > 0.5

    def test_no_recency_bias(self, engine):
        """Recent returns consistent with historical."""
        historical_returns = np.array([0.02, 0.01, -0.01, 0.03, 0.015, -0.005, 0.025])
        recent_returns = np.array([0.02, 0.015, 0.025])
        result = engine.detect_recency_bias(historical_returns, recent_returns)
        assert result["detected"] is False


# ─── Sentiment Analysis ─────────────────────────────────────────────────────

class TestSentimentAnalysis:
    def test_positive_sentiment(self, engine):
        """Bullish news sentiment."""
        news_items = [
            {"headline": "Company beats earnings expectations", "source": "Reuters"},
            {"headline": "Stock surges on strong guidance", "source": "Bloomberg"},
        ]
        result = engine.analyze_sentiment(news_items)
        assert result["sentiment"] > 0.3
        assert result["classification"] == "positive"

    def test_negative_sentiment(self, engine):
        """Bearish news sentiment."""
        news_items = [
            {"headline": "Company misses earnings badly", "source": "Reuters"},
            {"headline": "Stock plunges on weak outlook", "source": "Bloomberg"},
        ]
        result = engine.analyze_sentiment(news_items)
        assert result["sentiment"] < -0.3
        assert result["classification"] == "negative"

    def test_neutral_sentiment(self, engine):
        """Mixed/neutral news sentiment."""
        news_items = [
            {"headline": "Company reports earnings in line", "source": "Reuters"},
        ]
        result = engine.analyze_sentiment(news_items)
        assert abs(result["sentiment"]) <= 0.3
        assert result["classification"] == "neutral"

    def test_sentiment_empty_input(self, engine):
        """Empty news list should return neutral."""
        result = engine.analyze_sentiment([])
        assert result["sentiment"] == 0.0
        assert result["classification"] == "neutral"


# ─── Prospect Theory ────────────────────────────────────────────────────────

class TestProspectTheory:
    def test_value_function_gain(self, engine):
        """Concave value function for gains (risk aversion)."""
        result = engine.prospect_theory_value(100, alpha=0.88)
        assert result > 0
        assert result < 100  # Diminishing sensitivity

    def test_value_function_loss(self, engine):
        """Convex value function for losses (risk-seeking)."""
        result = engine.prospect_theory_value(-100, alpha=0.88, beta=0.88, lambda_=2.25)
        assert result < 0
        assert abs(result) > 100  # Losses loom larger

    def test_loss_aversion_ratio(self, engine):
        """Losses loom larger than gains (lambda > 1)."""
        gain_value = engine.prospect_theory_value(100)
        loss_value = engine.prospect_theory_value(-100)
        assert abs(loss_value) > gain_value

    def test_probability_weighting_overweight_small(self, engine):
        """Small probabilities are overweighted."""
        result = engine.probability_weighting(0.01, gamma=0.61)
        assert result > 0.01

    def test_probability_weighting_underweight_large(self, engine):
        """Large probabilities are underweighted."""
        result = engine.probability_weighting(0.99, gamma=0.61)
        assert result < 0.99

    def test_prospect_theory_evaluate(self, engine):
        """Full prospect theory evaluation of a gamble."""
        outcomes = np.array([-100, 50, 200])
        probabilities = np.array([0.3, 0.5, 0.2])
        result = engine.evaluate_gamble(outcomes, probabilities)
        assert "value" in result
        assert "certainty_equivalent" in result
        assert isinstance(result["value"], float)


# ─── Comprehensive Bias Report ──────────────────────────────────────────────

class TestComprehensiveReport:
    def test_full_bias_report(self, engine):
        """Engine produces a comprehensive bias report."""
        trades = [
            {"symbol": "AAPL", "action": "SELL", "pnl": 500, "holding_days": 3},
            {"symbol": "TSLA", "action": "HOLD", "pnl": -2000, "holding_days": 120},
        ]
        report = engine.generate_bias_report(trades=trades)
        assert "biases" in report
        assert "overall_risk_score" in report
        assert isinstance(report["overall_risk_score"], float)
        assert 0 <= report["overall_risk_score"] <= 1

    def test_report_includes_all_bias_types(self, engine):
        """Report covers all major behavioral biases."""
        report = engine.generate_bias_report(trades=[])
        expected_biases = {
            "loss_aversion",
            "overconfidence",
            "anchoring",
            "herding",
            "disposition_effect",
            "mental_accounting",
            "recency_bias",
        }
        assert expected_biases.issubset(set(report["biases"].keys()))
