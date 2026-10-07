"""Behavioral Finance Analytics Engine.

Detects behavioral biases in investment decisions using principles from
behavioral finance and investor psychology (Kahneman & Tversky, Thaler, etc.).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class BehavioralFinanceEngine:
    """Engine for detecting behavioral biases in investment decisions."""

    # ── Loss Aversion ────────────────────────────────────────────────────

    def detect_loss_aversion(
        self,
        trades: list[dict[str, Any]],
        winner_threshold: float = 0.0,
        holding_ratio_threshold: float = 2.0,
    ) -> dict[str, Any]:
        """Detect loss aversion: holding losers longer than winners.

        Loss aversion (Kahneman & Tversky, 1979): losses loom larger than gains.
        Investors hold losing positions too long and sell winning positions too quickly.
        """
        if not trades:
            return {"detected": False, "score": 0.0, "details": "No trades provided"}

        winners = [t for t in trades if t.get("pnl", 0) > winner_threshold]
        losers = [t for t in trades if t.get("pnl", 0) < winner_threshold]

        if not winners or not losers:
            return {"detected": False, "score": 0.0, "details": "Need both winners and losers"}

        avg_winner_holding = np.mean([t.get("holding_days", 0) for t in winners])
        avg_loser_holding = np.mean([t.get("holding_days", 0) for t in losers])

        if avg_winner_holding == 0:
            return {"detected": False, "score": 0.0, "details": "No winner holding data"}

        holding_ratio = avg_loser_holding / avg_winner_holding
        score = min(1.0, max(0.0, (holding_ratio - 1.0) / (holding_ratio_threshold - 1.0)))

        return {
            "detected": bool(holding_ratio > holding_ratio_threshold),
            "score": float(score),
            "details": {
                "avg_winner_holding_days": float(avg_winner_holding),
                "avg_loser_holding_days": float(avg_loser_holding),
                "holding_ratio": float(holding_ratio),
            },
        }

    # ── Overconfidence ────────────────────────────────────────────────────

    def detect_overconfidence(
        self,
        trades: list[dict[str, Any]],
        portfolio_value: float = 100_000,
        holdings: list[dict[str, Any]] | None = None,
        turnover_threshold: float = 2.0,
        concentration_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Detect overconfidence: excessive trading or concentrated positions.

        Overconfidence bias: investors overestimate their knowledge and ability,
        leading to excessive trading and under-diversification.
        """
        scores = []

        # Turnover-based detection
        if trades and portfolio_value > 0:
            # Annualized turnover proxy: number of trades relative to portfolio
            turnover_rate = len(trades) / (portfolio_value / 10_000)
            if turnover_rate > turnover_threshold:
                scores.append(min(1.0, turnover_rate / (turnover_threshold * 2)))

        # Concentration-based detection
        if holdings:
            weights = [h.get("weight", 0) for h in holdings]
            if weights:
                max_weight = max(weights)
                if max_weight > concentration_threshold:
                    scores.append(min(1.0, max_weight))

        if not scores:
            return {"detected": False, "score": 0.0, "details": "No overconfidence signals"}

        score = max(scores)
        return {
            "detected": bool(score > 0.5),
            "score": float(score),
            "details": {
                "turnover_rate": float(len(trades) / (portfolio_value / 10_000)) if portfolio_value > 0 else 0.0,
                "max_position_weight": float(max(h.get("weight", 0) for h in holdings)) if holdings else 0.0,
            },
        }

    # ── Anchoring Bias ────────────────────────────────────────────────────

    def detect_anchoring(
        self,
        decisions: list[dict[str, Any]],
        score_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Detect anchoring: fixating on a reference point (e.g., entry price).

        Anchoring bias: investors rely too heavily on an initial piece of information
        (anchor) when making decisions.
        """
        if not decisions:
            return {"detected": False, "score": 0.0, "details": "No decisions provided"}

        anchor_keywords = [
            "break even",
            "break-even",
            "entry price",
            "bought at",
            "original price",
            "waiting to recover",
            "get back to",
            "cost basis",
        ]

        anchored_count = 0
        for d in decisions:
            rationale = d.get("rationale", "").lower()
            if any(kw in rationale for kw in anchor_keywords):
                anchored_count += 1

        score = anchored_count / len(decisions)
        return {
            "detected": bool(score > score_threshold),
            "score": float(score),
            "details": {
                "anchored_decisions": anchored_count,
                "total_decisions": len(decisions),
            },
        }

    # ── Herding Bias ──────────────────────────────────────────────────────

    def detect_herding(
        self,
        own_trades: list[dict[str, Any]],
        social_signals: list[dict[str, Any]],
        volume_threshold: float = 1000,
        sentiment_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Detect herding: following the crowd rather than independent analysis.

        Herding bias: investors mimic the actions of a larger group,
        often ignoring their own analysis.
        """
        if not own_trades or not social_signals:
            return {"detected": False, "score": 0.0, "details": "Insufficient data"}

        # Check if social signals are strongly positive and own trades follow
        avg_social_volume = np.mean([s.get("social_volume", 0) for s in social_signals])
        avg_sentiment = np.mean([s.get("sentiment", 0) for s in social_signals])

        buy_count = sum(1 for t in own_trades if t.get("action", "").upper() == "BUY")
        buy_ratio = buy_count / len(own_trades)

        # Herding: high social volume + positive sentiment + buying along
        if avg_social_volume > volume_threshold and avg_sentiment > sentiment_threshold:
            if buy_ratio > 0.5:
                return {
                    "detected": True,
                    "score": float(min(1.0, (avg_sentiment + buy_ratio) / 2)),
                    "details": {
                        "avg_social_volume": float(avg_social_volume),
                        "avg_sentiment": float(avg_sentiment),
                        "buy_ratio": float(buy_ratio),
                    },
                }

        return {"detected": False, "score": 0.0, "details": "No herding signals detected"}

    # ── Disposition Effect ────────────────────────────────────────────────

    def detect_disposition_effect(
        self,
        trades: list[dict[str, Any]],
        holding_ratio_threshold: float = 1.5,
    ) -> dict[str, Any]:
        """Detect disposition effect: selling winners too early, holding losers too long.

        Disposition effect (Shefrin & Statman, 1985): investors sell assets that
        have increased in value and keep assets that have decreased in value.
        """
        if not trades:
            return {"detected": False, "score": 0.0, "details": "No trades provided"}

        winners = [t for t in trades if t.get("pnl_pct", 0) > 0]
        losers = [t for t in trades if t.get("pnl_pct", 0) < 0]

        if not winners or not losers:
            return {"detected": False, "score": 0.0, "details": "Need both winners and losers"}

        avg_winner_holding = np.mean([t.get("holding_days", 0) for t in winners])
        avg_loser_holding = np.mean([t.get("holding_days", 0) for t in losers])

        if avg_winner_holding == 0:
            return {"detected": False, "score": 0.0, "details": "No winner holding data"}

        holding_ratio = avg_loser_holding / avg_winner_holding
        score = min(1.0, max(0.0, (holding_ratio - 1.0) / (holding_ratio_threshold - 1.0)))

        return {
            "detected": bool(holding_ratio > holding_ratio_threshold),
            "score": float(score),
            "details": {
                "avg_winner_holding_days": float(avg_winner_holding),
                "avg_loser_holding_days": float(avg_loser_holding),
                "holding_ratio": float(holding_ratio),
            },
        }

    # ── Mental Accounting ─────────────────────────────────────────────────

    def detect_mental_accounting(
        self,
        accounts: list[dict[str, Any]],
        score_threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Detect mental accounting: treating money differently based on source/label.

        Mental accounting (Thaler, 1985): investors mentally categorize money
        into separate accounts, leading to irrational decision-making.
        """
        if not accounts or len(accounts) < 2:
            return {"detected": False, "score": 0.0, "details": "Need at least 2 accounts"}

        risk_profiles = [a.get("risk_profile", "").lower() for a in accounts]
        unique_profiles = set(risk_profiles)

        # Different risk profiles for different accounts = mental accounting
        if len(unique_profiles) > 1:
            # Score based on number of distinct risk profiles
            score = min(1.0, (len(unique_profiles) - 1) / 2.0)
            return {
                "detected": bool(score > score_threshold),
                "score": float(score),
                "details": {
                    "unique_risk_profiles": list(unique_profiles),
                    "num_accounts": len(accounts),
                },
            }

        return {"detected": False, "score": 0.0, "details": "Uniform risk treatment"}

    # ── Recency Bias ──────────────────────────────────────────────────────

    def detect_recency_bias(
        self,
        historical_returns: np.ndarray,
        recent_returns: np.ndarray,
        threshold_std: float = 2.0,
    ) -> dict[str, Any]:
        """Detect recency bias: overweighting recent information.

        Recency bias: investors give more weight to recent events
        than to historical patterns.
        """
        if len(historical_returns) == 0 or len(recent_returns) == 0:
            return {"detected": False, "score": 0.0, "details": "Insufficient data"}

        hist_mean = np.mean(historical_returns)
        hist_std = np.std(historical_returns)
        recent_mean = np.mean(recent_returns)

        if hist_std == 0:
            return {"detected": False, "score": 0.0, "details": "No historical variance"}

        # Z-score of recent returns relative to historical
        z_score = (recent_mean - hist_mean) / hist_std
        score = min(1.0, abs(z_score) / (threshold_std * 2))

        return {
            "detected": bool(abs(z_score) > threshold_std),
            "score": float(score),
            "details": {
                "historical_mean": float(hist_mean),
                "historical_std": float(hist_std),
                "recent_mean": float(recent_mean),
                "z_score": float(z_score),
            },
        }

    # ── Sentiment Analysis ────────────────────────────────────────────────

    def analyze_sentiment(
        self,
        news_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze sentiment of news headlines using keyword-based approach.

        Returns a sentiment score between -1 (very negative) and +1 (very positive).
        """
        if not news_items:
            return {"sentiment": 0.0, "classification": "neutral", "details": "No news items"}

        positive_keywords = [
            "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
            "strong", "growth", "profit", "profits", "gain", "gains", "bullish",
            "outperform", "upgrade", "upgraded", "record", "breakthrough", "positive",
            "exceed", "exceeds", "miss", "misses",  # miss is negative, handled below
        ]
        # Remove "miss" from positive — it's negative
        positive_keywords = [kw for kw in positive_keywords if kw not in ("miss", "misses")]

        negative_keywords = [
            "miss", "misses", "plunge", "plunges", "crash", "crashes", "bearish",
            "underperform", "downgrade", "downgraded", "loss", "losses", "weak",
            "decline", "declines", "fall", "falls", "drop", "drops", "negative",
            "investigation", "lawsuit", "fraud", "bankruptcy", "layoff", "layoffs",
            "warning", "warns", "concern", "concerns", "risk", "risks",
        ]

        sentiments = []
        for item in news_items:
            headline = item.get("headline", "").lower()
            pos_count = sum(1 for kw in positive_keywords if kw in headline)
            neg_count = sum(1 for kw in negative_keywords if kw in headline)

            if pos_count + neg_count == 0:
                sentiments.append(0.0)
            else:
                sentiments.append((pos_count - neg_count) / (pos_count + neg_count))

        avg_sentiment = float(np.mean(sentiments))

        if avg_sentiment > 0.3:
            classification = "positive"
        elif avg_sentiment < -0.3:
            classification = "negative"
        else:
            classification = "neutral"

        return {
            "sentiment": avg_sentiment,
            "classification": classification,
            "details": {
                "num_items": len(news_items),
                "individual_sentiments": sentiments,
            },
        }

    # ── Prospect Theory ───────────────────────────────────────────────────

    def prospect_theory_value(
        self,
        outcome: float,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
    ) -> float:
        """Prospect theory value function (Kahneman & Tversky, 1979).

        v(x) = x^alpha           if x >= 0 (gains, risk-averse)
        v(x) = -lambda * |x|^beta  if x < 0  (losses, risk-seeking)
        """
        if outcome >= 0:
            return float(outcome ** alpha)
        else:
            return float(-lambda_ * (abs(outcome) ** beta))

    def probability_weighting(
        self,
        probability: float,
        gamma: float = 0.61,
    ) -> float:
        """Probability weighting function (Kahneman & Tversky, 1992).

        w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)

        Overweights small probabilities, underweights large probabilities.
        """
        if probability <= 0:
            return 0.0
        if probability >= 1:
            return 1.0

        p = probability
        numerator = p ** gamma
        denominator = (p ** gamma + (1 - p) ** gamma) ** (1.0 / gamma)
        return float(numerator / denominator)

    def evaluate_gamble(
        self,
        outcomes: np.ndarray,
        probabilities: np.ndarray,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
        gamma: float = 0.61,
    ) -> dict[str, Any]:
        """Evaluate a gamble using prospect Theory.

        Computes the prospect theory value and certainty equivalent.
        """
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have same length")

        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1")

        # Compute weighted value
        total_value = 0.0
        for outcome, prob in zip(outcomes, probabilities):
            weighted_prob = self.probability_weighting(prob, gamma)
            value = self.prospect_theory_value(outcome, alpha, beta, lambda_)
            total_value += weighted_prob * value

        # Certainty equivalent: find outcome x such that v(x) = total_value
        if total_value >= 0:
            certainty_equivalent = total_value ** (1.0 / alpha)
        else:
            certainty_equivalent = -((-total_value) / lambda_) ** (1.0 / beta)

        return {
            "value": float(total_value),
            "certainty_equivalent": float(certainty_equivalent),
            "expected_value": float(np.sum(outcomes * probabilities)),
        }

    # ── Comprehensive Bias Report ─────────────────────────────────────────

    def generate_bias_report(
        self,
        trades: list[dict[str, Any]] | None = None,
        decisions: list[dict[str, Any]] | None = None,
        social_signals: list[dict[str, Any]] | None = None,
        accounts: list[dict[str, Any]] | None = None,
        historical_returns: np.ndarray | None = None,
        recent_returns: np.ndarray | None = None,
        news_items: list[dict[str, Any]] | None = None,
        portfolio_value: float = 100_000,
        holdings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive behavioral bias report.

        Analyzes all major behavioral biases and returns an overall risk score.
        """
        trades = trades or []
        decisions = decisions or []
        social_signals = social_signals or []
        accounts = accounts or []

        biases: dict[str, Any] = {}

        # Loss aversion — always include
        biases["loss_aversion"] = self.detect_loss_aversion(trades) if trades else {
            "detected": False, "score": 0.0, "details": "No trades provided"
        }

        # Overconfidence — always include
        biases["overconfidence"] = self.detect_overconfidence(
            trades, portfolio_value=portfolio_value, holdings=holdings
        )

        # Anchoring — always include
        biases["anchoring"] = self.detect_anchoring(decisions) if decisions else {
            "detected": False, "score": 0.0, "details": "No decisions provided"
        }

        # Herding — always include
        biases["herding"] = self.detect_herding(trades, social_signals) if social_signals else {
            "detected": False, "score": 0.0, "details": "No social signals provided"
        }

        # Disposition effect — always include
        biases["disposition_effect"] = self.detect_disposition_effect(trades) if trades else {
            "detected": False, "score": 0.0, "details": "No trades provided"
        }

        # Mental accounting — always include
        biases["mental_accounting"] = self.detect_mental_accounting(accounts) if accounts else {
            "detected": False, "score": 0.0, "details": "No accounts provided"
        }

        # Recency bias — always include
        if historical_returns is not None and recent_returns is not None:
            biases["recency_bias"] = self.detect_recency_bias(historical_returns, recent_returns)
        else:
            biases["recency_bias"] = {
                "detected": False, "score": 0.0, "details": "No return data provided"
            }

        # Sentiment — always include
        biases["sentiment"] = self.analyze_sentiment(news_items) if news_items else {
            "sentiment": 0.0, "classification": "neutral", "details": "No news items"
        }

        # Overall risk score: average of all bias scores
        scores = [b["score"] for b in biases.values() if "score" in b]
        overall_risk_score = float(np.mean(scores)) if scores else 0.0

        return {
            "biases": biases,
            "overall_risk_score": overall_risk_score,
            "num_biases_detected": sum(1 for b in biases.values() if b.get("detected", False)),
        }
