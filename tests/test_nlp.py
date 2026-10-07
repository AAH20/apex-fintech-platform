"""Tests for the NLP Analytics Engine (numpy + stdlib only)."""
import pytest

from src.nlp.analytics import NLPAnalyticsEngine


@pytest.fixture
def engine():
    return NLPAnalyticsEngine()


class TestSentimentAnalysis:
    def test_positive_sentiment(self, engine):
        result = engine.analyze_sentiment(
            "Outstanding growth and exceptional profitability drove record revenue."
        )
        assert result["label"] == "POSITIVE"
        assert result["score"] > 0

    def test_negative_sentiment(self, engine):
        result = engine.analyze_sentiment(
            "Revenue declined sharply amid worsening market conditions and massive losses."
        )
        assert result["label"] == "NEGATIVE"
        assert result["score"] < 0

    def test_neutral_sentiment(self, engine):
        result = engine.analyze_sentiment(
            "The company announced its quarterly results on Tuesday."
        )
        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.0


class TestTopicExtraction:
    def test_topic_extraction(self, engine):
        texts = [
            "Apple reported record quarterly earnings of $89.5 billion.",
            "The Federal Reserve raised interest rates by 25 basis points.",
            "Tesla stock surged 12% after announcing expansion.",
            "Goldman Sachs downgraded Microsoft citing slowdown.",
        ]
        result = engine.extract_topics(texts, n_topics=2)
        assert "topics" in result
        assert len(result["topics"]) == 2
        for topic in result["topics"]:
            assert "keywords" in topic
            assert len(topic["keywords"]) > 0


class TestEntityExtraction:
    def test_entity_extraction(self, engine):
        result = engine.extract_entities(
            "Apple Inc. reported earnings of $89.5 billion in New York."
        )
        assert isinstance(result, list)
        assert len(result) > 0
        for entity in result:
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity


class TestEdgeCases:
    def test_empty_text(self, engine):
        result = engine.analyze_sentiment("")
        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.0

    def test_single_word(self, engine):
        result = engine.analyze_sentiment("growth")
        assert result["label"] == "POSITIVE"

    def test_numbers_only(self, engine):
        result = engine.analyze_sentiment("123 456 789")
        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.0

    def test_empty_text_entities(self, engine):
        result = engine.extract_entities("")
        assert isinstance(result, list)
        assert len(result) == 0


class TestCombinedAnalysis:
    def test_combined_analysis(self, engine):
        result = engine.analyze("Apple Inc. reported exceptional quarterly earnings.")
        assert "sentiment" in result
        assert "entities" in result
        assert "topics" in result
        assert result["sentiment"]["label"] == "POSITIVE"
