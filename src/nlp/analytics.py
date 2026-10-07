"""NLP Analytics Engine — numpy + stdlib only (no transformers)."""
import re
from typing import Any, Dict, List

import numpy as np


class NLPAnalyticsEngine:
    """Rule-based sentiment, TF-IDF topics, regex NER for financial text."""

    POS_WORDS = {
        "outstanding", "growth", "exceptional", "profitability", "record",
        "excellent", "strong", "beat", "surge", "gain", "profit", "robust",
        "positive", "bullish", "upgrade", "buy", "outperform", "dividend",
        "buyback", "expansion", "synergies", "raised", "increased",
    }
    NEG_WORDS = {
        "declined", "sharply", "worsening", "losses", "loss", "decline",
        "plummet", "deteriorating", "miss", "weak", "concern", "risk",
        "fall", "drop", "negative", "crisis", "bearish", "downgrade",
        "sell", "underperform", "debt", "bankruptcy", "slowdown", "fell",
    }
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "of", "and", "or", "for", "with", "by", "from", "as", "its",
        "it", "this", "that", "after", "into", "across", "will", "be",
    }

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        words = self._tokenize(text)
        if not words:
            return {"label": "NEUTRAL", "score": 0.0}
        pos = sum(1 for w in words if w in self.POS_WORDS)
        neg = sum(1 for w in words if w in self.NEG_WORDS)
        score = (pos - neg) / max(len(words), 1)
        label = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
        return {"label": label, "score": round(score, 4)}

    def extract_topics(self, texts: List[str], n_topics: int = 2) -> Dict[str, Any]:
        if not texts:
            return {"topics": [], "document_topics": []}
        n_topics = min(n_topics, len(texts))
        vocab = sorted({w for t in texts for w in self._tokenize(t) if w not in self.STOP_WORDS})
        if not vocab:
            return {"topics": [{"keywords": []} for _ in range(n_topics)], "document_topics": []}
        vidx = {w: i for i, w in enumerate(vocab)}
        tf = np.zeros((len(texts), len(vocab)))
        for i, t in enumerate(texts):
            for w in self._tokenize(t):
                if w in vidx:
                    tf[i, vidx[w]] += 1
        df = (tf > 0).sum(axis=0)
        idf = np.log((1 + len(texts)) / (1 + df)) + 1
        tfidf = tf * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf = tfidf / norms
        try:
            _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
            topics = []
            for k in range(n_topics):
                top = np.argsort(np.abs(vt[k]))[-10:][::-1]
                topics.append({"keywords": [vocab[i] for i in top if i < len(vocab)]})
        except Exception:
            topics = [{"keywords": vocab[:10]} for _ in range(n_topics)]
        return {"topics": topics, "document_topics": tfidf.tolist()}

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        for m in re.finditer(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b", text):
            entities.append({"text": m.group(), "label": "ORG", "start": m.start(), "end": m.end()})
        for m in re.finditer(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|T))?", text, re.I):
            entities.append({"text": m.group(), "label": "MONEY", "start": m.start(), "end": m.end()})
        for m in re.finditer(r"\b[A-Z]{1,5}\b", text):
            if m.group() not in {"A", "I"}:
                entities.append({"text": m.group(), "label": "TICKER", "start": m.start(), "end": m.end()})
        return entities

    def analyze(self, text: str) -> Dict[str, Any]:
        return {
            "sentiment": self.analyze_sentiment(text),
            "entities": self.extract_entities(text),
            "topics": self.extract_topics([text], n_topics=1),
        }
