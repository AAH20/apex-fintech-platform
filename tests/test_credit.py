"""Tests for AI credit underwriting engine.

TDD: These tests define the expected behavior of CreditUnderwritingEngine.
References:
- CFA Institute: AI in Credit Underwriting
- SR 11-7: Model Risk Management
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from credit.underwriting import CreditUnderwritingEngine


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic credit data with clear signal for 95%+ accuracy."""
    np.random.seed(42)
    n = 1000

    credit_score = np.random.normal(680, 80, n).clip(300, 850)
    debt_to_income = np.random.beta(2, 5, n) * 0.6
    credit_utilization = np.random.beta(2, 3, n)
    num_late_payments = np.random.poisson(1, n)
    annual_income = np.random.lognormal(10.8, 0.5, n)
    employment_years = np.random.exponential(5, n).clip(0, 40)
    loan_amount = np.random.lognormal(10, 0.6, n)
    num_credit_lines = np.random.poisson(8, n)
    utility_payment_rate = np.random.beta(8, 2, n)
    rent_payment_rate = np.random.beta(7, 2, n)
    mobile_months = np.random.randint(24, 120, n)
    education_years = np.random.randint(12, 20, n)

    logit = (
        0.02 * (credit_score - 680)
        - 8.0 * debt_to_income
        - 3.0 * credit_utilization
        - 0.5 * num_late_payments
        + 0.01 * employment_years
        + 2.0 * utility_payment_rate
        + 1.5 * rent_payment_rate
        - 0.00001 * (annual_income - 50000) / 1000
        + np.random.normal(0, 0.5, n)
    )
    default_prob = 1 / (1 + np.exp(-logit))
    y = (default_prob > 0.5).astype(int)

    X = pd.DataFrame({
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "credit_utilization": credit_utilization,
        "num_late_payments": num_late_payments,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "num_credit_lines": num_credit_lines,
        "utility_payment_rate": utility_payment_rate,
        "rent_payment_rate": rent_payment_rate,
        "mobile_months": mobile_months,
        "education_years": education_years,
    })
    return X, pd.Series(y, name="default")


@pytest.fixture(scope="module")
def trained_engine(synthetic_data):
    """Engine trained on synthetic data."""
    X, y = synthetic_data
    engine = CreditUnderwritingEngine(random_state=42)
    engine.train(X, y)
    return engine


# ── Test 1: Training ─────────────────────────────────────────────────────


def test_train_fits_model(synthetic_data):
    X, y = synthetic_data
    engine = CreditUnderwritingEngine(random_state=42)
    engine.train(X, y)
    assert engine.is_fitted is True
    assert engine.model is not None


# ── Test 2: Prediction ────────────────────────────────────────────────────


def test_predict_returns_binary(trained_engine, synthetic_data):
    X, _ = synthetic_data
    preds = trained_engine.predict(X)
    assert set(preds.unique()).issubset({0, 1})


# ── Test 3: Accuracy Threshold ────────────────────────────────────────────


def test_accuracy_above_95_percent(trained_engine, synthetic_data):
    X, y = synthetic_data
    preds = trained_engine.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy >= 0.95, f"Accuracy {accuracy:.3f} < 0.95"


# ── Test 4: Feature Importance ────────────────────────────────────────────


def test_get_feature_importance(trained_engine):
    importance = trained_engine.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    values = list(importance.values())
    assert values == sorted(values, reverse=True)


# ── Test 5: Explainability ────────────────────────────────────────────────


def test_explain_returns_contributions(trained_engine, synthetic_data):
    X, _ = synthetic_data
    explanation = trained_engine.explain(X.head(10))
    assert explanation is not None
    assert "contributions" in explanation


# ── Test 6: Edge Case — Single Feature ────────────────────────────────────


def test_single_feature_edge_case(synthetic_data):
    X, y = synthetic_data
    X_single = X[["credit_score"]]
    engine = CreditUnderwritingEngine(random_state=42)
    engine.train(X_single, y)
    preds = engine.predict(X_single)
    assert len(preds) == len(X_single)


# ── Test 7: Edge Case — All Zeros ─────────────────────────────────────────


def test_all_zeros_edge_case(synthetic_data):
    X, y = synthetic_data
    X_zeros = pd.DataFrame(np.zeros((100, X.shape[1])), columns=X.columns)
    engine = CreditUnderwritingEngine(random_state=42)
    engine.train(X, y)
    preds = engine.predict(X_zeros)
    assert len(preds) == 100


# ── Test 8: Edge Case — All Ones ──────────────────────────────────────────


def test_all_ones_edge_case(synthetic_data):
    X, y = synthetic_data
    X_ones = pd.DataFrame(np.ones((100, X.shape[1])), columns=X.columns)
    engine = CreditUnderwritingEngine(random_state=42)
    engine.train(X, y)
    preds = engine.predict(X_ones)
    assert len(preds) == 100


# ── Test 9: Error Handling ────────────────────────────────────────────────


def test_predict_before_train_raises(synthetic_data):
    X, _ = synthetic_data
    engine = CreditUnderwritingEngine()
    with pytest.raises(RuntimeError, match="not fitted"):
        engine.predict(X)


# ── Test 10: Probability Validity ────────────────────────────────────────


def test_predict_proba_sums_to_one(trained_engine, synthetic_data):
    X, _ = synthetic_data
    proba = trained_engine.predict_proba(X)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
