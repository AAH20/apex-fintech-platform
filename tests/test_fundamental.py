"""Tests for fundamental analysis engine.

Covers ratio analysis, cash flow analysis, earnings quality, and forensic
anomaly detection per CFA Institute financial statement analysis standards.
"""
import numpy as np
import pandas as pd
import pytest

from fundamental.analysis import Anomaly, FundamentalAnalysisEngine


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _healthy_statements():
    """A financially healthy company with steady 10% growth (3 years)."""
    periods = ["2022", "2023", "2024"]
    income = pd.DataFrame(
        {
            "revenue": [1000.0, 1100.0, 1210.0],
            "cogs": [600.0, 660.0, 726.0],
            "gross_profit": [400.0, 440.0, 484.0],
            "operating_expenses": [200.0, 220.0, 242.0],
            "operating_income": [200.0, 220.0, 242.0],
            "interest_expense": [10.0, 11.0, 12.0],
            "net_income": [150.0, 165.0, 181.5],
            "ebitda": [250.0, 275.0, 302.5],
        },
        index=periods,
    ).T
    balance = pd.DataFrame(
        {
            "total_assets": [2000.0, 2200.0, 2420.0],
            "current_assets": [800.0, 880.0, 968.0],
            "current_liabilities": [400.0, 440.0, 484.0],
            "total_liabilities": [1000.0, 1100.0, 1210.0],
            "total_equity": [1000.0, 1100.0, 1210.0],
            "cash": [200.0, 220.0, 242.0],
            "inventory": [300.0, 330.0, 363.0],
            "receivables": [200.0, 220.0, 242.0],
            "total_debt": [600.0, 660.0, 726.0],
        },
        index=periods,
    ).T
    cashflow = pd.DataFrame(
        {
            "operating_cash_flow": [180.0, 198.0, 217.8],
            "investing_cash_flow": [-100.0, -110.0, -121.0],
            "financing_cash_flow": [-50.0, -55.0, -60.5],
            "capex": [100.0, 110.0, 121.0],
            "depreciation_amortization": [50.0, 55.0, 60.5],
        },
        index=periods,
    ).T
    return income, balance, cashflow


def _suspicious_statements():
    """A company showing classic earnings-manipulation red flags.

    Revenue jumps 36% while operating cash flow collapses 75%, receivables
    balloon (DSO doubles), and accruals spike — classic forensic warning signs.
    """
    periods = ["2022", "2023", "2024"]
    income = pd.DataFrame(
        {
            "revenue": [1000.0, 1100.0, 1500.0],
            "cogs": [600.0, 660.0, 900.0],
            "gross_profit": [400.0, 440.0, 600.0],
            "operating_expenses": [200.0, 220.0, 250.0],
            "operating_income": [200.0, 220.0, 350.0],
            "interest_expense": [10.0, 11.0, 12.0],
            "net_income": [150.0, 165.0, 400.0],
            "ebitda": [250.0, 275.0, 450.0],
        },
        index=periods,
    ).T
    balance = pd.DataFrame(
        {
            "total_assets": [2000.0, 2200.0, 2600.0],
            "current_assets": [800.0, 880.0, 1200.0],
            "current_liabilities": [400.0, 440.0, 484.0],
            "total_liabilities": [1000.0, 1100.0, 1400.0],
            "total_equity": [1000.0, 1100.0, 1200.0],
            "cash": [200.0, 220.0, 100.0],
            "inventory": [300.0, 330.0, 363.0],
            "receivables": [200.0, 220.0, 600.0],
            "total_debt": [600.0, 660.0, 800.0],
        },
        index=periods,
    ).T
    cashflow = pd.DataFrame(
        {
            "operating_cash_flow": [180.0, 198.0, 50.0],
            "investing_cash_flow": [-100.0, -110.0, -121.0],
            "financing_cash_flow": [-50.0, -55.0, -60.5],
            "capex": [100.0, 110.0, 121.0],
            "depreciation_amortization": [50.0, 55.0, 60.5],
        },
        index=periods,
    ).T
    return income, balance, cashflow


def _engine(kind="healthy"):
    if kind == "healthy":
        income, balance, cashflow = _healthy_statements()
    else:
        income, balance, cashflow = _suspicious_statements()
    return FundamentalAnalysisEngine(income, balance, cashflow)


# ---------------------------------------------------------------------------
# Ratio analysis
# ---------------------------------------------------------------------------


class TestProfitabilityRatios:
    """Profitability ratio computations (CFA FSA standards)."""

    def test_gross_margin(self):
        engine = _engine()
        ratios = engine.profitability_ratios()
        assert ratios.loc["gross_margin", "2024"] == pytest.approx(0.40, abs=1e-6)

    def test_net_margin(self):
        engine = _engine()
        ratios = engine.profitability_ratios()
        assert ratios.loc["net_margin", "2024"] == pytest.approx(0.15, abs=1e-6)

    def test_roe(self):
        engine = _engine()
        ratios = engine.profitability_ratios()
        assert ratios.loc["roe", "2024"] == pytest.approx(0.15, abs=1e-6)

    def test_ratios_dataframe_shape(self):
        engine = _engine()
        ratios = engine.profitability_ratios()
        assert ratios.shape[0] >= 5  # at least 5 profitability metrics
        assert list(ratios.columns) == ["2022", "2023", "2024"]


class TestLeverageAndLiquidityRatios:
    """Leverage and liquidity ratio computations."""

    def test_current_ratio(self):
        engine = _engine()
        ratios = engine.liquidity_ratios()
        assert ratios.loc["current_ratio", "2024"] == pytest.approx(2.0, abs=1e-6)

    def test_quick_ratio(self):
        engine = _engine()
        ratios = engine.liquidity_ratios()
        assert ratios.loc["quick_ratio", "2024"] == pytest.approx(1.25, abs=1e-6)

    def test_debt_to_equity(self):
        engine = _engine()
        ratios = engine.leverage_ratios()
        assert ratios.loc["debt_to_equity", "2024"] == pytest.approx(0.6, abs=1e-6)

    def test_interest_coverage(self):
        engine = _engine()
        ratios = engine.leverage_ratios()
        assert ratios.loc["interest_coverage", "2024"] == pytest.approx(242.0 / 12.0, abs=1e-6)


class TestEfficiencyRatios:
    """Efficiency / activity ratio computations."""

    def test_asset_turnover(self):
        engine = _engine()
        ratios = engine.efficiency_ratios()
        assert ratios.loc["asset_turnover", "2024"] == pytest.approx(0.5, abs=1e-6)

    def test_dso(self):
        engine = _engine()
        ratios = engine.efficiency_ratios()
        assert ratios.loc["dso", "2024"] == pytest.approx(73.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Cash flow analysis
# ---------------------------------------------------------------------------


class TestCashFlowAnalysis:
    """Cash flow statement analysis."""

    def test_free_cash_flow(self):
        engine = _engine()
        cf = engine.cash_flow_analysis()
        assert cf.loc["free_cash_flow", "2024"] == pytest.approx(96.8, abs=1e-6)

    def test_cash_conversion_ratio(self):
        engine = _engine()
        cf = engine.cash_flow_analysis()
        assert cf.loc["cash_conversion", "2024"] == pytest.approx(1.2, abs=1e-6)

    def test_operating_cf_growth(self):
        engine = _engine()
        trends = engine.cash_flow_trends()
        assert trends.loc["operating_cash_flow_growth", "2024"] == pytest.approx(0.10, abs=1e-6)

    def test_revenue_growth(self):
        engine = _engine()
        trends = engine.cash_flow_trends()
        assert trends.loc["revenue_growth", "2024"] == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# Earnings quality
# ---------------------------------------------------------------------------


class TestEarningsQuality:
    """Earnings quality and accrual analysis."""

    def test_accruals_ratio_healthy(self):
        """Healthy company has near-zero/negative accruals (cash-backed earnings)."""
        engine = _engine("healthy")
        eq = engine.earnings_quality()
        assert eq.loc["accruals_ratio", "2024"] == pytest.approx(-0.015, abs=1e-3)

    def test_accruals_ratio_suspicious(self):
        """Suspicious company shows large positive accruals."""
        engine = _engine("suspicious")
        eq = engine.earnings_quality()
        assert eq.loc["accruals_ratio", "2024"] > 0.10

    def test_earnings_quality_score_range(self):
        """Quality score is bounded between 0 and 1."""
        for kind in ("healthy", "suspicious"):
            engine = _engine(kind)
            eq = engine.earnings_quality()
            score = eq.loc["quality_score", "2024"]
            assert 0.0 <= score <= 1.0

    def test_healthy_scores_higher_than_suspicious(self):
        healthy = _engine("healthy").earnings_quality().loc["quality_score", "2024"]
        suspicious = _engine("suspicious").earnings_quality().loc["quality_score", "2024"]
        assert healthy > suspicious


# ---------------------------------------------------------------------------
# Anomaly detection (forensic accounting)
# ---------------------------------------------------------------------------


class TestAnomalyDetection:
    """Forensic anomaly detection — the core value proposition."""

    def test_healthy_company_few_anomalies(self):
        engine = _engine("healthy")
        anomalies = engine.detect_anomalies()
        assert len(anomalies) <= 2

    def test_suspicious_company_flags_divergence(self):
        engine = _engine("suspicious")
        anomalies = engine.detect_anomalies()
        categories = {a.category for a in anomalies}
        assert "revenue_cf_divergence" in categories

    def test_suspicious_company_flags_dso_spike(self):
        engine = _engine("suspicious")
        anomalies = engine.detect_anomalies()
        metrics = {a.metric for a in anomalies}
        assert "dso" in metrics

    def test_suspicious_company_flags_accruals(self):
        engine = _engine("suspicious")
        anomalies = engine.detect_anomalies()
        metrics = {a.metric for a in anomalies}
        assert "accruals_ratio" in metrics

    def test_anomaly_structure(self):
        engine = _engine("suspicious")
        anomalies = engine.detect_anomalies()
        assert len(anomalies) > 0
        for a in anomalies:
            assert isinstance(a, Anomaly)
            assert a.severity in ("low", "medium", "high")
            assert a.description

    def test_negative_fcf_flagged(self):
        engine = _engine("suspicious")
        anomalies = engine.detect_anomalies()
        metrics = {a.metric for a in anomalies}
        assert "free_cash_flow" in metrics


# ---------------------------------------------------------------------------
# Benford's law (forensic)
# ---------------------------------------------------------------------------


class TestBenfordAnalysis:
    """Benford's law first-digit analysis for forensic detection."""

    def test_benford_chi2_uniform_data_high(self):
        """Uniformly distributed digits should produce a high chi-square."""
        from fundamental.analysis import benford_chi2

        np.random.seed(42)
        uniform_data = np.random.uniform(1, 1000, 2000)
        chi2, p_value, _ = benford_chi2(uniform_data)
        assert chi2 > 15.507  # critical value, df=8, alpha=0.05
        assert p_value < 0.05

    def test_benford_chi2_benford_data_low(self):
        """Log-uniform (Benford-distributed) data should pass the test."""
        from fundamental.analysis import benford_chi2

        np.random.seed(42)
        benford_data = 10 ** np.random.uniform(0, 3, 5000)
        chi2, p_value, _ = benford_chi2(benford_data)
        assert chi2 < 15.507
        assert p_value > 0.05


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestFullAnalysis:
    """End-to-end analysis report generation."""

    def test_full_report_returns_dict(self):
        engine = _engine("healthy")
        report = engine.full_analysis()
        assert isinstance(report, dict)
        assert "ratios" in report
        assert "cash_flow" in report
        assert "earnings_quality" in report
        assert "anomalies" in report

    def test_full_report_anomalies_list(self):
        engine = _engine("suspicious")
        report = engine.full_analysis()
        assert isinstance(report["anomalies"], list)
        assert len(report["anomalies"]) >= 3

    def test_from_dicts_constructor(self):
        """Engine can be built from plain nested dicts."""
        income, balance, cashflow = _healthy_statements()
        engine = FundamentalAnalysisEngine.from_dicts(
            income.to_dict(), balance.to_dict(), cashflow.to_dict()
        )
        ratios = engine.profitability_ratios()
        assert ratios.loc["gross_margin", "2024"] == pytest.approx(0.40, abs=1e-6)
