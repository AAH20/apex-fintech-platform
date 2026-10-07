"""Tests for ESG analytics engine."""
import pytest
import pandas as pd

from src.esg.analytics import (
    ESGAnalyticsEngine,
    ESGScore,
    ImpactMetrics,
    GreenwashingReport,
    ESGDataInput,
)


class TestESGScoring:
    """Test ESG scoring functionality."""

    def test_score_single_company(self):
        """Test scoring a single company with complete data."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="COMP001",
            company_name="Test Corp",
            environmental_score=75.0,
            social_score=65.0,
            governance_score=80.0,
            carbon_emissions_tons=1000.0,
            revenue_millions=500.0,
        )
        score = engine.score_company(data)
        assert isinstance(score, ESGScore)
        assert score.company_id == "COMP001"
        assert 0 <= score.environmental <= 100
        assert 0 <= score.social <= 100
        assert 0 <= score.governance <= 100
        assert 0 <= score.overall <= 100

    def test_overall_score_weighted_average(self):
        """Test that overall score is weighted average of pillars."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="COMP002",
            company_name="Weighted Corp",
            environmental_score=90.0,
            social_score=60.0,
            governance_score=70.0,
        )
        score = engine.score_company(data)
        expected = (90.0 * 0.4 + 60.0 * 0.3 + 70.0 * 0.3)
        assert abs(score.overall - expected) < 0.01

    def test_score_with_missing_data(self):
        """Test scoring with missing ESG data uses defaults."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="COMP003",
            company_name="Missing Data Corp",
            environmental_score=None,
            social_score=50.0,
            governance_score=None,
        )
        score = engine.score_company(data)
        assert score.environmental == 50.0  # default
        assert score.social == 50.0
        assert score.governance == 50.0  # default

    def test_score_boundary_values(self):
        """Test scoring with boundary values (0 and 100)."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="COMP004",
            company_name="Boundary Corp",
            environmental_score=0.0,
            social_score=100.0,
            governance_score=50.0,
        )
        score = engine.score_company(data)
        assert score.environmental == 0.0
        assert score.social == 100.0
        assert score.governance == 50.0

    def test_carbon_intensity_calculation(self):
        """Test carbon intensity (emissions per revenue unit)."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="COMP005",
            company_name="Carbon Corp",
            environmental_score=70.0,
            social_score=70.0,
            governance_score=70.0,
            carbon_emissions_tons=2000.0,
            revenue_millions=1000.0,
        )
        score = engine.score_company(data)
        assert abs(score.carbon_intensity - 2.0) < 0.01


class TestImpactMeasurement:
    """Test portfolio impact measurement."""

    def test_portfolio_carbon_footprint(self):
        """Test portfolio-level carbon footprint aggregation."""
        engine = ESGAnalyticsEngine()
        holdings = [
            {"company_id": "A", "weight": 0.6, "carbon_emissions_tons": 1000},
            {"company_id": "B", "weight": 0.4, "carbon_emissions_tons": 2000},
        ]
        impact = engine.measure_portfolio_impact(holdings)
        assert isinstance(impact, ImpactMetrics)
        assert abs(impact.total_carbon_emissions - 1400.0) < 0.01

    def test_portfolio_weighted_esg_score(self):
        """Test portfolio weighted ESG score."""
        engine = ESGAnalyticsEngine()
        holdings = [
            {"company_id": "A", "weight": 0.5, "esg_score": 80.0},
            {"company_id": "B", "weight": 0.5, "esg_score": 60.0},
        ]
        impact = engine.measure_portfolio_impact(holdings)
        assert abs(impact.weighted_esg_score - 70.0) < 0.01

    def test_portfolio_normalized_weights(self):
        """Test that weights are normalized when they don't sum to 1."""
        engine = ESGAnalyticsEngine()
        holdings = [
            {"company_id": "A", "weight": 3, "carbon_emissions_tons": 1000},
            {"company_id": "B", "weight": 1, "carbon_emissions_tons": 2000},
        ]
        impact = engine.measure_portfolio_impact(holdings)
        # Normalized: A=0.75, B=0.25 -> 0.75*1000 + 0.25*2000 = 1250
        assert abs(impact.total_carbon_emissions - 1250.0) < 0.01

    def test_empty_portfolio(self):
        """Test impact measurement with empty portfolio."""
        engine = ESGAnalyticsEngine()
        impact = engine.measure_portfolio_impact([])
        assert impact.total_carbon_emissions == 0.0
        assert impact.weighted_esg_score == 0.0


class TestGreenwashingDetection:
    """Test greenwashing detection algorithms."""

    def test_detect_high_emissions_high_esg_score(self):
        """Test flagging companies with high emissions but high ESG scores."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="GW001",
            company_name="Greenwash Corp",
            environmental_score=95.0,
            social_score=90.0,
            governance_score=85.0,
            carbon_emissions_tons=50000.0,
            revenue_millions=100.0,
        )
        report = engine.detect_greenwashing(data)
        assert isinstance(report, GreenwashingReport)
        assert report.is_suspicious is True
        assert report.risk_level in ["HIGH", "MEDIUM", "LOW"]

    def test_detect_low_emissions_high_esg_score(self):
        """Test that low-emission high-ESG companies are not flagged."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="CLEAN001",
            company_name="Clean Corp",
            environmental_score=90.0,
            social_score=85.0,
            governance_score=80.0,
            carbon_emissions_tons=10.0,
            revenue_millions=1000.0,
        )
        report = engine.detect_greenwashing(data)
        assert report.is_suspicious is False

    def test_greenwashing_risk_levels(self):
        """Test different risk levels based on discrepancy magnitude."""
        engine = ESGAnalyticsEngine()
        # Small discrepancy -> LOW risk
        data_low = ESGDataInput(
            company_id="LOW001",
            company_name="Low Risk Corp",
            environmental_score=85.0,
            social_score=80.0,
            governance_score=75.0,
            carbon_emissions_tons=500.0,
            revenue_millions=100.0,
        )
        report_low = engine.detect_greenwashing(data_low)
        assert report_low.risk_level in ["LOW", "MEDIUM"]

        # Large discrepancy -> HIGH risk
        data_high = ESGDataInput(
            company_id="HIGH001",
            company_name="High Risk Corp",
            environmental_score=98.0,
            social_score=95.0,
            governance_score=90.0,
            carbon_emissions_tons=100000.0,
            revenue_millions=50.0,
        )
        report_high = engine.detect_greenwashing(data_high)
        assert report_high.risk_level == "HIGH"

    def test_greenwashing_report_fields(self):
        """Test that greenwashing report contains all required fields."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="GW002",
            company_name="Test Corp",
            environmental_score=80.0,
            social_score=75.0,
            governance_score=70.0,
            carbon_emissions_tons=1000.0,
            revenue_millions=200.0,
        )
        report = engine.detect_greenwashing(data)
        assert hasattr(report, "company_id")
        assert hasattr(report, "is_suspicious")
        assert hasattr(report, "risk_level")
        assert hasattr(report, "carbon_intensity")
        assert hasattr(report, "esg_score")
        assert hasattr(report, "discrepancy_score")


class TestESGDataInput:
    """Test ESG data input validation."""

    def test_valid_data_input(self):
        """Test creating valid ESG data input."""
        data = ESGDataInput(
            company_id="TEST001",
            company_name="Test Corp",
            environmental_score=75.0,
            social_score=65.0,
            governance_score=80.0,
        )
        assert data.company_id == "TEST001"
        assert data.environmental_score == 75.0

    def test_invalid_score_range(self):
        """Test that scores outside 0-100 raise validation error."""
        with pytest.raises(ValueError):
            ESGDataInput(
                company_id="TEST002",
                company_name="Bad Corp",
                environmental_score=150.0,
                social_score=65.0,
                governance_score=80.0,
            )

    def test_negative_emissions(self):
        """Test that negative carbon emissions raise validation error."""
        with pytest.raises(ValueError):
            ESGDataInput(
                company_id="TEST003",
                company_name="Negative Corp",
                environmental_score=75.0,
                social_score=65.0,
                governance_score=80.0,
                carbon_emissions_tons=-100.0,
            )


class TestESGAnalyticsEngineIntegration:
    """Integration tests for the full ESG analytics pipeline."""

    def test_full_pipeline(self):
        """Test complete ESG analysis pipeline."""
        engine = ESGAnalyticsEngine()
        data = ESGDataInput(
            company_id="FULL001",
            company_name="Full Pipeline Corp",
            environmental_score=85.0,
            social_score=75.0,
            governance_score=80.0,
            carbon_emissions_tons=500.0,
            revenue_millions=1000.0,
        )
        score = engine.score_company(data)
        impact = engine.measure_portfolio_impact([
            {"company_id": "FULL001", "weight": 1.0, "carbon_emissions_tons": 500}
        ])
        report = engine.detect_greenwashing(data)

        assert score.company_id == "FULL001"
        assert impact.total_carbon_emissions == 500.0
        assert report.company_id == "FULL001"

    def test_batch_scoring(self):
        """Test scoring multiple companies at once."""
        engine = ESGAnalyticsEngine()
        companies = [
            ESGDataInput(
                company_id=f"BATCH{i:03d}",
                company_name=f"Batch Corp {i}",
                environmental_score=70.0 + i,
                social_score=65.0 + i,
                governance_score=60.0 + i,
            )
            for i in range(5)
        ]
        scores = [engine.score_company(c) for c in companies]
        assert len(scores) == 5
        assert all(isinstance(s, ESGScore) for s in scores)

    def test_dataframe_input(self):
        """Test processing ESG data from pandas DataFrame."""
        engine = ESGAnalyticsEngine()
        df = pd.DataFrame({
            "company_id": ["DF001", "DF002"],
            "company_name": ["DF Corp 1", "DF Corp 2"],
            "environmental_score": [80.0, 70.0],
            "social_score": [75.0, 65.0],
            "governance_score": [70.0, 60.0],
            "carbon_emissions_tons": [100.0, 200.0],
            "revenue_millions": [500.0, 400.0],
        })
        results = engine.score_from_dataframe(df)
        assert len(results) == 2
        assert all(isinstance(s, ESGScore) for s in results)
