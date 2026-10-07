"""Tests for ethics and compliance engine."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.ethics.compliance import (
    ClientProfile,
    ComplexityLevel,
    EthicalPrinciple,
    EthicsComplianceEngine,
    EthicsReport,
    InvestmentProduct,
    InvestmentType,
    LiquidityLevel,
    RiskTolerance,
    SuitabilityAssessment,
    ViolationSeverity,
)


@pytest.fixture
def engine():
    """Create a fresh engine instance."""
    return EthicsComplianceEngine()


@pytest.fixture
def conservative_client():
    """Create a conservative client profile."""
    return ClientProfile(
        client_id="C001",
        name="Conservative Client",
        risk_tolerance=RiskTolerance.CONSERVATIVE,
        investment_horizon_years=10.0,
        net_worth=1_000_000,
        annual_income=150_000,
        investment_experience_years=15.0,
        liquidity_needs=LiquidityLevel.HIGH,
        investment_objectives=["capital_preservation", "income_generation"],
        restricted_investments=[InvestmentType.CRYPTOCURRENCY, InvestmentType.DERIVATIVES],
        accredited_investor=True,
    )


@pytest.fixture
def aggressive_client():
    """Create an aggressive client profile."""
    return ClientProfile(
        client_id="C002",
        name="Aggressive Client",
        risk_tolerance=RiskTolerance.AGGRESSIVE,
        investment_horizon_years=20.0,
        net_worth=5_000_000,
        annual_income=500_000,
        investment_experience_years=10.0,
        liquidity_needs=LiquidityLevel.LOW,
        investment_objectives=["capital_appreciation", "growth"],
        accredited_investor=True,
    )


@pytest.fixture
def low_risk_product():
    """Create a low-risk investment product."""
    return InvestmentProduct(
        product_id="P001",
        name="Treasury Bonds",
        investment_type=InvestmentType.FIXED_INCOME,
        risk_level=2,
        min_investment=10_000,
        liquidity=LiquidityLevel.HIGH,
        complexity=ComplexityLevel.SIMPLE,
    )


@pytest.fixture
def high_risk_product():
    """Create a high-risk investment product."""
    return InvestmentProduct(
        product_id="P002",
        name="Crypto Futures",
        investment_type=InvestmentType.CRYPTOCURRENCY,
        risk_level=9,
        min_investment=50_000,
        liquidity=LiquidityLevel.MEDIUM,
        complexity=ComplexityLevel.COMPLEX,
    )


class TestEngineInitialization:
    """Test engine initialization."""

    def test_engine_initializes_empty(self, engine):
        assert engine.clients == {}
        assert engine.products == {}
        assert engine.rules == {}
        assert engine.violations == {}
        assert engine.suitability_assessments == {}


class TestClientRegistration:
    """Test client registration."""

    def test_register_client(self, engine, conservative_client):
        result = engine.register_client(conservative_client)
        assert result.client_id == "C001"
        assert "C001" in engine.clients

    def test_register_duplicate_client_raises(self, engine, conservative_client):
        engine.register_client(conservative_client)
        with pytest.raises(ValueError, match="already registered"):
            engine.register_client(conservative_client)


class TestProductRegistration:
    """Test product registration."""

    def test_register_product(self, engine, low_risk_product):
        result = engine.register_product(low_risk_product)
        assert result.product_id == "P001"
        assert "P001" in engine.products

    def test_register_duplicate_product_raises(self, engine, low_risk_product):
        engine.register_product(low_risk_product)
        with pytest.raises(ValueError, match="already registered"):
            engine.register_product(low_risk_product)


class TestRuleRegistration:
    """Test ethical rule registration."""

    def test_register_rule(self, engine):
        from src.ethics.compliance import EthicalRule

        rule = EthicalRule(
            rule_id="ETH-001",
            principle=EthicalPrinciple.FIDUCIARY_DUTY,
            name="Fiduciary Duty",
            description="Act in the best interest of clients",
            severity=ViolationSeverity.CRITICAL,
        )
        result = engine.register_rule(rule)
        assert result.rule_id == "ETH-001"
        assert "ETH-001" in engine.rules

    def test_register_duplicate_rule_raises(self, engine):
        from src.ethics.compliance import EthicalRule

        rule = EthicalRule(
            rule_id="ETH-001",
            principle=EthicalPrinciple.INTEGRITY,
            name="Integrity",
            description="Act with integrity",
            severity=ViolationSeverity.HIGH,
        )
        engine.register_rule(rule)
        with pytest.raises(ValueError, match="already registered"):
            engine.register_rule(rule)


class TestSuitabilityAssessment:
    """Test suitability assessment."""

    def test_suitable_investment(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        result = engine.assess_suitability("C001", "P001")
        assert result.suitable is True
        assert result.risk_alignment is True
        assert result.liquidity_alignment is True
        assert len(result.reasons) == 0

    def test_risk_mismatch(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        result = engine.assess_suitability("C001", "P002")
        assert result.suitable is False
        assert result.risk_alignment is False
        assert len(result.reasons) > 0

    def test_liquidity_mismatch(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        result = engine.assess_suitability("C001", "P002")
        assert result.suitable is False
        assert result.liquidity_alignment is False

    def test_restricted_investment(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        result = engine.assess_suitability("C001", "P002")
        assert result.suitable is False
        assert any("restricted" in r.lower() for r in result.reasons)

    def test_nonexistent_client_raises(self, engine, low_risk_product):
        engine.register_product(low_risk_product)
        with pytest.raises(ValueError, match="not found"):
            engine.assess_suitability("NONEXISTENT", "P001")

    def test_nonexistent_product_raises(self, engine, conservative_client):
        engine.register_client(conservative_client)
        with pytest.raises(ValueError, match="not found"):
            engine.assess_suitability("C001", "NONEXISTENT")

    def test_assessment_stored(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        engine.assess_suitability("C001", "P001")
        assert "C001:P001" in engine.suitability_assessments


class TestFiduciaryDuty:
    """Test fiduciary duty checks."""

    def test_fiduciary_breach_unsuitable(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        engine.assess_suitability("C001", "P002")
        violations = engine.check_fiduciary_duty("C001", "P002")
        assert len(violations) > 0
        assert any(v.severity == ViolationSeverity.CRITICAL for v in violations)

    def test_fiduciary_compliant(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        engine.assess_suitability("C001", "P001")
        violations = engine.check_fiduciary_duty("C001", "P001")
        assert len(violations) == 0

    def test_fiduciary_undisclosed_conflict(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        violations = engine.check_fiduciary_duty(
            "C001", "P001", advisor_compensation=5000, conflicts_disclosed=False
        )
        assert len(violations) > 0
        assert any("not disclosed" in v.description for v in violations)


class TestConflictsOfInterest:
    """Test conflict of interest detection."""

    def test_advisor_owns_product(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        violations = engine.check_conflicts_of_interest(
            "C001", "P001", advisor_owns_product=True
        )
        assert len(violations) == 1
        assert "owns" in violations[0].description.lower()

    def test_referral_fee_conflict(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        violations = engine.check_conflicts_of_interest(
            "C001", "P001", referral_fee=1000
        )
        assert len(violations) == 1
        assert "referral" in violations[0].description.lower()

    def test_personal_account_trading(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        violations = engine.check_conflicts_of_interest(
            "C001", "P001", personal_account_trading=True
        )
        assert len(violations) == 1
        assert "personal account" in violations[0].description.lower()

    def test_no_conflicts(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        violations = engine.check_conflicts_of_interest("C001", "P001")
        assert len(violations) == 0


class TestViolationManagement:
    """Test violation management."""

    def test_resolve_violation(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        engine.assess_suitability("C001", "P002")
        violations = engine.get_active_violations()
        assert len(violations) > 0
        vio_id = violations[0].violation_id
        resolved = engine.resolve_violation(vio_id, "Client acknowledged risks")
        assert resolved.resolved is True
        assert resolved.resolution_notes == "Client acknowledged risks"

    def test_resolve_nonexistent_violation_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.resolve_violation("NONEXISTENT", "notes")

    def test_get_violations_by_severity(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        engine.assess_suitability("C001", "P002")
        # 3+ misalignments produce CRITICAL severity
        critical_violations = engine.get_violations_by_severity(ViolationSeverity.CRITICAL)
        assert len(critical_violations) > 0

    def test_get_violations_for_client(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        engine.assess_suitability("C001", "P002")
        client_violations = engine.get_violations_for_client("C001")
        assert len(client_violations) > 0


class TestEthicsReport:
    """Test ethics report generation."""

    def test_generate_report_empty(self, engine):
        report = engine.generate_ethics_report()
        assert isinstance(report, EthicsReport)
        assert report.total_clients == 0
        assert report.active_violations == 0

    def test_generate_report_with_data(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        engine.assess_suitability("C001", "P002")
        report = engine.generate_ethics_report()
        assert report.total_clients == 1
        assert report.total_products == 1
        assert report.active_violations > 0
        assert report.unsuitable_count == 1


class TestComplianceMonitoring:
    """Test compliance monitoring."""

    def test_monitor_compliant(self, engine, conservative_client, low_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(low_risk_product)
        engine.assess_suitability("C001", "P001")
        status = engine.monitor_compliance()
        assert status["compliance_status"] == "COMPLIANT"
        assert status["requires_immediate_action"] is False

    def test_monitor_non_compliant(self, engine, conservative_client, high_risk_product):
        engine.register_client(conservative_client)
        engine.register_product(high_risk_product)
        engine.assess_suitability("C001", "P002")
        status = engine.monitor_compliance()
        assert status["compliance_status"] == "NON_COMPLIANT"
        assert status["requires_immediate_action"] is True
        assert status["critical_violations"] > 0


class TestPydanticValidation:
    """Test pydantic model validation."""

    def test_client_profile_validation(self):
        with pytest.raises(ValidationError):
            ClientProfile(
                client_id="",
                name="Test",
                risk_tolerance=RiskTolerance.MODERATE,
                investment_horizon_years=5.0,
                net_worth=100_000,
                annual_income=50_000,
                investment_experience_years=2.0,
                liquidity_needs=LiquidityLevel.MEDIUM,
                investment_objectives=["growth"],
            )

    def test_investment_product_validation(self):
        with pytest.raises(ValidationError):
            InvestmentProduct(
                product_id="P001",
                name="Test",
                investment_type=InvestmentType.EQUITY,
                risk_level=15,  # Invalid: > 10
                min_investment=1000,
                liquidity=LiquidityLevel.HIGH,
                complexity=ComplexityLevel.SIMPLE,
            )

    def test_suitability_assessment_validation(self):
        assessment = SuitabilityAssessment(
            client_id="C001",
            product_id="P001",
            suitable=True,
            reasons=[],
            risk_alignment=True,
            horizon_alignment=True,
            liquidity_alignment=True,
            complexity_alignment=True,
        )
        assert assessment.suitable is True
        assert isinstance(assessment.assessed_at, datetime)
