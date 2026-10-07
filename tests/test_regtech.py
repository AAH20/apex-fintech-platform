"""Tests for RegTech compliance automation engine."""
import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.regtech.compliance import (
    ComplianceEngine,
    ComplianceFramework,
    ComplianceControl,
    Evidence,
    ComplianceStatus,
    ComplianceReport,
)


class TestComplianceEngineInit:
    """Test engine initialization and basic properties."""

    def test_engine_initializes_empty(self):
        engine = ComplianceEngine()
        assert engine.frameworks == {}
        assert engine.controls == {}
        assert engine.evidence == {}

    def test_engine_with_frameworks(self):
        engine = ComplianceEngine(frameworks=[ComplianceFramework.PCI_DSS])
        assert ComplianceFramework.PCI_DSS in engine.frameworks
        assert len(engine.frameworks) == 1


class TestControlMapping:
    """Test control registration and mapping."""

    def test_register_control(self):
        engine = ComplianceEngine()
        control = engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall configuration",
            description="Maintain firewall configuration standards",
        )
        assert control.control_id == "PCI-1.1"
        assert control.framework == ComplianceFramework.PCI_DSS
        assert control.status == ComplianceStatus.NOT_ASSESSED

    def test_register_duplicate_control_raises(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall configuration",
            description="Maintain firewall configuration standards",
        )
        with pytest.raises(ValueError, match="already registered"):
            engine.register_control(
                framework=ComplianceFramework.PCI_DSS,
                control_id="PCI-1.1",
                name="Duplicate",
                description="Duplicate control",
            )

    def test_get_controls_by_framework(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.register_control(
            framework=ComplianceFramework.SOX,
            control_id="SOX-404.1",
            name="ITGC",
            description="IT general controls",
        )
        pci_controls = engine.get_controls_by_framework(ComplianceFramework.PCI_DSS)
        assert len(pci_controls) == 1
        assert pci_controls[0].control_id == "PCI-1.1"


class TestEvidenceCollection:
    """Test evidence collection and management."""

    def test_add_evidence(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        evidence = engine.add_evidence(
            control_id="PCI-1.1",
            evidence_type="config_file",
            source="firewall_rules.json",
            data={"rules": 42, "last_reviewed": "2024-01-15"},
        )
        assert evidence.control_id == "PCI-1.1"
        assert evidence.evidence_type == "config_file"
        assert evidence.source == "firewall_rules.json"

    def test_add_evidence_to_nonexistent_control_raises(self):
        engine = ComplianceEngine()
        with pytest.raises(ValueError, match="not found"):
            engine.add_evidence(
                control_id="NONEXISTENT",
                evidence_type="config_file",
                source="test.json",
                data={},
            )

    def test_get_evidence_for_control(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.add_evidence(
            control_id="PCI-1.1",
            evidence_type="config_file",
            source="fw.json",
            data={},
        )
        engine.add_evidence(
            control_id="PCI-1.1",
            evidence_type="audit_log",
            source="audit.log",
            data={},
        )
        evidence_list = engine.get_evidence_for_control("PCI-1.1")
        assert len(evidence_list) == 2

    def test_evidence_has_timestamp(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        evidence = engine.add_evidence(
            control_id="PCI-1.1",
            evidence_type="config_file",
            source="fw.json",
            data={},
        )
        assert evidence.collected_at is not None
        assert isinstance(evidence.collected_at, datetime)


class TestComplianceStatus:
    """Test compliance status management."""

    def test_assess_control_compliant(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
        control = engine.controls["PCI-1.1"]
        assert control.status == ComplianceStatus.COMPLIANT

    def test_assess_control_non_compliant(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.assess_control("PCI-1.1", ComplianceStatus.NON_COMPLIANT)
        control = engine.controls["PCI-1.1"]
        assert control.status == ComplianceStatus.NON_COMPLIANT

    def test_assess_nonexistent_control_raises(self):
        engine = ComplianceEngine()
        with pytest.raises(ValueError, match="not found"):
            engine.assess_control("NONEXISTENT", ComplianceStatus.COMPLIANT)

    def test_status_transition_history(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.assess_control("PCI-1.1", ComplianceStatus.PENDING)
        engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
        control = engine.controls["PCI-1.1"]
        assert len(control.status_history) == 2
        assert control.status_history[0].status == ComplianceStatus.PENDING
        assert control.status_history[1].status == ComplianceStatus.COMPLIANT


class TestRegulatoryReporting:
    """Test compliance report generation."""

    def test_generate_report_empty(self):
        engine = ComplianceEngine()
        report = engine.generate_report()
        assert isinstance(report, ComplianceReport)
        assert report.total_controls == 0
        assert report.compliant_count == 0

    def test_generate_report_with_controls(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-2.1",
            name="Encryption",
            description="Data encryption",
        )
        engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
        engine.assess_control("PCI-2.1", ComplianceStatus.NON_COMPLIANT)
        report = engine.generate_report()
        assert report.total_controls == 2
        assert report.compliant_count == 1
        assert report.non_compliant_count == 1

    def test_report_includes_framework_breakdown(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.register_control(
            framework=ComplianceFramework.SOX,
            control_id="SOX-404.1",
            name="ITGC",
            description="IT general controls",
        )
        engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
        engine.assess_control("SOX-404.1", ComplianceStatus.COMPLIANT)
        report = engine.generate_report()
        assert ComplianceFramework.PCI_DSS in report.framework_breakdown
        assert ComplianceFramework.SOX in report.framework_breakdown

    def test_report_has_generated_at(self):
        engine = ComplianceEngine()
        report = engine.generate_report()
        assert report.generated_at is not None
        assert isinstance(report.generated_at, datetime)


class TestContinuousMonitoring:
    """Test continuous monitoring features."""

    def test_get_overdue_controls(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        # Manually set last_assessed to old date
        engine.controls["PCI-1.1"].last_assessed = datetime.now() - timedelta(days=120)
        overdue = engine.get_overdue_controls(max_age_days=90)
        assert len(overdue) == 1
        assert overdue[0].control_id == "PCI-1.1"

    def test_get_no_overdue_controls(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.controls["PCI-1.1"].last_assessed = datetime.now() - timedelta(days=30)
        overdue = engine.get_overdue_controls(max_age_days=90)
        assert len(overdue) == 0

    def test_reassess_control(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
        engine.reassess_control("PCI-1.1", ComplianceStatus.NON_COMPLIANT)
        control = engine.controls["PCI-1.1"]
        assert control.status == ComplianceStatus.NON_COMPLIANT
        assert len(control.status_history) == 2


class TestMultiFramework:
    """Test multi-framework compliance support."""

    def test_basel_iii_controls(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.BASEL_III,
            control_id="B3-CR-1",
            name="Credit Risk",
            description="Credit risk management framework",
        )
        controls = engine.get_controls_by_framework(ComplianceFramework.BASEL_III)
        assert len(controls) == 1
        assert controls[0].control_id == "B3-CR-1"

    def test_mixed_framework_report(self):
        engine = ComplianceEngine()
        engine.register_control(
            framework=ComplianceFramework.PCI_DSS,
            control_id="PCI-1.1",
            name="Firewall",
            description="Firewall config",
        )
        engine.register_control(
            framework=ComplianceFramework.SOX,
            control_id="SOX-404.1",
            name="ITGC",
            description="IT general controls",
        )
        engine.register_control(
            framework=ComplianceFramework.BASEL_III,
            control_id="B3-CR-1",
            name="Credit Risk",
            description="Credit risk management",
        )
        engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
        engine.assess_control("SOX-404.1", ComplianceStatus.COMPLIANT)
        engine.assess_control("B3-CR-1", ComplianceStatus.PENDING)
        report = engine.generate_report()
        assert report.total_controls == 3
        assert report.compliant_count == 2
        assert report.pending_count == 1


class TestPydanticValidation:
    """Test pydantic model validation."""

    def test_compliance_control_validation(self):
        with pytest.raises(ValidationError):
            ComplianceControl(
                control_id="",
                framework=ComplianceFramework.PCI_DSS,
                name="Test",
                description="Test description",
            )

    def test_evidence_validation(self):
        with pytest.raises(ValidationError):
            Evidence(
                control_id="PCI-1.1",
                evidence_type="",
                source="test.json",
                data={},
            )

    def test_compliance_report_validation(self):
        report = ComplianceReport(
            generated_at=datetime.now(),
            total_controls=10,
            compliant_count=5,
            non_compliant_count=3,
            pending_count=2,
            not_assessed_count=0,
            framework_breakdown={},
        )
        assert report.total_controls == 10
