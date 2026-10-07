"""RegTech compliance automation engine for PCI DSS, SOX, Basel III."""
from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ComplianceFramework(str, Enum):
    """Supported regulatory frameworks."""

    PCI_DSS = "PCI_DSS"
    SOX = "SOX"
    BASEL_III = "BASEL_III"


class ComplianceStatus(str, Enum):
    """Compliance status values."""

    NOT_ASSESSED = "NOT_ASSESSED"
    PENDING = "PENDING"
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"


class StatusTransition(BaseModel):
    """Record of a status change."""

    status: ComplianceStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    notes: str | None = None


class ComplianceControl(BaseModel):
    """A single compliance control."""

    control_id: str = Field(min_length=1)
    framework: ComplianceFramework
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    last_assessed: datetime | None = None
    status_history: list[StatusTransition] = Field(default_factory=list)


class Evidence(BaseModel):
    """Evidence collected for a control."""

    control_id: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    source: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)
    collected_at: datetime = Field(default_factory=datetime.now)


class ComplianceReport(BaseModel):
    """Generated compliance report."""

    generated_at: datetime = Field(default_factory=datetime.now)
    total_controls: int = 0
    compliant_count: int = 0
    non_compliant_count: int = 0
    pending_count: int = 0
    not_assessed_count: int = 0
    framework_breakdown: dict[ComplianceFramework, dict[str, int]] = Field(
        default_factory=dict
    )


class ComplianceEngine:
    """Main compliance automation engine."""

    def __init__(self, frameworks: list[ComplianceFramework] | None = None):
        self.frameworks: dict[ComplianceFramework, bool] = {}
        self.controls: dict[str, ComplianceControl] = {}
        self.evidence: dict[str, list[Evidence]] = {}

        if frameworks:
            for fw in frameworks:
                self.frameworks[fw] = True

    def register_control(
        self,
        framework: ComplianceFramework,
        control_id: str,
        name: str,
        description: str,
    ) -> ComplianceControl:
        """Register a new compliance control."""
        if control_id in self.controls:
            raise ValueError(f"Control {control_id} already registered")

        control = ComplianceControl(
            control_id=control_id,
            framework=framework,
            name=name,
            description=description,
        )
        self.controls[control_id] = control
        self.evidence[control_id] = []
        self.frameworks[framework] = True
        return control

    def get_controls_by_framework(
        self, framework: ComplianceFramework
    ) -> list[ComplianceControl]:
        """Get all controls for a specific framework."""
        return [
            c for c in self.controls.values() if c.framework == framework
        ]

    def add_evidence(
        self,
        control_id: str,
        evidence_type: str,
        source: str,
        data: dict[str, Any],
    ) -> Evidence:
        """Add evidence for a control."""
        if control_id not in self.controls:
            raise ValueError(f"Control {control_id} not found")

        ev = Evidence(
            control_id=control_id,
            evidence_type=evidence_type,
            source=source,
            data=data,
        )
        self.evidence[control_id].append(ev)
        return ev

    def get_evidence_for_control(self, control_id: str) -> list[Evidence]:
        """Get all evidence for a control."""
        return self.evidence.get(control_id, [])

    def assess_control(
        self, control_id: str, status: ComplianceStatus, notes: str | None = None
    ) -> None:
        """Assess a control's compliance status."""
        if control_id not in self.controls:
            raise ValueError(f"Control {control_id} not found")

        control = self.controls[control_id]
        control.status = status
        control.last_assessed = datetime.now()
        control.status_history.append(
            StatusTransition(status=status, notes=notes)
        )

    def reassess_control(
        self, control_id: str, status: ComplianceStatus, notes: str | None = None
    ) -> None:
        """Reassess a control (alias for assess_control with history)."""
        self.assess_control(control_id, status, notes)

    def get_overdue_controls(self, max_age_days: int = 90) -> list[ComplianceControl]:
        """Get controls that haven't been assessed recently."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        overdue = []
        for control in self.controls.values():
            if control.last_assessed is None:
                overdue.append(control)
            elif control.last_assessed < cutoff:
                overdue.append(control)
        return overdue

    def generate_report(self) -> ComplianceReport:
        """Generate a compliance report."""
        total = len(self.controls)
        compliant = sum(
            1 for c in self.controls.values()
            if c.status == ComplianceStatus.COMPLIANT
        )
        non_compliant = sum(
            1 for c in self.controls.values()
            if c.status == ComplianceStatus.NON_COMPLIANT
        )
        pending = sum(
            1 for c in self.controls.values()
            if c.status == ComplianceStatus.PENDING
        )
        not_assessed = sum(
            1 for c in self.controls.values()
            if c.status == ComplianceStatus.NOT_ASSESSED
        )

        framework_breakdown: dict[ComplianceFramework, dict[str, int]] = {}
        for fw in self.frameworks:
            fw_controls = self.get_controls_by_framework(fw)
            framework_breakdown[fw] = {
                "total": len(fw_controls),
                "compliant": sum(
                    1 for c in fw_controls
                    if c.status == ComplianceStatus.COMPLIANT
                ),
                "non_compliant": sum(
                    1 for c in fw_controls
                    if c.status == ComplianceStatus.NON_COMPLIANT
                ),
                "pending": sum(
                    1 for c in fw_controls
                    if c.status == ComplianceStatus.PENDING
                ),
                "not_assessed": sum(
                    1 for c in fw_controls
                    if c.status == ComplianceStatus.NOT_ASSESSED
                ),
            }

        return ComplianceReport(
            total_controls=total,
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            pending_count=pending,
            not_assessed_count=not_assessed,
            framework_breakdown=framework_breakdown,
        )
