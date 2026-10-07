"""Ethics and compliance engine for monitoring compliance and detecting ethical violations."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskTolerance(str, Enum):
    """Client risk tolerance levels."""

    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class LiquidityLevel(str, Enum):
    """Liquidity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ComplexityLevel(str, Enum):
    """Product complexity levels."""

    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"


class InvestmentType(str, Enum):
    """Investment product types."""

    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    DERIVATIVES = "DERIVATIVES"
    CRYPTOCURRENCY = "CRYPTOCURRENCY"
    COMMODITIES = "COMMODITIES"
    REAL_ESTATE = "REAL_ESTATE"


class EthicalPrinciple(str, Enum):
    """Ethical principles based on CFA Institute standards."""

    FIDUCIARY_DUTY = "FIDUCIARY_DUTY"
    INTEGRITY = "INTEGRITY"
    TRANSPARENCY = "TRANSPARENCY"
    FAIR_DEALING = "FAIR_DEALING"
    DUE_DILIGENCE = "DUE_DILIGENCE"
    CONFLICT_OF_INTEREST = "CONFLICT_OF_INTEREST"


class ViolationSeverity(str, Enum):
    """Violation severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ClientProfile(BaseModel):
    """Client profile for suitability and ethical analysis."""

    client_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    risk_tolerance: RiskTolerance
    investment_horizon_years: float = Field(gt=0)
    net_worth: float = Field(ge=0)
    annual_income: float = Field(ge=0)
    investment_experience_years: float = Field(ge=0)
    liquidity_needs: LiquidityLevel
    investment_objectives: list[str] = Field(default_factory=list)
    restricted_investments: list[InvestmentType] = Field(default_factory=list)
    accredited_investor: bool = False


class InvestmentProduct(BaseModel):
    """Investment product definition."""

    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    investment_type: InvestmentType
    risk_level: int = Field(ge=1, le=10)
    min_investment: float = Field(gt=0)
    liquidity: LiquidityLevel
    complexity: ComplexityLevel


class EthicalRule(BaseModel):
    """An ethical rule or standard."""

    rule_id: str = Field(min_length=1)
    principle: EthicalPrinciple
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: ViolationSeverity


class Violation(BaseModel):
    """An ethical or compliance violation."""

    violation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = Field(min_length=1)
    principle: EthicalPrinciple
    client_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    severity: ViolationSeverity
    description: str = Field(min_length=1)
    detected_at: datetime = Field(default_factory=datetime.now)
    resolved: bool = False
    resolution_notes: str | None = None
    resolved_at: datetime | None = None


class SuitabilityAssessment(BaseModel):
    """Result of a suitability assessment."""

    client_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    suitable: bool
    reasons: list[str] = Field(default_factory=list)
    risk_alignment: bool = True
    horizon_alignment: bool = True
    liquidity_alignment: bool = True
    complexity_alignment: bool = True
    assessed_at: datetime = Field(default_factory=datetime.now)


class EthicsReport(BaseModel):
    """Generated ethics and compliance report."""

    generated_at: datetime = Field(default_factory=datetime.now)
    total_clients: int = 0
    total_products: int = 0
    total_rules: int = 0
    active_violations: int = 0
    resolved_violations: int = 0
    unsuitable_count: int = 0
    suitable_count: int = 0
    violations_by_severity: dict[ViolationSeverity, int] = Field(default_factory=dict)
    violations_by_principle: dict[EthicalPrinciple, int] = Field(default_factory=dict)


class EthicsComplianceEngine:
    """Main ethics and compliance engine.

    Monitors compliance, detects ethical violations, and performs
    suitability checks based on CFA Institute ethical standards.
    """

    def __init__(self) -> None:
        self.clients: dict[str, ClientProfile] = {}
        self.products: dict[str, InvestmentProduct] = {}
        self.rules: dict[str, EthicalRule] = {}
        self.violations: dict[str, Violation] = {}
        self.suitability_assessments: dict[str, SuitabilityAssessment] = {}

    def register_client(self, client: ClientProfile) -> ClientProfile:
        """Register a client profile."""
        if client.client_id in self.clients:
            raise ValueError(f"Client {client.client_id} already registered")
        self.clients[client.client_id] = client
        return client

    def register_product(self, product: InvestmentProduct) -> InvestmentProduct:
        """Register an investment product."""
        if product.product_id in self.products:
            raise ValueError(f"Product {product.product_id} already registered")
        self.products[product.product_id] = product
        return product

    def register_rule(self, rule: EthicalRule) -> EthicalRule:
        """Register an ethical rule."""
        if rule.rule_id in self.rules:
            raise ValueError(f"Rule {rule.rule_id} already registered")
        self.rules[rule.rule_id] = rule
        return rule

    def assess_suitability(
        self, client_id: str, product_id: str
    ) -> SuitabilityAssessment:
        """Assess whether a product is suitable for a client.

        Evaluates risk alignment, horizon alignment, liquidity alignment,
        complexity alignment, and restricted investments.
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")
        if product_id not in self.products:
            raise ValueError(f"Product {product_id} not found")

        client = self.clients[client_id]
        product = self.products[product_id]
        reasons: list[str] = []

        # Risk alignment
        risk_alignment = self._check_risk_alignment(client, product)
        if not risk_alignment:
            reasons.append(
                f"Product risk level {product.risk_level} exceeds "
                f"client {client.risk_tolerance.value} risk tolerance"
            )

        # Horizon alignment
        horizon_alignment = self._check_horizon_alignment(client, product)
        if not horizon_alignment:
            reasons.append(
                f"Product liquidity {product.liquidity.value} insufficient "
                f"for client horizon {client.investment_horizon_years} years"
            )

        # Liquidity alignment
        liquidity_alignment = self._check_liquidity_alignment(client, product)
        if not liquidity_alignment:
            reasons.append(
                f"Product liquidity {product.liquidity.value} does not meet "
                f"client liquidity needs {client.liquidity_needs.value}"
            )

        # Complexity alignment
        complexity_alignment = self._check_complexity_alignment(client, product)
        if not complexity_alignment:
            reasons.append(
                f"Product complexity {product.complexity.value} exceeds "
                f"client experience level"
            )

        # Restricted investments
        if product.investment_type in client.restricted_investments:
            reasons.append(
                f"Product type {product.investment_type.value} is restricted "
                f"for this client"
            )

        suitable = (
            risk_alignment
            and horizon_alignment
            and liquidity_alignment
            and complexity_alignment
            and product.investment_type not in client.restricted_investments
        )

        assessment = SuitabilityAssessment(
            client_id=client_id,
            product_id=product_id,
            suitable=suitable,
            reasons=reasons,
            risk_alignment=risk_alignment,
            horizon_alignment=horizon_alignment,
            liquidity_alignment=liquidity_alignment,
            complexity_alignment=complexity_alignment,
        )
        self.suitability_assessments[f"{client_id}:{product_id}"] = assessment

        # Auto-generate violations for unsuitable products
        if not suitable:
            self._create_suitability_violations(client_id, product_id, reasons)

        return assessment

    def _create_suitability_violations(
        self, client_id: str, product_id: str, reasons: list[str]
    ) -> list[Violation]:
        """Create violations for an unsuitable product recommendation."""
        violations: list[Violation] = []

        # Determine severity based on number of misalignments
        if len(reasons) >= 3:
            severity = ViolationSeverity.CRITICAL
        elif len(reasons) >= 2:
            severity = ViolationSeverity.HIGH
        else:
            severity = ViolationSeverity.MEDIUM

        violations.append(
            Violation(
                rule_id="ETH-SUIT-001",
                principle=EthicalPrinciple.FIDUCIARY_DUTY,
                client_id=client_id,
                product_id=product_id,
                severity=severity,
                description=f"Unsuitable product recommendation: {'; '.join(reasons)}",
            )
        )

        for v in violations:
            self.violations[v.violation_id] = v

        return violations

    def _check_risk_alignment(
        self, client: ClientProfile, product: InvestmentProduct
    ) -> bool:
        """Check if product risk aligns with client risk tolerance."""
        max_risk = {
            RiskTolerance.CONSERVATIVE: 3,
            RiskTolerance.MODERATE: 6,
            RiskTolerance.AGGRESSIVE: 10,
        }
        return product.risk_level <= max_risk[client.risk_tolerance]

    def _check_horizon_alignment(
        self, client: ClientProfile, product: InvestmentProduct
    ) -> bool:
        """Check if product liquidity aligns with client investment horizon."""
        if client.investment_horizon_years < 2:
            return product.liquidity == LiquidityLevel.HIGH
        if client.investment_horizon_years < 5:
            return product.liquidity in (LiquidityLevel.MEDIUM, LiquidityLevel.HIGH)
        return True

    def _check_liquidity_alignment(
        self, client: ClientProfile, product: InvestmentProduct
    ) -> bool:
        """Check if product liquidity meets client liquidity needs."""
        if client.liquidity_needs == LiquidityLevel.HIGH:
            return product.liquidity == LiquidityLevel.HIGH
        if client.liquidity_needs == LiquidityLevel.MEDIUM:
            return product.liquidity in (LiquidityLevel.MEDIUM, LiquidityLevel.HIGH)
        return True

    def _check_complexity_alignment(
        self, client: ClientProfile, product: InvestmentProduct
    ) -> bool:
        """Check if product complexity aligns with client experience."""
        if product.complexity == ComplexityLevel.SIMPLE:
            return True
        if product.complexity == ComplexityLevel.MODERATE:
            return client.investment_experience_years >= 2
        return client.investment_experience_years >= 5

    def check_fiduciary_duty(
        self,
        client_id: str,
        product_id: str,
        advisor_compensation: float = 0,
        conflicts_disclosed: bool = True,
    ) -> list[Violation]:
        """Check for fiduciary duty violations.

        A fiduciary must act in the best interest of the client.
        """
        violations: list[Violation] = []

        # Check if suitability assessment exists
        key = f"{client_id}:{product_id}"
        if key in self.suitability_assessments:
            assessment = self.suitability_assessments[key]
            if not assessment.suitable:
                violations.append(
                    Violation(
                        rule_id="ETH-FID-001",
                        principle=EthicalPrinciple.FIDUCIARY_DUTY,
                        client_id=client_id,
                        product_id=product_id,
                        severity=ViolationSeverity.CRITICAL,
                        description=(
                            f"Recommending unsuitable product: "
                            f"{'; '.join(assessment.reasons)}"
                        ),
                    )
                )

        # Check for undisclosed conflicts
        if advisor_compensation > 0 and not conflicts_disclosed:
            violations.append(
                Violation(
                    rule_id="ETH-FID-002",
                    principle=EthicalPrinciple.FIDUCIARY_DUTY,
                    client_id=client_id,
                    product_id=product_id,
                    severity=ViolationSeverity.HIGH,
                    description=(
                        f"Advisor compensation of ${advisor_compensation:,.2f} "
                        f"not disclosed to client"
                    ),
                )
            )

        for v in violations:
            self.violations[v.violation_id] = v

        return violations

    def check_conflicts_of_interest(
        self,
        client_id: str,
        product_id: str,
        advisor_owns_product: bool = False,
        referral_fee: float = 0,
        personal_account_trading: bool = False,
    ) -> list[Violation]:
        """Check for conflicts of interest violations."""
        violations: list[Violation] = []

        if advisor_owns_product:
            violations.append(
                Violation(
                    rule_id="ETH-COI-001",
                    principle=EthicalPrinciple.CONFLICT_OF_INTEREST,
                    client_id=client_id,
                    product_id=product_id,
                    severity=ViolationSeverity.HIGH,
                    description="Advisor owns the recommended product",
                )
            )

        if referral_fee > 0:
            violations.append(
                Violation(
                    rule_id="ETH-COI-002",
                    principle=EthicalPrinciple.CONFLICT_OF_INTEREST,
                    client_id=client_id,
                    product_id=product_id,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Referral fee of ${referral_fee:,.2f} creates conflict of interest",
                )
            )

        if personal_account_trading:
            violations.append(
                Violation(
                    rule_id="ETH-COI-003",
                    principle=EthicalPrinciple.CONFLICT_OF_INTEREST,
                    client_id=client_id,
                    product_id=product_id,
                    severity=ViolationSeverity.HIGH,
                    description="Advisor engages in personal account trading in same product",
                )
            )

        for v in violations:
            self.violations[v.violation_id] = v

        return violations

    def resolve_violation(self, violation_id: str, notes: str) -> Violation:
        """Resolve a violation with notes."""
        if violation_id not in self.violations:
            raise ValueError(f"Violation {violation_id} not found")
        violation = self.violations[violation_id]
        violation.resolved = True
        violation.resolution_notes = notes
        violation.resolved_at = datetime.now()
        return violation

    def get_active_violations(self) -> list[Violation]:
        """Get all unresolved violations."""
        return [v for v in self.violations.values() if not v.resolved]

    def get_violations_by_severity(self, severity: ViolationSeverity) -> list[Violation]:
        """Get violations filtered by severity."""
        return [
            v for v in self.violations.values() if v.severity == severity
        ]

    def get_violations_for_client(self, client_id: str) -> list[Violation]:
        """Get all violations for a specific client."""
        return [v for v in self.violations.values() if v.client_id == client_id]

    def generate_ethics_report(self) -> EthicsReport:
        """Generate a comprehensive ethics and compliance report."""
        active = self.get_active_violations()
        resolved = [v for v in self.violations.values() if v.resolved]

        violations_by_severity: dict[ViolationSeverity, int] = {}
        for v in active:
            violations_by_severity[v.severity] = (
                violations_by_severity.get(v.severity, 0) + 1
            )

        violations_by_principle: dict[EthicalPrinciple, int] = {}
        for v in active:
            violations_by_principle[v.principle] = (
                violations_by_principle.get(v.principle, 0) + 1
            )

        suitable_count = sum(
            1 for a in self.suitability_assessments.values() if a.suitable
        )
        unsuitable_count = sum(
            1 for a in self.suitability_assessments.values() if not a.suitable
        )

        return EthicsReport(
            total_clients=len(self.clients),
            total_products=len(self.products),
            total_rules=len(self.rules),
            active_violations=len(active),
            resolved_violations=len(resolved),
            unsuitable_count=unsuitable_count,
            suitable_count=suitable_count,
            violations_by_severity=violations_by_severity,
            violations_by_principle=violations_by_principle,
        )

    def monitor_compliance(self) -> dict[str, Any]:
        """Monitor overall compliance status.

        Returns a compliance status dictionary with key metrics.
        """
        active = self.get_active_violations()
        critical = [v for v in active if v.severity == ViolationSeverity.CRITICAL]
        high = [v for v in active if v.severity == ViolationSeverity.HIGH]

        if critical:
            status = "NON_COMPLIANT"
        elif high:
            status = "REVIEW_REQUIRED"
        elif active:
            status = "COMPLIANT_WITH_WARNINGS"
        else:
            status = "COMPLIANT"

        return {
            "compliance_status": status,
            "active_violations": len(active),
            "critical_violations": len(critical),
            "high_violations": len(high),
            "requires_immediate_action": len(critical) > 0,
            "monitored_at": datetime.now().isoformat(),
        }
