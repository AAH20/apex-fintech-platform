"""ESG analytics engine for scoring, impact measurement, and greenwashing detection."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ESGDataInput(BaseModel):
    """Input data for ESG analysis."""

    company_id: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    environmental_score: Optional[float] = Field(default=None, ge=0, le=100)
    social_score: Optional[float] = Field(default=None, ge=0, le=100)
    governance_score: Optional[float] = Field(default=None, ge=0, le=100)
    carbon_emissions_tons: Optional[float] = Field(default=None, ge=0)
    revenue_millions: Optional[float] = Field(default=None, ge=0)

    @field_validator("environmental_score", "social_score", "governance_score")
    @classmethod
    def validate_score_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 100):
            raise ValueError("ESG scores must be between 0 and 100")
        return v

    @field_validator("carbon_emissions_tons")
    @classmethod
    def validate_emissions(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Carbon emissions cannot be negative")
        return v


class ESGScore(BaseModel):
    """ESG score output for a single company."""

    company_id: str
    company_name: str
    environmental: float = Field(ge=0, le=100)
    social: float = Field(ge=0, le=100)
    governance: float = Field(ge=0, le=100)
    overall: float = Field(ge=0, le=100)
    carbon_intensity: Optional[float] = None


class ImpactMetrics(BaseModel):
    """Portfolio-level impact metrics."""

    total_carbon_emissions: float = 0.0
    weighted_esg_score: float = 0.0
    portfolio_carbon_intensity: Optional[float] = None
    holdings_count: int = 0


class GreenwashingReport(BaseModel):
    """Greenwashing detection report."""

    company_id: str
    company_name: str
    is_suspicious: bool
    risk_level: str  # LOW, MEDIUM, HIGH
    carbon_intensity: Optional[float]
    esg_score: float
    discrepancy_score: float
    flags: list[str] = Field(default_factory=list)


class ESGAnalyticsEngine:
    """ESG analytics engine for scoring, impact measurement, and greenwashing detection.

    Scoring methodology aligns with MSCI ESG Research and Sustainalytics frameworks:
    - Environmental: 40% weight (climate, pollution, resource use)
    - Social: 30% weight (labor, human capital, community)
    - Governance: 30% weight (board, ethics, transparency)
    """

    # Industry-standard pillar weights (MSCI/Sustainalytics aligned)
    ENV_WEIGHT = 0.40
    SOC_WEIGHT = 0.30
    GOV_WEIGHT = 0.30

    # Default score for missing data (neutral)
    DEFAULT_SCORE = 50.0

    # Greenwashing detection thresholds
    CARBON_INTENSITY_HIGH = 10.0  # tons per $M revenue
    CARBON_INTENSITY_MEDIUM = 5.0
    ESG_SCORE_HIGH = 80.0
    ESG_SCORE_VERY_HIGH = 90.0

    def score_company(self, data: ESGDataInput) -> ESGScore:
        """Calculate ESG score for a single company.

        Uses weighted average of E/S/G pillars. Missing values default to 50.0.
        """
        env = data.environmental_score if data.environmental_score is not None else self.DEFAULT_SCORE
        soc = data.social_score if data.social_score is not None else self.DEFAULT_SCORE
        gov = data.governance_score if data.governance_score is not None else self.DEFAULT_SCORE

        overall = (
            env * self.ENV_WEIGHT
            + soc * self.SOC_WEIGHT
            + gov * self.GOV_WEIGHT
        )

        carbon_intensity = None
        if data.carbon_emissions_tons is not None and data.revenue_millions is not None and data.revenue_millions > 0:
            carbon_intensity = data.carbon_emissions_tons / data.revenue_millions

        return ESGScore(
            company_id=data.company_id,
            company_name=data.company_name,
            environmental=round(env, 2),
            social=round(soc, 2),
            governance=round(gov, 2),
            overall=round(overall, 2),
            carbon_intensity=round(carbon_intensity, 4) if carbon_intensity is not None else None,
        )

    def measure_portfolio_impact(self, holdings: list[dict]) -> ImpactMetrics:
        """Measure portfolio-level ESG impact.

        Args:
            holdings: List of dicts with keys:
                - company_id: str
                - weight: float (portfolio weight, will be normalized)
                - carbon_emissions_tons: float (optional)
                - esg_score: float (optional)
                - revenue_millions: float (optional)

        Returns:
            ImpactMetrics with aggregated portfolio metrics.
        """
        if not holdings:
            return ImpactMetrics()

        # Normalize weights
        total_weight = sum(h.get("weight", 0) for h in holdings)
        if total_weight == 0:
            return ImpactMetrics()

        normalized_holdings = []
        for h in holdings:
            norm_weight = h.get("weight", 0) / total_weight
            normalized_holdings.append({**h, "weight": norm_weight})

        # Aggregate carbon emissions
        total_carbon = sum(
            h.get("carbon_emissions_tons", 0) * h["weight"]
            for h in normalized_holdings
        )

        # Weighted ESG score
        weighted_esg = sum(
            h.get("esg_score", 0) * h["weight"]
            for h in normalized_holdings
        )

        # Portfolio carbon intensity
        total_revenue = sum(
            h.get("revenue_millions", 0) * h["weight"]
            for h in normalized_holdings
        )
        portfolio_carbon_intensity = None
        if total_revenue > 0:
            portfolio_carbon_intensity = total_carbon / total_revenue

        return ImpactMetrics(
            total_carbon_emissions=round(total_carbon, 2),
            weighted_esg_score=round(weighted_esg, 2),
            portfolio_carbon_intensity=round(portfolio_carbon_intensity, 4) if portfolio_carbon_intensity else None,
            holdings_count=len(holdings),
        )

    def detect_greenwashing(self, data: ESGDataInput) -> GreenwashingReport:
        """Detect potential greenwashing by analyzing ESG score vs. actual environmental impact.

        Flags companies with high carbon intensity but high ESG scores as suspicious.
        Risk level increases with the magnitude of discrepancy.

        Reference: EU SFDR Article 8/9, CFA Institute ESG Investing standards.
        """
        score = self.score_company(data)

        flags = []
        is_suspicious = False
        risk_level = "LOW"
        discrepancy_score = 0.0

        carbon_intensity = score.carbon_intensity

        if carbon_intensity is not None:
            # High carbon intensity + high ESG score = potential greenwashing
            if carbon_intensity > self.CARBON_INTENSITY_HIGH and score.overall >= self.ESG_SCORE_HIGH:
                is_suspicious = True
                flags.append("High carbon intensity with high ESG score")

                # Risk level based on carbon intensity magnitude
                if carbon_intensity > self.CARBON_INTENSITY_HIGH * 5:  # > 50 tons/$M
                    risk_level = "HIGH"
                    flags.append("Severe discrepancy between ESG claims and environmental impact")
                elif carbon_intensity > self.CARBON_INTENSITY_HIGH * 2:  # > 20 tons/$M
                    risk_level = "MEDIUM"
                    flags.append("Moderate discrepancy between ESG claims and environmental impact")
                else:
                    risk_level = "LOW"

                # Discrepancy: gap between ESG score and expected score given carbon intensity
                # Higher carbon intensity → lower expected environmental score
                expected_score = max(0.0, 100.0 - (carbon_intensity / self.CARBON_INTENSITY_HIGH) * 25.0)
                discrepancy_score = score.overall - expected_score

            elif carbon_intensity > self.CARBON_INTENSITY_MEDIUM and score.overall >= self.ESG_SCORE_VERY_HIGH:
                is_suspicious = True
                flags.append("Elevated carbon intensity with very high ESG score")
                risk_level = "MEDIUM"
                discrepancy_score = score.overall - 50.0

        # Additional check: very low environmental score but high overall
        if score.environmental < 40 and score.overall > 70:
            is_suspicious = True
            flags.append("Low environmental score but high overall ESG rating")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
            discrepancy_score = max(discrepancy_score, score.overall - score.environmental)

        return GreenwashingReport(
            company_id=data.company_id,
            company_name=data.company_name,
            is_suspicious=is_suspicious,
            risk_level=risk_level,
            carbon_intensity=carbon_intensity,
            esg_score=score.overall,
            discrepancy_score=round(discrepancy_score, 2),
            flags=flags,
        )

    def score_from_dataframe(self, df: pd.DataFrame) -> list[ESGScore]:
        """Score multiple companies from a pandas DataFrame.

        Expected columns: company_id, company_name, environmental_score,
        social_score, governance_score, carbon_emissions_tons, revenue_millions
        """
        results = []
        for _, row in df.iterrows():
            data = ESGDataInput(
                company_id=str(row["company_id"]),
                company_name=str(row["company_name"]),
                environmental_score=float(row["environmental_score"]) if pd.notna(row.get("environmental_score")) else None,
                social_score=float(row["social_score"]) if pd.notna(row.get("social_score")) else None,
                governance_score=float(row["governance_score"]) if pd.notna(row.get("governance_score")) else None,
                carbon_emissions_tons=float(row["carbon_emissions_tons"]) if pd.notna(row.get("carbon_emissions_tons")) else None,
                revenue_millions=float(row["revenue_millions"]) if pd.notna(row.get("revenue_millions")) else None,
            )
            results.append(self.score_company(data))
        return results
