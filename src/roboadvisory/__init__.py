"""Robo-advisory engine: risk profiling, goal-based investing, tax-loss harvesting."""

from .engine import (
    EQUITY_ASSETS,
    EXPECTED_RETURNS,
    COVARIANCE,
    ASSET_CLASSES,
    Goal,
    GoalPlan,
    InvestmentPlan,
    RebalanceTrade,
    RiskProfile,
    RiskTolerance,
    RoboAdvisoryEngine,
    TaxLot,
    TaxLossOpportunity,
)

__all__ = [
    "ASSET_CLASSES",
    "COVARIANCE",
    "EQUITY_ASSETS",
    "EXPECTED_RETURNS",
    "Goal",
    "GoalPlan",
    "InvestmentPlan",
    "RebalanceTrade",
    "RiskProfile",
    "RiskTolerance",
    "RoboAdvisoryEngine",
    "TaxLot",
    "TaxLossOpportunity",
]
