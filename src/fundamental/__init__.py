"""Fundamental financial statement analysis engine."""
from .analysis import Anomaly, FundamentalAnalysisEngine, benford_chi2

__all__ = ["Anomaly", "FundamentalAnalysisEngine", "benford_chi2"]
