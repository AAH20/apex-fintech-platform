"""Decision analysis engine — decision trees, real options, scenario analysis.

Implements CFA Institute decision analysis frameworks:
- Decision trees with expected value maximization
- Real options valuation (expand, abandon, defer)
- Scenario analysis with probability-weighted NPV
- Sensitivity analysis (one-way, two-way, tornado)
- Binomial option pricing
"""

from .analysis import (
    DecisionAnalysisEngine,
    DecisionNode,
    RealOption,
    Scenario,
    SensitivityResult,
)

__all__ = [
    "DecisionAnalysisEngine",
    "DecisionNode",
    "RealOption",
    "Scenario",
    "SensitivityResult",
]
