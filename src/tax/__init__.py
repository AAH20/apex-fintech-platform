"""Tax optimization engine for high-net-worth individuals.

Implements tax-loss harvesting, estate planning, and charitable giving
strategies with CFA Institute-aligned tax optimization principles.
"""

from .optimization import (
    AssetLot,
    CharitableGivingStrategy,
    EstatePlan,
    TaxBracket,
    TaxOptimizationEngine,
    TaxStrategy,
)

__all__ = [
    "AssetLot",
    "CharitableGivingStrategy",
    "EstatePlan",
    "TaxBracket",
    "TaxOptimizationEngine",
    "TaxStrategy",
]
