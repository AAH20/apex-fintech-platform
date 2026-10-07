"""Insurtech analytics engine for pricing, reserving, and risk pooling."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class MortalityTable:
    """Mortality table with ages and q_x (probability of death within one year)."""

    ages: NDArray[np.int64]
    q_x: NDArray[np.float64]

    def __post_init__(self):
        if len(self.ages) != len(self.q_x):
            raise ValueError("ages and q_x must have same length")
        if np.any(self.q_x < 0) or np.any(self.q_x > 1):
            raise ValueError("q_x must be between 0 and 1")

    def q_at_age(self, age: int) -> float:
        """Get q_x for a given age, interpolating if necessary."""
        if age in self.ages:
            idx = np.where(self.ages == age)[0][0]
            return float(self.q_x[idx])
        # Linear interpolation
        return float(np.interp(age, self.ages, self.q_x))

    def survival_probability(self, age: int, years: int) -> float:
        """Probability of surviving `years` years from `age`."""
        prob = 1.0
        for t in range(years):
            q = self.q_at_age(age + t)
            prob *= (1 - q)
        return prob

    def death_probability(self, age: int, years: int) -> float:
        """Probability of dying within `years` years from `age`."""
        return 1.0 - self.survival_probability(age, years)


@dataclass
class Policy:
    """Insurance policy parameters."""

    sum_insured: float
    age: int
    term_years: int
    mortality_table: MortalityTable


@dataclass
class ReserveResult:
    """Result of a reserve calculation."""

    ultimate: NDArray[np.float64]
    reserve: float
    triangle: NDArray[np.float64]
    development_factors: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class RiskPoolResult:
    """Result of risk pooling calculation."""

    pooled_variance: float
    diversification_benefit: float
    coefficient_of_variation: float


class InsurtechEngine:
    """Main insurtech analytics engine."""

    def __init__(
        self,
        mortality_table: MortalityTable,
        interest_rate: float = 0.05,
    ):
        self.mortality_table = mortality_table
        self.interest_rate = interest_rate

    # ------------------------------------------------------------------
    # Premium Calculation
    # ------------------------------------------------------------------

    def calculate_net_premium(self, policy: Policy) -> float:
        """Calculate net premium (expected present value of claims)."""
        premium = 0.0
        survival_prob = 1.0
        for t in range(policy.term_years):
            q = self.mortality_table.q_at_age(policy.age + t)
            discount = 1.0 / (1.0 + self.interest_rate) ** (t + 1)
            premium += policy.sum_insured * survival_prob * q * discount
            survival_prob *= (1 - q)
        return premium

    def calculate_gross_premium(self, policy: Policy, loading_factor: float) -> float:
        """Calculate gross premium = net premium * (1 + loading_factor)."""
        net = self.calculate_net_premium(policy)
        return net * (1.0 + loading_factor)

    # ------------------------------------------------------------------
    # Reserving
    # ------------------------------------------------------------------

    def chain_ladder(self, triangle: NDArray[np.float64]) -> ReserveResult:
        """Chain ladder method for reserve calculation."""
        triangle = np.array(triangle, dtype=np.float64)
        n = triangle.shape[0]

        # Calculate age-to-age development factors
        factors = []
        for j in range(n - 1):
            # Sum of column j (excluding last row)
            sum_col_j = np.sum(triangle[: n - 1 - j, j])
            # Sum of column j+1 (excluding last row)
            sum_col_j1 = np.sum(triangle[: n - 1 - j, j + 1])
            if sum_col_j > 0:
                factors.append(sum_col_j1 / sum_col_j)
            else:
                factors.append(1.0)

        # Project ultimate losses
        ultimate = np.zeros(n)
        latest_diagonal = np.array([triangle[i, n - 1 - i] for i in range(n)])

        for i in range(n):
            ult = latest_diagonal[i]
            for k in range(n - 1 - i):
                ult *= factors[k]
            ultimate[i] = ult

        # Build full projected triangle
        projected = np.zeros_like(triangle)
        for i in range(n):
            for j in range(n):
                if j >= n - 1 - i:
                    # On or above diagonal
                    projected[i, j] = ultimate[i]
                else:
                    projected[i, j] = triangle[i, j]

        reserve = float(np.sum(ultimate - latest_diagonal))

        return ReserveResult(
            ultimate=ultimate,
            reserve=reserve,
            triangle=projected,
            development_factors=np.array(factors),
        )

    def bornhuetter_ferguson(
        self,
        triangle: NDArray[np.float64],
        expected_losses: NDArray[np.float64],
        weight: float = 0.5,
    ) -> ReserveResult:
        """Bornhuetter-Ferguson method blending chain ladder and expected losses."""
        triangle = np.array(triangle, dtype=np.float64)
        expected_losses = np.array(expected_losses, dtype=np.float64)
        n = triangle.shape[0]

        # Get chain ladder result
        cl_result = self.chain_ladder(triangle)

        # Latest diagonal (paid to date)
        latest_diagonal = np.array([triangle[i, n - 1 - i] for i in range(n)])

        # BF ultimate = weight * CL_ultimate + (1 - weight) * expected
        bf_ultimate = weight * cl_result.ultimate + (1 - weight) * expected_losses

        reserve = float(np.sum(bf_ultimate - latest_diagonal))

        return ReserveResult(
            ultimate=bf_ultimate,
            reserve=reserve,
            triangle=cl_result.triangle,
            development_factors=cl_result.development_factors,
        )

    def loss_ratio_method(
        self,
        earned_premium: NDArray[np.float64],
        paid_losses: NDArray[np.float64],
        expected_loss_ratio: float,
    ) -> ReserveResult:
        """Loss ratio method: reserve = expected_loss_ratio * earned_premium - paid."""
        earned_premium = np.array(earned_premium, dtype=np.float64)
        paid_losses = np.array(paid_losses, dtype=np.float64)

        ultimate = expected_loss_ratio * earned_premium
        reserve = float(np.sum(ultimate - paid_losses))

        return ReserveResult(
            ultimate=ultimate,
            reserve=reserve,
            triangle=np.array([]),
            development_factors=np.array([]),
        )

    # ------------------------------------------------------------------
    # Risk Pooling
    # ------------------------------------------------------------------

    def risk_pool(
        self,
        individual_variances: NDArray[np.float64],
        correlation: float = 0.0,
    ) -> RiskPoolResult:
        """Calculate pooled risk metrics.

        The pooled variance is the variance of the average loss per risk:
        Var(average) = (σ²/n) * [1 + (n-1)*ρ]

        For ρ=0: Var(average) = σ²/n
        For ρ=1: Var(average) = σ²
        """
        individual_variances = np.array(individual_variances, dtype=np.float64)
        n = len(individual_variances)

        if n == 0:
            raise ValueError("individual_variances cannot be empty")

        # Average variance
        avg_variance = np.mean(individual_variances)

        # Pooled variance (variance of the average)
        pooled_variance = (avg_variance / n) * (1 + (n - 1) * correlation)

        # Diversification benefit = sum of individual variances - n * pooled_variance
        # This represents the reduction in total variance from pooling
        total_individual_variance = np.sum(individual_variances)
        diversification_benefit = total_individual_variance - n * pooled_variance

        # Coefficient of variation
        mean_variance = np.mean(individual_variances)
        if mean_variance > 0:
            cv = np.sqrt(pooled_variance) / mean_variance
        else:
            cv = 0.0

        return RiskPoolResult(
            pooled_variance=float(pooled_variance),
            diversification_benefit=float(diversification_benefit),
            coefficient_of_variation=float(cv),
        )
