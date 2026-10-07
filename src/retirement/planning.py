"""Retirement planning engine with Monte Carlo simulation."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation."""

    final_balances: np.ndarray
    success_rate: float
    percentile_10: float
    percentile_50: float
    percentile_90: float
    median_balances: Optional[np.ndarray] = None


@dataclass
class WithdrawalStrategy:
    """Withdrawal strategy for retirement."""

    name: str
    initial_rate: float
    floor_rate: float
    ceiling_rate: float
    guardrail_threshold: float = 0.0

    def calculate_withdrawal(
        self,
        current_portfolio: float,
        initial_portfolio: float,
        base_withdrawal: float,
    ) -> float:
        """Calculate the withdrawal amount for the current year."""
        if self.name == "fixed_percentage":
            return self.initial_rate * current_portfolio

        # Guardrails strategy
        ratio = current_portfolio / initial_portfolio
        threshold = self.guardrail_threshold

        if ratio > 1 + threshold:
            # Portfolio up significantly - increase withdrawal
            withdrawal = base_withdrawal * (ratio / (1 + threshold))
        elif ratio < 1 - threshold:
            # Portfolio down significantly - decrease withdrawal to floor
            withdrawal = self.floor_rate * current_portfolio
        else:
            withdrawal = base_withdrawal

        # Clamp to floor and ceiling
        floor = self.floor_rate * current_portfolio
        ceiling = self.ceiling_rate * current_portfolio
        return max(floor, min(ceiling, withdrawal))


@dataclass
class RetirementPlan:
    """Complete retirement plan."""

    monte_carlo: MonteCarloResult
    social_security_age: int
    withdrawal_strategy: WithdrawalStrategy
    retirement_age: int
    success_rate: float
    projected_balances: np.ndarray


class RetirementPlanningEngine:
    """Engine for retirement planning with Monte Carlo simulation."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run_monte_carlo(
        self,
        n_simulations: int,
        annual_return_mean: float,
        annual_return_std: float,
        **params,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if annual_return_std < 0:
            raise ValueError("annual_return_std must be non-negative")

        current_age = params["current_age"]
        retirement_age = params["retirement_age"]
        life_expectancy = params["life_expectancy"]
        current_savings = params["current_savings"]
        annual_contribution = params["annual_contribution"]
        annual_retirement_spending = params["annual_retirement_spending"]
        social_security_age = params["social_security_age"]
        social_security_monthly = params["social_security_monthly"]

        total_years = life_expectancy - current_age
        accumulation_years = retirement_age - current_age
        distribution_years = life_expectancy - retirement_age

        # Generate random returns for all simulations at once
        returns = self.rng.normal(
            annual_return_mean,
            annual_return_std,
            (n_simulations, total_years),
        )

        # Initialize portfolios
        portfolios = np.full(n_simulations, current_savings, dtype=float)

        # Track median balances at each year
        median_balances = []

        # Accumulation phase
        for year in range(accumulation_years):
            portfolios = portfolios * (1 + returns[:, year]) + annual_contribution
            median_balances.append(np.median(portfolios))

        # Distribution phase
        for year in range(distribution_years):
            age = retirement_age + year
            spending = annual_retirement_spending
            if age >= social_security_age:
                spending -= social_security_monthly * 12
            portfolios = (
                portfolios * (1 + returns[:, accumulation_years + year]) - spending
            )
            median_balances.append(np.median(portfolios))

        # Calculate results
        final_balances = portfolios
        success_rate = float(np.mean(final_balances > 0))
        percentile_10 = float(np.percentile(final_balances, 10))
        percentile_50 = float(np.percentile(final_balances, 50))
        percentile_90 = float(np.percentile(final_balances, 90))

        return MonteCarloResult(
            final_balances=final_balances,
            success_rate=success_rate,
            percentile_10=percentile_10,
            percentile_50=percentile_50,
            percentile_90=percentile_90,
            median_balances=np.array(median_balances),
        )

    def optimize_social_security(
        self,
        current_age: int,
        life_expectancy: int,
        monthly_benefit_at_62: float,
        full_retirement_age: int,
        monthly_benefit_at_fra: float,
        delayed_retirement_credit: float,
    ) -> dict:
        """Optimize Social Security claiming age."""
        best_age = 62
        best_total = 0.0

        for age in range(62, 71):
            if age < full_retirement_age:
                monthly = monthly_benefit_at_62
            elif age == full_retirement_age:
                monthly = monthly_benefit_at_fra
            else:
                years_delayed = age - full_retirement_age
                monthly = monthly_benefit_at_fra * (
                    1 + delayed_retirement_credit
                ) ** years_delayed

            years_receiving = life_expectancy - age
            total = monthly * 12 * years_receiving

            if total > best_total:
                best_total = total
                best_age = age

        return {"optimal_claiming_age": best_age, "total_benefit": best_total}

    def generate_plan(
        self,
        n_simulations: int,
        annual_return_mean: float,
        annual_return_std: float,
        **params,
    ) -> RetirementPlan:
        """Generate a complete retirement plan."""
        # Run Monte Carlo
        mc_result = self.run_monte_carlo(
            n_simulations=n_simulations,
            annual_return_mean=annual_return_mean,
            annual_return_std=annual_return_std,
            **params,
        )

        # Optimize Social Security
        ss_result = self.optimize_social_security(
            current_age=params["current_age"],
            life_expectancy=params["life_expectancy"],
            monthly_benefit_at_62=params["social_security_monthly"] * 0.75,
            full_retirement_age=67,
            monthly_benefit_at_fra=params["social_security_monthly"],
            delayed_retirement_credit=0.08,
        )

        # Create withdrawal strategy
        withdrawal_strategy = WithdrawalStrategy(
            name="guardrails",
            initial_rate=0.04,
            floor_rate=0.03,
            ceiling_rate=0.05,
            guardrail_threshold=0.20,
        )

        return RetirementPlan(
            monte_carlo=mc_result,
            social_security_age=ss_result["optimal_claiming_age"],
            withdrawal_strategy=withdrawal_strategy,
            retirement_age=params["retirement_age"],
            success_rate=mc_result.success_rate,
            projected_balances=mc_result.median_balances,
        )
