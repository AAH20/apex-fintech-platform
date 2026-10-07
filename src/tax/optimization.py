"""Tax optimization engine for high-net-worth individuals.

Implements tax-loss harvesting, estate planning, and charitable giving
strategies with CFA Institute-aligned tax optimization principles.

References:
    CFA Institute. (2024). Taxation of Investments.
    IRS Publication 559: Survivors, Executors, and Administrators.
    IRS Publication 526: Charitable Contributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TaxBracket:
    """Represents a single tax bracket."""

    lower: float
    upper: float
    rate: float

    @classmethod
    def single_2024(cls) -> list[TaxBracket]:
        """2024 federal tax brackets for single filers."""
        return [
            cls(0, 11_600, 0.10),
            cls(11_600, 47_150, 0.12),
            cls(47_150, 100_525, 0.22),
            cls(100_525, 191_950, 0.24),
            cls(191_950, 243_725, 0.32),
            cls(243_725, 609_350, 0.35),
            cls(609_350, math.inf, 0.37),
        ]

    @classmethod
    def married_filing_jointly_2024(cls) -> list[TaxBracket]:
        """2024 federal tax brackets for married filing jointly."""
        return [
            cls(0, 23_200, 0.10),
            cls(23_200, 94_300, 0.12),
            cls(94_300, 201_050, 0.22),
            cls(201_050, 383_900, 0.24),
            cls(383_900, 487_450, 0.32),
            cls(487_450, 731_200, 0.35),
            cls(731_200, math.inf, 0.37),
        ]

    @staticmethod
    def compute_tax(income: float, brackets: list[TaxBracket]) -> float:
        """Compute progressive tax owed given income and brackets."""
        if income <= 0:
            return 0.0

        tax = 0.0
        remaining = income
        for bracket in brackets:
            if remaining <= 0:
                break
            taxable_in_bracket = min(remaining, bracket.upper - bracket.lower)
            if taxable_in_bracket > 0:
                tax += taxable_in_bracket * bracket.rate
                remaining -= taxable_in_bracket
        return tax

    @staticmethod
    def get_marginal_rate(income: float, brackets: list[TaxBracket]) -> float:
        """Get the marginal tax rate for a given income level."""
        for bracket in brackets:
            if income <= bracket.upper:
                return bracket.rate
        return brackets[-1].rate if brackets else 0.0


@dataclass
class AssetLot:
    """Represents a tax lot of a security holding."""

    symbol: str
    shares: float
    cost_basis: float  # per share
    current_price: float  # per share

    @property
    def cost_basis_total(self) -> float:
        return self.shares * self.cost_basis

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_gain(self) -> float:
        return self.market_value - self.cost_basis_total

    @property
    def unrealized_gain_pct(self) -> float:
        if self.cost_basis_total == 0:
            return 0.0
        return self.unrealized_gain / self.cost_basis_total


@dataclass
class EstatePlan:
    """Estate planning parameters."""

    gross_estate: float
    exemption: float = 13_610_000  # 2024 single exemption
    annual_gifts: float = 0.0
    marital_deduction: bool = False
    charitable_deduction: float = 0.0
    state_estate_tax: float = 0.0


@dataclass
class CharitableGivingStrategy:
    """Charitable giving parameters."""

    agi: float
    cash_donations: float = 0.0
    appreciated_asset_donations: float = 0.0  # FMV
    asset_cost_basis: float = 0.0
    donor_advised_fund_balance: float = 0.0
    intended_annual_giving: float = 0.0


@dataclass
class TaxStrategy:
    """Represents a tax optimization strategy recommendation."""

    strategy_type: str
    description: str
    estimated_savings: float
    priority: int  # 1 = highest
    action_items: list[str] = field(default_factory=list)


@dataclass
class HarvestRecommendation:
    """Tax-loss harvesting recommendation."""

    should_harvest: bool
    estimated_savings: float
    lots_to_harvest: list[AssetLot]
    total_loss: float


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class TaxOptimizationEngine:
    """Tax optimization engine for high-net-worth individuals.

    Provides comprehensive tax planning across:
    - Tax-loss harvesting
    - Estate planning
    - Charitable giving
    - Roth conversion analysis
    - Income shifting
    - Bracket optimization
    """

    # 2024 LTCG brackets (single filer)
    LTCG_BRACKETS_SINGLE = [
        (0, 47_025, 0.0),
        (47_025, 518_900, 0.15),
        (518_900, math.inf, 0.20),
    ]

    # 2024 LTCG brackets (MFJ)
    LTCG_BRACKETS_MFJ = [
        (0, 94_050, 0.0),
        (94_050, 583_750, 0.15),
        (583_750, math.inf, 0.20),
    ]

    # Annual gift exclusion (2024)
    ANNUAL_GIFT_EXCLUSION = 18_000

    # Estate tax rate
    ESTATE_TAX_RATE = 0.40

    # Charitable deduction limits (% of AGI)
    CASH_DEDUCTION_LIMIT = 0.60
    APPRECIATED_ASSET_DEDUCTION_LIMIT = 0.30

    def __init__(
        self,
        filing_status: str = "single",
        state_ca: bool = False,
        ordinary_income: float = 0.0,
        state_tax_rate: float = 0.0,
    ) -> None:
        """Initialize the tax optimization engine.

        Args:
            filing_status: "single" or "married_jointly"
            state_ca: Whether subject to California state tax
            ordinary_income: Annual ordinary income
            state_tax_rate: State marginal tax rate
        """
        if filing_status not in ("single", "married_jointly"):
            raise ValueError("filing_status must be 'single' or 'married_jointly'")

        self.filing_status = filing_status
        self.state_ca = state_ca
        self.ordinary_income = ordinary_income
        self.state_tax_rate = state_tax_rate

        # Initialize brackets
        if filing_status == "single":
            self.brackets = TaxBracket.single_2024()
            self.ltcg_brackets = self.LTCG_BRACKETS_SINGLE
            self.exemption = 13_610_000
        else:
            self.brackets = TaxBracket.married_filing_jointly_2024()
            self.ltcg_brackets = self.LTCG_BRACKETS_MFJ
            self.exemption = 27_220_000

        # Track sales for wash sale detection
        self._sales_history: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Tax-Loss Harvesting
    # -----------------------------------------------------------------------

    def identify_loss_lots(self, lots: list[AssetLot]) -> list[AssetLot]:
        """Identify lots with unrealized losses.

        Args:
            lots: List of asset lots to analyze

        Returns:
            List of lots with negative unrealized gains
        """
        return [lot for lot in lots if lot.unrealized_gain < 0]

    def calculate_harvest_savings(
        self, lots: list[AssetLot], ordinary_income_rate: float | None = None
    ) -> float:
        """Calculate tax savings from harvesting losses.

        Args:
            lots: Lots to harvest
            ordinary_income_rate: Marginal rate (auto-detected if None)

        Returns:
            Estimated tax savings
        """
        if ordinary_income_rate is None:
            ordinary_income_rate = self.get_marginal_rate()

        total_loss = sum(abs(lot.unrealized_gain) for lot in lots if lot.unrealized_gain < 0)
        return total_loss * ordinary_income_rate

    def record_sale(
        self,
        symbol: str,
        shares: float,
        sale_price: float,
        cost_basis: float,
        days_ago: int = 0,
    ) -> None:
        """Record a sale for wash sale tracking.

        Args:
            symbol: Security symbol
            shares: Number of shares sold
            sale_price: Price per share at sale
            cost_basis: Original cost basis per share
            days_ago: How many days ago the sale occurred
        """
        self._sales_history.append(
            {
                "symbol": symbol,
                "shares": shares,
                "sale_price": sale_price,
                "cost_basis": cost_basis,
                "days_ago": days_ago,
                "loss": (sale_price - cost_basis) * shares,
            }
        )

    def is_wash_sale(self, symbol: str, days_after_sale: int = 30) -> bool:
        """Check if a repurchase would trigger wash sale rules.

        Args:
            symbol: Security to check
            days_after_sale: Days since the sale

        Returns:
            True if wash sale rules apply
        """
        for sale in self._sales_history:
            if sale["symbol"] == symbol and sale["loss"] < 0:
                if days_after_sale <= 30:
                    return True
        return False

    def net_capital_gains(self, lots: list[AssetLot]) -> float:
        """Calculate net capital gains across all lots.

        Args:
            lots: List of asset lots

        Returns:
            Net capital gains (positive = net gain, negative = net loss)
        """
        return sum(lot.unrealized_gain for lot in lots)

    def recommend_harvest(
        self, lots: list[AssetLot], min_loss_threshold: float = 1_000
    ) -> HarvestRecommendation:
        """Generate tax-loss harvesting recommendation.

        Args:
            lots: Portfolio lots to analyze
            min_loss_threshold: Minimum loss to trigger recommendation

        Returns:
            HarvestRecommendation with analysis
        """
        loss_lots = self.identify_loss_lots(lots)
        total_loss = sum(abs(lot.unrealized_gain) for lot in loss_lots)

        if total_loss < min_loss_threshold:
            return HarvestRecommendation(
                should_harvest=False,
                estimated_savings=0.0,
                lots_to_harvest=[],
                total_loss=total_loss,
            )

        marginal_rate = self.get_marginal_rate()
        estimated_savings = total_loss * marginal_rate

        return HarvestRecommendation(
            should_harvest=True,
            estimated_savings=estimated_savings,
            lots_to_harvest=loss_lots,
            total_loss=total_loss,
        )

    # -----------------------------------------------------------------------
    # Estate Planning
    # -----------------------------------------------------------------------

    def compute_estate_tax(self, plan: EstatePlan) -> float:
        """Compute federal estate tax.

        Args:
            plan: Estate plan parameters

        Returns:
            Federal estate tax owed
        """
        if plan.marital_deduction:
            return 0.0

        taxable_estate = (
            plan.gross_estate
            - plan.exemption
            - plan.annual_gifts
            - plan.charitable_deduction
        )
        taxable_estate = max(0, taxable_estate)

        federal_tax = taxable_estate * self.ESTATE_TAX_RATE
        return federal_tax + plan.state_estate_tax

    def recommend_estate_strategies(self, plan: EstatePlan) -> list[TaxStrategy]:
        """Generate estate planning strategies.

        Args:
            plan: Estate plan parameters

        Returns:
            List of recommended strategies
        """
        strategies = []
        taxable_estate = plan.gross_estate - plan.exemption - plan.annual_gifts

        if taxable_estate > 0:
            # Annual gifting strategy
            annual_gift_capacity = self.ANNUAL_GIFT_EXCLUSION * 2  # Per couple
            strategies.append(
                TaxStrategy(
                    strategy_type="annual_gifting",
                    description=f"Gift up to ${annual_gift_capacity:,.0f} annually to reduce taxable estate",
                    estimated_savings=annual_gift_capacity * self.ESTATE_TAX_RATE,
                    priority=1,
                    action_items=[
                        "Set up annual exclusion gifts to heirs",
                        "Consider 529 plan superfunding (5-year election)",
                        "Document all gifts for IRS reporting",
                    ],
                )
            )

            # Irrevocable trust
            strategies.append(
                TaxStrategy(
                    strategy_type="irrevocable_trust",
                    description="Transfer assets to irrevocable trust to remove from estate",
                    estimated_savings=taxable_estate * 0.15,
                    priority=2,
                    action_items=[
                        "Establish irrevocable life insurance trust (ILIT)",
                        "Consider grantor retained annuity trust (GRAT)",
                        "Evaluate qualified personal residence trust (QPRT)",
                    ],
                )
            )

            # Charitable remainder trust
            strategies.append(
                TaxStrategy(
                    strategy_type="charitable_remainder_trust",
                    description="CRT provides income stream and reduces estate",
                    estimated_savings=taxable_estate * 0.10,
                    priority=3,
                    action_items=[
                        "Fund CRT with appreciated assets",
                        "Receive charitable income tax deduction",
                        "Heirs receive remainder after trust term",
                    ],
                )
            )

        return strategies

    # -----------------------------------------------------------------------
    # Charitable Giving
    # -----------------------------------------------------------------------

    def compute_charitable_deduction(self, strategy: CharitableGivingStrategy) -> float:
        """Compute total charitable deduction.

        Args:
            strategy: Charitable giving parameters

        Returns:
            Total deductible amount
        """
        # Cash donations capped at 60% of AGI
        cash_deduction = min(
            strategy.cash_donations,
            strategy.agi * self.CASH_DEDUCTION_LIMIT,
        )

        # Appreciated assets capped at 30% of AGI
        asset_deduction = min(
            strategy.appreciated_asset_donations,
            strategy.agi * self.APPRECIATED_ASSET_DEDUCTION_LIMIT,
        )

        return cash_deduction + asset_deduction

    def compute_charitable_tax_savings(
        self, strategy: CharitableGivingStrategy, ltcg_rate: float = 0.20
    ) -> float:
        """Compute total tax savings from charitable giving.

        Args:
            strategy: Charitable giving parameters
            ltcg_rate: Long-term capital gains tax rate

        Returns:
            Total tax savings
        """
        deduction = self.compute_charitable_deduction(strategy)
        # Use strategy AGI for marginal rate if engine income not set
        income_for_rate = self.ordinary_income if self.ordinary_income > 0 else strategy.agi
        marginal_rate = TaxBracket.get_marginal_rate(income_for_rate, self.brackets)

        # Income tax savings from deduction
        income_tax_savings = deduction * marginal_rate

        # Capital gains tax avoided on appreciated assets
        if strategy.appreciated_asset_donations > 0:
            gain = strategy.appreciated_asset_donations - strategy.asset_cost_basis
            cg_tax_avoided = gain * ltcg_rate
        else:
            cg_tax_avoided = 0.0

        return income_tax_savings + cg_tax_avoided

    def recommend_charitable_strategy(
        self, strategy: CharitableGivingStrategy
    ) -> dict[str, Any]:
        """Generate charitable giving recommendations.

        Args:
            strategy: Charitable giving parameters

        Returns:
            Dictionary with recommendations
        """
        agi = strategy.agi
        use_daf = agi > 1_000_000 or strategy.intended_annual_giving > 50_000
        bunching_recommended = agi > 500_000

        recommendations = {
            "use_donor_advised_fund": use_daf,
            "bunching_recommended": bunching_recommended,
            "optimal_donation_type": "appreciated_assets" if strategy.appreciated_asset_donations > 0 else "cash",
            "estimated_deduction": self.compute_charitable_deduction(strategy),
            "estimated_savings": self.compute_charitable_tax_savings(strategy),
        }

        if bunching_recommended:
            recommendations["bunching_strategy"] = (
                "Bunch multiple years of donations into one year to exceed standard deduction"
            )

        if use_daf:
            recommendations["daf_strategy"] = (
                "Fund DAF in high-income years, grant to charities over time"
            )

        return recommendations

    # -----------------------------------------------------------------------
    # Roth Conversion Analysis
    # -----------------------------------------------------------------------

    def analyze_roth_conversion(self, conversion_amount: float) -> dict[str, Any]:
        """Analyze the tax impact of a Roth conversion.

        Args:
            conversion_amount: Amount to convert from traditional to Roth

        Returns:
            Analysis results
        """
        new_income = self.ordinary_income + conversion_amount
        current_tax = TaxBracket.compute_tax(self.ordinary_income, self.brackets)
        new_tax = TaxBracket.compute_tax(new_income, self.brackets)

        additional_tax = new_tax - current_tax
        marginal_rate = additional_tax / conversion_amount if conversion_amount > 0 else 0.0

        return {
            "conversion_amount": conversion_amount,
            "additional_tax": additional_tax,
            "marginal_rate": marginal_rate,
            "current_tax": current_tax,
            "new_tax": new_tax,
            "effective_rate_on_conversion": marginal_rate,
        }

    # -----------------------------------------------------------------------
    # Income Shifting
    # -----------------------------------------------------------------------

    def recommend_income_shifting(self) -> dict[str, Any]:
        """Recommend income shifting strategies.

        Returns:
            Income shifting recommendations
        """
        high_earner_threshold = 400_000
        recommended = self.ordinary_income > high_earner_threshold

        potential_savings = 0.0
        if recommended:
            # Estimate savings from income shifting to family members in lower brackets
            shiftable_income = min(self.ordinary_income * 0.10, 50_000)
            rate_differential = self.get_marginal_rate() - 0.12  # Assume 12% bracket for family
            potential_savings = shiftable_income * max(0, rate_differential)

        return {
            "recommended": recommended,
            "potential_savings": potential_savings,
            "strategies": [
                "Hire family members in family business",
                "Shift investment income to children (kiddie tax rules apply)",
                "Establish family limited partnership (FLP)",
                "Use irrevocable trusts for income distribution",
            ] if recommended else [],
        }

    # -----------------------------------------------------------------------
    # Bracket Optimization
    # -----------------------------------------------------------------------

    def optimize_bracket_position(self, target_bracket: float) -> dict[str, Any]:
        """Optimize tax bracket positioning.

        Args:
            target_bracket: Target marginal tax rate

        Returns:
            Bracket optimization analysis
        """
        current_rate = self.get_marginal_rate()
        current_bracket = current_rate

        recommendation = None
        if current_rate > target_bracket:
            # Recommend strategies to lower bracket
            recommendation = (
                f"Consider Roth conversions in low-income years, "
                f"accelerate deductions, or harvest losses to drop from "
                f"{current_rate:.0%} to {target_bracket:.0%} bracket"
            )
        elif current_rate < target_bracket:
            recommendation = (
                "Consider accelerating income or Roth conversions to "
                "fill lower brackets before rates increase"
            )

        return {
            "current_bracket": current_bracket,
            "target_bracket": target_bracket,
            "recommendation": recommendation,
            "income": self.ordinary_income,
        }

    # -----------------------------------------------------------------------
    # Capital Gains Tax
    # -----------------------------------------------------------------------

    def compute_capital_gains_tax(
        self,
        gains: float,
        holding_period_years: float,
        ordinary_income: float | None = None,
    ) -> float:
        """Compute capital gains tax.

        Args:
            gains: Capital gains amount
            holding_period_years: Years held
            ordinary_income: Ordinary income for bracket determination

        Returns:
            Capital gains tax owed
        """
        if gains <= 0:
            return 0.0

        if ordinary_income is None:
            ordinary_income = self.ordinary_income

        if holding_period_years < 1:
            # Short-term: taxed as ordinary income
            return TaxBracket.compute_tax(ordinary_income + gains, self.brackets) - TaxBracket.compute_tax(
                ordinary_income, self.brackets
            )

        # Long-term: preferential rates
        ltcg_tax = 0.0
        remaining = gains
        for lower, upper, rate in self.ltcg_brackets:
            if remaining <= 0:
                break
            taxable = min(remaining, upper - lower)
            if taxable > 0:
                ltcg_tax += taxable * rate
                remaining -= taxable

        # Net Investment Income Tax (NIIT) 3.8% for high earners
        niit_threshold = 200_000 if self.filing_status == "single" else 250_000
        if ordinary_income + gains > niit_threshold:
            niit = gains * 0.038
            ltcg_tax += niit

        return ltcg_tax

    # -----------------------------------------------------------------------
    # Comprehensive Optimization
    # -----------------------------------------------------------------------

    def get_marginal_rate(self) -> float:
        """Get current marginal tax rate."""
        return TaxBracket.get_marginal_rate(self.ordinary_income, self.brackets)

    def optimize(self) -> dict[str, Any]:
        """Run basic optimization analysis.

        Returns:
            Optimization results
        """
        return {
            "tax_loss_harvesting": {
                "description": "Harvest tax losses to offset gains",
                "applicable": True,
            },
            "estate_planning": {
                "description": "Estate tax minimization strategies",
                "applicable": self.ordinary_income > 500_000,
            },
            "charitable_giving": {
                "description": "Optimize charitable contributions",
                "applicable": True,
            },
        }

    def optimize_portfolio_tax(
        self,
        lots: list[AssetLot],
        target_allocation: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Optimize portfolio for tax efficiency using numpy vectorized operations.

        Args:
            lots: Portfolio asset lots
            target_allocation: Target portfolio weights (optional)

        Returns:
            Portfolio tax optimization results
        """
        if not lots:
            return {"tax_drag": 0.0, "recommendations": []}

        # Build numpy arrays for vectorized computation
        symbols = np.array([lot.symbol for lot in lots])
        cost_bases = np.array([lot.cost_basis_total for lot in lots])
        market_values = np.array([lot.market_value for lot in lots])
        unrealized_gains = market_values - cost_bases

        # Compute tax drag (unrealized gains * potential tax rate)
        marginal_rate = self.get_marginal_rate()
        tax_drag = np.sum(np.maximum(unrealized_gains, 0)) * marginal_rate

        # Identify lots with losses (harvest candidates)
        loss_mask = unrealized_gains < 0
        loss_lots = symbols[loss_mask]
        total_harvestable_loss = abs(np.sum(unrealized_gains[loss_mask]))

        # Compute portfolio concentration risk
        total_value = np.sum(market_values)
        if total_value > 0:
            weights = market_values / total_value
            max_weight = np.max(weights)
            concentration_risk = max_weight > 0.25  # >25% in single position
        else:
            concentration_risk = False

        recommendations = []
        if total_harvestable_loss > 1_000:
            recommendations.append(
                f"Harvest ${total_harvestable_loss:,.0f} in losses from: {', '.join(loss_lots)}"
            )
        if concentration_risk:
            recommendations.append(
                f"Reduce concentration: largest position is {max_weight:.1%} of portfolio"
            )

        return {
            "tax_drag": float(tax_drag),
            "total_harvestable_loss": float(total_harvestable_loss),
            "loss_lots": loss_lots.tolist(),
            "concentration_risk": bool(concentration_risk),
            "recommendations": recommendations,
            "portfolio_value": float(total_value),
        }

    def comprehensive_optimize(
        self,
        lots: list[AssetLot],
        estate: EstatePlan,
        charitable: CharitableGivingStrategy,
    ) -> dict[str, Any]:
        """Run comprehensive tax optimization.

        Args:
            lots: Portfolio asset lots
            estate: Estate plan parameters
            charitable: Charitable giving strategy

        Returns:
            Complete optimization results
        """
        strategies = []
        total_savings = 0.0

        # Tax-loss harvesting
        harvest_rec = self.recommend_harvest(lots)
        if harvest_rec.should_harvest:
            strategies.append(
                TaxStrategy(
                    strategy_type="tax_loss_harvesting",
                    description=f"Harvest ${harvest_rec.total_loss:,.0f} in losses",
                    estimated_savings=harvest_rec.estimated_savings,
                    priority=1,
                    action_items=[f"Sell {lot.symbol} to realize loss" for lot in harvest_rec.lots_to_harvest],
                )
            )
            total_savings += harvest_rec.estimated_savings

        # Estate planning
        estate_tax = self.compute_estate_tax(estate)
        if estate_tax > 0:
            estate_strategies = self.recommend_estate_strategies(estate)
            for es in estate_strategies:
                strategies.append(es)
                total_savings += es.estimated_savings

        # Charitable giving
        charitable_savings = self.compute_charitable_tax_savings(charitable)
        if charitable_savings > 0:
            strategies.append(
                TaxStrategy(
                    strategy_type="charitable_giving",
                    description="Optimize charitable contributions",
                    estimated_savings=charitable_savings,
                    priority=2,
                    action_items=[
                        "Donate appreciated assets instead of cash",
                        "Consider donor-advised fund for bunching",
                        "Maximize deduction within AGI limits",
                    ],
                )
            )
            total_savings += charitable_savings

        # Roth conversion analysis
        if self.ordinary_income > 200_000:
            roth_analysis = self.analyze_roth_conversion(50_000)
            if roth_analysis["marginal_rate"] < 0.30:
                strategies.append(
                    TaxStrategy(
                        strategy_type="roth_conversion",
                        description="Partial Roth conversion in low bracket years",
                        estimated_savings=roth_analysis["additional_tax"] * 0.20,  # Future tax savings
                        priority=3,
                        action_items=[
                            f"Convert up to ${roth_analysis['conversion_amount']:,.0f} to Roth",
                            f"Current marginal rate: {roth_analysis['marginal_rate']:.1%}",
                        ],
                    )
                )

        return {
            "total_tax_savings": total_savings,
            "strategies": strategies,
            "estate_tax_owed": estate_tax,
            "harvest_recommendation": harvest_rec,
            "charitable_recommendation": self.recommend_charitable_strategy(charitable),
        }
