"""Decision analysis engine — decision trees, real options, scenario analysis.

Implements CFA Institute decision analysis frameworks:
- Decision trees with expected value maximization
- Real options valuation (expand, abandon, defer)
- Scenario analysis with probability-weighted NPV
- Sensitivity analysis (one-way, two-way, tornado)
- Binomial option pricing
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class DecisionNode:
    """A node in a decision tree.

    node_type: "decision" (choose max), "chance" (probability-weighted), or "terminal" (leaf value).
    """

    name: str
    node_type: str  # "decision" | "chance" | "terminal"
    children: list[DecisionNode] = field(default_factory=list)
    value: float | None = None  # For terminal nodes
    probability: float | None = None  # For chance node children


@dataclass
class RealOption:
    """A real option valuation input.

    option_type: "expand" | "abandon" | "defer"
    underlying_value: Current value of the project/asset
    strike_cost: Cost to exercise (expansion cost, salvage value, etc.)
    upside_value: Value in the good state
    downside_value: Value in the bad state
    prob_up: Probability of the good state
    prob_down: Probability of the bad state
    """

    option_type: str
    underlying_value: float
    strike_cost: float
    upside_value: float
    downside_value: float
    prob_up: float
    prob_down: float


@dataclass
class Scenario:
    """A scenario for scenario analysis."""

    name: str
    probability: float
    cash_flows: list[float]


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""

    impacts: dict[str, np.ndarray]  # param_name -> array of output values
    ranking: list[str]  # param names sorted by impact (descending)


class DecisionAnalysisEngine:
    """Engine for investment decision analysis.

    Combines decision trees, real options, scenario analysis, and sensitivity
    analysis to evaluate investment decisions under uncertainty.
    """

    # -----------------------------------------------------------------------
    # Decision Trees
    # -----------------------------------------------------------------------

    def evaluate_decision_tree(self, node: DecisionNode) -> dict[str, Any]:
        """Evaluate a decision tree and return the optimal expected value.

        Returns:
            dict with keys:
                - expected_value: float
                - optimal_path: list of node names from root to optimal leaf
        """
        ev, path = self._evaluate_node(node)
        return {"expected_value": ev, "optimal_path": path}

    def _evaluate_node(self, node: DecisionNode) -> tuple[float, list[str]]:
        """Recursively evaluate a node, returning (expected_value, path)."""
        if node.node_type == "terminal":
            if node.value is None:
                raise ValueError(f"Terminal node '{node.name}' must have a value")
            return node.value, [node.name]

        if node.node_type == "decision":
            if not node.children:
                raise ValueError(f"Decision node '{node.name}' must have children")
            best_ev = -math.inf
            best_path: list[str] = []
            for child in node.children:
                ev, path = self._evaluate_node(child)
                if ev > best_ev:
                    best_ev = ev
                    best_path = path
            return best_ev, [node.name] + best_path

        if node.node_type == "chance":
            if not node.children:
                raise ValueError(f"Chance node '{node.name}' must have children")
            # Validate probabilities
            probs = [c.probability for c in node.children]
            if any(p is None for p in probs):
                raise ValueError(
                    f"Chance node '{node.name}' children must have probabilities"
                )
            total = sum(probs)  # type: ignore[arg-type]
            if not math.isclose(total, 1.0, abs_tol=1e-9):
                raise ValueError(
                    f"Chance node '{node.name}' probabilities must sum to 1, got {total}"
                )
            ev = 0.0
            for child in node.children:
                child_ev, _ = self._evaluate_node(child)
                ev += child.probability * child_ev  # type: ignore[operator]
            # For chance nodes, we don't track a single optimal path
            return ev, [node.name]

        raise ValueError(f"Unknown node type: {node.node_type}")

    # -----------------------------------------------------------------------
    # Real Options
    # -----------------------------------------------------------------------

    def value_real_option(self, option: RealOption) -> dict[str, Any]:
        """Value a real option using a binomial-like framework.

        Returns:
            dict with keys:
                - option_value: float
                - exercise: bool (whether to exercise the option)
                - option_premium: float (option_value - underlying_value)
        """
        if option.option_type == "expand":
            # In each state, choose max(expanded_value - strike, underlying)
            up_val = max(option.upside_value - option.strike_cost, option.underlying_value)
            down_val = max(option.downside_value - option.strike_cost, option.underlying_value)
        elif option.option_type == "abandon":
            # In each state, choose max(project_value, salvage)
            up_val = max(option.upside_value, option.strike_cost)
            down_val = max(option.downside_value, option.strike_cost)
        elif option.option_type == "defer":
            # In each state, choose max(future_value, 0) — invest only if positive
            up_val = max(option.upside_value, 0.0)
            down_val = max(option.downside_value, 0.0)
        else:
            raise ValueError(f"Unknown option type: {option.option_type}")

        ev_exercise = option.prob_up * up_val + option.prob_down * down_val
        option_value = max(ev_exercise, option.underlying_value)
        exercise = option_value > option.underlying_value
        option_premium = option_value - option.underlying_value

        return {
            "option_value": option_value,
            "exercise": exercise,
            "option_premium": option_premium,
        }

    def binomial_option_price(
        self,
        spot: float,
        strike: float,
        rate: float,
        sigma: float,
        maturity: float,
        steps: int,
        option_type: str = "call",
    ) -> float:
        """Price a European option using the Cox-Ross-Rubinstein binomial model.

        Args:
            spot: Current asset price
            strike: Strike price
            rate: Risk-free rate (annualized)
            sigma: Volatility (annualized)
            maturity: Time to maturity (years)
            steps: Number of time steps
            option_type: "call" or "put"

        Returns:
            Option price
        """
        if option_type not in ("call", "put"):
            raise ValueError(f"Unknown option type: {option_type}")

        dt = maturity / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp(rate * dt) - d) / (u - d)
        disc = math.exp(-rate * dt)

        # Asset prices at maturity: S_T = spot * u^j * d^(steps-j)
        j = np.arange(steps + 1)
        ST = spot * (u ** j) * (d ** (steps - j))

        # Option payoff at maturity
        if option_type == "call":
            values = np.maximum(ST - strike, 0.0)
        else:
            values = np.maximum(strike - ST, 0.0)

        # Backward induction
        for _ in range(steps):
            values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

        return float(values[0])

    # -----------------------------------------------------------------------
    # Scenario Analysis
    # -----------------------------------------------------------------------

    def scenario_analysis(
        self, scenarios: list[Scenario], discount_rate: float
    ) -> dict[str, Any]:
        """Perform probability-weighted scenario analysis.

        Args:
            scenarios: List of Scenario objects
            discount_rate: Annual discount rate

        Returns:
            dict with keys:
                - expected_npv: probability-weighted NPV
                - scenario_npvs: dict mapping scenario name to NPV
                - npv_mean: unweighted mean of scenario NPVs
                - npv_std: unweighted std of scenario NPVs
                - npv_min: minimum scenario NPV
                - npv_max: maximum scenario NPV
        """
        if not scenarios:
            raise ValueError("At least one scenario is required")

        # Validate probabilities
        total_prob = sum(s.probability for s in scenarios)
        if not math.isclose(total_prob, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"Scenario probabilities must sum to 1, got {total_prob}"
            )

        # Compute NPV for each scenario
        scenario_npvs: dict[str, float] = {}
        for s in scenarios:
            npv = self._compute_npv(s.cash_flows, discount_rate)
            scenario_npvs[s.name] = npv

        # Probability-weighted expected NPV
        expected_npv = sum(
            s.probability * scenario_npvs[s.name] for s in scenarios
        )

        # Unweighted statistics
        npv_values = np.array(list(scenario_npvs.values()))
        npv_mean = float(np.mean(npv_values))
        npv_std = float(np.std(npv_values))
        npv_min = float(np.min(npv_values))
        npv_max = float(np.max(npv_values))

        return {
            "expected_npv": expected_npv,
            "scenario_npvs": scenario_npvs,
            "npv_mean": npv_mean,
            "npv_std": npv_std,
            "npv_min": npv_min,
            "npv_max": npv_max,
        }

    def _compute_npv(self, cash_flows: list[float], discount_rate: float) -> float:
        """Compute NPV of a cash flow stream."""
        return sum(
            cf / (1.0 + discount_rate) ** t for t, cf in enumerate(cash_flows)
        )

    # -----------------------------------------------------------------------
    # Sensitivity Analysis
    # -----------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        base_params: dict[str, float],
        param_ranges: dict[str, np.ndarray],
        evaluation_func: Callable[..., float],
    ) -> SensitivityResult:
        """Perform one-way sensitivity analysis.

        For each parameter, vary it over its range while holding others at base values.

        Args:
            base_params: Dictionary of base parameter values
            param_ranges: Dictionary mapping param name to array of values to test
            evaluation_func: Function that takes keyword arguments and returns a float

        Returns:
            SensitivityResult with impacts and ranking
        """
        impacts: dict[str, np.ndarray] = {}

        for param_name, values in param_ranges.items():
            results = []
            for val in values:
                params = dict(base_params)
                params[param_name] = float(val)
                results.append(evaluation_func(**params))
            impacts[param_name] = np.array(results)

        # Rank by absolute impact (max - min)
        ranking = sorted(
            impacts.keys(),
            key=lambda p: np.max(impacts[p]) - np.min(impacts[p]),
            reverse=True,
        )

        return SensitivityResult(impacts=impacts, ranking=ranking)

    def two_way_sensitivity(
        self,
        func: Callable[..., float],
        param1_name: str,
        param1_values: np.ndarray,
        param2_name: str,
        param2_values: np.ndarray,
    ) -> np.ndarray:
        """Perform two-way sensitivity analysis.

        Returns:
            2D array where result[i, j] = func(param1_values[i], param2_values[j])
        """
        result = np.zeros((len(param1_values), len(param2_values)))
        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                result[i, j] = func(**{param1_name: v1, param2_name: v2})
        return result

    def tornado_analysis(
        self,
        base_params: dict[str, float],
        variation_pct: float,
        evaluation_func: Callable[..., float],
    ) -> dict[str, dict[str, float]]:
        """Perform tornado sensitivity analysis.

        For each parameter, compute the output at base ± variation_pct.

        Returns:
            Dict mapping param name to {"high": value, "low": value}
        """
        result: dict[str, dict[str, float]] = {}
        for param_name, base_val in base_params.items():
            # High value
            params_high = dict(base_params)
            params_high[param_name] = base_val * (1.0 + variation_pct)
            high_val = evaluation_func(**params_high)

            # Low value
            params_low = dict(base_params)
            params_low[param_name] = base_val * (1.0 - variation_pct)
            low_val = evaluation_func(**params_low)

            result[param_name] = {"high": high_val, "low": low_val}

        return result
