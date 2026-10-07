"""Tests for decision analysis engine — decision trees, real options, scenario analysis."""

import numpy as np
import pytest

from decision import (
    DecisionAnalysisEngine,
    DecisionNode,
    RealOption,
    Scenario,
    SensitivityResult,
)


# ---------------------------------------------------------------------------
# Decision Trees
# ---------------------------------------------------------------------------


class TestDecisionTrees:
    """Tests for decision tree construction and evaluation."""

    def test_simple_decision_tree_ev(self):
        """Two-action decision tree with known expected values."""
        engine = DecisionAnalysisEngine()
        # Action A: 60% chance of 100, 40% chance of -50 → EV = 0.6*100 + 0.4*(-50) = 40
        # Action B: certain 30 → EV = 30
        tree = DecisionNode(
            name="root",
            node_type="decision",
            children=[
                DecisionNode(
                    name="A",
                    node_type="chance",
                    probability=None,
                    children=[
                        DecisionNode(name="A_good", node_type="terminal", value=100.0, probability=0.6),
                        DecisionNode(name="A_bad", node_type="terminal", value=-50.0, probability=0.4),
                    ],
                ),
                DecisionNode(name="B", node_type="terminal", value=30.0),
            ],
        )
        result = engine.evaluate_decision_tree(tree)
        assert result["expected_value"] == pytest.approx(40.0)
        assert result["optimal_path"] == ["root", "A"]

    def test_nested_decision_tree(self):
        """Multi-level decision tree with nested chance nodes."""
        engine = DecisionAnalysisEngine()
        # Root decision: Expand or Wait
        # Expand: 70% → 200, 30% → -100 → EV = 110
        # Wait: then decide Small (certain 50) or Large (50% → 120, 50% → -20 → EV=50)
        # Wait EV = max(50, 50) = 50
        # Root EV = max(110, 50) = 110
        tree = DecisionNode(
            name="root",
            node_type="decision",
            children=[
                DecisionNode(
                    name="expand",
                    node_type="chance",
                    children=[
                        DecisionNode(name="expand_good", node_type="terminal", value=200.0, probability=0.7),
                        DecisionNode(name="expand_bad", node_type="terminal", value=-100.0, probability=0.3),
                    ],
                ),
                DecisionNode(
                    name="wait",
                    node_type="decision",
                    children=[
                        DecisionNode(name="small", node_type="terminal", value=50.0),
                        DecisionNode(
                            name="large",
                            node_type="chance",
                            children=[
                                DecisionNode(name="large_good", node_type="terminal", value=120.0, probability=0.5),
                                DecisionNode(name="large_bad", node_type="terminal", value=-20.0, probability=0.5),
                            ],
                        ),
                    ],
                ),
            ],
        )
        result = engine.evaluate_decision_tree(tree)
        assert result["expected_value"] == pytest.approx(110.0)
        assert result["optimal_path"] == ["root", "expand"]

    def test_decision_tree_all_terminal(self):
        """Decision node with only terminal children picks the max."""
        engine = DecisionAnalysisEngine()
        tree = DecisionNode(
            name="root",
            node_type="decision",
            children=[
                DecisionNode(name="low", node_type="terminal", value=10.0),
                DecisionNode(name="mid", node_type="terminal", value=25.0),
                DecisionNode(name="high", node_type="terminal", value=15.0),
            ],
        )
        result = engine.evaluate_decision_tree(tree)
        assert result["expected_value"] == pytest.approx(25.0)
        assert result["optimal_path"] == ["root", "mid"]

    def test_chance_node_probability_validation(self):
        """Chance node probabilities must sum to 1."""
        engine = DecisionAnalysisEngine()
        tree = DecisionNode(
            name="root",
            node_type="chance",
            children=[
                DecisionNode(name="a", node_type="terminal", value=100.0, probability=0.5),
                DecisionNode(name="b", node_type="terminal", value=200.0, probability=0.3),
            ],
        )
        with pytest.raises(ValueError, match="probabilities"):
            engine.evaluate_decision_tree(tree)

    def test_single_terminal_node(self):
        """A single terminal node returns its own value."""
        engine = DecisionAnalysisEngine()
        tree = DecisionNode(name="only", node_type="terminal", value=42.0)
        result = engine.evaluate_decision_tree(tree)
        assert result["expected_value"] == pytest.approx(42.0)
        assert result["optimal_path"] == ["only"]


# ---------------------------------------------------------------------------
# Real Options
# ---------------------------------------------------------------------------


class TestRealOptions:
    """Tests for real options valuation (expand, abandon, defer)."""

    def test_expand_option(self):
        """Option to expand: pay cost to get additional value if conditions are good."""
        engine = DecisionAnalysisEngine()
        # Current project value = 100
        # Expansion cost = 30
        # If good (60%): value becomes 180, net = 180 - 30 = 150
        # If bad (40%): value stays 100, don't expand, net = 100
        # Option value = 0.6 * 150 + 0.4 * 100 = 130
        option = RealOption(
            option_type="expand",
            underlying_value=100.0,
            strike_cost=30.0,
            upside_value=180.0,
            downside_value=100.0,
            prob_up=0.6,
            prob_down=0.4,
        )
        result = engine.value_real_option(option)
        assert result["option_value"] == pytest.approx(130.0)
        assert result["exercise"] is True
        assert result["option_premium"] == pytest.approx(30.0)

    def test_abandon_option(self):
        """Option to abandon: receive salvage value instead of continuing."""
        engine = DecisionAnalysisEngine()
        # Continue value = 80, salvage = 100
        # Option value = max(80, 100) = 100
        option = RealOption(
            option_type="abandon",
            underlying_value=80.0,
            strike_cost=100.0,  # salvage value
            upside_value=80.0,
            downside_value=100.0,
            prob_up=0.5,
            prob_down=0.5,
        )
        result = engine.value_real_option(option)
        assert result["option_value"] == pytest.approx(100.0)
        assert result["exercise"] is True

    def test_defer_option(self):
        """Option to defer: wait for more information before committing."""
        engine = DecisionAnalysisEngine()
        # Invest now: NPV = 50
        # Wait: 70% chance NPV becomes 120, 30% chance NPV becomes -20 (don't invest)
        # Deferral value = 0.7 * 120 + 0.3 * 0 = 84
        option = RealOption(
            option_type="defer",
            underlying_value=50.0,
            strike_cost=0.0,
            upside_value=120.0,
            downside_value=0.0,
            prob_up=0.7,
            prob_down=0.3,
        )
        result = engine.value_real_option(option)
        assert result["option_value"] == pytest.approx(84.0)
        assert result["exercise"] is True

    def test_option_not_exercised(self):
        """Option with negative premium is not exercised."""
        engine = DecisionAnalysisEngine()
        option = RealOption(
            option_type="expand",
            underlying_value=100.0,
            strike_cost=80.0,
            upside_value=110.0,
            downside_value=100.0,
            prob_up=0.5,
            prob_down=0.5,
        )
        result = engine.value_real_option(option)
        # EV of exercising = 0.5*(110-80) + 0.5*0 = 15
        # But underlying is 100, so option value = 100 + 15 = 115? No...
        # Actually: if expand: 0.5*110 + 0.5*100 - 80 = 15; if not: 100
        # Option value = max(15, 100) = 100, exercise = False
        assert result["exercise"] is False
        assert result["option_value"] == pytest.approx(100.0)

    def test_binomial_option_pricing(self):
        """Binomial tree option pricing for a financial option."""
        engine = DecisionAnalysisEngine()
        # S=100, K=100, r=5%, sigma=20%, T=1, steps=100
        # European call should be close to Black-Scholes ~10.45
        result = engine.binomial_option_price(
            spot=100.0, strike=100.0, rate=0.05, sigma=0.20, maturity=1.0, steps=100, option_type="call"
        )
        assert 9.0 < result < 12.0  # Approximate BS value

    def test_binomial_put_option(self):
        """Binomial put option pricing."""
        engine = DecisionAnalysisEngine()
        result = engine.binomial_option_price(
            spot=100.0, strike=100.0, rate=0.05, sigma=0.20, maturity=1.0, steps=100, option_type="put"
        )
        assert 5.0 < result < 7.0  # Approximate BS put ~5.57


# ---------------------------------------------------------------------------
# Scenario Analysis
# ---------------------------------------------------------------------------


class TestScenarioAnalysis:
    """Tests for multi-scenario investment analysis."""

    def test_three_scenario_npv(self):
        """Compute NPV under base, upside, and downside scenarios."""
        engine = DecisionAnalysisEngine()
        scenarios = [
            Scenario(name="downside", probability=0.25, cash_flows=[-100, 20, 30, 30, 30]),
            Scenario(name="base", probability=0.50, cash_flows=[-100, 40, 50, 50, 50]),
            Scenario(name="upside", probability=0.25, cash_flows=[-100, 60, 70, 80, 90]),
        ]
        result = engine.scenario_analysis(scenarios, discount_rate=0.10)
        assert "expected_npv" in result
        assert "scenario_npvs" in result
        assert "downside" in result["scenario_npvs"]
        assert "base" in result["scenario_npvs"]
        assert "upside" in result["scenario_npvs"]
        # Expected NPV = 0.25*NPV_down + 0.50*NPV_base + 0.25*NPV_up
        expected = (
            0.25 * result["scenario_npvs"]["downside"]
            + 0.50 * result["scenario_npvs"]["base"]
            + 0.25 * result["scenario_npvs"]["upside"]
        )
        assert result["expected_npv"] == pytest.approx(expected)

    def test_scenario_probabilities_sum_to_one(self):
        """Scenario probabilities must sum to 1."""
        engine = DecisionAnalysisEngine()
        scenarios = [
            Scenario(name="a", probability=0.3, cash_flows=[-100, 50]),
            Scenario(name="b", probability=0.3, cash_flows=[-100, 60]),
        ]
        with pytest.raises(ValueError, match="probabilities"):
            engine.scenario_analysis(scenarios, discount_rate=0.10)

    def test_scenario_statistics(self):
        """Scenario analysis returns mean, std, min, max of NPVs."""
        engine = DecisionAnalysisEngine()
        scenarios = [
            Scenario(name="low", probability=0.2, cash_flows=[-100, 10, 10]),
            Scenario(name="mid", probability=0.6, cash_flows=[-100, 50, 50]),
            Scenario(name="high", probability=0.2, cash_flows=[-100, 100, 100]),
        ]
        result = engine.scenario_analysis(scenarios, discount_rate=0.10)
        npvs = [result["scenario_npvs"][s.name] for s in scenarios]
        assert result["npv_mean"] == pytest.approx(np.mean(npvs))
        assert result["npv_std"] == pytest.approx(np.std(npvs))
        assert result["npv_min"] == pytest.approx(min(npvs))
        assert result["npv_max"] == pytest.approx(max(npvs))


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    """Tests for one-way and two-way sensitivity analysis."""

    def test_one_way_sensitivity(self):
        """Vary one parameter and measure NPV impact."""
        engine = DecisionAnalysisEngine()

        def npv_func(revenue: float) -> float:
            return -100 + revenue * 3 / 1.1 + revenue * 3 / 1.1**2

        result = engine.sensitivity_analysis(
            base_params={"revenue": 50.0},
            param_ranges={"revenue": np.linspace(30.0, 70.0, 9)},
            evaluation_func=npv_func,
        )
        assert isinstance(result, SensitivityResult)
        assert "revenue" in result.impacts
        # NPV should increase with revenue
        values = result.impacts["revenue"]
        assert values[-1] > values[0]

    def test_sensitivity_ranking(self):
        """Parameters ranked by absolute impact on output."""
        engine = DecisionAnalysisEngine()

        def npv_func(revenue: float, cost: float) -> float:
            return -100 + (revenue - cost) * 3 / 1.1

        result = engine.sensitivity_analysis(
            base_params={"revenue": 50.0, "cost": 20.0},
            param_ranges={
                "revenue": np.linspace(40.0, 60.0, 5),
                "cost": np.linspace(15.0, 25.0, 5),
            },
            evaluation_func=npv_func,
        )
        # Revenue has larger impact per unit change
        assert result.ranking[0] == "revenue"

    def test_two_way_sensitivity(self):
        """Two-way sensitivity produces a matrix of results."""
        engine = DecisionAnalysisEngine()

        def npv_func(price: float, volume: float) -> float:
            return -500 + price * volume / 1.1

        result = engine.two_way_sensitivity(
            func=npv_func,
            param1_name="price",
            param1_values=np.array([8.0, 10.0, 12.0]),
            param2_name="volume",
            param2_values=np.array([50.0, 100.0, 150.0]),
        )
        assert result.shape == (3, 3)
        # NPV increases with both price and volume
        assert result[2, 2] > result[0, 0]
        assert result[0, 0] < 0  # Low price, low volume → negative NPV
        assert result[2, 2] > 0  # High price, high volume → positive NPV

    def test_tornado_data(self):
        """Tornado analysis returns low/high deviations for each parameter."""
        engine = DecisionAnalysisEngine()

        def npv_func(revenue: float, cost: float) -> float:
            return -100 + (revenue - cost) * 5 / 1.1

        result = engine.tornado_analysis(
            base_params={"revenue": 50.0, "cost": 20.0},
            variation_pct=0.20,
            evaluation_func=npv_func,
        )
        assert "revenue" in result
        assert "cost" in result
        # Revenue increase → NPV increase; cost increase → NPV decrease
        assert result["revenue"]["high"] > result["revenue"]["low"]
        assert result["cost"]["high"] < result["cost"]["low"]


# ---------------------------------------------------------------------------
# Integration / End-to-End
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end tests combining multiple analysis types."""

    def test_full_investment_analysis(self):
        """Complete analysis: decision tree + scenarios + sensitivity."""
        engine = DecisionAnalysisEngine()

        # Decision tree: invest now or wait
        tree = DecisionNode(
            name="root",
            node_type="decision",
            children=[
                DecisionNode(
                    name="invest",
                    node_type="chance",
                    children=[
                        DecisionNode(name="success", node_type="terminal", value=200.0, probability=0.6),
                        DecisionNode(name="failure", node_type="terminal", value=-80.0, probability=0.4),
                    ],
                ),
                DecisionNode(name="wait", node_type="terminal", value=20.0),
            ],
        )
        tree_result = engine.evaluate_decision_tree(tree)
        assert tree_result["expected_value"] == pytest.approx(88.0)  # 0.6*200 + 0.4*(-80)

        # Scenario analysis on the invest branch
        scenarios = [
            Scenario(name="poor", probability=0.3, cash_flows=[-100, 10, 20, 20]),
            Scenario(name="expected", probability=0.5, cash_flows=[-100, 40, 50, 60]),
            Scenario(name="strong", probability=0.2, cash_flows=[-100, 80, 90, 100]),
        ]
        scenario_result = engine.scenario_analysis(scenarios, discount_rate=0.12)
        assert scenario_result["expected_npv"] > 0

        # Sensitivity on revenue
        def npv_func(rev: float) -> float:
            return -100 + rev / 1.12 + rev / 1.12**2 + rev / 1.12**3

        sens_result = engine.sensitivity_analysis(
            base_params={"rev": 50.0},
            param_ranges={"rev": np.linspace(30.0, 70.0, 9)},
            evaluation_func=npv_func,
        )
        assert sens_result.impacts["rev"][-1] > sens_result.impacts["rev"][0]

    def test_engine_handles_empty_scenarios(self):
        """Empty scenario list raises an error."""
        engine = DecisionAnalysisEngine()
        with pytest.raises(ValueError, match="(?i)at least one scenario"):
            engine.scenario_analysis([], discount_rate=0.10)

    def test_real_option_in_decision_tree(self):
        """Real option valued within a decision tree framework."""
        engine = DecisionAnalysisEngine()
        # Project worth 100, option to expand for 30 cost
        # Good state (60%): 180, Bad state (40%): 100
        option = RealOption(
            option_type="expand",
            underlying_value=100.0,
            strike_cost=30.0,
            upside_value=180.0,
            downside_value=100.0,
            prob_up=0.6,
            prob_down=0.4,
        )
        opt_result = engine.value_real_option(option)
        # Use option value in a decision tree
        tree = DecisionNode(
            name="root",
            node_type="decision",
            children=[
                DecisionNode(name="with_option", node_type="terminal", value=opt_result["option_value"]),
                DecisionNode(name="without", node_type="terminal", value=100.0),
            ],
        )
        tree_result = engine.evaluate_decision_tree(tree)
        assert tree_result["expected_value"] == pytest.approx(opt_result["option_value"])
        assert tree_result["optimal_path"] == ["root", "with_option"]
