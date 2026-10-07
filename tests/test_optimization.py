"""Tests for the optimization engine — LP, IP, QP, and constraint optimization."""

import numpy as np
import pytest

from optimization import OptimizationEngine, OptimizationResult


# ---------------------------------------------------------------------------
# Linear Programming (LP)
# ---------------------------------------------------------------------------


class TestLinearProgramming:
    """Tests for linear programming problems."""

    def test_simple_lp_maximization(self):
        """Maximize 3x + 2y subject to x + y <= 4, x >= 0, y >= 0."""
        engine = OptimizationEngine()
        result = engine.solve_lp(
            c=np.array([-3.0, -2.0]),  # negative because we minimize
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([4.0]),
            bounds=[(0, None), (0, None)],
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(-12.0, abs=1e-6)
        assert result.x[0] == pytest.approx(4.0, abs=1e-4)
        assert result.x[1] == pytest.approx(0.0, abs=1e-4)

    def test_lp_with_equality_constraints(self):
        """Minimize x + y subject to x + y = 3, x >= 0, y >= 0."""
        engine = OptimizationEngine()
        result = engine.solve_lp(
            c=np.array([1.0, 1.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([3.0]),
            bounds=[(0, None), (0, None)],
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(3.0, abs=1e-6)
        assert result.x[0] + result.x[1] == pytest.approx(3.0, abs=1e-4)

    def test_lp_production_planning(self):
        """Classic production planning: maximize profit from two products.

        Product A: profit 40/unit, requires 2h labor + 1kg material
        Product B: profit 30/unit, requires 1h labor + 2kg material
        Available: 100h labor, 80kg material
        """
        engine = OptimizationEngine()
        result = engine.solve_lp(
            c=np.array([-40.0, -30.0]),
            A_ub=np.array([[2.0, 1.0], [1.0, 2.0]]),
            b_ub=np.array([100.0, 80.0]),
            bounds=[(0, None), (0, None)],
        )
        assert result.status == "optimal"
        # Optimal: x=40, y=20, profit=2200
        assert result.fun == pytest.approx(-2200.0, abs=1e-4)
        assert result.x[0] == pytest.approx(40.0, abs=1e-3)
        assert result.x[1] == pytest.approx(20.0, abs=1e-3)

    def test_lp_infeasible(self):
        """Infeasible LP: x >= 2 and x <= 1."""
        engine = OptimizationEngine()
        result = engine.solve_lp(
            c=np.array([1.0]),
            A_ub=np.array([[-1.0], [1.0]]),
            b_ub=np.array([-2.0, 1.0]),
            bounds=[(None, None)],
        )
        assert result.status != "optimal"

    def test_lp_unbounded(self):
        """Unbounded LP: minimize -x with no upper bound."""
        engine = OptimizationEngine()
        result = engine.solve_lp(
            c=np.array([-1.0]),
            bounds=[(0, None)],
        )
        assert result.status != "optimal"


# ---------------------------------------------------------------------------
# Integer Programming (IP)
# ---------------------------------------------------------------------------


class TestIntegerProgramming:
    """Tests for integer and mixed-integer programming problems."""

    def test_simple_ip(self):
        """Maximize 5x + 4y subject to x + y <= 5, x, y >= 0, integer."""
        engine = OptimizationEngine()
        result = engine.solve_ip(
            c=np.array([-5.0, -4.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([5.0]),
            bounds=[(0, None), (0, None)],
            integrality=np.array([1, 1]),
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(-25.0, abs=1e-4)
        assert result.x[0] == pytest.approx(5.0, abs=1e-4)
        assert result.x[1] == pytest.approx(0.0, abs=1e-4)

    def test_ip_knapsack(self):
        """0-1 knapsack: maximize value subject to weight constraint.

        Items: (weight, value) = [(2,3), (3,4), (4,5), (5,6)]
        Capacity: 8
        Optimal: items 1, 3 -> weight=8, value=10
        """
        engine = OptimizationEngine()
        result = engine.solve_ip(
            c=np.array([-3.0, -4.0, -5.0, -6.0]),
            A_ub=np.array([[2.0, 3.0, 4.0, 5.0]]),
            b_ub=np.array([8.0]),
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
            integrality=np.array([1, 1, 1, 1]),
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(-10.0, abs=1e-4)
        total_weight = 2 * result.x[0] + 3 * result.x[1] + 4 * result.x[2] + 5 * result.x[3]
        assert total_weight <= 8.0 + 1e-6

    def test_mixed_integer_programming(self):
        """Mixed IP: x integer, y continuous.

        Maximize 3x + 2y subject to x + y <= 4.5, x >= 0, y >= 0.
        Optimal: x=4, y=0.5, obj=13
        """
        engine = OptimizationEngine()
        result = engine.solve_milp(
            c=np.array([-3.0, -2.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([4.5]),
            bounds=[(0, None), (0, None)],
            integrality=np.array([1, 0]),
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(-13.0, abs=1e-4)
        assert result.x[0] == pytest.approx(4.0, abs=1e-4)
        assert result.x[1] == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Quadratic Programming (QP)
# ---------------------------------------------------------------------------


class TestQuadraticProgramming:
    """Tests for quadratic programming problems."""

    def test_simple_qp(self):
        """Minimize 0.5*(x^2 + y^2) subject to x + y = 1.

        Optimal: x = y = 0.5, obj = 0.25
        """
        engine = OptimizationEngine()
        result = engine.solve_qp(
            P=np.array([[1.0, 0.0], [0.0, 1.0]]),
            q=np.array([0.0, 0.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.0]),
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(0.25, abs=1e-4)
        assert result.x[0] == pytest.approx(0.5, abs=1e-4)
        assert result.x[1] == pytest.approx(0.5, abs=1e-4)

    def test_qp_with_linear_term(self):
        """Minimize 0.5*(x^2 + y^2) - 2x - 3y subject to x + y <= 1, x,y >= 0.

        Unconstrained min: x=2, y=3. With constraint x+y<=1, x,y>=0:
        Optimal: x=0, y=1, obj=0.5-3=-2.5
        """
        engine = OptimizationEngine()
        result = engine.solve_qp(
            P=np.array([[1.0, 0.0], [0.0, 1.0]]),
            q=np.array([-2.0, -3.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([1.0]),
            bounds=[(0, None), (0, None)],
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(-2.5, abs=1e-4)
        assert result.x[0] == pytest.approx(0.0, abs=1e-4)
        assert result.x[1] == pytest.approx(1.0, abs=1e-4)

    def test_qp_portfolio_optimization(self):
        """Markowitz portfolio optimization.

        Minimize 0.5 * w^T Sigma w subject to sum(w) = 1, w >= 0.
        """
        engine = OptimizationEngine()
        sigma = np.array([[0.1, 0.02], [0.02, 0.08]])
        result = engine.solve_qp(
            P=sigma,
            q=np.array([0.0, 0.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.0]),
            bounds=[(0, None), (0, None)],
        )
        assert result.status == "optimal"
        assert result.x[0] + result.x[1] == pytest.approx(1.0, abs=1e-4)
        assert result.x[0] >= -1e-6
        assert result.x[1] >= -1e-6


# ---------------------------------------------------------------------------
# Constraint Optimization (Nonlinear)
# ---------------------------------------------------------------------------


class TestConstraintOptimization:
    """Tests for general nonlinear constraint optimization."""

    def test_rosenbrock_unconstrained(self):
        """Minimize Rosenbrock function: (1-x)^2 + 100(y-x^2)^2.

        Global minimum at (1, 1) with f=0.
        """
        engine = OptimizationEngine()
        result = engine.solve_constraint(
            fun=lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
            x0=np.array([0.0, 0.0]),
            method="SLSQP",
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(0.0, abs=1e-4)
        assert result.x[0] == pytest.approx(1.0, abs=1e-3)
        assert result.x[1] == pytest.approx(1.0, abs=1e-3)

    def test_nonlinear_with_equality_constraint(self):
        """Minimize x^2 + y^2 subject to x + y = 1.

        Optimal: x = y = 0.5, obj = 0.5
        """
        engine = OptimizationEngine()
        result = engine.solve_constraint(
            fun=lambda x: x[0] ** 2 + x[1] ** 2,
            x0=np.array([0.0, 0.0]),
            constraints=[{"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0}],
            method="SLSQP",
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(0.5, abs=1e-4)
        assert result.x[0] == pytest.approx(0.5, abs=1e-4)
        assert result.x[1] == pytest.approx(0.5, abs=1e-4)

    def test_nonlinear_with_inequality_constraint(self):
        """Minimize (x-2)^2 + (y-1)^2 subject to x + y <= 2, x,y >= 0.

        Unconstrained min at (2,1). With x+y<=2: optimal at (1.5, 0.5), obj=0.5
        """
        engine = OptimizationEngine()
        result = engine.solve_constraint(
            fun=lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
            x0=np.array([0.0, 0.0]),
            constraints=[{"type": "ineq", "fun": lambda x: 2.0 - x[0] - x[1]}],
            bounds=[(0, None), (0, None)],
            method="SLSQP",
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(0.5, abs=1e-3)
        assert result.x[0] + result.x[1] <= 2.0 + 1e-6


# ---------------------------------------------------------------------------
# Engine Infrastructure
# ---------------------------------------------------------------------------


class TestEngineInfrastructure:
    """Tests for engine initialization and result structure."""

    def test_engine_initialization(self):
        """Engine initializes with default solver."""
        engine = OptimizationEngine()
        assert engine is not None

    def test_result_dataclass_fields(self):
        """OptimizationResult has all required fields."""
        result = OptimizationResult(
            status="optimal",
            fun=1.0,
            x=np.array([1.0]),
            nit=10,
            message="Optimization terminated successfully",
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(1.0)
        assert result.x[0] == pytest.approx(1.0)
        assert result.nit == 10
        assert "success" in result.message.lower()

    def test_engine_with_custom_solver(self):
        """Engine accepts custom solver parameter."""
        engine = OptimizationEngine(solver="CLARABEL")
        assert engine is not None

    def test_lp_with_cvxpy_backend(self):
        """LP can be solved via cvxpy backend."""
        engine = OptimizationEngine(backend="cvxpy")
        result = engine.solve_lp(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([4.0]),
            bounds=[(0, None), (0, None)],
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(0.0, abs=1e-4)

    def test_qp_with_cvxpy_backend(self):
        """QP can be solved via cvxpy backend."""
        engine = OptimizationEngine(backend="cvxpy")
        result = engine.solve_qp(
            P=np.array([[2.0, 0.0], [0.0, 2.0]]),
            q=np.array([0.0, 0.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.0]),
        )
        assert result.status == "optimal"
        assert result.fun == pytest.approx(0.5, abs=1e-4)
