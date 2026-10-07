"""Optimization engine for LP, IP, QP, and constraint optimization.

Provides a unified interface for solving:
- Linear Programming (LP)
- Integer Programming (IP)
- Mixed-Integer Linear Programming (MILP)
- Quadratic Programming (QP)
- General nonlinear constraint optimization

Backends:
- scipy (default): uses scipy.optimize (linprog, milp, minimize)
- cvxpy: uses cvxpy for LP and QP

References:
    CFA Institute. (2024). Quantitative Methods for Investment Management.
    Boyd, S. & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
    Nocedal, J. & Wright, S. (2006). Numerical Optimization. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize


@dataclass
class OptimizationResult:
    """Result of an optimization problem.

    Attributes:
        status: Solution status ("optimal", "infeasible", "unbounded", etc.)
        fun: Optimal objective value
        x: Optimal solution vector
        nit: Number of iterations
        message: Solver message
    """

    status: str
    fun: float
    x: np.ndarray
    nit: int
    message: str


class OptimizationEngine:
    """Unified optimization engine.

    Supports LP, IP, MILP, QP, and general nonlinear constraint optimization
    via scipy (default) or cvxpy backends.

    Args:
        backend: Solver backend ("scipy" or "cvxpy")
        solver: Specific solver to use (mainly for cvxpy backend)
    """

    def __init__(self, backend: str = "scipy", solver: Optional[str] = None) -> None:
        if backend not in ("scipy", "cvxpy"):
            raise ValueError(f"Unknown backend: {backend!r}. Use 'scipy' or 'cvxpy'.")
        self.backend = backend
        self.solver = solver

    # -----------------------------------------------------------------------
    # Linear Programming
    # -----------------------------------------------------------------------

    def solve_lp(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[Sequence] = None,
    ) -> OptimizationResult:
        """Solve a linear programming problem.

        Minimize c^T x subject to:
            A_ub @ x <= b_ub
            A_eq @ x == b_eq
            bounds[i][0] <= x[i] <= bounds[i][1]
        """
        if self.backend == "cvxpy":
            return self._solve_lp_cvxpy(c, A_ub, b_ub, A_eq, b_eq, bounds)
        return self._solve_lp_scipy(c, A_ub, b_ub, A_eq, b_eq, bounds)

    def _solve_lp_scipy(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )
        return OptimizationResult(
            status="optimal" if result.success else result.message,
            fun=result.fun,
            x=result.x if result.x is not None else np.array([]),
            nit=result.nit,
            message=result.message,
        )

    def _solve_lp_cvxpy(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        import cvxpy as cp

        n = len(c)
        x = cp.Variable(n)
        constraints = []
        if A_ub is not None:
            constraints.append(A_ub @ x <= b_ub)
        if A_eq is not None:
            constraints.append(A_eq @ x == b_eq)
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    constraints.append(x[i] >= lb)
                if ub is not None:
                    constraints.append(x[i] <= ub)
        prob = cp.Problem(cp.Minimize(c @ x), constraints)
        prob.solve(solver=self.solver)
        return OptimizationResult(
            status=prob.status,
            fun=prob.value,
            x=x.value if x.value is not None else np.array([]),
            nit=prob.solver_stats.num_iters if prob.solver_stats else 0,
            message=str(prob.status),
        )

    # -----------------------------------------------------------------------
    # Integer / Mixed-Integer Programming
    # -----------------------------------------------------------------------

    def solve_ip(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[Sequence] = None,
        integrality: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Solve an integer programming problem (all variables integer)."""
        if integrality is None:
            integrality = np.ones(len(c))
        return self._solve_milp_scipy(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)

    def solve_milp(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[Sequence] = None,
        integrality: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Solve a mixed-integer linear programming problem."""
        return self._solve_milp_scipy(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)

    def _solve_milp_scipy(self, c, A_ub, b_ub, A_eq, b_eq, bounds, integrality):
        # Build constraints
        constraints = []
        if A_ub is not None:
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
        if A_eq is not None:
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

        # Build bounds
        if bounds is not None:
            lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
            ub = [b[1] if b[1] is not None else np.inf for b in bounds]
            scipy_bounds = Bounds(lb, ub)
        else:
            scipy_bounds = Bounds(-np.inf, np.inf)

        # Default integrality: all continuous
        if integrality is None:
            integrality = np.zeros(len(c))

        result = milp(c=c, integrality=integrality, bounds=scipy_bounds, constraints=constraints)
        return OptimizationResult(
            status="optimal" if result.success else result.message,
            fun=result.fun,
            x=result.x if result.x is not None else np.array([]),
            nit=getattr(result, "nit", 0),
            message=result.message,
        )

    # -----------------------------------------------------------------------
    # Quadratic Programming
    # -----------------------------------------------------------------------

    def solve_qp(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[Sequence] = None,
    ) -> OptimizationResult:
        """Solve a quadratic programming problem.

        Minimize 0.5 * x^T P x + q^T x subject to:
            A_ub @ x <= b_ub
            A_eq @ x == b_eq
            bounds[i][0] <= x[i] <= bounds[i][1]
        """
        if self.backend == "cvxpy":
            return self._solve_qp_cvxpy(P, q, A_ub, b_ub, A_eq, b_eq, bounds)
        return self._solve_qp_scipy(P, q, A_ub, b_ub, A_eq, b_eq, bounds)

    def _solve_qp_scipy(self, P, q, A_ub, b_ub, A_eq, b_eq, bounds):
        def objective(x):
            return 0.5 * x @ P @ x + q @ x

        def gradient(x):
            return P @ x + q

        constraints = []
        if A_ub is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda x: b_ub - A_ub @ x, "jac": lambda x: -A_ub}
            )
        if A_eq is not None:
            constraints.append(
                {"type": "eq", "fun": lambda x: A_eq @ x - b_eq, "jac": lambda x: A_eq}
            )

        x0 = np.zeros(len(q))
        result = minimize(
            objective, x0, jac=gradient, method="SLSQP", bounds=bounds, constraints=constraints
        )
        return OptimizationResult(
            status="optimal" if result.success else result.message,
            fun=result.fun,
            x=result.x if result.x is not None else np.array([]),
            nit=result.nit,
            message=result.message,
        )

    def _solve_qp_cvxpy(self, P, q, A_ub, b_ub, A_eq, b_eq, bounds):
        import cvxpy as cp

        n = len(q)
        x = cp.Variable(n)
        constraints = []
        if A_ub is not None:
            constraints.append(A_ub @ x <= b_ub)
        if A_eq is not None:
            constraints.append(A_eq @ x == b_eq)
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    constraints.append(x[i] >= lb)
                if ub is not None:
                    constraints.append(x[i] <= ub)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), constraints)
        prob.solve(solver=self.solver)
        return OptimizationResult(
            status=prob.status,
            fun=prob.value,
            x=x.value if x.value is not None else np.array([]),
            nit=prob.solver_stats.num_iters if prob.solver_stats else 0,
            message=str(prob.status),
        )

    # -----------------------------------------------------------------------
    # General Nonlinear Constraint Optimization
    # -----------------------------------------------------------------------

    def solve_constraint(
        self,
        fun: Callable,
        x0: np.ndarray,
        constraints: Optional[Sequence] = None,
        bounds: Optional[Sequence] = None,
        method: str = "SLSQP",
    ) -> OptimizationResult:
        """Solve a general nonlinear constraint optimization problem.

        Minimize fun(x) subject to:
            constraints[i]["fun"](x) >= 0  if constraints[i]["type"] == "ineq"
            constraints[i]["fun"](x) == 0  if constraints[i]["type"] == "eq"
            bounds[i][0] <= x[i] <= bounds[i][1]
        """
        if constraints is None:
            constraints = []
        result = minimize(
            fun, np.asarray(x0, dtype=float), method=method, bounds=bounds, constraints=constraints
        )
        return OptimizationResult(
            status="optimal" if result.success else result.message,
            fun=result.fun,
            x=result.x if result.x is not None else np.array([]),
            nit=result.nit,
            message=result.message,
        )
