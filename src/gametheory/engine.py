"""Game theory engine — Nash equilibrium, auctions, mechanism design.

Implements core game theory concepts for strategic analysis:
- Pure and mixed strategy Nash equilibrium
- Auction theory (first-price, second-price, all-pay)
- Mechanism design (Myerson, VCG, revenue equivalence)
- Zero-sum games (minimax)

References:
    - Nash, J. (1950). Equilibrium points in n-person games.
    - Vickrey, W. (1961). Counterspeculation, auctions, and competitive sealed tenders.
    - Myerson, R. (1981). Optimal auction design.
    - Mas-Colell, Whinston & Green (1995). Microeconomic Theory.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


class GameTheoryEngine:
    """Game theory engine for solving strategic interaction problems.

    Supports:
    - Pure strategy Nash equilibrium (2-player games)
    - Mixed strategy Nash equilibrium (2x2 games)
    - Dominated strategy elimination
    - Auction theory (first-price, second-price, all-pay)
    - Mechanism design (Myerson optimal auction, VCG)
    - Zero-sum game solving (minimax)
    """

    # ── Nash Equilibrium ────────────────────────────────────────────────

    def find_pure_nash(self, payoff_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Find all pure strategy Nash equilibria in a 2-player game.

        Args:
            payoff_matrix: Shape (m, n, 2) where [i, j, 0] is row player's payoff
                          and [i, j, 1] is column player's payoff.

        Returns:
            List of (row_strategy, col_strategy) tuples that are Nash equilibria.
        """
        m, n = payoff_matrix.shape[0], payoff_matrix.shape[1]
        equilibria = []

        for i in range(m):
            for j in range(n):
                row_payoff = payoff_matrix[i, j, 0]
                col_payoff = payoff_matrix[i, j, 1]

                # Check if row player can deviate profitably
                row_best = all(
                    payoff_matrix[k, j, 0] <= row_payoff + 1e-10
                    for k in range(m)
                )
                # Check if col player can deviate profitably
                col_best = all(
                    payoff_matrix[i, k, 1] <= col_payoff + 1e-10
                    for k in range(n)
                )

                if row_best and col_best:
                    equilibria.append((i, j))

        return equilibria

    def find_mixed_nash_2x2(
        self, payoff_matrix: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Find mixed strategy Nash equilibrium for a 2x2 game.

        For a 2x2 game with payoff matrix:
            [[(a1, b1), (a2, b2)],
             [(a3, b3), (a4, b4)]]

        Row player mixes with probability p on row 0.
        Col player mixes with probability q on col 0.

        Returns:
            (p, q) tuple or None if no mixed equilibrium exists.
        """
        a1 = payoff_matrix[0, 0, 0]
        a2 = payoff_matrix[0, 1, 0]
        a3 = payoff_matrix[1, 0, 0]
        a4 = payoff_matrix[1, 1, 0]

        b1 = payoff_matrix[0, 0, 1]
        b2 = payoff_matrix[0, 1, 1]
        b3 = payoff_matrix[1, 0, 1]
        b4 = payoff_matrix[1, 1, 1]

        # Col player's indifference determines row's mix p:
        # p*b1 + (1-p)*b3 = p*b2 + (1-p)*b4
        # => p*(b1 - b2 - b3 + b4) = b4 - b3
        denom_p = b1 - b2 - b3 + b4
        if abs(denom_p) < 1e-12:
            return None
        p = (b4 - b3) / denom_p

        # Row player's indifference determines col's mix q:
        # q*a1 + (1-q)*a2 = q*a3 + (1-q)*a4
        # => q*(a1 - a2 - a3 + a4) = a4 - a2
        denom_q = a1 - a2 - a3 + a4
        if abs(denom_q) < 1e-12:
            return None
        q = (a4 - a2) / denom_q

        if not (0 <= p <= 1 and 0 <= q <= 1):
            return None

        return (p, q)

    def find_dominated_strategies(
        self, payoff_matrix: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Find strictly dominated strategies for each player.

        Returns:
            Tuple of (dominated_row_strategies, dominated_col_strategies).
        """
        m, n = payoff_matrix.shape[0], payoff_matrix.shape[1]
        dominated_rows = []
        dominated_cols = []

        # Check row player
        for i in range(m):
            for k in range(m):
                if i == k:
                    continue
                if all(
                    payoff_matrix[k, j, 0] > payoff_matrix[i, j, 0]
                    for j in range(n)
                ):
                    dominated_rows.append(i)
                    break

        # Check col player
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                if all(
                    payoff_matrix[i, k, 1] > payoff_matrix[i, j, 1]
                    for i in range(m)
                ):
                    dominated_cols.append(j)
                    break

        return (dominated_rows, dominated_cols)

    # ── Auction Theory ─────────────────────────────────────────────────

    def second_price_equilibrium_bids(self, values: List[float]) -> List[float]:
        """Second-price (Vickrey) auction: truthful bidding is dominant.

        Args:
            values: List of bidder valuations.

        Returns:
            List of equilibrium bids (equal to values).
        """
        return list(values)

    def second_price_payment(self, values: List[float]) -> float:
        """Payment in a second-price auction: second-highest bid.

        Args:
            values: List of bidder valuations.

        Returns:
            Second-highest value (the payment).
        """
        if len(values) < 2:
            return 0.0
        sorted_vals = sorted(values, reverse=True)
        return sorted_vals[1]

    def first_price_symmetric_bid(self, value: float, n_bidders: int) -> float:
        """Symmetric equilibrium bid in a first-price auction.

        For uniform [0, max] values with n bidders:
            b(v) = (n-1)/n * v

        Args:
            value: Bidder's valuation.
            n_bidders: Number of bidders.

        Returns:
            Equilibrium bid.
        """
        if n_bidders <= 1:
            return 0.0
        return (n_bidders - 1) / n_bidders * value

    def first_price_best_response(self, value: float, n_bids: int) -> float:
        """Best response bid in a first-price auction.

        Given others use symmetric strategy, the best response is the same.

        Args:
            value: Bidder's valuation.
            n_bids: Number of other bidders.

        Returns:
            Best response bid.
        """
        return self.first_price_symmetric_bid(value, n_bids + 1)

    def all_pay_symmetric_bid(self, value: float, n_bidders: int) -> float:
        """Symmetric equilibrium bid in an all-pay auction.

        For uniform [0,1] values with n bidders:
            b(v) = (n-1)/n * v^n

        Args:
            value: Bidder's valuation.
            n_bidders: Number of bidders.

        Returns:
            Equilibrium bid.
        """
        if n_bidders <= 1:
            return 0.0
        return (n_bidders - 1) / n_bidders * value ** n_bidders

    def expected_revenue_first_price(self, n_bidders: int, max_value: float) -> float:
        """Expected revenue in a first-price auction with uniform values.

        For uniform [0, max] with n bidders:
            E[revenue] = (n-1)/(n+1) * max

        Args:
            n_bidders: Number of bidders.
            max_value: Maximum possible value.

        Returns:
            Expected revenue.
        """
        if n_bidders <= 1:
            return 0.0
        return (n_bidders - 1) / (n_bidders + 1) * max_value

    def expected_revenue_second_price(self, n_bidders: int, max_value: float) -> float:
        """Expected revenue in a second-price auction with uniform values.

        For uniform [0, max] with n bidders:
            E[revenue] = (n-1)/(n+1) * max

        Args:
            n_bidders: Number of bidders.
            max_value: Maximum possible value.

        Returns:
            Expected revenue.
        """
        if n_bidders <= 1:
            return 0.0
        return (n_bidders - 1) / (n_bidders + 1) * max_value

    # ── Mechanism Design ───────────────────────────────────────────────

    def myerson_optimal_reserve(self, n_bidders: int, max_value: float) -> float:
        """Myerson's optimal reserve price for uniform distribution.

        For uniform [0, max], the optimal reserve is max/2.

        Args:
            n_bidders: Number of bidders.
            max_value: Maximum possible value.

        Returns:
            Optimal reserve price.
        """
        return max_value / 2.0

    def myerson_virtual_value(
        self, v: float, distribution: str = "uniform", max_value: float = 1.0
    ) -> float:
        """Myerson's virtual value: φ(v) = v - (1-F(v))/f(v).

        For uniform [0, max]: F(v)=v/max, f(v)=1/max
        => φ(v) = v - (1 - v/max) / (1/max) = v - (max - v) = 2v - max

        Args:
            v: Value.
            distribution: Distribution type ("uniform").
            max_value: Maximum value for uniform distribution.

        Returns:
            Virtual value.
        """
        if distribution == "uniform":
            return 2 * v - max_value
        raise ValueError(f"Unsupported distribution: {distribution}")

    def vcg_equilibrium_bids(self, values: List[float]) -> List[float]:
        """VCG mechanism: truthful bidding is dominant strategy.

        Args:
            values: List of bidder valuations.

        Returns:
            List of equilibrium bids (equal to values).
        """
        return list(values)

    def vcg_payments(self, values: List[float]) -> List[float]:
        """VCG payments: each bidder pays the externality they impose.

        For a single-item auction, the winner pays the second-highest value,
        and all others pay 0.

        Args:
            values: List of bidder valuations.

        Returns:
            List of VCG payments.
        """
        n = len(values)
        payments = [0.0] * n

        if n < 2:
            return payments

        winner_idx = int(np.argmax(values))
        sorted_vals = sorted(values, reverse=True)
        payments[winner_idx] = sorted_vals[1]

        return payments

    # ── Zero-Sum Games ─────────────────────────────────────────────────

    def solve_zero_sum(
        self, payoff_matrix: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Solve a zero-sum game using linear programming (minimax).

        For a zero-sum game with row player's payoff matrix A:
        - Row player maximizes minimum gain
        - Col player minimizes maximum loss

        Uses scipy.optimize.linprog if available, otherwise uses
        a simple iterative method for 2x2 games.

        Args:
            payoff_matrix: Shape (m, n, 2) where [i, j, 0] is row player's payoff.
                          For zero-sum, [i, j, 1] = -[i, j, 0].

        Returns:
            (game_value, row_strategy, col_strategy) tuple.
        """
        A = payoff_matrix[:, :, 0]  # Row player's payoff matrix
        m, n = A.shape

        # Try scipy first
        try:
            from scipy.optimize import linprog

            # Row player's problem: max v s.t. A^T x >= v*1, sum(x)=1, x>=0
            # Convert to: min -v s.t. -A^T x + v*1 <= 0, sum(x)=1, x>=0
            # Variables: [x_0, ..., x_{m-1}, v]

            c = np.zeros(m + 1)
            c[-1] = -1  # Minimize -v

            # Constraints: -A^T x + v <= 0 for each column j
            A_ub = np.zeros((n, m + 1))
            A_ub[:, :m] = -A.T
            A_ub[:, -1] = 1
            b_ub = np.zeros(n)

            # Equality: sum(x) = 1
            A_eq = np.zeros((1, m + 1))
            A_eq[0, :m] = 1
            b_eq = np.array([1.0])

            # Bounds: x >= 0, v unbounded
            bounds = [(0, None)] * m + [(None, None)]

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

            if result.success:
                x = result.x[:m]
                v = result.x[-1]
                row_strategy = x / x.sum() if x.sum() > 0 else np.ones(m) / m

                # Col player's problem: min v s.t. A y <= v*1, sum(y)=1, y>=0
                c2 = np.zeros(n + 1)
                c2[-1] = 1  # Minimize v

                A_ub2 = np.zeros((m, n + 1))
                A_ub2[:, :n] = A
                A_ub2[:, -1] = -1
                b_ub2 = np.zeros(m)

                A_eq2 = np.zeros((1, n + 1))
                A_eq2[0, :n] = 1
                b_eq2 = np.array([1.0])

                bounds2 = [(0, None)] * n + [(None, None)]

                result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2)

                if result2.success:
                    y = result2.x[:n]
                    col_strategy = y / y.sum() if y.sum() > 0 else np.ones(n) / n
                    return (float(v), row_strategy, col_strategy)

        except ImportError:
            pass

        # Fallback: simple iterative method for 2x2 games
        if m == 2 and n == 2:
            return self._solve_zero_sum_2x2(A)

        # General fallback: uniform strategy
        return (0.0, np.ones(m) / m, np.ones(n) / n)

    def _solve_zero_sum_2x2(
        self, A: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Solve a 2x2 zero-sum game analytically.

        Args:
            A: 2x2 payoff matrix for row player.

        Returns:
            (game_value, row_strategy, col_strategy) tuple.
        """
        a, b = A[0, 0], A[0, 1]
        c, d = A[1, 0], A[1, 1]

        # Check for saddle point
        row_mins = np.min(A, axis=1)
        col_maxs = np.max(A, axis=0)
        maximin = np.max(row_mins)
        minimax = np.min(col_maxs)

        if abs(maximin - minimax) < 1e-10:
            # Saddle point exists
            i = int(np.argmax(row_mins))
            j = int(np.argmin(col_maxs))
            row_strategy = np.zeros(2)
            col_strategy = np.zeros(2)
            row_strategy[i] = 1.0
            col_strategy[j] = 1.0
            return (float(maximin), row_strategy, col_strategy)

        # Mixed strategy equilibrium
        # Row player: p*a + (1-p)*c = p*b + (1-p)*d
        # => p = (d - c) / (a - b - c + d)
        denom = a - b - c + d
        if abs(denom) < 1e-12:
            return (0.0, np.array([0.5, 0.5]), np.array([0.5, 0.5]))

        p = (d - c) / denom
        p = np.clip(p, 0, 1)

        # Col player: q*a + (1-q)*b = q*c + (1-q)*d
        # => q = (d - b) / (a - b - c + d)
        q = (d - b) / denom
        q = np.clip(q, 0, 1)

        row_strategy = np.array([p, 1 - p])
        col_strategy = np.array([q, 1 - q])

        # Game value
        value = p * q * a + p * (1 - q) * b + (1 - p) * q * c + (1 - p) * (1 - q) * d

        return (float(value), row_strategy, col_strategy)
