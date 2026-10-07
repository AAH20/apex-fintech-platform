"""Tests for game theory engine — Nash equilibrium, auctions, mechanism design.

TDD: These tests define the expected behavior of GameTheoryEngine.

References:
    - Nash, J. (1950). Equilibrium points in n-person games.
    - Vickrey, W. (1961). Counterspeculation, auctions, and competitive sealed tenders.
    - Myerson, R. (1981). Optimal auction design.
    - Mas-Colell, Whinston & Green (1995). Microeconomic Theory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gametheory import GameTheoryEngine


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    return GameTheoryEngine()


@pytest.fixture
def prisoner_dilemma():
    """Classic Prisoner's Dilemma payoff matrix (row player, col player)."""
    return np.array([
        [[3, 3], [0, 5]],
        [[5, 0], [1, 1]],
    ])


@pytest.fixture
def battle_of_sexes():
    """Battle of the Sexes — two pure NE, one mixed."""
    return np.array([
        [[2, 1], [0, 0]],
        [[0, 0], [1, 2]],
    ])


@pytest.fixture
def matching_pennies():
    """Matching Pennies — no pure NE, unique mixed NE at (0.5, 0.5)."""
    return np.array([
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]],
    ])


@pytest.fixture
def zero_sum_game():
    """Simple zero-sum game for minimax testing."""
    return np.array([
        [[4, -4], [-2, 2]],
        [[-1, 1], [3, -3]],
    ])


# ── Nash Equilibrium Tests ────────────────────────────────────────────────


class TestNashEquilibrium:
    """Tests for pure and mixed strategy Nash equilibrium computation."""

    def test_prisoners_dilemma_unique_ne(self, engine, prisoner_dilemma):
        """Prisoner's Dilemma has a unique pure NE: (Defect, Defect)."""
        result = engine.find_pure_nash(prisoner_dilemma)
        assert len(result) == 1
        assert result[0] == (1, 1)

    def test_battle_of_sexes_two_pure_ne(self, engine, battle_of_sexes):
        """Battle of the Sexes has two pure NE: (O,O) and (F,F)."""
        result = engine.find_pure_nash(battle_of_sexes)
        assert len(result) == 2
        assert (0, 0) in result
        assert (1, 1) in result

    def test_matching_pennies_no_pure_ne(self, engine, matching_pennies):
        """Matching Pennies has no pure strategy NE."""
        result = engine.find_pure_nash(matching_pennies)
        assert len(result) == 0

    def test_matching_pennies_mixed_ne(self, engine, matching_pennies):
        """Matching Pennies has a unique mixed NE at (0.5, 0.5)."""
        result = engine.find_mixed_nash_2x2(matching_pennies)
        assert result is not None
        p, q = result
        assert p == pytest.approx(0.5, abs=1e-6)
        assert q == pytest.approx(0.5, abs=1e-6)

    def test_battle_of_sexes_mixed_ne(self, engine, battle_of_sexes):
        """Battle of the Sexes has a mixed NE."""
        result = engine.find_mixed_nash_2x2(battle_of_sexes)
        assert result is not None
        p, q = result
        assert p == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert q == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_nash_equilibrium_is_best_response(self, engine, prisoner_dilemma):
        """Each player's strategy in NE is a best response to the other."""
        result = engine.find_pure_nash(prisoner_dilemma)
        ne = result[0]
        row_payoff = prisoner_dilemma[ne[0], ne[1], 0]
        col_payoff = prisoner_dilemma[ne[0], ne[1], 1]
        for i in range(2):
            assert prisoner_dilemma[i, ne[1], 0] <= row_payoff + 1e-10
        for j in range(2):
            assert prisoner_dilemma[ne[0], j, 1] <= col_payoff + 1e-10

    def test_dominated_strategy_elimination(self, engine, prisoner_dilemma):
        """Defect strictly dominates Cooperate in Prisoner's Dilemma."""
        dominated = engine.find_dominated_strategies(prisoner_dilemma)
        assert 0 in dominated[0]
        assert 0 in dominated[1]


# ── Auction Theory Tests ──────────────────────────────────────────────────


class TestAuctionTheory:
    """Tests for auction theory: first-price, second-price, all-pay."""

    def test_second_price_truthful_bidding(self, engine):
        """In second-price auction, bidding one's value is dominant."""
        values = [10.0, 20.0, 30.0]
        bids = engine.second_price_equilibrium_bids(values)
        for v, b in zip(values, bids):
            assert b == pytest.approx(v, abs=1e-10)

    def test_first_price_symmetric_equilibrium(self, engine):
        """In symmetric first-price auction with uniform values, b(v) = (n-1)/n * v."""
        n_bidders = 4
        value = 100.0
        bid = engine.first_price_symmetric_bid(value, n_bidders)
        expected = (n_bidders - 1) / n_bidders * value
        assert bid == pytest.approx(expected, abs=1e-10)

    def test_first_price_bid_shading(self, engine):
        """First-price bids are shaded below true value."""
        value = 100.0
        for n in [2, 3, 5, 10]:
            bid = engine.first_price_symmetric_bid(value, n)
            assert bid < value
            assert bid > 0

    def test_first_price_more_bidders_higher_bid(self, engine):
        """More bidders → less shading → higher bid."""
        value = 100.0
        bids = [engine.first_price_symmetric_bid(value, n) for n in [2, 3, 5, 10]]
        for i in range(len(bids) - 1):
            assert bids[i] < bids[i + 1]

    def test_all_pay_symmetric_equilibrium(self, engine):
        """All-pay auction: symmetric equilibrium bid for uniform [0,1] values."""
        n_bidders = 3
        value = 0.5
        bid = engine.all_pay_symmetric_bid(value, n_bidders)
        expected = (n_bidders - 1) / n_bidders * value ** n_bidders
        assert bid == pytest.approx(expected, abs=1e-10)

    def test_auction_revenue_ranking(self, engine):
        """Expected revenue: first-price = second-price (revenue equivalence)."""
        n_bidders = 5
        max_value = 100.0
        rev_first = engine.expected_revenue_first_price(n_bidders, max_value)
        rev_second = engine.expected_revenue_second_price(n_bidders, max_value)
        assert rev_first == pytest.approx(rev_second, rel=1e-6)

    def test_auction_winner_pays_second_highest(self, engine):
        """In second-price auction, winner pays the second-highest bid."""
        values = [10.0, 25.0, 15.0, 30.0, 20.0]
        bids = engine.second_price_equilibrium_bids(values)
        winner_idx = np.argmax(bids)
        assert values[winner_idx] == max(values)
        payment = engine.second_price_payment(values)
        sorted_vals = sorted(values, reverse=True)
        assert payment == pytest.approx(sorted_vals[1], abs=1e-10)


# ── Mechanism Design Tests ────────────────────────────────────────────────


class TestMechanismDesign:
    """Tests for mechanism design: revenue equivalence, Myerson, VCG."""

    def test_revenue_equivalence_theorem(self, engine):
        """Revenue Equivalence: any standard auction with same reserve yields same revenue."""
        n_bidders = 4
        max_value = 100.0
        rev_fp = engine.expected_revenue_first_price(n_bidders, max_value)
        rev_sp = engine.expected_revenue_second_price(n_bidders, max_value)
        assert rev_fp == pytest.approx(rev_sp, rel=1e-6)

    def test_myerson_optimal_auction_reserve(self, engine):
        """Myerson's optimal auction uses a reserve price."""
        n_bidders = 3
        max_value = 100.0
        reserve = engine.myerson_optimal_reserve(n_bidders, max_value)
        assert reserve == pytest.approx(max_value / 2.0, abs=1e-10)

    def test_myerson_virtual_value(self, engine):
        """Myerson's virtual value: φ(v) = v - (1-F(v))/f(v)."""
        v = 0.7
        virtual_val = engine.myerson_virtual_value(v, distribution="uniform", max_value=1.0)
        expected = 2 * v - 1
        assert virtual_val == pytest.approx(expected, abs=1e-10)

    def test_vcg_mechanism_truthful(self, engine):
        """VCG mechanism: truthful bidding is dominant strategy."""
        values = [10.0, 20.0, 30.0]
        bids = engine.vcg_equilibrium_bids(values)
        for v, b in zip(values, bids):
            assert b == pytest.approx(v, abs=1e-10)

    def test_vcg_payment(self, engine):
        """VCG payment: winner pays externality imposed on others."""
        values = [10.0, 20.0, 30.0]
        payments = engine.vcg_payments(values)
        winner_idx = np.argmax(values)
        sorted_vals = sorted(values, reverse=True)
        assert payments[winner_idx] == pytest.approx(sorted_vals[1], abs=1e-10)
        for i, p in enumerate(payments):
            if i != winner_idx:
                assert p == pytest.approx(0.0, abs=1e-10)

    def test_vcg_total_payment(self, engine):
        """VCG total payment equals second-highest value."""
        values = [10.0, 20.0, 30.0, 25.0, 15.0]
        payments = engine.vcg_payments(values)
        sorted_vals = sorted(values, reverse=True)
        total = sum(payments)
        assert total == pytest.approx(sorted_vals[1], abs=1e-10)


# ── Zero-Sum Game Tests ───────────────────────────────────────────────────


class TestZeroSumGames:
    """Tests for zero-sum game solving via minimax."""

    def test_minimax_value_exists(self, engine, zero_sum_game):
        """Every finite zero-sum game has a minimax value."""
        value, row_strategy, col_strategy = engine.solve_zero_sum(zero_sum_game)
        assert isinstance(value, float)
        assert len(row_strategy) == 2
        assert len(col_strategy) == 2

    def test_minimax_strategies_are_distributions(self, engine, zero_sum_game):
        """Minimax strategies are valid probability distributions."""
        _, row_strategy, col_strategy = engine.solve_zero_sum(zero_sum_game)
        assert abs(sum(row_strategy) - 1.0) < 1e-10
        assert abs(sum(col_strategy) - 1.0) < 1e-10
        assert all(s >= -1e-10 for s in row_strategy)
        assert all(s >= -1e-10 for s in col_strategy)

    def test_minimax_value_bounds(self, engine, zero_sum_game):
        """Minimax value is between min and max of the payoff matrix."""
        value, _, _ = engine.solve_zero_sum(zero_sum_game)
        payoffs = zero_sum_game[:, :, 0]
        assert payoffs.min() - 1e-10 <= value <= payoffs.max() + 1e-10


# ── Integration Tests ─────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests combining multiple game theory concepts."""

    def test_auction_as_mechanism(self, engine):
        """Auction can be viewed as a mechanism design problem."""
        values = [10.0, 20.0, 30.0]
        sp_bids = engine.second_price_equilibrium_bids(values)
        assert all(b == v for b, v in zip(sp_bids, values))
        vcg_bids = engine.vcg_equilibrium_bids(values)
        assert all(b == v for b, v in zip(vcg_bids, values))

    def test_nash_in_auction_context(self, engine):
        """Symmetric bidding in first-price auction is a Nash equilibrium."""
        n_bidders = 3
        values = [50.0, 75.0, 100.0]
        bids = [engine.first_price_symmetric_bid(v, n_bidders) for v in values]
        for i, (v, b) in enumerate(zip(values, bids)):
            best_response = engine.first_price_best_response(v, n_bids=n_bidders - 1)
            assert b == pytest.approx(best_response, rel=1e-4)

    def test_engine_handles_edge_cases(self, engine):
        """Engine handles edge cases gracefully."""
        bid = engine.first_price_symmetric_bid(100.0, 1)
        assert bid == pytest.approx(0.0, abs=1e-10)
        bid = engine.first_price_symmetric_bid(0.0, 3)
        assert bid == pytest.approx(0.0, abs=1e-10)
        bid = engine.first_price_symmetric_bid(100.0, 100)
        assert bid > 0
        assert bid <= 100.0
