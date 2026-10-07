"""Tests for M&A valuation engine — DCF, comps, and precedent transactions."""
import numpy as np
import pytest

from ma import ValuationEngine, DCFResult, CompsResult, PrecedentResult


# ---------------------------------------------------------------------------
# DCF Valuation
# ---------------------------------------------------------------------------


class TestDCFValuation:
    """Tests for Discounted Cash Flow valuation."""

    def test_dcf_basic_valuation(self):
        """DCF produces a positive enterprise value for a profitable company."""
        engine = ValuationEngine()
        result = engine.dcf_valuation(
            free_cash_flows=[100.0, 110.0, 121.0, 133.1, 146.4],
            wacc=0.10,
            terminal_growth=0.03,
        )
        assert isinstance(result, DCFResult)
        assert result.enterprise_value > 0
        assert result.equity_value > 0

    def test_dcf_enterprise_value_calculation(self):
        """DCF enterprise value equals PV of FCFs plus PV of terminal value."""
        engine = ValuationEngine()
        fcfs = [100.0, 100.0, 100.0]
        wacc = 0.10
        tg = 0.02
        result = engine.dcf_valuation(
            free_cash_flows=fcfs, wacc=wacc, terminal_growth=tg
        )
        # Manual calculation
        pv_fcfs = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(fcfs))
        terminal_value = fcfs[-1] * (1 + tg) / (wacc - tg)
        pv_terminal = terminal_value / (1 + wacc) ** len(fcfs)
        expected_ev = pv_fcfs + pv_terminal
        assert result.enterprise_value == pytest.approx(expected_ev, rel=1e-6)

    def test_dcf_equity_value_subtracts_debt(self):
        """Equity value = Enterprise value - Net debt."""
        engine = ValuationEngine()
        result = engine.dcf_valuation(
            free_cash_flows=[100.0, 110.0, 121.0],
            wacc=0.10,
            terminal_growth=0.03,
            net_debt=200.0,
        )
        assert result.equity_value == pytest.approx(
            result.enterprise_value - 200.0, rel=1e-6
        )

    def test_dcf_higher_wacc_lowers_value(self):
        """Higher discount rate reduces enterprise value."""
        engine = ValuationEngine()
        fcfs = [100.0, 110.0, 121.0, 133.1, 146.4]
        result_low = engine.dcf_valuation(fcfs, wacc=0.08, terminal_growth=0.03)
        result_high = engine.dcf_valuation(fcfs, wacc=0.15, terminal_growth=0.03)
        assert result_high.enterprise_value < result_low.enterprise_value

    def test_dcf_higher_terminal_growth_increases_value(self):
        """Higher terminal growth rate increases enterprise value."""
        engine = ValuationEngine()
        fcfs = [100.0, 110.0, 121.0, 133.1, 146.4]
        result_low = engine.dcf_valuation(fcfs, wacc=0.10, terminal_growth=0.01)
        result_high = engine.dcf_valuation(fcfs, wacc=0.10, terminal_growth=0.05)
        assert result_high.enterprise_value > result_low.enterprise_value

    def test_dcf_terminal_growth_must_be_less_than_wacc(self):
        """Terminal growth must be less than WACC for convergence."""
        engine = ValuationEngine()
        with pytest.raises(ValueError, match="terminal_growth must be less than wacc"):
            engine.dcf_valuation(
                free_cash_flows=[100.0, 110.0],
                wacc=0.05,
                terminal_growth=0.06,
            )

    def test_dcf_perpetuity_no_growth(self):
        """With zero terminal growth, TV = FCF / WACC."""
        engine = ValuationEngine()
        fcfs = [100.0, 100.0, 100.0]
        wacc = 0.10
        result = engine.dcf_valuation(fcfs, wacc=wacc, terminal_growth=0.0)
        pv_fcfs = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(fcfs))
        tv = 100.0 / wacc
        pv_tv = tv / (1 + wacc) ** 3
        assert result.enterprise_value == pytest.approx(pv_fcfs + pv_tv, rel=1e-6)

    def test_dcf_result_has_implied_multiple(self):
        """DCF result includes implied EV/EBITDA multiple."""
        engine = ValuationEngine()
        result = engine.dcf_valuation(
            free_cash_flows=[100.0, 110.0, 121.0],
            wacc=0.10,
            terminal_growth=0.03,
            ebitda=200.0,
        )
        assert result.implied_ev_ebitda > 0
        assert result.implied_ev_ebitda == pytest.approx(
            result.enterprise_value / 200.0, rel=1e-6
        )


# ---------------------------------------------------------------------------
# Comparable Company Analysis
# ---------------------------------------------------------------------------


class TestCompsValuation:
    """Tests for comparable company analysis."""

    def test_comps_basic_valuation(self):
        """Comps produces a valuation range from peer multiples."""
        engine = ValuationEngine()
        result = engine.comps_valuation(
            metric=100.0,
            peer_multiples=[8.0, 10.0, 12.0, 9.0, 11.0],
        )
        assert isinstance(result, CompsResult)
        assert result.low_value < result.mid_value < result.high_value

    def test_comps_uses_median_for_mid(self):
        """Mid-case uses median multiple."""
        engine = ValuationEngine()
        metric = 100.0
        multiples = [8.0, 10.0, 12.0, 9.0, 11.0]
        result = engine.comps_valuation(metric=metric, peer_multiples=multiples)
        median_multiple = float(np.median(multiples))
        assert result.mid_value == pytest.approx(metric * median_multiple, rel=1e-6)

    def test_comps_low_high_use_percentiles(self):
        """Low and high use 25th and 75th percentile multiples."""
        engine = ValuationEngine()
        metric = 100.0
        multiples = [8.0, 9.0, 10.0, 11.0, 12.0]
        result = engine.comps_valuation(metric=metric, peer_multiples=multiples)
        p25 = float(np.percentile(multiples, 25))
        p75 = float(np.percentile(multiples, 75))
        assert result.low_value == pytest.approx(metric * p25, rel=1e-6)
        assert result.high_value == pytest.approx(metric * p75, rel=1e-6)

    def test_comps_with_ev_ebitda(self):
        """Comps works with EV/EBITDA multiples."""
        engine = ValuationEngine()
        result = engine.comps_valuation(
            metric=50.0,
            peer_multiples=[6.0, 7.0, 8.0, 7.5, 6.5],
            metric_name="EBITDA",
        )
        assert result.metric_name == "EBITDA"
        assert result.mid_value > 0

    def test_comps_requires_at_least_3_peers(self):
        """Comps requires at least 3 peer multiples."""
        engine = ValuationEngine()
        with pytest.raises(ValueError, match="at least 3 peer multiples"):
            engine.comps_valuation(metric=100.0, peer_multiples=[8.0, 10.0])


# ---------------------------------------------------------------------------
# Precedent Transactions
# ---------------------------------------------------------------------------


class TestPrecedentTransactions:
    """Tests for precedent transaction analysis."""

    def test_precedent_basic_valuation(self):
        """Precedent transactions produce a valuation range."""
        engine = ValuationEngine()
        result = engine.precedent_transactions_valuation(
            metric=100.0,
            deal_multiples=[10.0, 12.0, 14.0, 11.0, 13.0],
        )
        assert isinstance(result, PrecedentResult)
        assert result.low_value < result.mid_value < result.high_value

    def test_precedent_includes_control_premium(self):
        """Precedent transactions include control premium over unaffected price."""
        engine = ValuationEngine()
        result = engine.precedent_transactions_valuation(
            metric=100.0,
            deal_multiples=[10.0, 12.0, 14.0, 11.0, 13.0],
            control_premium=0.25,
        )
        # With control premium, values should be higher
        result_no_premium = engine.precedent_transactions_valuation(
            metric=100.0,
            deal_multiples=[10.0, 12.0, 14.0, 11.0, 13.0],
            control_premium=0.0,
        )
        assert result.mid_value > result_no_premium.mid_value

    def test_precedent_control_premium_scales_value(self):
        """Control premium directly scales the valuation."""
        engine = ValuationEngine()
        metric = 100.0
        multiples = [10.0, 12.0, 14.0]
        premium = 0.30
        result = engine.precedent_transactions_valuation(
            metric=metric,
            deal_multiples=multiples,
            control_premium=premium,
        )
        median_multiple = float(np.median(multiples))
        expected = metric * median_multiple * (1 + premium)
        assert result.mid_value == pytest.approx(expected, rel=1e-6)

    def test_precedent_requires_at_least_2_deals(self):
        """Precedent transactions require at least 2 deals."""
        engine = ValuationEngine()
        with pytest.raises(ValueError, match="at least 2 deal multiples"):
            engine.precedent_transactions_valuation(
                metric=100.0, deal_multiples=[10.0]
            )


# ---------------------------------------------------------------------------
# Integrated / Combined Valuation
# ---------------------------------------------------------------------------


class TestCombinedValuation:
    """Tests for combined valuation approach."""

    def test_combined_valuation_weights(self):
        """Combined valuation weights DCF, comps, and precedent."""
        engine = ValuationEngine()
        result = engine.combined_valuation(
            dcf_value=1000.0,
            comps_mid=1100.0,
            precedent_mid=1200.0,
            weights=(0.5, 0.25, 0.25),
        )
        expected = 0.5 * 1000.0 + 0.25 * 1100.0 + 0.25 * 1200.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_combined_valuation_default_weights(self):
        """Default weights are equal (1/3 each)."""
        engine = ValuationEngine()
        result = engine.combined_valuation(
            dcf_value=900.0,
            comps_mid=1200.0,
            precedent_mid=1500.0,
        )
        expected = (900.0 + 1200.0 + 1500.0) / 3.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_combined_valuation_weights_must_sum_to_one(self):
        """Weights must sum to 1.0."""
        engine = ValuationEngine()
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            engine.combined_valuation(
                dcf_value=1000.0,
                comps_mid=1100.0,
                precedent_mid=1200.0,
                weights=(0.5, 0.5, 0.5),
            )


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    """Tests for DCF sensitivity analysis."""

    def test_dcf_sensitivity_grid(self):
        """Sensitivity analysis produces a grid of values."""
        engine = ValuationEngine()
        wacc_range = np.linspace(0.08, 0.12, 5)
        tg_range = np.linspace(0.01, 0.05, 5)
        grid = engine.dcf_sensitivity(
            free_cash_flows=[100.0, 110.0, 121.0],
            wacc_range=wacc_range,
            terminal_growth_range=tg_range,
        )
        assert grid.shape == (5, 5)
        # All values should be positive
        assert np.all(grid > 0)

    def test_dcf_sensitivity_higher_wacc_lower_value(self):
        """In sensitivity grid, higher WACC → lower value."""
        engine = ValuationEngine()
        wacc_range = np.linspace(0.08, 0.12, 5)
        tg_range = np.linspace(0.01, 0.05, 5)
        grid = engine.dcf_sensitivity(
            free_cash_flows=[100.0, 110.0, 121.0],
            wacc_range=wacc_range,
            terminal_growth_range=tg_range,
        )
        # For any fixed tg, increasing wacc should decrease value
        for j in range(5):
            for i in range(4):
                assert grid[i + 1, j] < grid[i, j]

    def test_dcf_sensitivity_higher_tg_higher_value(self):
        """In sensitivity grid, higher terminal growth → higher value."""
        engine = ValuationEngine()
        wacc_range = np.linspace(0.08, 0.12, 5)
        tg_range = np.linspace(0.01, 0.05, 5)
        grid = engine.dcf_sensitivity(
            free_cash_flows=[100.0, 110.0, 121.0],
            wacc_range=wacc_range,
            terminal_growth_range=tg_range,
        )
        # For any fixed wacc, increasing tg should increase value
        for i in range(5):
            for j in range(4):
                assert grid[i, j + 1] > grid[i, j]
