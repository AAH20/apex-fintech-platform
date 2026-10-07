"""Tests for statistical inference engine — hypothesis testing, CIs, regression, ANOVA."""

import numpy as np
import pytest

from stat_inference import StatisticalInferenceEngine


# ---------------------------------------------------------------------------
# One-sample t-test
# ---------------------------------------------------------------------------


class TestOneSampleTTest:
    """Tests for one-sample t-test."""

    def test_reject_null_when_mean_differs(self):
        """Reject H0 when sample mean differs significantly from population mean."""
        engine = StatisticalInferenceEngine()
        sample = np.array([10.5, 11.2, 9.8, 10.9, 11.5, 10.1, 10.7, 11.0])
        result = engine.one_sample_ttest(sample, pop_mean=10.0, alpha=0.05)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05
        assert result["statistic"] > 0

    def test_fail_to_reject_null_when_mean_close(self):
        """Fail to reject H0 when sample mean is close to population mean."""
        engine = StatisticalInferenceEngine()
        sample = np.array([10.1, 9.9, 10.0, 10.2, 9.8, 10.05, 9.95, 10.0])
        result = engine.one_sample_ttest(sample, pop_mean=10.0, alpha=0.05)
        assert result["reject_null"] is False
        assert result["p_value"] > 0.05

    def test_ttest_returns_correct_keys(self):
        """Result dict contains all expected keys."""
        engine = StatisticalInferenceEngine()
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = engine.one_sample_ttest(sample, pop_mean=3.0)
        expected_keys = {"statistic", "p_value", "reject_null", "df", "ci_lower", "ci_upper"}
        assert expected_keys.issubset(result.keys())

    def test_ttest_two_sided(self):
        """Two-sided test detects difference in either direction."""
        engine = StatisticalInferenceEngine()
        sample = np.array([7.0, 7.5, 6.8, 7.2, 7.1, 6.9, 7.3, 7.0])
        result = engine.one_sample_ttest(sample, pop_mean=10.0, alpha=0.05)
        assert result["reject_null"] is True
        assert result["statistic"] < 0


# ---------------------------------------------------------------------------
# Two-sample t-test
# ---------------------------------------------------------------------------


class TestTwoSampleTTest:
    """Tests for independent two-sample t-test."""

    def test_different_means_reject_null(self):
        """Reject H0 when two samples have clearly different means."""
        engine = StatisticalInferenceEngine()
        group1 = np.array([10.0, 10.5, 9.8, 11.0, 10.2, 10.8, 9.9, 10.4])
        group2 = np.array([12.0, 12.5, 11.8, 13.0, 12.2, 12.8, 11.9, 12.4])
        result = engine.two_sample_ttest(group1, group2, alpha=0.05)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05

    def test_similar_means_fail_to_reject(self):
        """Fail to reject H0 when two samples have similar means."""
        engine = StatisticalInferenceEngine()
        group1 = np.array([10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7])
        group2 = np.array([10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 10.0, 9.9])
        result = engine.two_sample_ttest(group1, group2, alpha=0.05)
        assert result["reject_null"] is False
        assert result["p_value"] > 0.05

    def test_welch_ttest_unequal_variances(self):
        """Welch's t-test handles unequal variances."""
        engine = StatisticalInferenceEngine()
        group1 = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02])
        group2 = np.array([5.0, 5.5, 4.5, 5.2, 4.8, 5.3, 4.9, 5.1])
        result = engine.welch_ttest(group1, group2, alpha=0.05)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_paired_detects_systematic_difference(self):
        """Paired t-test detects systematic difference between paired samples."""
        engine = StatisticalInferenceEngine()
        before = np.array([100.0, 102.0, 98.0, 105.0, 101.0, 99.0, 103.0, 100.0])
        after = np.array([105.0, 107.0, 103.0, 110.0, 106.0, 104.0, 108.0, 105.0])
        result = engine.paired_ttest(before, after, alpha=0.05)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05

    def test_paired_no_difference(self):
        """Paired t-test fails to reject when differences are noise."""
        engine = StatisticalInferenceEngine()
        before = np.array([10.0, 10.5, 9.8, 10.2, 9.9, 10.1, 10.0, 10.3])
        after = np.array([10.1, 10.4, 9.9, 10.1, 10.0, 10.2, 9.9, 10.2])
        result = engine.paired_ttest(before, after, alpha=0.05)
        assert result["reject_null"] is False


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    """Tests for confidence interval estimation."""

    def test_ci_mean_contains_true_mean(self):
        """95% CI for mean contains the true population mean."""
        engine = StatisticalInferenceEngine()
        rng = np.random.default_rng(42)
        sample = rng.normal(loc=50.0, scale=10.0, size=100)
        result = engine.confidence_interval_mean(sample, confidence=0.95)
        assert result["ci_lower"] < 50.0 < result["ci_upper"]
        assert result["confidence"] == 0.95

    def test_ci_mean_higher_confidence_wider(self):
        """Higher confidence level produces wider interval."""
        engine = StatisticalInferenceEngine()
        rng = np.random.default_rng(42)
        sample = rng.normal(loc=0.0, scale=1.0, size=50)
        ci_90 = engine.confidence_interval_mean(sample, confidence=0.90)
        ci_99 = engine.confidence_interval_mean(sample, confidence=0.99)
        width_90 = ci_90["ci_upper"] - ci_90["ci_lower"]
        width_99 = ci_99["ci_upper"] - ci_99["ci_lower"]
        assert width_99 > width_90

    def test_ci_proportion(self):
        """Confidence interval for a proportion."""
        engine = StatisticalInferenceEngine()
        result = engine.confidence_interval_proportion(successes=30, trials=100, confidence=0.95)
        assert result["ci_lower"] < 0.30 < result["ci_upper"]
        assert result["proportion"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Chi-square goodness of fit
# ---------------------------------------------------------------------------


class TestChiSquareGOF:
    """Tests for chi-square goodness-of-fit test."""

    def test_uniform_distribution(self):
        """Chi-square GOF detects non-uniform distribution."""
        engine = StatisticalInferenceEngine()
        observed = np.array([30, 10, 10, 10, 10])  # Clearly non-uniform
        expected = np.array([14, 14, 14, 14, 14])  # Uniform expectation
        result = engine.chi_square_gof(observed, expected)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05

    def test_expected_distribution(self):
        """Chi-square GOF fails to reject when observed matches expected."""
        engine = StatisticalInferenceEngine()
        observed = np.array([14, 15, 13, 14, 14])
        expected = np.array([14, 14, 14, 14, 14])
        result = engine.chi_square_gof(observed, expected)
        assert result["reject_null"] is False
        assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


class TestANOVA:
    """Tests for one-way ANOVA."""

    def test_anova_detects_group_differences(self):
        """ANOVA detects significant differences between group means."""
        engine = StatisticalInferenceEngine()
        group1 = np.array([10.0, 10.5, 9.8, 10.2, 10.1])
        group2 = np.array([15.0, 15.5, 14.8, 15.2, 15.1])
        group3 = np.array([20.0, 20.5, 19.8, 20.2, 20.1])
        result = engine.anova([group1, group2, group3], alpha=0.05)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05
        assert result["f_statistic"] > 1.0

    def test_anova_no_difference(self):
        """ANOVA fails to reject when group means are similar."""
        engine = StatisticalInferenceEngine()
        rng = np.random.default_rng(42)
        group1 = rng.normal(10.0, 1.0, 20)
        group2 = rng.normal(10.1, 1.0, 20)
        group3 = rng.normal(9.9, 1.0, 20)
        result = engine.anova([group1, group2, group3], alpha=0.05)
        assert result["reject_null"] is False
        assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    """Tests for simple linear regression."""


    def test_regression_recovers_known_coefficients(self):
        """Regression recovers known slope and intercept."""
        engine = StatisticalInferenceEngine()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = 2.0 * x + 5.0  # True: slope=2, intercept=5
        result = engine.linear_regression(x, y)
        assert result["slope"] == pytest.approx(2.0, abs=1e-10)
        assert result["intercept"] == pytest.approx(5.0, abs=1e-10)

    def test_regression_r_squared_perfect_fit(self):
        """R-squared is 1.0 for a perfect linear fit."""
        engine = StatisticalInferenceEngine()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x - 1.0
        result = engine.linear_regression(x, y)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-10)

    def test_regression_r_squared_noisy_data(self):
        """R-squared is between 0 and 1 for noisy data."""
        engine = StatisticalInferenceEngine()
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + rng.normal(0, 1.0, 50)
        result = engine.linear_regression(x, y)
        assert 0.0 < result["r_squared"] < 1.0

    def test_regression_predict(self):
        """Regression model can predict new values."""
        engine = StatisticalInferenceEngine()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        result = engine.linear_regression(x, y)
        prediction = result["predict"](6.0)
        assert prediction == pytest.approx(13.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Non-parametric tests
# ---------------------------------------------------------------------------


class TestNonParametric:
    """Tests for non-parametric tests."""

    def test_mann_whitney_detects_difference(self):
        """Mann-Whitney U detects difference between distributions."""
        engine = StatisticalInferenceEngine()
        group1 = np.array([1.0, 1.5, 1.2, 1.8, 1.3, 1.6, 1.1, 1.4])
        group2 = np.array([5.0, 5.5, 5.2, 5.8, 5.3, 5.6, 5.1, 5.4])
        result = engine.mann_whitney_u(group1, group2, alpha=0.05)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05

    def test_mann_whitney_no_difference(self):
        """Mann-Whitney U fails to reject for similar distributions."""
        engine = StatisticalInferenceEngine()
        rng = np.random.default_rng(42)
        group1 = rng.normal(10.0, 1.0, 30)
        group2 = rng.normal(10.1, 1.0, 30)
        result = engine.mann_whitney_u(group1, group2, alpha=0.05)
        assert result["reject_null"] is False
