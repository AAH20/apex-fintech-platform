"""Statistical inference engine.

Provides hypothesis testing, confidence intervals, regression analysis,
ANOVA, and non-parametric tests using scipy.stats.

References:
    CFA Institute. (2024). Quantitative Methods. CFA Program Curriculum.
    Casella, G. & Berger, R.L. (2002). Statistical Inference. Duxbury.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


class StatisticalInferenceEngine:
    """Statistical inference engine for hypothesis testing and estimation.

    Implements core statistical methods used in quantitative finance:
    - One-sample, two-sample, and paired t-tests
    - Confidence intervals for means and proportions
    - Chi-square goodness-of-fit tests
    - One-way ANOVA
    - Simple linear regression
    - Mann-Whitney U test (non-parametric)
    """

    # ------------------------------------------------------------------
    # Hypothesis tests
    # ------------------------------------------------------------------

    def one_sample_ttest(
        self,
        sample: np.ndarray,
        pop_mean: float,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> dict[str, Any]:
        """One-sample t-test for the mean.

        Tests H0: sample mean == pop_mean vs. H1: sample mean != pop_mean.

        Args:
            sample: Sample data.
            pop_mean: Hypothesized population mean under H0.
            alpha: Significance level.
            alternative: 'two-sided', 'less', or 'greater'.

        Returns:
            Dictionary with statistic, p_value, reject_null, df, ci_lower, ci_upper.
        """
        sample = np.asarray(sample, dtype=float)
        t_stat, p_value = stats.ttest_1samp(sample, pop_mean, alternative=alternative)
        df = len(sample) - 1
        ci = stats.t.interval(1 - alpha, df, loc=np.mean(sample), scale=stats.sem(sample))
        return {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "reject_null": bool(p_value < alpha),
            "df": int(df),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
        }

    def two_sample_ttest(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
        equal_var: bool = True,
    ) -> dict[str, Any]:
        """Independent two-sample t-test.

        Tests H0: mean(group1) == mean(group2).

        Args:
            group1: First sample.
            group2: Second sample.
            alpha: Significance level.
            equal_var: If True, use Student's t-test (equal variances).
                       If False, use Welch's t-test.

        Returns:
            Dictionary with statistic, p_value, reject_null, df.
        """
        group1 = np.asarray(group1, dtype=float)
        group2 = np.asarray(group2, dtype=float)
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        if equal_var:
            df = len(group1) + len(group2) - 2
        else:
            # Welch-Satterthwaite degrees of freedom
            s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            numerator = (s1 / n1 + s2 / n2) ** 2
            denominator = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
            df = numerator / denominator
        return {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "reject_null": bool(p_value < alpha),
            "df": float(df),
        }

    def welch_ttest(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Welch's t-test for unequal variances.

        Args:
            group1: First sample.
            group2: Second sample.
            alpha: Significance level.

        Returns:
            Dictionary with statistic, p_value, reject_null, df.
        """
        return self.two_sample_ttest(group1, group2, alpha=alpha, equal_var=False)

    def paired_ttest(
        self,
        before: np.ndarray,
        after: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Paired t-test for dependent samples.

        Tests H0: mean(after - before) == 0.

        Args:
            before: Measurements before treatment.
            after: Measurements after treatment.
            alpha: Significance level.

        Returns:
            Dictionary with statistic, p_value, reject_null, df.
        """
        before = np.asarray(before, dtype=float)
        after = np.asarray(after, dtype=float)
        t_stat, p_value = stats.ttest_rel(after, before)
        df = len(before) - 1
        return {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "reject_null": bool(p_value < alpha),
            "df": int(df),
        }

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def confidence_interval_mean(
        self,
        sample: np.ndarray,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Confidence interval for the population mean.

        Uses the t-distribution (appropriate for small samples).

        Args:
            sample: Sample data.
            confidence: Confidence level (e.g., 0.95 for 95%).

        Returns:
            Dictionary with mean, ci_lower, ci_upper, confidence, margin_of_error.
        """
        sample = np.asarray(sample, dtype=float)
        n = len(sample)
        mean = np.mean(sample)
        se = stats.sem(sample)
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * se
        return {
            "mean": float(mean),
            "ci_lower": float(mean - margin),
            "ci_upper": float(mean + margin),
            "confidence": confidence,
            "margin_of_error": float(margin),
        }

    def confidence_interval_proportion(
        self,
        successes: int,
        trials: int,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Confidence interval for a population proportion.

        Uses the normal approximation (Wald interval).

        Args:
            successes: Number of successes.
            trials: Total number of trials.
            confidence: Confidence level.

        Returns:
            Dictionary with proportion, ci_lower, ci_upper, confidence.
        """
        p_hat = successes / trials
        alpha = 1 - confidence
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(p_hat * (1 - p_hat) / trials)
        margin = z_crit * se
        return {
            "proportion": float(p_hat),
            "ci_lower": float(max(0.0, p_hat - margin)),
            "ci_upper": float(min(1.0, p_hat + margin)),
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Chi-square tests
    # ------------------------------------------------------------------

    def chi_square_gof(
        self,
        observed: np.ndarray,
        expected: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Chi-square goodness-of-fit test.

        Tests H0: observed frequencies match expected frequencies.

        Args:
            observed: Observed frequencies.
            expected: Expected frequencies under H0.
            alpha: Significance level.

        Returns:
            Dictionary with statistic, p_value, reject_null, df.
        """
        observed = np.asarray(observed, dtype=float)
        expected = np.asarray(expected, dtype=float)
        chi2_stat, p_value = stats.chisquare(observed, expected)
        df = len(observed) - 1
        return {
            "statistic": float(chi2_stat),
            "p_value": float(p_value),
            "reject_null": bool(p_value < alpha),
            "df": int(df),
        }

    # ------------------------------------------------------------------
    # ANOVA
    # ------------------------------------------------------------------

    def anova(
        self,
        groups: list[np.ndarray],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """One-way ANOVA.

        Tests H0: all group means are equal.

        Args:
            groups: List of sample arrays, one per group.
            alpha: Significance level.

        Returns:
            Dictionary with f_statistic, p_value, reject_null, df_between, df_within.
        """
        f_stat, p_value = stats.f_oneway(*groups)
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        df_between = k - 1
        df_within = n_total - k
        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "reject_null": bool(p_value < alpha),
            "df_between": int(df_between),
            "df_within": int(df_within),
        }

    # ------------------------------------------------------------------
    # Linear regression
    # ------------------------------------------------------------------

    def linear_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Simple linear regression: y = slope * x + intercept.

        Args:
            x: Independent variable.
            y: Dependent variable.

        Returns:
            Dictionary with slope, intercept, r_squared, p_value, std_err, predict.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value**2

        def predict(x_new: float) -> float:
            """Predict y for a new x value."""
            return slope * x_new + intercept

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "predict": predict,
        }

    # ------------------------------------------------------------------
    # Non-parametric tests
    # ------------------------------------------------------------------

    def mann_whitney_u(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> dict[str, Any]:
        """Mann-Whitney U test (non-parametric alternative to t-test).

        Tests H0: distributions of both groups are equal.

        Args:
            group1: First sample.
            group2: Second sample.
            alpha: Significance level.
            alternative: 'two-sided', 'less', or 'greater'.

        Returns:
            Dictionary with statistic, p_value, reject_null.
        """
        group1 = np.asarray(group1, dtype=float)
        group2 = np.asarray(group2, dtype=float)
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        return {
            "statistic": float(u_stat),
            "p_value": float(p_value),
            "reject_null": bool(p_value < alpha),
        }
