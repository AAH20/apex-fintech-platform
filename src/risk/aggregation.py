"""Risk aggregation engine with dependence uncertainty.

Implements the Rearrangement Algorithm (Puccetti & Rüschendorf, 2012)
to compute sharp bounds on aggregate portfolio risk when the dependence
structure between individual risks is unknown.

Given marginal distributions for n risks, the sharp bounds on the
distribution of the sum S = X_1 + ... + X_n are:
  - Upper bound (worst case): comonotonic coupling — all risks move together
  - Lower bound (best case): independent coupling — risks diversify

The comonotonic sum gives the sharp upper bound on VaR/CVaR. The
independent sum gives a valid lower bound (the true sharp lower bound
for n > 2 requires solving an optimization problem; the independent
coupling provides a conservative, computable lower bound).
"""

import numpy as np


class RiskAggregationEngine:
    """Compute worst-case and best-case VaR/CVaR bounds for a portfolio.

    Uses the rearrangement algorithm to obtain bounds on the
    distribution of the sum of risks with given marginals but unknown
    dependence structure.

    Parameters
    ----------
    confidence_level : float
        Confidence level for VaR/CVaR (e.g., 0.95). Must be in (0, 1).
    n_samples : int
        Number of samples per marginal for the rearrangement algorithm.
        Default 10000 gives good accuracy while staying fast.
    """

    def __init__(self, confidence_level: float = 0.95, n_samples: int = 10000):
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        self.confidence_level = confidence_level
        self.n_samples = n_samples

    def rearrangement_algorithm(
        self, marginal_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute bounds on the sum distribution via rearrangement.

        Parameters
        ----------
        marginal_samples : np.ndarray
            Array of shape (n_risks, n_samples) where each row contains
            samples from one risk's marginal distribution.

        Returns
        -------
        lower_sum : np.ndarray
            Samples of the independent sum (best-case / lower bound).
        upper_sum : np.ndarray
            Samples of the comonotonic sum (worst-case / upper bound).
        """
        samples = np.asarray(marginal_samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # Comonotonic (worst case): sort all ascending, sum element-wise
        sorted_samples = np.sort(samples, axis=1)
        upper_sum = sorted_samples.sum(axis=0)

        # Independent (best case): sum the original (unsorted) samples
        lower_sum = samples.sum(axis=0)

        return lower_sum, upper_sum

    def compute_var_bounds(
        self, marginal_samples: np.ndarray
    ) -> tuple[float, float]:
        """Compute lower and upper bounds on VaR.

        Parameters
        ----------
        marginal_samples : np.ndarray
            Array of shape (n_risks, n_samples) with samples from each
            risk's marginal distribution.

        Returns
        -------
        lower_var : float
            Lower bound on VaR (best case, independent coupling).
        upper_var : float
            Upper bound on VaR (worst case, comonotonic coupling).
        """
        lower_sum, upper_sum = self.rearrangement_algorithm(marginal_samples)
        alpha = self.confidence_level * 100
        lower_var = float(np.percentile(lower_sum, alpha))
        upper_var = float(np.percentile(upper_sum, alpha))
        return lower_var, upper_var

    def compute_cvar_bounds(
        self, marginal_samples: np.ndarray
    ) -> tuple[float, float]:
        """Compute lower and upper bounds on CVaR (Expected Shortfall).

        Parameters
        ----------
        marginal_samples : np.ndarray
            Array of shape (n_risks, n_samples) with samples from each
            risk's marginal distribution.

        Returns
        -------
        lower_cvar : float
            Lower bound on CVaR (best case, independent coupling).
        upper_cvar : float
            Upper bound on CVaR (worst case, comonotonic coupling).
        """
        lower_sum, upper_sum = self.rearrangement_algorithm(marginal_samples)
        alpha = self.confidence_level * 100

        lower_var = np.percentile(lower_sum, alpha)
        upper_var = np.percentile(upper_sum, alpha)

        lower_cvar = float(np.mean(lower_sum[lower_sum >= lower_var]))
        upper_cvar = float(np.mean(upper_sum[upper_sum >= upper_var]))
        return lower_cvar, upper_cvar
