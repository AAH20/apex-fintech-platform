"""Asset allocation engine — strategic, tactical, factor, risk parity.
All implementations numpy-only: no scipy dependency.
References:
    Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
    Black, F. & Litterman, R. (1992). Global Portfolio Optimization.
    Roncalli, T. (2013). Introduction to Risk Parity and Budgeting.
"""

from __future__ import annotations

import numpy as np


class AssetAllocationEngine:
    """Multi-strategy asset allocation engine.

    Supports:
    - Strategic (mean-variance) allocation
    - Tactical allocation (tilts from benchmark)
    - Factor-based allocation
    - Risk parity
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Strategic allocation (mean-variance optimization) — numpy active-set
    # ------------------------------------------------------------------

    def strategic_allocation(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 2.0,
    ) -> np.ndarray:
        """Mean-variance optimal weights (long-only, fully invested).

        Maximizes: w'mu - (lambda/2) * w'Sigma*w
        Subject to: sum(w) = 1, w >= 0

        Uses a simple projected-gradient active-set method.
        """

        mu = np.asarray(expected_returns, dtype=float)
        cov = np.asarray(cov_matrix, dtype=float)
        n = len(mu)

        if cov.shape != (n, n):
            raise ValueError(
                f"Covariance matrix shape {cov.shape} incompatible with {n} assets"
            )
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")

        # Projected gradient descent with projection onto simplex
        def portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
            return w @ cov @ w

        def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
            return w @ mu

        # Active-set style: start with all weights, project onto non-negative + sum=1
        w = np.ones(n) / n
        learning_rate = 0.01
        max_iter = 5000
        tolerance = 1e-10

        for i in range(max_iter):
            # Gradient of negative utility: -mu + lambda * cov * w
            grad = -mu + risk_aversion * cov @ w

            # Gradient step
            w_new = w - learning_rate * grad

            # Project onto simplex (sum=1, w >= 0)
            # Duchi et al. simplex projection
            rho = n
            for _ in range(n):  # simple binary search for projection
                sorted_w = np.sort(w_new)[::-1]
                cssv = np.cumsum(sorted_w) - 1.0
                rho = np.searchsorted(cssv, 0, side="right")
                if rho < n and cssv[rho] > 0:
                    break
                rho -= 1

            # Compute theta for projection
            if rho >= 0:
                theta = cssv[rho] / (rho + 1)
            else:
                theta = 0.0

            w = np.maximum(w_new - theta, 0.0)
            w = w / np.sum(w)  # renormalize for numerical stability

            # Check convergence: KKT residual
            # KKT: there exists lambda such that:
            #   -mu + lambda*cov*w + nu = 0  (stationarity)
            #   nu >= 0, w >= 0, nu_i * w_i = 0  (complementary slackness)
            #   sum(w) = 1  (primal feasibility)

            # Compute Lagrange multiplier estimate
            # From stationarity: lambda approx = (mu - grad_nu) / ... 
            # Simplified: check if all positive weights have same marginal return
            marginal_returns = cov @ w

            if i > 0:
                # Check KKT residual change
                utility_change = abs(
                    portfolio_return(w, mu) - portfolio_return(w_prev, mu)
                    + risk_aversion / 2 * (portfolio_variance(w, cov) - portfolio_variance(w_prev, cov))
                )
                if utility_change < tolerance:
                    break

            w_prev = w.copy()

        # Final clip and normalize
        w = np.clip(w, 0.0, None)
        if np.sum(w) > 0:
            w = w / np.sum(w)
        return w

    # ------------------------------------------------------------------
    # Tactical allocation
    # ------------------------------------------------------------------

    def tactical_allocation(
        self,
        strategic_weights: np.ndarray,
        tilts: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        max_deviation: float = 0.1,
        risk_aversion: float = 2.0,
    ) -> np.ndarray:
        """Apply tactical tilts to strategic benchmark, then re-optimize.

        Tilts are capped at max_deviation, then the portfolio is re-optimized
        around the tilted benchmark.
        """

        strategic = np.asarray(strategic_weights, dtype=float)
        tilts = np.asarray(tilts, dtype=float)
        mu = np.asarray(expected_returns, dtype=float)
        cov = np.asarray(cov_matrix, dtype=float)
        n = len(strategic)

        if len(tilts) != n:
            raise ValueError(f"tilts length {len(tilts)} != {n} assets")
        if cov.shape != (n, n):
            raise ValueError("Covariance matrix dimension mismatch")

        # Cap tilts
        capped_tilts = np.clip(tilts, -max_deviation, max_deviation)
        # Tilted benchmark (not necessarily summing to 1)
        tilted = strategic + capped_tilts
        tilted = np.clip(tilted, 0.0, None)

        # Re-optimize with tilted benchmark as reference using projected gradient
        w = tilted / np.sum(tilted) if np.sum(tilted) > 0 else np.ones(n) / n
        learning_rate = 0.01
        max_iter = 5000
        tolerance = 1e-10

        for i in range(max_iter):
            # Gradient of tracking-aware utility
            # Tracking term: w - tilted
            marginal = cov @ w
            tracking = w - tilted

            # Negative utility: -w'mu + 0.5*lambda*w'Sigma*w + 0.5*lambda*tracking'Sigma*tracking
            # Gradient: -mu + lambda*cov*w + lambda*cov*tracking
            grad = -mu + risk_aversion * cov @ (w + tracking)

            # Gradient step
            w_new = w - learning_rate * grad

            # Project onto simplex
            rho = n
            sorted_w = np.sort(w_new)[::-1]
            cssv = np.cumsum(sorted_w) - 1.0
            for r in range(n, 0, -1):
                if cssv[r - 1] <= 0:
                    rho = r - 1
                    break

            if rho >= 0:
                theta = cssv[rho] / (rho + 1) if rho < n else 0.0
            else:
                theta = 1.0

            w = np.maximum(w_new - theta, 0.0)
            w = w / np.sum(w) if np.sum(w) > 0 else np.ones(n) / n

            # Check convergence
            if i > 0:
                utility_prev = (
                    -w_prev @ mu
                    + 0.5 * risk_aversion * w_prev @ cov @ w_prev
                    + 0.5 * risk_aversion * (w_prev - tilted) @ cov @ (w_prev - tilted)
                )
                utility_curr = (
                    -w @ mu
                    + 0.5 * risk_aversion * w @ cov @ w
                    + 0.5 * risk_aversion * (w - tilted) @ cov @ (w - tilted)
                )
                if abs(utility_curr - utility_prev) < tolerance:
                    break

            w_prev = w.copy()

        w = np.clip(w, 0.0, None)
        return w / np.sum(w) if np.sum(w) > 0 else np.ones(n) / n

    # ------------------------------------------------------------------
    # Factor allocation
    # ------------------------------------------------------------------

    def factor_allocation(
        self,
        factor_exposures: np.ndarray,
        factor_returns: np.ndarray,
    ) -> np.ndarray:
        """Allocate based on factor exposures and expected factor returns.

        Computes implied asset returns from factor model, then optimizes.
        """

        B = np.asarray(factor_exposures, dtype=float)  # (n_assets, n_factors)
        f = np.asarray(factor_returns, dtype=float)  # (n_factors,)

        if B.shape[1] != len(f):
            raise ValueError(
                f"Factor exposures have {B.shape[1]} factors but {len(f)} factor returns"
            )

        # Implied returns: mu_i = sum_k B_ik * f_k
        mu = B @ f
        # Simple diagonal covariance from factor model
        diag = np.sum(B**2, axis=1) * 0.01 + 1e-4
        cov = np.diag(diag)

        return self.strategic_allocation(mu, cov, risk_aversion=1.0)

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def black_litterman(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        views: np.ndarray,
        view_assets: np.ndarray,
        tau: float = 0.05,
        omega: np.ndarray | None = None,
    ) -> np.ndarray:
        """Black-Litterman model: blend market equilibrium with investor views.

        Args:
            cov_matrix: (n, n) covariance matrix
            market_weights: (n,) market capitalization weights
            views: (k,) expected returns for viewed assets
            view_assets: (k,) indices of assets with views
            tau: scaling factor for prior uncertainty
            omega: (k, k) view uncertainty matrix (default: diagonal from cov)
        """

        cov = np.asarray(cov_matrix, dtype=float)
        w_mkt = np.asarray(market_weights, dtype=float)
        views = np.asarray(views, dtype=float)
        view_assets = np.asarray(view_assets, dtype=int)
        n = len(w_mkt)

        if cov.shape != (n, n):
            raise ValueError("Covariance matrix dimension mismatch")
        if len(views) != len(view_assets):
            raise ValueError("views and view_assets must have same length")

        # No views → return market weights
        if len(views) == 0:
            return w_mkt / np.sum(w_mkt)

        # Reverse optimization: implied equilibrium returns
        # pi = delta * Sigma * w_mkt (using risk aversion = 2.5)
        delta = 2.5
        pi = delta * cov @ w_mkt

        # View matrix P (k x n): each row picks out the viewed asset
        k = len(views)
        P = np.zeros((k, n))
        for i, asset_idx in enumerate(view_assets):
            P[i, asset_idx] = 1.0

        # View uncertainty
        if omega is None:
            omega = np.diag(np.diag(P @ cov @ P.T)) * 0.5

        # Posterior returns (He & Litterman 1999)
        # mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
        tau_sigma = tau * cov
        # Use solve instead of inv for stability
        try:
            tau_sigma_inv = np.linalg.solve(tau_sigma, np.eye(n))
        except np.linalg.LinAlgError:
            tau_sigma_inv = np.linalg.inv(tau_sigma + 1e-12 * np.eye(n))

        omega_inv = np.linalg.inv(omega + 1e-12 * np.eye(k))

        M_inv = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        mu_bl = M_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ views)

        # Posterior covariance
        cov_bl = cov + M_inv

        # Optimize with BL returns
        return self.strategic_allocation(mu_bl, cov_bl, risk_aversion=delta)

    # ------------------------------------------------------------------
    # Risk parity
    # ------------------------------------------------------------------

    def risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Equal risk contribution (risk parity) allocation.

        Finds weights such that each asset contributes equally to portfolio risk.
        """

        cov = np.asarray(cov_matrix, dtype=float)
        n = cov.shape[0]

        if cov.shape != (n, n):
            raise ValueError("Covariance matrix must be square")
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("Covariance matrix must be symmetric")

        # Minimize sum of squared differences in risk contributions
        # using projected gradient with simplex constraint

        def objective(w: np.ndarray) -> float:
            port_var = w @ cov @ w
            if port_var <= 0:
                return 1e10
            marginal = cov @ w
            risk_contrib = w * marginal / port_var
            target = 1.0 / n
            return np.sum((risk_contrib - target) ** 2)

        # Projected gradient descent with simplex projection
        w = np.ones(n) / n
        learning_rate = 0.01
        max_iter = 10000
        tolerance = 1e-12

        for i in range(max_iter):
            # Compute gradient numerically (avoiding issues with objective non-smoothness)
            eps = 1e-6
            grad = np.zeros(n)
            for j in range(n):
                w_perturb = w.copy()
                w_perturb[j] += eps
                # Project perturbed weights onto simplex
                w_perturb_proj = self._project_simplex(w_perturb)
                f_perturb = objective(w_perturb_proj)
                f_base = objective(w)
                grad[j] = (f_perturb - f_base) / eps

            # Gradient step
            w_new = w - learning_rate * grad

            # Project onto simplex
            w = self._project_simplex(w_new)

            # Check convergence: change in objective
            if i > 0:
                obj_change = abs(objective(w) - obj_prev)
                if obj_change < tolerance:
                    break

            obj_prev = objective(w)

        w = np.clip(w, 0.0, None)
        return w / np.sum(w) if np.sum(w) > 0 else np.ones(n) / n

    # ------------------------------------------------------------------
    # Helper: simplex projection (Duchi et al.)
    # ------------------------------------------------------------------

    @staticmethod
    def _project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
        """Project vector v onto the simplex {w | sum(w)=z, w>=0}.

        Duchi, Shalev-Shwartz, Singer (2008) — O(n log n) algorithm.
        """
        # Sort in descending order
        sorted_v = np.sort(v)[::-1]
        # Cumulative sum
        cssv = np.cumsum(sorted_v)

        # Find rho: largest such that sorted_v[rho] > (cssv[rho] - z) / (rho + 1)
        rho = 0
        for i in range(len(sorted_v)):
            if sorted_v[i] > (cssv[i] - z) / (i + 1):
                rho = i + 1

        # Compute theta
        theta = (cssv[rho] - z) / (rho + 1) if rho > 0 else 0.0

        # Project: max(v - theta, 0)
        w = np.maximum(v - theta, 0.0)
        # Renormalize to target sum
        return w * z / np.sum(w)