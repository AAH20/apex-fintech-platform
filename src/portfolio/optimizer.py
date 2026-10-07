"""Cardinality-constrained portfolio optimizer (numpy only).
Solves: min w^T Σ w - λ μ^T w  s.t. sum(w)=1, w>=0, |{i: w_i>0}| <= k
Plus optional sector constraints. NP-hard MIQP (Bienstock 1996).
Greedy heuristic: (1) Solve unconstrained QP, (2) Keep top-K, (3) Re-solve."""
from typing import Optional

import numpy as np

class CardinalityConstrainedOptimizer:
    """Portfolio optimizer with cardinality constraints.
    Parameters: mu (n,), Sigma (n,n), k, risk_aversion, sectors, sector_limits."""

    def __init__(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        k: int,
        risk_aversion: float = 1.0,
        sectors: Optional[np.ndarray] = None,
        sector_limits: Optional[dict] = None,
    ):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.n = len(self.mu)
        self.k = min(k, self.n)
        self.risk_aversion = risk_aversion
        self.sectors = np.asarray(sectors) if sectors is not None else None
        self.sector_limits = sector_limits
        self._weights: Optional[np.ndarray] = None
        self._objective: Optional[float] = None

    def _solve_eq(self, idx: np.ndarray) -> np.ndarray:
        """Solve equality-constrained QP via KKT, clip negatives, renormalize."""
        m = len(idx)
        if m == 0:
            return np.zeros(self.n)
        mu_s, Sigma_s = self.mu[idx], self.Sigma[np.ix_(idx, idx)]
        KKT = np.zeros((m + 1, m + 1))
        KKT[:m, :m] = 2.0 * Sigma_s
        KKT[:m, m] = 1.0
        KKT[m, :m] = 1.0
        rhs = np.concatenate([self.risk_aversion * mu_s, [1.0]])
        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            return np.full(self.n, 1.0 / self.n)
        w = np.zeros(self.n)
        w[idx] = np.maximum(sol[:m], 0.0)
        if w.sum() > 0:
            w /= w.sum()
        else:
            w[idx] = 1.0 / m
        return w
    def _solve_nonneg(self, idx: np.ndarray) -> np.ndarray:
        """Solve QP with non-negativity via active-set."""
        active = list(idx)
        for _ in range(len(idx)):
            if not active:
                return np.zeros(self.n)
            w = self._solve_eq(np.array(active))
            neg = [i for i in active if w[i] < -1e-10]
            if not neg:
                return w
            active.remove(min(neg, key=lambda i: w[i]))
        return self._solve_eq(np.array(active)) if active else np.zeros(self.n)
    def _select_top_k(self, w: np.ndarray) -> np.ndarray:
        """Select top-K by weight, ensuring sector diversity."""
        order = np.argsort(-w)
        selected: list[int] = []
        sector_totals: dict[int, float] = {}
        has_sectors = self.sectors is not None and self.sector_limits is not None
        if has_sectors:
            for sec in self.sector_limits:
                for i in order:
                    if self.sectors[i] == sec:
                        selected.append(i)
                        sector_totals[sec] = w[i]
                        break
        for i in order:
            if len(selected) >= self.k:
                break
            if i in selected:
                continue
            if has_sectors:
                sec = self.sectors[i]
                limit = self.sector_limits.get(sec, 1.0)
                if sector_totals.get(sec, 0.0) + w[i] > limit:
                    continue
                sector_totals[sec] = sector_totals.get(sec, 0.0) + w[i]
            selected.append(i)
        return np.array(selected[: self.k])

    def _apply_sector_limits(self, w: np.ndarray) -> np.ndarray:
        """Iteratively cap over-limit sectors and redistribute."""
        if self.sectors is None or self.sector_limits is None:
            return w
        w = w.copy()
        for _ in range(100):
            capped = np.zeros(self.n, dtype=bool)
            for sec, limit in self.sector_limits.items():
                mask = self.sectors == sec
                if w[mask].sum() > limit:
                    w[mask] *= limit / w[mask].sum()
                    capped |= mask
            excess = 1.0 - w.sum()
            if excess > 1e-10 and (~capped).any():
                w[~capped] += excess / (~capped).sum()
            over = any(
                w[self.sectors == sec].sum() > lim + 1e-10
                for sec, lim in self.sector_limits.items()
            )
            if not over:
                break
        return w

    def optimize(self) -> None:
        """Run the greedy heuristic and store results."""
        if self.k <= 0:
            self._weights = np.zeros(self.n)
            self._objective = np.inf
            return
        w_full = self._solve_eq(np.arange(self.n))
        top_k = self._select_top_k(w_full)
        w = self._solve_nonneg(top_k)
        w = self._apply_sector_limits(w)
        w[w < 1e-8] = 0.0
        if w.sum() > 0:
            w /= w.sum()
        self._weights = w
        idx = np.where(w > 1e-8)[0]
        if len(idx) > 0:
            self._objective = (
                w[idx] @ self.Sigma[np.ix_(idx, idx)] @ w[idx]
                - self.risk_aversion * self.mu[idx] @ w[idx]
            )
        else:
            self._objective = np.inf

    def get_weights(self) -> np.ndarray:
        """Return optimized weights."""
        if self._weights is None:
            self.optimize()
        return self._weights

    def get_objective_value(self) -> float:
        """Return objective value of the optimized portfolio."""
        if self._objective is None:
            self.optimize()
        return self._objective
