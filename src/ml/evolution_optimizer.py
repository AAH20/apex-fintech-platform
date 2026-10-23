"""Evolutionary optimization for SignalRefiner hyperparameters — numpy-only genetic algorithm.

Optimizes vol_target, momentum_lookback, mean_reversion_speed, regime_lookback,
min_signal_strength for enhanced alpha generation. Tournament selection, blend crossover,
Gaussian mutation. Convergence detection. Fitness: Sharpe/Return/Sortino via BacktestEngine.

Anti-loop: 199 lines, single file, no helpers, test-first ready.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    from ml.signal_refiner import SignalRefiner
    from ml.pipeline import run_backtest
except ImportError:
    from signal_refiner import SignalRefiner  # fallback
    from pipeline import run_backtest


class EvolutionOptimizer:
    """Genetic algorithm optimizer for SignalRefiner hyperparameters."""

    def __init__(self, population_size: int = 40, generations: int = 30,
                 mutation_rate: float = 0.2, mutation_sigma: float = 0.2,
                 crossover_rate: float = 0.9, crossover_alpha: float = 0.5,
                 elitism_count: int = 2, tournament_size: int = 3,
                 seed: int = 42, fitness_metric: str = "sharpe_ratio",
                 allow_short: bool = False):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.crossover_alpha = crossover_alpha
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.seed = seed
        self.fitness_metric = fitness_metric
        self.allow_short = allow_short
        self.rng = np.random.default_rng(seed)

        self.param_bounds = {"vol_target": (0.05, 0.50),
                             "momentum_lookback": (1, 20),
                             "mean_reversion_speed": (0.01, 0.5),
                             "regime_lookback": (10, 60),
                             "min_signal_strength": (0.1, 0.8)}
        self.dim = len(self.param_bounds)
        self.fitness_history: List[float] = []
        self.best_fitness: float = float("-inf")
        self.best_params: Optional[Dict[str, float]] = None
        self.fitness_eval_count = 0

    def _decode(self, genes: np.ndarray) -> Dict[str, float]:
        params = {}
        for i, (name, (lo, hi)) in enumerate(self.param_bounds.items()):
            bounded = lo + (hi - lo) * (1 + np.tanh(float(genes[i]))) / 2.0
            params[name] = max(lo, min(hi, round(bounded, 6)))
        return params

    def _evaluate(self, genes: np.ndarray, predictions: np.ndarray,
                  returns: np.ndarray) -> float:
        params = self._decode(genes)
        try:
            refiner = SignalRefiner(vol_target=params["vol_target"],
                                    momentum_lookback=int(params["momentum_lookback"]),
                                    mean_reversion_speed=params["mean_reversion_speed"],
                                    regime_lookback=int(params["regime_lookback"]),
                                    min_signal_strength=params["min_signal_strength"])
            signals = refiner.refine(predictions, returns)
            metrics = run_backtest(predictions=signals, returns=returns,
                                   allow_short=self.allow_short)
            self.fitness_eval_count += 1
            if self.fitness_metric in metrics and np.isfinite(metrics[self.fitness_metric]):
                return float(metrics[self.fitness_metric])
            if "sharpe_ratio" in metrics and np.isfinite(metrics["sharpe_ratio"]):
                return float(metrics["sharpe_ratio"])
            if "total_return" in metrics and np.isfinite(metrics["total_return"]):
                return float(metrics["total_return"])
        except Exception:
            pass
        return float("-inf")

    def _tournament_select(self, population: List[np.ndarray],
                           fitnesses: List[float]) -> np.ndarray:
        indices = self.rng.choice(len(population), self.tournament_size, replace=False)
        tournament = [(population[i], fitnesses[i]) for i in indices]
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]

    def _blend_crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c1 = np.zeros_like(p1); c2 = np.zeros_like(p2)
        for i in range(len(p1)):
            lo, hi = min(p1[i], p2[i]), max(p1[i], p2[i])
            span = hi - lo
            c1[i] = (p1[i] + p2[i]) / 2.0 - self.crossover_alpha * span
            c2[i] = (p1[i] + p2[i]) / 2.0 + self.crossover_alpha * span
        return c1, c2

    def _mutate(self, gene: float) -> float:
        if self.rng.random() >= self.mutation_rate:
            return gene
        return gene + self.rng.normal(0, self.mutation_sigma)

    def _crossover_mutate(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < self.crossover_rate:
            c1, c2 = self._blend_crossover(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()
        c1 = np.array([self._mutate(g) for g in c1])
        c2 = np.array([self._mutate(g) for g in c2])
        return c1, c2

    def evolve(self, predictions: np.ndarray, returns: np.ndarray,
               baseline_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        population = [self.rng.uniform(-3, 3, size=self.dim) for _ in range(self.population_size)]
        fitnesses = [self._evaluate(p, predictions, returns) for p in population]
        self.fitness_eval_count += self.population_size

        best_idx = int(np.argmax(fitnesses))
        best_fitness = fitnesses[best_idx]
        best_genes = population[best_idx]
        self.best_fitness = best_fitness
        self.best_params = self._decode(best_genes)
        self.fitness_history = [best_fitness]

        for _ in range(1, self.generations + 1):
            elite_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:self.elitism_count]
            new_pop = [population[i].copy() for i in elite_idx]

            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover_mutate(p1, p2)
                new_pop.extend([c1, c2])

            population = new_pop[:self.population_size]
            fitnesses = [self._evaluate(p, predictions, returns) for p in population]
            self.fitness_eval_count += len(population)

            gen_best = max(fitnesses)
            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                self.best_params = self._decode(population[np.argmax(fitnesses)])
            self.fitness_history.append(gen_best)

            if len(self.fitness_history) > 5 and np.all(np.abs(np.diff(self.fitness_history[-5:])) < 1e-6):
                break

        improvement_pct = 0.0
        if baseline_params:
            try:
                base = self._evaluate(self._decode(baseline_params), predictions, returns)
                if np.isfinite(base) and self.best_fitness > base:
                    improvement_pct = (self.best_fitness - base) / abs(base) * 100.0
            except Exception:
                pass

        return {"best_params": self.best_params,
                "best_fitness": float(self.best_fitness) if np.isfinite(self.best_fitness) else 0.0,
                "fitness_history": [float(f) for f in self.fitness_history],
                "improvement_pct": float(improvement_pct),
                "evaluations": self.fitness_eval_count}


def optimize_signal_refiner_parameters(predictions: np.ndarray, returns: np.ndarray,
                                       population_size: int = 40, generations: int = 30,
                                       fitness_metric: str = "sharpe_ratio",
                                       allow_short: bool = False,
                                       baseline_params: Optional[Dict[str, float]] = None,
                                       seed: int = 42) -> Dict[str, Any]:
    """Convenience function: optimize SignalRefiner params via EA for alpha enhancement."""
    opt = EvolutionOptimizer(population_size=population_size, generations=generations,
                             fitness_metric=fitness_metric, allow_short=allow_short, seed=seed)
    return opt.evolve(predictions, returns, baseline_params)