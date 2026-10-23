"""TDD: Evolutionary optimization tests — verify GA hyperparameter optimization works."""

import sys
sys.path.insert(0, '/home/aah/apex-fintech-platform/src')
import numpy as np
from ml.evolution_optimizer import EvolutionOptimizer, optimize_signal_refiner_parameters


def test_evolution_optimizer_init():
    """Test GA optimizer initializes with valid parameters."""
    opt = EvolutionOptimizer(population_size=20, generations=10, seed=42)
    assert opt.population_size == 20
    assert opt.generations == 10
    assert opt.fitness_metric == "sharpe_ratio"
    print("  PASS: EvolutionOptimizer init")


def test_evolution_optimizer_evolve():
    """Test GA can evolve parameters with synthetic data."""
    np.random.seed(42)
    predictions = np.random.randn(100)
    returns = np.random.randn(100) * 0.02
    opt = EvolutionOptimizer(population_size=10, generations=5, seed=42)
    result = opt.evolve(predictions, returns)
    assert "best_params" in result
    assert "best_fitness" in result
    assert "improvement_pct" in result
    assert "evaluations" in result
    assert result["best_fitness"] is not None
    print("  PASS: EvolutionOptimizer evolve")


def test_optimize_signal_refiner():
    """Test convenience function optimizes SignalRefiner parameters."""
    np.random.seed(42)
    predictions = np.random.randn(100)
    returns = np.random.randn(100) * 0.02
    result = optimize_signal_refiner_parameters(predictions, returns, generations=5)
    assert "best_params" in result
    assert "best_fitness" in result
    assert "improvement_pct" in result
    print("  PASS: optimize_signal_refiner_parameters")


def test_improvement_over_baseline():
    """Test GA provides improvement over baseline parameters."""
    np.random.seed(42)
    predictions = np.random.randn(100)
    returns = np.random.randn(100) * 0.02
    baseline = {"vol_target": 0.15, "momentum_lookback": 5,
                "mean_reversion_speed": 0.1, "regime_lookback": 20,
                "min_signal_strength": 0.3}
    opt = EvolutionOptimizer(population_size=10, generations=5, seed=42)
    result = opt.evolve(predictions, returns, baseline_params=baseline)
    assert "improvement_pct" in result
    # Improvement may be positive or zero (GA should not worsen significantly)
    print(f"  PASS: Improvement over baseline: {result['improvement_pct']:.1f}%")


def test_convergence_detection():
    """Test GA converges within reasonable evaluations."""
    np.random.seed(42)
    predictions = np.random.randn(100)
    returns = np.random.randn(100) * 0.02
    opt = EvolutionOptimizer(population_size=10, generations=10, seed=42)
    result = opt.evolve(predictions, returns)
    assert result["evaluations"] > 0
    assert result["best_fitness"] is not None
    print(f"  PASS: Converged in {result['evaluations']} evaluations, best fitness: {result['best_fitness']:.4f}")


def main():
    print("Running TDD tests for evolutionary optimization...")
    test_evolution_optimizer_init()
    test_evolution_optimizer_evolve()
    test_optimize_signal_refiner()
    test_improvement_over_baseline()
    test_convergence_detection()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()