#!/usr/bin/env python3
"""
Recipe: A/B Testing Different Algorithm Implementations
=========================================================

Problem: You want to compare two different algorithm implementations 
(e.g., PPO vs SAC, or two versions of the same algorithm) in a controlled way.

Solution: Use Axiom's experiment system to run both algorithms with identical
configurations and seeds, then compare their results systematically.

This recipe demonstrates:
- Running multiple algorithm variants with same configuration
- Ensuring fair comparison through seed control
- Collecting and comparing metrics
- Statistical analysis of results
- Generating comparison reports
"""

import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from metta.sweep.axiom.core import Pipeline
from metta.sweep.axiom.experiment import AxiomExperiment, RunHandle
from metta.sweep.axiom.experiment_spec import ExperimentSpec, AxiomControls
from metta.sweep.axiom.train_and_eval import (
    TrainAndEvalSpec,
    TrainAndEvalExperiment,
    create_quick_experiment
)


# =============================================================================
# ALGORITHM IMPLEMENTATIONS
# =============================================================================

def algorithm_a_train(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Algorithm A implementation (e.g., standard PPO).
    """
    print("  [Algorithm A] Training with standard PPO")
    
    # In real scenario, this would be actual PPO training
    # For demo, we'll simulate with predictable results
    np.random.seed(state.get("seed", 42))
    
    rewards = []
    for step in range(100):
        # Algorithm A has steady improvement
        reward = 50 + step * 0.5 + np.random.normal(0, 5)
        rewards.append(reward)
    
    state["training_rewards"] = rewards
    state["final_reward"] = rewards[-1]
    state["algorithm"] = "PPO_Standard"
    state["convergence_step"] = 60
    
    return state


def algorithm_b_train(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Algorithm B implementation (e.g., modified PPO with different advantage computation).
    """
    print("  [Algorithm B] Training with modified PPO")
    
    np.random.seed(state.get("seed", 42))
    
    rewards = []
    for step in range(100):
        # Algorithm B has faster initial learning but plateaus
        if step < 40:
            reward = 30 + step * 1.5 + np.random.normal(0, 7)
        else:
            reward = 85 + np.random.normal(0, 3)
        rewards.append(reward)
    
    state["training_rewards"] = rewards
    state["final_reward"] = rewards[-1]
    state["algorithm"] = "PPO_Modified"
    state["convergence_step"] = 40
    
    return state


def algorithm_c_train(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Algorithm C implementation (e.g., SAC - Soft Actor-Critic).
    """
    print("  [Algorithm C] Training with SAC")
    
    np.random.seed(state.get("seed", 42))
    
    rewards = []
    for step in range(100):
        # Algorithm C has slow start but best final performance
        if step < 20:
            reward = 20 + step * 0.3 + np.random.normal(0, 10)
        else:
            reward = 40 + (step - 20) * 1.0 + np.random.normal(0, 4)
        rewards.append(reward)
    
    state["training_rewards"] = rewards
    state["final_reward"] = rewards[-1]
    state["algorithm"] = "SAC"
    state["convergence_step"] = 80
    
    return state


# =============================================================================
# A/B TESTING FRAMEWORK
# =============================================================================

@dataclass
class ABTestResult:
    """Results from an A/B test comparison."""
    algorithm: str
    metrics: Dict[str, float]
    run_handle: RunHandle
    raw_data: Dict[str, Any]


@dataclass
class ComparisonReport:
    """Statistical comparison report between algorithms."""
    winner: str
    confidence: float
    metrics_comparison: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Any]
    recommendation: str


class ABTestRunner:
    """
    Runner for A/B testing different algorithm implementations.
    """
    
    def __init__(self, base_spec: ExperimentSpec, algorithms: Dict[str, Any]):
        """
        Initialize A/B test runner.
        
        Args:
            base_spec: Base experiment configuration
            algorithms: Dict mapping algorithm names to implementations
        """
        self.base_spec = base_spec
        self.algorithms = algorithms
        self.results: Dict[str, ABTestResult] = {}
        
    def run_algorithm(self, name: str, implementation: Any, 
                     seed: Optional[int] = None) -> ABTestResult:
        """
        Run a single algorithm implementation.
        
        Args:
            name: Algorithm name
            implementation: Algorithm implementation (function or pipeline)
            seed: Random seed for reproducibility
            
        Returns:
            Test results for this algorithm
        """
        print(f"\n{'='*60}")
        print(f"Running Algorithm: {name}")
        print(f"{'='*60}")
        
        # Create spec for this algorithm
        spec = ExperimentSpec(
            name=f"{self.base_spec.name}_{name}",
            description=f"A/B test run for {name}",
            config={**self.base_spec.config, "algorithm": name},
            controls=AxiomControls(
                seed=seed or self.base_spec.controls.seed,
                enforce_determinism=True
            )
        )
        
        # Create pipeline with the algorithm
        def pipeline_factory(config):
            pipeline = Pipeline(name=f"ab_test_{name}")
            pipeline.stage("initialize", lambda _: {"seed": spec.controls.seed})
            
            if callable(implementation):
                pipeline.stage("train", implementation)
            else:
                pipeline.join("train", implementation)
            
            pipeline.stage("evaluate", self._evaluate_algorithm)
            pipeline.stage("collect_metrics", self._collect_metrics)
            
            return pipeline
        
        # Run experiment
        experiment = AxiomExperiment(spec, pipeline_factory)
        experiment.prepare()
        run_handle = experiment.run(tag=f"algorithm_{name}")
        
        # Extract results
        manifest = run_handle.manifest()
        result = ABTestResult(
            algorithm=name,
            metrics=manifest["pipeline_result"].get("metrics", {}),
            run_handle=run_handle,
            raw_data=manifest["pipeline_result"]
        )
        
        self.results[name] = result
        return result
    
    def _evaluate_algorithm(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained algorithm."""
        # In real scenario, this would run evaluation episodes
        # For demo, we'll compute metrics from training data
        
        rewards = state.get("training_rewards", [])
        
        if rewards:
            state["eval_mean_reward"] = np.mean(rewards[-20:])  # Last 20 episodes
            state["eval_std_reward"] = np.std(rewards[-20:])
            state["eval_max_reward"] = max(rewards)
            state["improvement_rate"] = (rewards[-1] - rewards[0]) / len(rewards)
        
        return state
    
    def _collect_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics for comparison."""
        state["metrics"] = {
            "final_reward": state.get("final_reward", 0),
            "mean_reward": state.get("eval_mean_reward", 0),
            "std_reward": state.get("eval_std_reward", 0),
            "max_reward": state.get("eval_max_reward", 0),
            "convergence_step": state.get("convergence_step", 100),
            "improvement_rate": state.get("improvement_rate", 0),
            "training_time": state.get("training_time", 0),  # Would be actual time in real scenario
        }
        return state
    
    def run_all(self, seeds: Optional[List[int]] = None) -> Dict[str, List[ABTestResult]]:
        """
        Run all algorithms with multiple seeds.
        
        Args:
            seeds: List of seeds for multiple runs
            
        Returns:
            Dict mapping algorithm names to list of results across seeds
        """
        if seeds is None:
            seeds = [42]  # Single seed by default
        
        all_results = {name: [] for name in self.algorithms}
        
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"# Running with seed: {seed}")
            print(f"{'#'*60}")
            
            for name, implementation in self.algorithms.items():
                result = self.run_algorithm(name, implementation, seed)
                all_results[name].append(result)
        
        return all_results
    
    def compare(self, results: Optional[Dict[str, List[ABTestResult]]] = None) -> ComparisonReport:
        """
        Compare algorithm results and generate report.
        
        Args:
            results: Results to compare (uses self.results if None)
            
        Returns:
            Comparison report with statistical analysis
        """
        if results is None:
            results = {name: [self.results[name]] for name in self.results}
        
        print("\n" + "#"*60)
        print("# COMPARISON REPORT")
        print("#"*60)
        
        # Aggregate metrics across seeds
        aggregated = {}
        for name, result_list in results.items():
            metrics_lists = {key: [] for key in result_list[0].metrics}
            
            for result in result_list:
                for key, value in result.metrics.items():
                    metrics_lists[key].append(value)
            
            aggregated[name] = {
                key: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values),
                }
                for key, values in metrics_lists.items()
            }
        
        # Determine winner based on final reward
        winner = max(aggregated.keys(), 
                    key=lambda x: aggregated[x]["final_reward"]["mean"])
        
        # Calculate confidence (simplified - in reality would use proper statistical tests)
        winner_mean = aggregated[winner]["final_reward"]["mean"]
        winner_std = aggregated[winner]["final_reward"]["std"]
        
        confidence_scores = []
        for name in aggregated:
            if name != winner:
                other_mean = aggregated[name]["final_reward"]["mean"]
                other_std = aggregated[name]["final_reward"]["std"]
                
                # Simplified confidence calculation
                mean_diff = winner_mean - other_mean
                combined_std = np.sqrt(winner_std**2 + other_std**2)
                if combined_std > 0:
                    z_score = mean_diff / combined_std
                    confidence = min(0.99, max(0.5, 0.5 + z_score * 0.2))
                else:
                    confidence = 0.99
                confidence_scores.append(confidence)
        
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.99
        
        # Generate recommendation
        if overall_confidence > 0.95:
            recommendation = f"Strong evidence to use {winner}"
        elif overall_confidence > 0.80:
            recommendation = f"Moderate evidence to use {winner}"
        else:
            recommendation = "No clear winner - more testing needed"
        
        # Print comparison table
        print("\nMetric Comparison:")
        print("-" * 80)
        print(f"{'Metric':<20} | " + " | ".join(f"{name:<15}" for name in aggregated.keys()))
        print("-" * 80)
        
        for metric in ["final_reward", "convergence_step", "improvement_rate"]:
            values = [f"{aggregated[name][metric]['mean']:.2f}" 
                     for name in aggregated.keys()]
            print(f"{metric:<20} | " + " | ".join(f"{v:<15}" for v in values))
        
        print("\nWinner Analysis:")
        print("-" * 40)
        print(f"Winner: {winner}")
        print(f"Confidence: {overall_confidence:.1%}")
        print(f"Recommendation: {recommendation}")
        
        return ComparisonReport(
            winner=winner,
            confidence=overall_confidence,
            metrics_comparison=aggregated,
            statistical_tests={"simplified_z_test": True},  # In reality, would include proper tests
            recommendation=recommendation
        )


# =============================================================================
# EXAMPLE: COMPARING PPO VARIANTS
# =============================================================================

def compare_ppo_variants():
    """
    Example: Compare different PPO implementation variants.
    """
    print("="*70)
    print("COMPARING PPO VARIANTS")
    print("="*70)
    
    # Base configuration
    base_spec = ExperimentSpec(
        name="ppo_comparison",
        description="Comparing PPO implementation variants",
        config={
            "env": "CartPole-v1",
            "total_timesteps": 100000,
            "batch_size": 64,
        },
        controls=AxiomControls(seed=42, enforce_determinism=True)
    )
    
    # Define algorithm variants
    algorithms = {
        "PPO_Standard": algorithm_a_train,
        "PPO_Modified": algorithm_b_train,
    }
    
    # Run A/B test
    runner = ABTestRunner(base_spec, algorithms)
    
    # Run with multiple seeds for statistical validity
    results = runner.run_all(seeds=[42, 123, 456])
    
    # Generate comparison report
    report = runner.compare(results)
    
    return report


# =============================================================================
# EXAMPLE: ALGORITHM FAMILY COMPARISON
# =============================================================================

def compare_algorithm_families():
    """
    Example: Compare different algorithm families (PPO vs SAC).
    """
    print("\n" + "="*70)
    print("COMPARING ALGORITHM FAMILIES")
    print("="*70)
    
    # Base configuration
    base_spec = ExperimentSpec(
        name="algorithm_family_comparison",
        description="Comparing different RL algorithm families",
        config={
            "env": "HalfCheetah-v3",
            "total_timesteps": 500000,
        },
        controls=AxiomControls(seed=42, enforce_determinism=True)
    )
    
    # Define algorithms from different families
    algorithms = {
        "PPO": algorithm_a_train,
        "SAC": algorithm_c_train,
    }
    
    # Run comparison
    runner = ABTestRunner(base_spec, algorithms)
    results = runner.run_all(seeds=[42, 123])
    report = runner.compare(results)
    
    return report


# =============================================================================
# ADVANCED: PROGRESSIVE A/B TESTING
# =============================================================================

class ProgressiveABTester:
    """
    Progressive A/B testing that stops early if clear winner emerges.
    """
    
    def __init__(self, base_spec: ExperimentSpec, algorithms: Dict[str, Any]):
        self.base_spec = base_spec
        self.algorithms = algorithms
        self.runner = ABTestRunner(base_spec, algorithms)
    
    def run_progressive(self, max_seeds: int = 10, 
                       confidence_threshold: float = 0.95) -> ComparisonReport:
        """
        Run progressive A/B test, stopping early if confidence threshold met.
        
        Args:
            max_seeds: Maximum number of seeds to test
            confidence_threshold: Confidence level to stop testing
            
        Returns:
            Final comparison report
        """
        print("\n" + "="*70)
        print("PROGRESSIVE A/B TESTING")
        print("="*70)
        
        seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606][:max_seeds]
        all_results = {name: [] for name in self.algorithms}
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n### Round {i}/{max_seeds} (seed: {seed}) ###")
            
            # Run algorithms with this seed
            for name, impl in self.algorithms.items():
                result = self.runner.run_algorithm(name, impl, seed)
                all_results[name].append(result)
            
            # Check if we have enough confidence
            if i >= 3:  # Need at least 3 runs for meaningful statistics
                report = self.runner.compare(all_results)
                
                print(f"\nCurrent confidence: {report.confidence:.1%}")
                
                if report.confidence >= confidence_threshold:
                    print(f"\n✓ Confidence threshold reached after {i} rounds!")
                    print(f"  Winner: {report.winner}")
                    return report
        
        # Final report after all seeds
        print(f"\nCompleted all {max_seeds} rounds")
        return self.runner.compare(all_results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution demonstrating A/B testing workflows.
    """
    print("="*70)
    print("A/B TESTING ALGORITHMS COOKBOOK")
    print("="*70)
    print("\nThis recipe demonstrates how to compare different algorithm")
    print("implementations in a fair, reproducible way.\n")
    
    # Example 1: Compare PPO variants
    print("\n" + "="*70)
    print("EXAMPLE 1: PPO Variants Comparison")
    print("="*70)
    ppo_report = compare_ppo_variants()
    
    # Example 2: Compare algorithm families
    print("\n" + "="*70)
    print("EXAMPLE 2: Algorithm Families Comparison")
    print("="*70)
    family_report = compare_algorithm_families()
    
    # Example 3: Progressive testing
    print("\n" + "="*70)
    print("EXAMPLE 3: Progressive A/B Testing")
    print("="*70)
    
    spec = ExperimentSpec(
        name="progressive_test",
        config={"env": "CartPole-v1"},
        controls=AxiomControls(seed=42)
    )
    
    progressive = ProgressiveABTester(
        spec,
        {"Fast": algorithm_b_train, "Stable": algorithm_a_train}
    )
    
    progressive_report = progressive.run_progressive(
        max_seeds=6, 
        confidence_threshold=0.90
    )
    
    print("\n" + "="*70)
    print("A/B TESTING COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Always use multiple seeds for statistical validity")
    print("2. Control all variables except the algorithm being tested")
    print("3. Use deterministic seeds for reproducibility")
    print("4. Consider early stopping if clear winner emerges")
    print("5. Save all results for future analysis and comparison")


if __name__ == "__main__":
    main()