#!/usr/bin/env python3
"""Comprehensive evaluation of outpost agent across all experiments with different configs."""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List

from cogames.cogs_vs_clips.exploration_experiments import (
    Experiment1Mission,
    Experiment2Mission,
    Experiment3Mission,
    Experiment4Mission,
    Experiment5Mission,
    Experiment6Mission,
    Experiment7Mission,
    Experiment8Mission,
    Experiment9Mission,
    Experiment10Mission,
)
from cogames.policy.scripted_agent_outpost import ScriptedAgentPolicy, Hyperparameters
from mettagrid import MettaGridEnv

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Hyperparameter configuration to test."""

    name: str
    recharge_start: int = 65
    energy_buffer: int = 20
    min_energy_for_silicon: int = 70
    charger_search_threshold: int = 40
    prefer_nearby: bool = True


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    experiment: str
    config_name: str
    total_reward: float
    hearts_assembled: int
    steps_taken: int
    final_energy: int
    extractors_used: Dict[str, int]
    success: bool  # True if reward > 0


# Configurations to test
CONFIGS = [
    EvalConfig(
        name="baseline",
        recharge_start=65,
        energy_buffer=20,
        min_energy_for_silicon=70,
    ),
    EvalConfig(
        name="conservative",
        recharge_start=70,
        energy_buffer=25,
        min_energy_for_silicon=80,
    ),
    EvalConfig(
        name="aggressive",
        recharge_start=60,
        energy_buffer=15,
        min_energy_for_silicon=65,
    ),
    EvalConfig(
        name="silicon_focused",
        recharge_start=75,
        energy_buffer=20,
        min_energy_for_silicon=90,
    ),
]

# All experiments
EXPERIMENTS = [
    ("exp1", Experiment1Mission),
    ("exp2", Experiment2Mission),
    ("exp3", Experiment3Mission),
    ("exp4", Experiment4Mission),
    ("exp5", Experiment5Mission),
    ("exp6", Experiment6Mission),
    ("exp7", Experiment7Mission),
    ("exp8", Experiment8Mission),
    ("exp9", Experiment9Mission),
    ("exp10", Experiment10Mission),
]


def run_evaluation(
    experiment_name: str, mission_class, config: EvalConfig, max_steps: int = 1000
) -> EvalResult:
    """Run a single evaluation."""
    print(f"\n{'='*80}")
    print(f"Testing {experiment_name} with config '{config.name}'")
    print(f"{'='*80}")

    # Create mission and environment
    mission = mission_class()
    mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
    env_config = mission.make_env()
    env = MettaGridEnv(env_config)

    # Create policy with custom hyperparameters
    policy = ScriptedAgentPolicy(env)
    impl = policy._impl

    # Apply hyperparameters
    impl.RECHARGE_START = config.recharge_start
    impl.hyperparams.energy_buffer = config.energy_buffer
    impl.hyperparams.min_energy_for_silicon = config.min_energy_for_silicon
    impl.hyperparams.charger_search_threshold = config.charger_search_threshold
    impl.hyperparams.prefer_nearby = config.prefer_nearby

    # Reset and run
    observations = env.reset()
    agent_policy = policy.agent_policy(0)

    total_reward = 0.0
    for step in range(max_steps):
        action = agent_policy.step(observations[0])
        observations, rewards, dones, truncated, info = env.step([action])
        total_reward += rewards[0]

        if dones[0] or truncated[0]:
            break

    # Gather results
    state = agent_policy._state

    # Count extractor uses
    extractors_used = {}
    for resource_type in ["carbon", "oxygen", "germanium", "silicon"]:
        total_uses = sum(
            e.total_harvests for e in impl.extractor_memory.get_by_type(resource_type)
        )
        if total_uses > 0:
            extractors_used[resource_type] = total_uses

    result = EvalResult(
        experiment=experiment_name,
        config_name=config.name,
        total_reward=total_reward,
        hearts_assembled=state.hearts_assembled,
        steps_taken=step + 1,
        final_energy=state.energy,
        extractors_used=extractors_used,
        success=total_reward > 0,
    )

    # Print summary
    print(f"\nResults:")
    print(f"  Reward: {result.total_reward:.2f}")
    print(f"  Hearts: {result.hearts_assembled}")
    print(f"  Steps: {result.steps_taken}/{max_steps}")
    print(f"  Final Energy: {result.final_energy}")
    print(f"  Extractors Used: {extractors_used}")
    print(f"  Success: {'✅' if result.success else '❌'}")

    return result


def main():
    """Run all evaluations."""
    print("\n" + "=" * 80)
    print("PHASE 1 COMPREHENSIVE EVALUATION")
    print("Testing all experiments with multiple hyperparameter configurations")
    print("=" * 80)

    all_results: List[EvalResult] = []

    # Run each experiment with each config
    for exp_name, mission_class in EXPERIMENTS:
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT: {exp_name.upper()}")
        print(f"{'#'*80}")

        for config in CONFIGS:
            try:
                result = run_evaluation(exp_name, mission_class, config, max_steps=1000)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error running {exp_name} with {config.name}: {e}")
                logger.exception("Evaluation error")

    # Generate summary report
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)

    # Group by experiment
    by_experiment: Dict[str, List[EvalResult]] = {}
    for result in all_results:
        if result.experiment not in by_experiment:
            by_experiment[result.experiment] = []
        by_experiment[result.experiment].append(result)

    # Print per-experiment summary
    for exp_name in sorted(by_experiment.keys()):
        results = by_experiment[exp_name]
        print(f"\n{exp_name.upper()}:")

        for result in results:
            status = "✅" if result.success else "❌"
            print(
                f"  {status} {result.config_name:15s}: "
                f"Reward={result.total_reward:.1f}, "
                f"Hearts={result.hearts_assembled}, "
                f"Steps={result.steps_taken}"
            )

    # Best config per experiment
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION PER EXPERIMENT")
    print("=" * 80)

    for exp_name in sorted(by_experiment.keys()):
        results = by_experiment[exp_name]
        best = max(results, key=lambda r: (r.total_reward, -r.steps_taken))
        print(
            f"{exp_name:6s}: {best.config_name:15s} "
            f"(reward={best.total_reward:.1f}, hearts={best.hearts_assembled})"
        )

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    success_count = sum(1 for r in all_results if r.success)
    total_count = len(all_results)
    avg_reward = sum(r.total_reward for r in all_results) / total_count
    avg_hearts = sum(r.hearts_assembled for r in all_results) / total_count

    print(f"Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Hearts Assembled: {avg_hearts:.2f}")

    # Config performance
    print("\n" + "=" * 80)
    print("CONFIGURATION PERFORMANCE")
    print("=" * 80)

    by_config: Dict[str, List[EvalResult]] = {}
    for result in all_results:
        if result.config_name not in by_config:
            by_config[result.config_name] = []
        by_config[result.config_name].append(result)

    for config_name in sorted(by_config.keys()):
        results = by_config[config_name]
        successes = sum(1 for r in results if r.success)
        avg_reward = sum(r.total_reward for r in results) / len(results)
        avg_hearts = sum(r.hearts_assembled for r in results) / len(results)

        print(
            f"{config_name:15s}: {successes}/{len(results)} success, "
            f"avg_reward={avg_reward:.2f}, avg_hearts={avg_hearts:.2f}"
        )

    # Save detailed results to JSON
    output_file = "phase1_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, cls=CustomJsonEncoder)
    print(f"\nDetailed results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()

