#!/usr/bin/env python3
"""Comprehensive evaluation of eval missions with multiple hyperparameter configs."""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from cogames.cogs_vs_clips.eval_missions import EVAL_MISSIONS
from cogames.policy.scripted_agent_outpost import Hyperparameters, ScriptedAgentPolicy
from mettagrid import MettaGridEnv

# Set up logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    name: str
    hyperparams: Hyperparameters


@dataclass
class EvaluationResult:
    experiment_name: str
    config_name: str
    reward: float
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    final_energy: int
    extractors_used: dict[str, int]
    success: bool
    error: Optional[str] = None


# Define hyperparameter configurations to test
HYPERPARAMETER_CONFIGS = {
    "baseline": HyperparameterConfig(
        name="baseline",
        hyperparams=Hyperparameters(
            energy_buffer=20,
            charger_search_threshold=40,
            prefer_nearby=True,
            cooldown_tolerance=20,
            depletion_threshold=0.2,
            track_efficiency=True,
            efficiency_weight=0.3,
            min_energy_for_silicon=70,
        ),
    ),
    "conservative": HyperparameterConfig(
        name="conservative",
        hyperparams=Hyperparameters(
            energy_buffer=30,
            charger_search_threshold=30,
            prefer_nearby=True,
            cooldown_tolerance=10,
            depletion_threshold=0.3,
            track_efficiency=True,
            efficiency_weight=0.5,
            min_energy_for_silicon=80,
            max_wait_turns=30,
        ),
    ),
    "aggressive": HyperparameterConfig(
        name="aggressive",
        hyperparams=Hyperparameters(
            energy_buffer=10,
            charger_search_threshold=50,
            prefer_nearby=False,
            cooldown_tolerance=30,
            depletion_threshold=0.1,
            track_efficiency=True,
            efficiency_weight=0.2,
            min_energy_for_silicon=60,
            max_wait_turns=75,
        ),
    ),
    "silicon_focused": HyperparameterConfig(
        name="silicon_focused",
        hyperparams=Hyperparameters(
            energy_buffer=15,
            charger_search_threshold=50,
            prefer_nearby=True,
            cooldown_tolerance=25,
            depletion_threshold=0.2,
            track_efficiency=True,
            efficiency_weight=0.3,
            min_energy_for_silicon=85,
            max_wait_turns=40,
        ),
    ),
}


class CustomJsonEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def evaluate_mission(
    mission_cls: type, config: HyperparameterConfig, max_steps: int = 1000
) -> EvaluationResult:
    """Evaluate a single mission with a given hyperparameter configuration."""
    mission = mission_cls()
    experiment_name = f"{mission.site.name}.{mission.name}"

    try:
        # Instantiate mission
        mission = mission.instantiate(mission.site.map_builder, num_cogs=1)

        # Create environment
        env_config = mission.make_env()
        env = MettaGridEnv(env_config)

        # Create policy with specific hyperparameters
        policy = ScriptedAgentPolicy(env)
        policy._impl.hyperparams = config.hyperparams

        # Reset and get agent policies
        obs = env.reset()
        agent_policies = [policy.agent_policy(i) for i in range(env.c_env.num_agents)]

        # Run episode
        total_reward = 0.0
        hearts_assembled = 0

        for step in range(max_steps):
            actions = [ap.step(obs[i]) for i, ap in enumerate(agent_policies)]
            obs, rewards, terminated, truncated, info = env.step(actions)

            total_reward += sum(rewards)
            if sum(rewards) > 0:
                hearts_assembled += 1

            if terminated[0] or truncated[0]:
                break

        # Get final state
        state = agent_policies[0]._state
        final_energy = int(state.energy) if hasattr(state, "energy") else 0

        # Get extractor usage from memory
        impl = policy._impl
        extractors_used = {}
        for resource_type in ["carbon", "oxygen", "germanium", "silicon", "charger"]:
            extractors = impl.extractor_memory.get_by_type(resource_type)
            total_uses = sum(e.total_harvests for e in extractors)
            if total_uses > 0:
                extractors_used[resource_type] = total_uses

        success = total_reward > 0

        return EvaluationResult(
            experiment_name=experiment_name,
            config_name=config.name,
            reward=float(total_reward),
            hearts_assembled=hearts_assembled,
            steps_taken=step + 1,
            max_steps=max_steps,
            final_energy=final_energy,
            extractors_used=extractors_used,
            success=success,
        )

    except Exception as e:
        logger.error(f"Error evaluating {experiment_name} with {config.name}: {e}")
        return EvaluationResult(
            experiment_name=experiment_name,
            config_name=config.name,
            reward=0.0,
            hearts_assembled=0,
            steps_taken=0,
            max_steps=max_steps,
            final_energy=0,
            extractors_used={},
            success=False,
            error=str(e),
        )


def main():
    print("=" * 80)
    print("EVAL MISSIONS COMPREHENSIVE EVALUATION")
    print("Testing all eval missions with multiple hyperparameter configurations")
    print("=" * 80)
    print()

    all_results = []
    max_steps = 1000

    for mission_cls in EVAL_MISSIONS:
        mission = mission_cls()
        print()
        print("#" * 80)
        print(f"# MISSION: {mission.name.upper()}")
        print("#" * 80)
        print()

        for config_name, config in HYPERPARAMETER_CONFIGS.items():
            print("=" * 80)
            print(f"Testing {mission.site.name}.{mission.name} with config '{config_name}'")
            print("=" * 80)
            print()

            result = evaluate_mission(mission_cls, config, max_steps)
            all_results.append(result)

            # Print results
            success_icon = "✅" if result.success else "❌"
            print("Results:")
            print(f"  Reward: {result.reward:.2f}")
            print(f"  Hearts: {result.hearts_assembled}")
            print(f"  Steps: {result.steps_taken}/{result.max_steps}")
            print(f"  Final Energy: {result.final_energy}")
            print(f"  Extractors Used: {result.extractors_used}")
            print(f"  Success: {success_icon}")
            if result.error:
                print(f"  Error: {result.error}")
            print()

    # Print summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print()

    # Group by mission
    missions_dict = {}
    for result in all_results:
        if result.experiment_name not in missions_dict:
            missions_dict[result.experiment_name] = []
        missions_dict[result.experiment_name].append(result)

    for mission_name in sorted(missions_dict.keys()):
        results = missions_dict[mission_name]
        print(f"{mission_name.upper()}:")
        for result in results:
            success_icon = "✅" if result.success else "❌"
            print(
                f"  {success_icon} {result.config_name:15s}: "
                f"Reward={result.reward:.1f}, Hearts={result.hearts_assembled}, Steps={result.steps_taken}"
            )
        print()

    # Print best config per mission
    print("=" * 80)
    print("BEST CONFIGURATION PER MISSION")
    print("=" * 80)
    for mission_name in sorted(missions_dict.keys()):
        results = missions_dict[mission_name]
        best = max(results, key=lambda r: (r.reward, r.hearts_assembled))
        short_name = mission_name.split(".")[-1]
        print(
            f"{short_name:20s}: {best.config_name:15s} "
            f"(reward={best.reward:.1f}, hearts={best.hearts_assembled})"
        )

    # Overall statistics
    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    total_runs = len(all_results)
    successful_runs = sum(1 for r in all_results if r.success)
    success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
    avg_reward = sum(r.reward for r in all_results) / total_runs if total_runs > 0 else 0
    avg_hearts = sum(r.hearts_assembled for r in all_results) / total_runs if total_runs > 0 else 0

    print(f"Success Rate: {successful_runs}/{total_runs} ({success_rate:.1f}%)")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Hearts Assembled: {avg_hearts:.2f}")

    # Config performance
    print()
    print("=" * 80)
    print("CONFIGURATION PERFORMANCE")
    print("=" * 80)
    config_stats = {}
    for config_name in HYPERPARAMETER_CONFIGS.keys():
        config_results = [r for r in all_results if r.config_name == config_name]
        if config_results:
            success_count = sum(1 for r in config_results if r.success)
            total_count = len(config_results)
            avg_reward = sum(r.reward for r in config_results) / total_count
            avg_hearts = sum(r.hearts_assembled for r in config_results) / total_count
            config_stats[config_name] = {
                "success": success_count,
                "total": total_count,
                "avg_reward": avg_reward,
                "avg_hearts": avg_hearts,
            }

    for config_name in sorted(config_stats.keys(), key=lambda k: -config_stats[k]["avg_reward"]):
        stats = config_stats[config_name]
        print(
            f"{config_name:15s}: {stats['success']}/{stats['total']} success, "
            f"avg_reward={stats['avg_reward']:.2f}, avg_hearts={stats['avg_hearts']:.2f}"
        )

    # Save detailed results to JSON
    output_file = "eval_missions_results.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, cls=CustomJsonEncoder)
    print(f"\nDetailed results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
