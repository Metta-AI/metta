#!/usr/bin/env -S uv run
"""
Hyperparameter Search for Unclipping Scripted Agent

Tests multiple hyperparameter presets on a subset of missions (including clipping variants)
to identify optimal configurations for the unclipping agent.

Usage:
  uv run python packages/cogames/scripts/hyperparameter_search.py
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from cogames.cogs_vs_clips.evals.difficulty_variants import get_difficulty
from cogames.cogs_vs_clips.evals.eval_missions import (
    CollectResourcesClassic,
    CollectResourcesSpread,
    EnergyStarved,
    ExtractorHub30,
    GoTogether,
    OxygenBottleneck,
)
from cogames.cogs_vs_clips.mission import Mission, NumCogsVariant
from cogames.policy.scripted_agent.unclipping_agent import (
    UnclippingHyperparameters,
    UnclippingPolicy,
)
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    preset: str
    experiment: str
    num_cogs: int
    difficulty: str
    total_reward: float  # Average reward per agent
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    success: bool


# Missions to test (6 representative missions)
TEST_MISSIONS = {
    "energy_starved": EnergyStarved,
    "oxygen_bottleneck": OxygenBottleneck,
    "extractor_hub_30": ExtractorHub30,
    "collect_resources_classic": CollectResourcesClassic,
    "go_together": GoTogether,
    "collect_resources_spread": CollectResourcesSpread,
}

# Difficulties to test (including clipping variants)
TEST_DIFFICULTIES = [
    "standard",
    "hard",
    "clipped_oxygen",
    "clipped_carbon",
    "clipped_germanium",
    "clipped_silicon",
]

# Agent counts to test
TEST_COGS = [1, 2, 4, 8]


def create_hyperparameter_presets() -> Dict[str, UnclippingHyperparameters]:
    """Create hyperparameter presets to test."""
    presets = {}

    # 1. Default (thorough_exploration - best performing preset)
    presets["default"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=40,
        exploration_area_check_window=35,
        exploration_area_size_threshold=9,
        exploration_escape_duration=8,
        exploration_direction_persistence=18,
        exploration_assembler_distance_threshold=12,
    )

    # 2. Conservative (larger area threshold, longer persistence, shorter escape)
    presets["conservative"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=40,
        exploration_area_check_window=40,
        exploration_area_size_threshold=10,
        exploration_escape_duration=8,
        exploration_direction_persistence=15,
        exploration_assembler_distance_threshold=12,
    )

    # 3. Aggressive (smaller area threshold, shorter persistence, longer escape)
    presets["aggressive"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=25,
        exploration_area_check_window=25,
        exploration_area_size_threshold=5,
        exploration_escape_duration=15,
        exploration_direction_persistence=7,
        exploration_assembler_distance_threshold=8,
    )

    # 4. Fast exploration (short persistence, quick escape triggers)
    presets["fast_exploration"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=30,
        exploration_area_check_window=25,
        exploration_area_size_threshold=6,
        exploration_escape_duration=12,
        exploration_direction_persistence=5,
        exploration_assembler_distance_threshold=8,
    )

    # 5. Thorough exploration (long persistence, larger area tolerance)
    presets["thorough_exploration"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=40,
        exploration_area_check_window=35,
        exploration_area_size_threshold=9,
        exploration_escape_duration=8,
        exploration_direction_persistence=18,
        exploration_assembler_distance_threshold=12,
    )

    # 6. Quick escape (small area, frequent escape attempts)
    presets["quick_escape"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=25,
        exploration_area_check_window=20,
        exploration_area_size_threshold=5,
        exploration_escape_duration=8,
        exploration_direction_persistence=8,
        exploration_assembler_distance_threshold=6,
    )

    # 7. Patient exploration (very long persistence, large area tolerance)
    presets["patient_exploration"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=50,
        exploration_area_check_window=50,
        exploration_area_size_threshold=12,
        exploration_escape_duration=10,
        exploration_direction_persistence=25,
        exploration_assembler_distance_threshold=15,
    )

    # 8. Balanced tuned (middle ground with slight optimizations)
    presets["balanced_tuned"] = UnclippingHyperparameters(
        recharge_threshold_low=35,
        recharge_threshold_high=85,
        stuck_detection_enabled=True,
        stuck_escape_distance=12,
        position_history_size=35,
        exploration_area_check_window=30,
        exploration_area_size_threshold=8,
        exploration_escape_duration=12,
        exploration_direction_persistence=12,
        exploration_assembler_distance_threshold=10,
    )

    return presets


def run_evaluation(
    preset_name: str,
    hyperparams: UnclippingHyperparameters,
    experiments: Dict[str, Mission],
    difficulties: List[str],
    cogs_list: List[int],
    max_steps: int = 1000,
    seed: int = 42,
) -> List[EvalResult]:
    """Run evaluation for a hyperparameter preset."""
    results = []

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluating preset: {preset_name}")
    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Difficulties: {len(difficulties)}")
    logger.info(f"Agent counts: {cogs_list}")
    logger.info(f"{'=' * 80}\n")

    total_tests = len(experiments) * len(difficulties) * len(cogs_list)
    completed = 0

    for exp_name, base_mission in experiments.items():
        for difficulty_name in difficulties:
            try:
                difficulty = get_difficulty(difficulty_name)
            except Exception as e:
                logger.error(f"Unknown difficulty: {difficulty_name}: {e}")
                continue

            for num_cogs in cogs_list:
                completed += 1
                logger.info(f"[{completed}/{total_tests}] {exp_name} | {difficulty_name} | {num_cogs} agent(s)")

                # Create mission and apply difficulty
                mission = base_mission.with_variants([difficulty, NumCogsVariant(num_cogs=num_cogs)])

                try:
                    env_config = mission.make_env()
                    # Only override max_steps if difficulty doesn't specify it
                    if not hasattr(difficulty, "max_steps_override") or difficulty.max_steps_override is None:
                        env_config.game.max_steps = max_steps

                    # Get the actual max_steps from env_config
                    actual_max_steps = env_config.game.max_steps

                    # Create policy with PolicyEnvInterface and hyperparameters
                    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_config)
                    policy = UnclippingPolicy(policy_env_info, hyperparams)
                    agent_policies = [policy.agent_policy(i) for i in range(num_cogs)]

                    # Create rollout and run episode
                    rollout = Rollout(
                        env_config,
                        agent_policies,
                        render_mode="none",
                        seed=seed,
                    )
                    rollout.run_until_done()

                    # Get results - average reward per agent
                    total_reward = float(sum(rollout._sim.episode_rewards)) / num_cogs
                    final_step = rollout._sim.current_step

                    # Record result
                    result = EvalResult(
                        preset=preset_name,
                        experiment=exp_name,
                        num_cogs=num_cogs,
                        difficulty=difficulty_name,
                        total_reward=total_reward,
                        hearts_assembled=int(total_reward),
                        steps_taken=final_step + 1,
                        max_steps=actual_max_steps,
                        success=total_reward > 0,
                    )
                    results.append(result)

                    status = "✓" if result.success else "✗"
                    logger.info(f"  {status} Reward: {total_reward:.1f}, Steps: {final_step + 1}/{actual_max_steps}")

                except Exception as e:
                    logger.error(f"  ✗ Error: {e}")
                    # Record failure
                    result = EvalResult(
                        preset=preset_name,
                        experiment=exp_name,
                        num_cogs=num_cogs,
                        difficulty=difficulty_name,
                        total_reward=0.0,
                        hearts_assembled=0,
                        steps_taken=0,
                        max_steps=max_steps,
                        success=False,
                    )
                    results.append(result)

    return results


def print_summary(results: List[EvalResult]):
    """Print summary statistics by preset."""
    if not results:
        logger.info("\nNo results to summarize.")
        return

    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY BY PRESET")
    logger.info(f"{'=' * 80}\n")

    # Group by preset
    by_preset = defaultdict(list)
    for r in results:
        by_preset[r.preset].append(r)

    # Sort presets by average reward
    preset_avg_rewards = {
        preset: sum(r.total_reward for r in results) / len(results) for preset, results in by_preset.items()
    }
    sorted_presets = sorted(preset_avg_rewards.items(), key=lambda x: x[1], reverse=True)

    for preset_name, _avg_reward in sorted_presets:
        preset_results = by_preset[preset_name]
        total = len(preset_results)
        successes = sum(1 for r in preset_results if r.success)
        avg_reward_actual = sum(r.total_reward for r in preset_results) / total
        avg_hearts = sum(r.hearts_assembled for r in preset_results) / total

        logger.info(f"{preset_name}:")
        logger.info(f"  Success rate: {successes}/{total} ({100 * successes / total:.1f}%)")
        logger.info(f"  Avg reward: {avg_reward_actual:.2f}")
        logger.info(f"  Avg hearts: {avg_hearts:.2f}")
        logger.info("")

    # Overall best
    best_preset = sorted_presets[0][0]
    logger.info(f"Best preset: {best_preset} (avg reward: {sorted_presets[0][1]:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for baseline scripted agent")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="hyperparameter_search_results.json", help="Output JSON file")
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Specific presets to test (default: all)",
    )

    args = parser.parse_args()

    # Create hyperparameter presets
    all_presets = create_hyperparameter_presets()

    # Filter presets if specified
    if args.presets:
        presets_to_test = {name: all_presets[name] for name in args.presets if name in all_presets}
        if not presets_to_test:
            logger.error(f"No valid presets found. Available: {list(all_presets.keys())}")
            return
    else:
        presets_to_test = all_presets

    logger.info(f"Testing {len(presets_to_test)} hyperparameter presets")
    logger.info(f"Missions: {list(TEST_MISSIONS.keys())}")
    logger.info(f"Difficulties: {TEST_DIFFICULTIES}")
    logger.info(f"Agent counts: {TEST_COGS}")

    all_results = []

    # Run evaluation for each preset
    for preset_name, hyperparams in presets_to_test.items():
        results = run_evaluation(
            preset_name=preset_name,
            hyperparams=hyperparams,
            experiments=TEST_MISSIONS,
            difficulties=TEST_DIFFICULTIES,
            cogs_list=TEST_COGS,
            max_steps=args.steps,
            seed=args.seed,
        )
        all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Save results to JSON
    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
