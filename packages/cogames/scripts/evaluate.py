#!/usr/bin/env python3
"""
Unified Evaluation Script for Scripted Agent

This script consolidates all evaluation suites:
1. Training Facility: Test on hand-designed training maps
2. Outpost Experiments: Test on exploration experiments (EXP1-10) and eval missions (EVAL1-10)
3. Difficulty Variants: Test all experiments × difficulties × hyperparameter presets

Usage:
  # Training Facility suite
  uv run python -u packages/cogames/scripts/evaluate.py training-facility \\
      --cogs 1 --episodes 3 --steps 500

  # Outpost Experiments suite
  uv run python -u packages/cogames/scripts/evaluate.py outpost \\
      --experiments EXP1 EXP2 EXP4 --hyperparams conservative aggressive

  # Difficulty Variants suite (comprehensive)
  uv run python -u packages/cogames/scripts/evaluate.py difficulty \\
      --difficulties easy medium hard --hyperparams all

  # All suites
  uv run python -u packages/cogames/scripts/evaluate.py all
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from cogames.cogs_vs_clips.difficulty_variants import DIFFICULTY_LEVELS, apply_difficulty
from cogames.cogs_vs_clips.eval_missions import (
    CarbonDesert,
    EnergyStarved,
    GermaniumClutch,
    GermaniumRush,
    HighRegenSprint,
    OxygenBottleneck,
    SiliconWorkbench,
    SingleUseWorld,
    SlowOxygen,
    SparseBalanced,
)
from cogames.cogs_vs_clips.exploration_experiments import (
    Experiment1Mission,
    Experiment2Mission,
    Experiment4Mission,
    Experiment5Mission,
    Experiment6Mission,
    Experiment7Mission,
    Experiment8Mission,
    Experiment9Mission,
    Experiment10Mission,
)
from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.hyperparameter_presets import HYPERPARAMETER_PRESETS
from cogames.policy.scripted_agent import ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    suite: str  # "training_facility", "outpost", "difficulty"
    test_name: str  # Map name, experiment name, or experiment+difficulty
    preset: str | None  # Hyperparameter preset used (if applicable)
    total_reward: float
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    final_energy: int
    success: bool  # True if reward > 0


# =============================================================================
# Test Suite 1: Training Facility
# =============================================================================

TF_MAPS: List[str] = [
    "training_facility_open_1.map",
    "training_facility_open_2.map",
    "training_facility_open_3.map",
    "training_facility_tight_4.map",
    "training_facility_tight_5.map",
]


def run_training_facility_suite(
    maps: List[str] = None,
    cogs: int = 2,
    episodes: int = 3,
    max_steps: int = 500,
    seed: int = 42,
) -> List[EvalResult]:
    """Run training facility evaluation suite."""
    print("\n" + "=" * 80)
    print("TRAINING FACILITY SUITE")
    print("=" * 80)

    if maps is None:
        maps = TF_MAPS

    results = []

    for map_name in maps:
        print(f"\n## Evaluating: {map_name}")
        try:
            env_cfg = make_game(num_cogs=cogs, map_name=map_name)
            env = MettaGridEnv(env_cfg=env_cfg)
        except Exception as e:
            logger.error(f"SKIP {map_name}: failed to create env ({e})")
            continue

        per_map_sum = 0.0
        total_hearts = 0
        total_steps = 0

        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            policy = ScriptedAgentPolicy(env)
            agents = [policy.agent_policy(i) for i in range(env.num_agents)]

            episode_reward = 0.0
            for step in range(max_steps):
                actions = np.zeros(env.num_agents, dtype=dtype_actions)
                for i in range(env.num_agents):
                    actions[i] = int(agents[i].step(obs[i]))
                obs, rewards, done, truncated, _ = env.step(actions)
                episode_reward += float(rewards.sum())
                if all(done) or all(truncated):
                    break

            per_map_sum += episode_reward
            # Try to get hearts (may not be available in all envs)
            try:
                total_hearts += agents[0]._state.hearts_assembled
            except AttributeError:
                pass
            total_steps += step + 1

        env.close()

        avg_reward = per_map_sum / episodes if episodes > 0 else 0.0
        avg_hearts = total_hearts / episodes if episodes > 0 else 0
        avg_steps = total_steps / episodes

        result = EvalResult(
            suite="training_facility",
            test_name=map_name,
            preset=None,  # Training facility doesn't use presets
            total_reward=float(avg_reward),
            hearts_assembled=int(avg_hearts),
            steps_taken=int(avg_steps),
            max_steps=max_steps,
            final_energy=0,  # Not tracked for training facility
            success=avg_reward > 0,
        )
        results.append(result)

        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Hearts: {avg_hearts:.1f}")
        print(f"  Avg Steps: {avg_steps:.0f}")
        print(f"  {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    return results


# =============================================================================
# Test Suite 2: Outpost Experiments
# =============================================================================

EXPERIMENTS = [
    # Exploration Experiments (EXP1-10)
    ("EXP1", Experiment1Mission),
    ("EXP2", Experiment2Mission),
    ("EXP4", Experiment4Mission),
    ("EXP5", Experiment5Mission),
    ("EXP6", Experiment6Mission),
    ("EXP7", Experiment7Mission),
    ("EXP8", Experiment8Mission),
    ("EXP9", Experiment9Mission),
    ("EXP10", Experiment10Mission),
    # Eval Missions (EVAL1-10)
    ("EVAL1_EnergyStarved", EnergyStarved),
    ("EVAL2_OxygenBottleneck", OxygenBottleneck),
    ("EVAL3_GermaniumRush", GermaniumRush),
    ("EVAL4_SiliconWorkbench", SiliconWorkbench),
    ("EVAL5_CarbonDesert", CarbonDesert),
    ("EVAL6_SingleUseWorld", SingleUseWorld),
    ("EVAL7_SlowOxygen", SlowOxygen),
    ("EVAL8_HighRegenSprint", HighRegenSprint),
    ("EVAL9_SparseBalanced", SparseBalanced),
    ("EVAL10_GermaniumClutch", GermaniumClutch),
]

EXPERIMENT_MAP = dict(EXPERIMENTS)


def get_max_steps_for_mission(mission_class) -> int:
    """Determine appropriate step limit based on map size."""
    mission = mission_class()
    map_builder = mission.site.map_builder

    width = map_builder.width if hasattr(map_builder, "width") else 30
    height = map_builder.height if hasattr(map_builder, "height") else 30
    map_size = max(width, height)

    if map_size >= 100:
        return 3000
    elif map_size >= 80:
        return 2500
    elif map_size >= 60:
        return 2000
    elif map_size >= 50:
        return 1500
    else:
        return 1000


def run_outpost_suite(
    experiments: List[str] = None,
    hyperparams: List[str] = None,
) -> List[EvalResult]:
    """Run outpost experiments evaluation suite."""
    print("\n" + "=" * 80)
    print("OUTPOST EXPERIMENTS SUITE")
    print("=" * 80)

    if experiments is None:
        experiments = [name for name, _ in EXPERIMENTS]
    if hyperparams is None:
        hyperparams = ["adaptive"]  # Default to adaptive preset

    results = []
    success_count = 0
    total_count = 0

    for exp_name in experiments:
        if exp_name not in EXPERIMENT_MAP:
            logger.error(f"Unknown experiment: {exp_name}")
            continue

        mission_class = EXPERIMENT_MAP[exp_name]
        max_steps = get_max_steps_for_mission(mission_class)

        for preset_name in hyperparams:
            if preset_name not in HYPERPARAMETER_PRESETS:
                logger.error(f"Unknown preset: {preset_name}")
                continue

            test_name = f"{exp_name}_{preset_name}"
            print(f"\n## [{success_count}/{total_count}] {test_name}")

            try:
                mission = mission_class()
                mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
                env_config = mission.make_env()
                env_config.game.max_steps = max_steps

                env = MettaGridEnv(env_config)
                hyperparams_obj = HYPERPARAMETER_PRESETS[preset_name]
                policy = ScriptedAgentPolicy(env, hyperparams=hyperparams_obj)

                obs, info = env.reset()
                policy.reset(obs, info)
                agent_policy = policy.agent_policy(0)

                total_reward = 0.0
                for step in range(max_steps):
                    action = agent_policy.step(obs[0])
                    obs, rewards, dones, truncated, info = env.step([action])
                    total_reward += float(rewards[0])
                    if dones[0] or truncated[0]:
                        break

                agent_state = agent_policy._state
                hearts_assembled = agent_state.hearts_assembled
                steps_taken = step + 1
                final_energy = agent_state.energy
                success = total_reward > 0

                result = EvalResult(
                    suite="outpost",
                    test_name=exp_name,
                    preset=preset_name,
                    total_reward=float(total_reward),
                    hearts_assembled=int(hearts_assembled),
                    steps_taken=int(steps_taken),
                    max_steps=max_steps,
                    final_energy=int(final_energy),
                    success=success,
                )
                results.append(result)
                total_count += 1
                if success:
                    success_count += 1

                print(f"  Reward: {total_reward:.1f}")
                print(f"  Hearts: {hearts_assembled}")
                print(f"  Steps: {steps_taken}/{max_steps}")
                print(f"  {'✅ SUCCESS' if success else '❌ FAILED'}")

            except Exception as e:
                logger.error(f"Error in {test_name}: {e}")
                total_count += 1

    return results


# =============================================================================
# Test Suite 3: Difficulty Variants
# =============================================================================


def get_max_steps_for_difficulty(mission_class, difficulty_name: str) -> int:
    """Determine appropriate step limit based on map size and difficulty."""
    mission = mission_class()
    map_builder = mission.site.map_builder

    width = map_builder.width if hasattr(map_builder, "width") else 30
    height = map_builder.height if hasattr(map_builder, "height") else 30
    map_size = max(width, height)

    # Base steps on map size
    if map_size >= 100:
        base_steps = 3000
    elif map_size >= 80:
        base_steps = 2500
    elif map_size >= 60:
        base_steps = 2000
    elif map_size >= 50:
        base_steps = 1500
    else:
        base_steps = 1000

    # Adjust for difficulty
    difficulty_multipliers = {
        "easy": 0.8,
        "medium": 1.0,
        "hard": 1.3,
        "extreme": 1.5,
        "single_use": 1.2,
        "speed_run": 0.9,
        "energy_crisis": 1.4,
    }

    multiplier = difficulty_multipliers.get(difficulty_name, 1.0)
    return int(base_steps * multiplier)


def run_difficulty_suite(
    experiments: List[str] = None,
    difficulties: List[str] = None,
    hyperparams: List[str] = None,
) -> List[EvalResult]:
    """Run difficulty variants evaluation suite."""
    print("\n" + "=" * 80)
    print("DIFFICULTY VARIANTS SUITE")
    print("=" * 80)

    if experiments is None:
        experiments = [name for name, _ in EXPERIMENTS]
    if difficulties is None:
        difficulties = ["easy", "medium", "hard", "extreme"]
    if hyperparams is None or hyperparams == ["all"]:
        hyperparams = [
            "conservative",
            "aggressive",
            "efficient",
            "adaptive",
            "easy_mode",
            "hard_mode",
            "extreme_mode",
            "oxygen_hunter",
            "germanium_focused",
        ]

    print(f"\nExperiments: {len(experiments)}")
    print(f"Difficulties: {len(difficulties)}")
    print(f"Presets: {len(hyperparams)}")
    print(f"Total tests: {len(experiments) * len(difficulties) * len(hyperparams)}")

    results = []
    success_count = 0
    total_count = 0

    for exp_name in experiments:
        if exp_name not in EXPERIMENT_MAP:
            logger.error(f"Unknown experiment: {exp_name}")
            continue

        mission_class = EXPERIMENT_MAP[exp_name]

        for difficulty_name in difficulties:
            if difficulty_name not in DIFFICULTY_LEVELS:
                logger.error(f"Unknown difficulty: {difficulty_name}")
                continue

            for preset_name in hyperparams:
                if preset_name not in HYPERPARAMETER_PRESETS:
                    logger.error(f"Unknown preset: {preset_name}")
                    continue

                max_steps = get_max_steps_for_difficulty(mission_class, difficulty_name)
                test_name = f"{exp_name}_{difficulty_name}_{preset_name}"

                print(f"\n{'=' * 80}")
                status = f"[{success_count}/{total_count}]"
                print(f"{status} {exp_name} | {difficulty_name} | {preset_name} (max_steps={max_steps})")
                print(f"{'=' * 80}")

                try:
                    mission = mission_class()
                    difficulty = DIFFICULTY_LEVELS[difficulty_name]
                    apply_difficulty(mission, difficulty)

                    mission = mission.instantiate(mission.site.map_builder, num_cogs=1)
                    env_config = mission.make_env()
                    env_config.game.max_steps = max_steps

                    env = MettaGridEnv(env_config)
                    hyperparams_obj = HYPERPARAMETER_PRESETS[preset_name]
                    policy = ScriptedAgentPolicy(env, hyperparams=hyperparams_obj)

                    obs, info = env.reset()
                    policy.reset(obs, info)
                    agent_policy = policy.agent_policy(0)

                    total_reward = 0.0
                    for step in range(max_steps):
                        action = agent_policy.step(obs[0])
                        obs, rewards, dones, truncated, info = env.step([action])
                        total_reward += float(rewards[0])
                        if dones[0] or truncated[0]:
                            break

                    agent_state = agent_policy._state
                    hearts_assembled = agent_state.hearts_assembled
                    steps_taken = step + 1
                    final_energy = agent_state.energy
                    success = total_reward > 0

                    result = EvalResult(
                        suite="difficulty",
                        test_name=f"{exp_name}_{difficulty_name}",
                        preset=preset_name,
                        total_reward=float(total_reward),
                        hearts_assembled=int(hearts_assembled),
                        steps_taken=int(steps_taken),
                        max_steps=max_steps,
                        final_energy=int(final_energy),
                        success=success,
                    )
                    results.append(result)
                    total_count += 1
                    if success:
                        success_count += 1

                    print(f"  Reward: {total_reward:.1f}")
                    print(f"  Hearts: {hearts_assembled}")
                    print(f"  Steps: {steps_taken}/{max_steps}")
                    print(f"  Energy: {final_energy}")
                    print(f"  {'✅ SUCCESS' if success else '❌ FAILED'}")

                except Exception as e:
                    logger.error(f"Error in {test_name}: {e}")
                    total_count += 1

    return results


# =============================================================================
# Main CLI
# =============================================================================


def print_summary(results: List[EvalResult], suite_name: str):
    """Print summary statistics for results."""
    print("\n" + "=" * 80)
    print(f"{suite_name.upper()} - SUMMARY")
    print("=" * 80)

    successes = sum(1 for r in results if r.success)
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"Successes: {successes}")
    print(f"Failures: {total - successes}")
    print(f"Success rate: {100 * successes / total:.1f}%" if total > 0 else "N/A")

    # Suite-specific summaries
    if results[0].suite == "difficulty":
        # Group by difficulty
        difficulties = {}
        for r in results:
            diff = r.test_name.split("_")[-1]
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(r)

        print("\n## Success Rate by Difficulty")
        for diff, diff_results in sorted(difficulties.items()):
            diff_success = sum(1 for r in diff_results if r.success)
            print(f"  {diff:15s}: {diff_success}/{len(diff_results)} ({100 * diff_success / len(diff_results):.1f}%)")

        # Group by preset
        presets = {}
        for r in results:
            if r.preset not in presets:
                presets[r.preset] = []
            presets[r.preset].append(r)

        print("\n## Success Rate by Preset")
        for preset, preset_results in sorted(presets.items()):
            preset_success = sum(1 for r in preset_results if r.success)
            print(
                f"  {preset:20s}: {preset_success}/{len(preset_results)} "
                f"({100 * preset_success / len(preset_results):.1f}%)"
            )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for scripted agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="suite", help="Evaluation suite to run")

    # Training facility suite
    tf_parser = subparsers.add_parser("training-facility", help="Run training facility suite")
    tf_parser.add_argument("--maps", nargs="*", default=None, help="Maps to test (default: all)")
    tf_parser.add_argument("--cogs", type=int, default=2, help="Number of agents")
    tf_parser.add_argument("--episodes", type=int, default=3, help="Episodes per map")
    tf_parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    tf_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Outpost experiments suite
    outpost_parser = subparsers.add_parser("outpost", help="Run outpost experiments suite")
    outpost_parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiments to test (e.g., EXP1 EXP2)",
    )
    outpost_parser.add_argument(
        "--hyperparams",
        nargs="*",
        default=None,
        help="Hyperparameter presets to test",
    )

    # Difficulty variants suite
    diff_parser = subparsers.add_parser("difficulty", help="Run difficulty variants suite")
    diff_parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiments to test (e.g., EXP1 EXP2)",
    )
    diff_parser.add_argument(
        "--difficulties",
        nargs="*",
        default=None,
        help="Difficulties to test (e.g., easy medium hard)",
    )
    diff_parser.add_argument(
        "--hyperparams",
        nargs="*",
        default=None,
        help="Hyperparameter presets to test (use 'all' for all presets)",
    )

    # All suites
    subparsers.add_parser("all", help="Run all evaluation suites")

    # Output options
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    args = parser.parse_args()

    if args.suite is None:
        parser.print_help()
        return

    all_results = []

    # Run requested suite(s)
    if args.suite in ["training-facility", "all"]:
        tf_args = {} if args.suite == "all" else vars(args)
        tf_results = run_training_facility_suite(
            maps=tf_args.get("maps"),
            cogs=tf_args.get("cogs", 2),
            episodes=tf_args.get("episodes", 3),
            max_steps=tf_args.get("steps", 500),
            seed=tf_args.get("seed", 42),
        )
        all_results.extend(tf_results)
        print_summary(tf_results, "Training Facility")

    if args.suite in ["outpost", "all"]:
        outpost_args = {} if args.suite == "all" else vars(args)
        outpost_results = run_outpost_suite(
            experiments=outpost_args.get("experiments"),
            hyperparams=outpost_args.get("hyperparams"),
        )
        all_results.extend(outpost_results)
        print_summary(outpost_results, "Outpost Experiments")

    if args.suite in ["difficulty", "all"]:
        diff_args = {} if args.suite == "all" else vars(args)
        diff_results = run_difficulty_suite(
            experiments=diff_args.get("experiments"),
            difficulties=diff_args.get("difficulties"),
            hyperparams=diff_args.get("hyperparams"),
        )
        all_results.extend(diff_results)
        print_summary(diff_results, "Difficulty Variants")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
