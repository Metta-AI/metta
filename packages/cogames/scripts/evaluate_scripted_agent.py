#!/usr/bin/env python3
"""
Simplified Evaluation Script for Scripted Agent

Two main test suites:
1. Training Facility: Test on hand-designed training maps
2. Full Evaluation: Test all experiments × difficulties × hyperparameters × clipping × agent counts

Usage:
  # Training Facility (quick test)
  uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py training-facility

  # Full evaluation (comprehensive)
  uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py full

  # Both
  uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py all
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

from cogames.cogs_vs_clips.evals import (
    DIFFICULTY_LEVELS,
    apply_difficulty,
)
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.scripted_agent import HYPERPARAMETER_PRESETS, ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    suite: str
    test_name: str
    num_cogs: int
    preset: str | None
    difficulty: str | None
    clip_mode: str | None
    clip_rate: float | None
    total_reward: float
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    final_energy: int
    success: bool


# =============================================================================
# Test Suite 1: Training Facility (Simple)
# =============================================================================

TF_MAPS = [
    "training_facility_open_1.map",
    "training_facility_open_2.map",
    "training_facility_open_3.map",
]


def run_training_facility_suite(
    maps: Optional[List[str]] = None,
    cogs_list: Optional[List[int]] = None,
    episodes: int = 3,
    max_steps: int = 800,
    seed: int = 42,
) -> List[EvalResult]:
    """Run training facility evaluation suite with multiple agent counts."""
    print("\n" + "=" * 80)
    print("TRAINING FACILITY SUITE")
    print("=" * 80)

    if maps is None:
        maps = TF_MAPS
    if cogs_list is None:
        cogs_list = [1, 2, 4]

    results = []

    for cogs in cogs_list:
        print(f"\n### Testing with {cogs} agent(s) ###")

        for map_name in maps:
            print(f"\n## Evaluating: {map_name} (cogs={cogs})")
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
                last_step = 0
                for step in range(max_steps):
                    actions = np.zeros(env.num_agents, dtype=dtype_actions)
                    for i in range(env.num_agents):
                        actions[i] = int(agents[i].step(obs[i]))
                    obs, rewards, done, truncated, _ = env.step(actions)
                    episode_reward += float(rewards.sum())
                    last_step = step
                    if all(done) or all(truncated):
                        break

                per_map_sum += episode_reward
                agent0_state = getattr(agents[0], "_state", None)
                total_hearts += int(getattr(agent0_state, "hearts_assembled", 0))
                total_steps += last_step + 1

            env.close()

            avg_reward = per_map_sum / episodes if episodes > 0 else 0.0
            avg_hearts = total_hearts / episodes if episodes > 0 else 0
            avg_steps = total_steps / episodes

            result = EvalResult(
                suite="training_facility",
                test_name=map_name,
                num_cogs=cogs,
                preset=None,
                difficulty=None,
                clip_mode=None,
                clip_rate=None,
                total_reward=float(avg_reward),
                hearts_assembled=int(avg_hearts),
                steps_taken=int(avg_steps),
                max_steps=max_steps,
                final_energy=0,
                success=avg_reward > 0,
            )
            results.append(result)

            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Hearts: {avg_hearts:.1f}")
            print(f"  Avg Steps: {avg_steps:.0f}")
            print(f"  {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    return results


# =============================================================================
# Test Suite 2: Full Evaluation (Comprehensive)
# =============================================================================

# All evaluation missions (each will be tested with all difficulty levels)
# Difficulty variants (easy/medium/hard/extreme) are applied at runtime
EXPERIMENTS = [(cls.__name__, cls) for cls in EVAL_MISSIONS]

EXPERIMENT_MAP = dict(EXPERIMENTS)


def run_full_evaluation_suite(
    experiments: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    hyperparams: Optional[List[str]] = None,
    clip_modes: Optional[List[str]] = None,
    clip_rates: Optional[List[float]] = None,
    cogs_list: Optional[List[int]] = None,
    max_steps: int = 1000,
) -> List[EvalResult]:
    """
    Run comprehensive evaluation:
    experiments × difficulties × hyperparameters × clipping × agent counts
    """
    print("\n" + "=" * 80)
    print("FULL EVALUATION SUITE")
    print("=" * 80)

    # Defaults
    if experiments is None:
        experiments = [name for name, _ in EXPERIMENTS]
    if difficulties is None:
        difficulties = list(DIFFICULTY_LEVELS.keys())  # easy, medium, hard, extreme
    if hyperparams is None:
        hyperparams = list(HYPERPARAMETER_PRESETS.keys())
    if clip_modes is None:
        clip_modes = []
    if clip_rates is None:
        clip_rates = [0.0]
    if cogs_list is None:
        cogs_list = [1, 2, 4, 8]

    total_tests = (
        len(experiments)
        * len(difficulties)
        * len(clip_rates)
        * len(cogs_list)
        * len(hyperparams)
    )

    print(f"\nExperiments: {len(experiments)}")
    print(f"Difficulties: {len(difficulties)}")
    print(f"Hyperparams: {len(hyperparams)}")
    print("Clip strategy: unclipped only")
    print(f"Clip rates: {clip_rates}")
    print(f"Agent counts: {cogs_list}")
    print(f"Total tests: {total_tests}\n")

    results = []
    success_count = 0
    test_count = 0

    for exp_name in experiments:
        if exp_name not in EXPERIMENT_MAP:
            logger.error(f"Unknown experiment: {exp_name}")
            continue

        mission_class = EXPERIMENT_MAP[exp_name]

        for difficulty_name in difficulties:
            if difficulty_name not in DIFFICULTY_LEVELS:
                logger.error(f"Unknown difficulty: {difficulty_name}")
                continue

            clip_mode = "none"
            for clip_rate in clip_rates:
                if clip_rate > 0:
                    continue

                for num_cogs in cogs_list:
                    for preset_name in hyperparams:
                        if preset_name not in HYPERPARAMETER_PRESETS:
                            logger.error(f"Unknown preset: {preset_name}")
                            continue

                        test_name = f"{exp_name}_{difficulty_name}_clip{clip_rate}_cogs{num_cogs}_{preset_name}"
                        print(f"\n[{test_count}/{total_tests}] {test_name}")

                        try:
                            # Build mission
                            mission = mission_class()
                            difficulty = DIFFICULTY_LEVELS[difficulty_name]
                            apply_difficulty(mission, difficulty)

                            # clipped variants are disabled in this sweep

                            map_builder = mission.site.map_builder if mission.site else None
                            mission = mission.instantiate(map_builder, num_cogs=num_cogs)
                            env_config = mission.make_env()
                            env_config.game.max_steps = max_steps

                            # Run episode
                            env = MettaGridEnv(env_config)
                            hyperparams_obj = HYPERPARAMETER_PRESETS[preset_name]
                            policy = ScriptedAgentPolicy(env, hyperparams=hyperparams_obj)

                            obs, info = env.reset()
                            policy.reset(obs, info)
                            agents = [policy.agent_policy(i) for i in range(num_cogs)]

                            total_reward = 0.0
                            last_step = 0
                            for step in range(max_steps):
                                actions = np.zeros(num_cogs, dtype=dtype_actions)
                                for i in range(num_cogs):
                                    actions[i] = int(agents[i].step(obs[i]))
                                obs, rewards, dones, truncated, info = env.step(actions)
                                total_reward += float(rewards.sum())
                                last_step = step
                                if all(dones) or all(truncated):
                                    break

                            agent_state = getattr(agents[0], "_state", None)
                            hearts_assembled = int(getattr(agent_state, "hearts_assembled", 0))
                            steps_taken = last_step + 1
                            final_energy = int(getattr(agent_state, "energy", 0))
                            success = total_reward > 0

                            env.close()

                            result = EvalResult(
                                suite="full",
                                test_name=test_name,
                                num_cogs=num_cogs,
                                preset=preset_name,
                                difficulty=difficulty_name,
                                clip_mode=clip_mode,
                                clip_rate=clip_rate,
                                total_reward=float(total_reward),
                                hearts_assembled=int(hearts_assembled),
                                steps_taken=int(steps_taken),
                                max_steps=max_steps,
                                final_energy=int(final_energy),
                                success=success,
                            )
                            results.append(result)
                            test_count += 1
                            if success:
                                success_count += 1

                            print(f"  Reward: {total_reward:.1f}")
                            print(f"  Steps: {steps_taken}/{max_steps}")
                            print(f"  {'✅ SUCCESS' if success else '❌ FAILED'}")

                        except Exception as e:
                            logger.error(f"Error in {test_name}: {e}")
                            test_count += 1

    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {success_count}/{test_count} successful")
    print(f"{'=' * 80}")

    return results


# =============================================================================
# Summary and Main
# =============================================================================


def print_summary(results: List[EvalResult], suite_name: str):
    """Print summary statistics for results."""
    print("\n" + "=" * 80)
    print(f"{suite_name.upper()} - SUMMARY")
    print("=" * 80)

    if not results:
        print("\nNo results to summarize.")
        return

    successes = sum(1 for r in results if r.success)
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"Successes: {successes}")
    print(f"Failures: {total - successes}")
    print(f"Success rate: {100 * successes / total:.1f}%" if total > 0 else "N/A")

    # Group by agent count
    by_cogs = {}
    for r in results:
        if r.num_cogs not in by_cogs:
            by_cogs[r.num_cogs] = []
        by_cogs[r.num_cogs].append(r)

    print("\n## Success Rate by Agent Count")
    for cogs in sorted(by_cogs.keys()):
        cogs_results = by_cogs[cogs]
        cogs_success = sum(1 for r in cogs_results if r.success)
        avg_reward = sum(r.total_reward for r in cogs_results) / len(cogs_results)
        print(
            f"  {cogs} agent(s): {cogs_success}/{len(cogs_results)} "
            f"({100 * cogs_success / len(cogs_results):.1f}%) "
            f"avg_reward={avg_reward:.2f}"
        )

    # Group by difficulty (if applicable)
    if results[0].difficulty:
        by_diff = {}
        for r in results:
            if r.difficulty not in by_diff:
                by_diff[r.difficulty] = []
            by_diff[r.difficulty].append(r)

        print("\n## Success Rate by Difficulty")
        for diff in sorted(by_diff.keys()):
            diff_results = by_diff[diff]
            diff_success = sum(1 for r in diff_results if r.success)
            print(f"  {diff:15s}: {diff_success}/{len(diff_results)} ({100 * diff_success / len(diff_results):.1f}%)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simplified evaluation script for scripted agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    subparsers = parser.add_subparsers(dest="suite", help="Evaluation suite to run")

    # Training facility suite
    tf_parser = subparsers.add_parser("training-facility", help="Run training facility suite")
    tf_parser.add_argument("--maps", nargs="*", default=None, help="Maps to test (default: all)")
    tf_parser.add_argument("--cogs", nargs="*", type=int, default=None, help="Agent counts (default: 1 2 4)")
    tf_parser.add_argument("--episodes", type=int, default=3, help="Episodes per map")
    tf_parser.add_argument("--steps", type=int, default=800, help="Max steps per episode")

    # Full evaluation suite
    full_parser = subparsers.add_parser("full", help="Run full evaluation suite")
    full_parser.add_argument("--experiments", nargs="*", default=None, help="Experiments to test (default: all)")
    full_parser.add_argument("--difficulties", nargs="*", default=None, help="Difficulties (default: all)")
    full_parser.add_argument("--hyperparams", nargs="*", default=None, help="Hyperparameter presets (default: all)")
    full_parser.add_argument("--clip-modes", nargs="*", default=None, help="Clip modes (default: none + all resources)")
    full_parser.add_argument("--clip-rates", nargs="*", type=float, default=None, help="Clip rates (default: 0.0 0.25)")
    full_parser.add_argument("--cogs", nargs="*", type=int, default=None, help="Agent counts (default: 1 2 4 8)")
    full_parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")

    # All suites
    subparsers.add_parser("all", help="Run all evaluation suites")

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
            cogs_list=tf_args.get("cogs"),
            episodes=tf_args.get("episodes", 3),
            max_steps=tf_args.get("steps", 800),
        )
        all_results.extend(tf_results)
        print_summary(tf_results, "Training Facility")

    if args.suite in ["full", "all"]:
        full_args = {} if args.suite == "all" else vars(args)
        full_results = run_full_evaluation_suite(
            experiments=full_args.get("experiments"),
            difficulties=full_args.get("difficulties"),
            hyperparams=full_args.get("hyperparams"),
            clip_modes=full_args.get("clip_modes"),
            clip_rates=full_args.get("clip_rates"),
            cogs_list=full_args.get("cogs"),
            max_steps=full_args.get("steps", 1000),
        )
        all_results.extend(full_results)
        print_summary(full_results, "Full Evaluation")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
