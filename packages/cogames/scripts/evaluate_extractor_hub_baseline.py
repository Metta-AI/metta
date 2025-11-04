#!/usr/bin/env python3
"""
Focused Evaluation Script for Extractor Hub 30 Baseline

Tests only extractor_hub_30 mission with minimal_baseline preset
to establish a clean performance baseline for ablation studies.

Usage:
  uv run python packages/cogames/scripts/evaluate_extractor_hub_baseline.py
  uv run python packages/cogames/scripts/evaluate_extractor_hub_baseline.py --output baseline_results.json
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass

import numpy as np

from cogames.cogs_vs_clips.evals import DIFFICULTY_LEVELS, apply_difficulty
from cogames.cogs_vs_clips.evals.eval_missions import ExtractorHub30
from cogames.policy.scripted_agent import HYPERPARAMETER_PRESETS, ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    mission: str
    num_cogs: int
    preset: str
    difficulty: str
    total_reward: float
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    final_energy: int
    success: bool
    final_carbon: int = 0
    final_oxygen: int = 0
    final_germanium: int = 0
    final_silicon: int = 0


def run_episode(
    mission_cls,
    mission_name: str,
    num_cogs: int,
    preset_name: str,
    difficulty_name: str,
    max_steps: int = 1000,
    seed: int | None = None,
) -> EvalResult:
    """Run a single episode and return results."""
    # Instantiate mission
    mission = mission_cls()

    # Apply difficulty
    difficulty = DIFFICULTY_LEVELS[difficulty_name]
    apply_difficulty(mission, difficulty)

    # Instantiate mission with num_cogs
    map_builder = mission.site.map_builder if mission.site else None
    mission_inst = mission.instantiate(map_builder, num_cogs=num_cogs)

    # Build env config and wrap
    env_config = mission_inst.make_env()
    env_config.game.max_steps = max_steps
    env = MettaGridEnv(env_config)

    # Create policy
    hyperparams = HYPERPARAMETER_PRESETS[preset_name]
    if seed is not None:
        hyperparams.seed = seed
    policy = ScriptedAgentPolicy(env, hyperparams=hyperparams)

    # Run episode
    obs, info = env.reset()
    policy.reset(obs, info)
    agents = [policy.agent_policy(i) for i in range(num_cogs)]

    done = False
    total_reward = 0.0
    step = 0

    actions = np.zeros(num_cogs, dtype=dtype_actions)

    while not done and step < max_steps:
        for agent_id in range(num_cogs):
            actions[agent_id] = int(agents[agent_id].step(obs[agent_id]))

        obs, rewards, dones, truncs, info = env.step(actions)
        total_reward += float(np.sum(rewards))
        done = bool(np.any(dones) or np.any(truncs))
        step += 1

    # Extract final metrics
    hearts_assembled = 0
    final_energy = 0
    final_carbon = 0
    final_oxygen = 0
    final_germanium = 0
    final_silicon = 0

    # TODO: Parse final inventory from observations if needed
    # For now, rely on reward-based success detection

    success = total_reward >= 3.0  # Heuristic: 3+ reward means at least 1 heart

    return EvalResult(
        mission=mission_name,
        num_cogs=num_cogs,
        preset=preset_name,
        difficulty=difficulty_name,
        total_reward=total_reward,
        hearts_assembled=hearts_assembled,
        steps_taken=step,
        max_steps=max_steps,
        final_energy=final_energy,
        success=success,
        final_carbon=final_carbon,
        final_oxygen=final_oxygen,
        final_germanium=final_germanium,
        final_silicon=final_silicon,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate extractor_hub_30 with minimal_baseline preset")
    parser.add_argument("--output", type=str, default="baseline_results.json", help="Output JSON file")
    parser.add_argument("--cogs", type=int, nargs="+", default=[2, 4], help="Agent counts to test")
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=["story_mode", "standard", "hard", "brutal"],
        help="Difficulty variants to test",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    args = parser.parse_args()

    mission_cls = ExtractorHub30
    preset_name = "minimal_baseline"
    mission_name = "extractor_hub_30"

    logger.info("=" * 80)
    logger.info("Extractor Hub 30 Baseline Evaluation")
    logger.info(f"Mission: {mission_name}")
    logger.info(f"Preset: {preset_name}")
    logger.info(f"Agent counts: {args.cogs}")
    logger.info(f"Difficulties: {args.difficulties}")
    logger.info("=" * 80)

    results = []

    for num_cogs in args.cogs:
        for difficulty_name in args.difficulties:
            logger.info(f"\nRunning: {num_cogs} cogs, {difficulty_name} difficulty")
            try:
                result = run_episode(
                    mission_cls=mission_cls,
                    mission_name=mission_name,
                    num_cogs=num_cogs,
                    preset_name=preset_name,
                    difficulty_name=difficulty_name,
                    max_steps=args.max_steps,
                    seed=args.seed,
                )
                results.append(result)

                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                logger.info(
                    f"  {status} | Reward: {result.total_reward:.2f} | "
                    f"Steps: {result.steps_taken}/{result.max_steps} | "
                    f"Hearts: {result.hearts_assembled}"
                )
            except Exception as e:
                logger.error(f"  ERROR: {e}", exc_info=True)

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    total_runs = len(results)
    successes = sum(1 for r in results if r.success)
    success_rate = (successes / total_runs * 100) if total_runs > 0 else 0.0

    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Success rate: {success_rate:.1f}%")

    # Per agent count
    for num_cogs in args.cogs:
        cog_results = [r for r in results if r.num_cogs == num_cogs]
        cog_successes = sum(1 for r in cog_results if r.success)
        cog_rate = (cog_successes / len(cog_results) * 100) if cog_results else 0.0
        logger.info(f"  {num_cogs} cogs: {cog_successes}/{len(cog_results)} ({cog_rate:.1f}%)")

    # Per difficulty
    for difficulty in args.difficulties:
        diff_results = [r for r in results if r.difficulty == difficulty]
        diff_successes = sum(1 for r in diff_results if r.success)
        diff_rate = (diff_successes / len(diff_results) * 100) if diff_results else 0.0
        logger.info(f"  {difficulty}: {diff_successes}/{len(diff_results)} ({diff_rate:.1f}%)")

    # Write results to JSON
    output_data = {
        "mission": mission_name,
        "preset": preset_name,
        "agent_counts": args.cogs,
        "difficulties": args.difficulties,
        "total_runs": total_runs,
        "successes": successes,
        "success_rate": success_rate,
        "results": [asdict(r) for r in results],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
