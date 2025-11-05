#!/usr/bin/env python3
"""
Evaluation Script for Baseline Scripted Agents

Tests two baseline policies:
- BaselinePolicy: Single/multi-agent resource gathering and heart assembly
- UnclippingPolicy: Extends baseline with extractor unclipping

Usage:
  # Quick test
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py \\
      --agent simple --experiments OxygenBottleneck --cogs 1

  # Full evaluation
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py --agent all

  # Specific configuration
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py \\
      --agent unclipping --experiments ExtractorHub30 ExtractorHub50 --cogs 1 2 4
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List

from cogames.cogs_vs_clips.evals import CANONICAL_DIFFICULTY_ORDER, DIFFICULTY_LEVELS, apply_difficulty
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.policy.scripted_agent import BaselinePolicy, UnclippingPolicy
from mettagrid.simulator.rollout import Rollout

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from a single evaluation run."""

    agent: str
    experiment: str
    num_cogs: int
    difficulty: str
    clip_rate: float
    total_reward: float
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    success: bool


@dataclass
class AgentConfig:
    """Configuration for a baseline agent."""

    key: str
    label: str
    policy_factory: Callable[[], Any]
    cogs_list: List[int]
    difficulties: List[str]


def is_clipping_difficulty(name: str) -> bool:
    """Check if a difficulty involves clipping."""
    return "clipped" in name.lower() or "clipping" in name.lower()


# Available agents
AGENT_CONFIGS: Dict[str, AgentConfig] = {
    "baseline": AgentConfig(
        key="baseline",
        label="Baseline",
        policy_factory=lambda: BaselinePolicy(),
        cogs_list=[1, 2, 4, 8],
        difficulties=[d for d in CANONICAL_DIFFICULTY_ORDER if not is_clipping_difficulty(d)],
    ),
    "unclipping": AgentConfig(
        key="unclipping",
        label="UnclippingAgent",
        policy_factory=lambda: UnclippingPolicy(),
        cogs_list=[1, 2, 4, 8],
        difficulties=CANONICAL_DIFFICULTY_ORDER,  # With and without clipping
    ),
}

# All evaluation missions
EXPERIMENT_MAP = {cls.__name__: cls for cls in EVAL_MISSIONS}


def run_evaluation(
    agent_config: AgentConfig,
    experiments: List[str],
    difficulties: List[str],
    cogs_list: List[int],
    max_steps: int = 1000,
    seed: int = 42,
) -> List[EvalResult]:
    """Run evaluation for an agent configuration."""
    results = []

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluating: {agent_config.label}")
    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Difficulties: {len(difficulties)}")
    logger.info(f"Agent counts: {cogs_list}")
    logger.info(f"{'=' * 80}\n")

    total_tests = len(experiments) * len(difficulties) * len(cogs_list)
    completed = 0

    for exp_name in experiments:
        if exp_name not in EXPERIMENT_MAP:
            logger.error(f"Unknown experiment: {exp_name}")
            continue

        mission_class = EXPERIMENT_MAP[exp_name]

        for difficulty_name in difficulties:
            if difficulty_name not in DIFFICULTY_LEVELS:
                logger.error(f"Unknown difficulty: {difficulty_name}")
                continue

            difficulty = DIFFICULTY_LEVELS[difficulty_name]

            for num_cogs in cogs_list:
                completed += 1
                logger.info(f"[{completed}/{total_tests}] {exp_name} | {difficulty_name} | {num_cogs} agent(s)")

                # Create mission and apply difficulty
                mission = mission_class()
                apply_difficulty(mission, difficulty)

                # Get clip rate for metadata
                clip_rate = getattr(difficulty, "extractor_clip_rate", 0.0)

                try:
                    # Instantiate mission and create environment config
                    map_builder = mission.site.map_builder if mission.site else None
                    mission_inst = mission.instantiate(map_builder, num_cogs=num_cogs)
                    env_config = mission_inst.make_env()
                    env_config.game.max_steps = max_steps

                    # Create policy (scripted agents will get simulation from Rollout)
                    policy = agent_config.policy_factory()
                    agent_policies = [policy] * num_cogs

                    # Create rollout and run episode
                    rollout = Rollout(
                        env_config,
                        agent_policies,
                        render_mode="none",
                        seed=seed,
                        pass_sim_to_policies=True,
                    )
                    rollout.run_until_done()

                    # Get results
                    total_reward = float(sum(rollout._sim.episode_rewards))
                    final_step = rollout._sim.current_step

                    # Record result
                    result = EvalResult(
                        agent=agent_config.label,
                        experiment=exp_name,
                        num_cogs=num_cogs,
                        difficulty=difficulty_name,
                        clip_rate=clip_rate,
                        total_reward=total_reward,
                        hearts_assembled=int(total_reward),
                        steps_taken=final_step + 1,
                        max_steps=max_steps,
                        success=total_reward > 0,
                    )
                    results.append(result)

                    status = "✓" if result.success else "✗"
                    logger.info(f"  {status} Reward: {total_reward:.1f}, Steps: {final_step + 1}/{max_steps}")

                except Exception as e:
                    logger.error(f"  ✗ Error: {e}")
                    # Record failure
                    result = EvalResult(
                        agent=agent_config.label,
                        experiment=exp_name,
                        num_cogs=num_cogs,
                        difficulty=difficulty_name,
                        clip_rate=0.0,
                        total_reward=0.0,
                        hearts_assembled=0,
                        steps_taken=0,
                        max_steps=max_steps,
                        success=False,
                    )
                    results.append(result)

    return results


def print_summary(results: List[EvalResult]):
    """Print summary statistics."""
    if not results:
        logger.info("\nNo results to summarize.")
        return

    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}\n")

    total = len(results)
    successes = sum(1 for r in results if r.success)
    logger.info(f"Total tests: {total}")
    logger.info(f"Successes: {successes}/{total} ({100 * successes / total:.1f}%)")

    # By agent
    logger.info("\n## By Agent")
    agents = sorted(set(r.agent for r in results))
    for agent in agents:
        agent_results = [r for r in results if r.agent == agent]
        agent_successes = sum(1 for r in agent_results if r.success)
        avg_reward = sum(r.total_reward for r in agent_results) / len(agent_results)
        logger.info(
            f"  {agent}: {agent_successes}/{len(agent_results)} "
            f"({100 * agent_successes / len(agent_results):.1f}%) "
            f"avg_reward={avg_reward:.2f}"
        )

    # By agent count
    logger.info("\n## By Agent Count")
    cogs = sorted(set(r.num_cogs for r in results))
    for num_cogs in cogs:
        cogs_results = [r for r in results if r.num_cogs == num_cogs]
        cogs_successes = sum(1 for r in cogs_results if r.success)
        avg_reward = sum(r.total_reward for r in cogs_results) / len(cogs_results)
        logger.info(
            f"  {num_cogs} agent(s): {cogs_successes}/{len(cogs_results)} "
            f"({100 * cogs_successes / len(cogs_results):.1f}%) "
            f"avg_reward={avg_reward:.2f}"
        )

    # By difficulty
    logger.info("\n## By Difficulty")
    difficulties = sorted(set(r.difficulty for r in results))
    for diff in difficulties:
        diff_results = [r for r in results if r.difficulty == diff]
        diff_successes = sum(1 for r in diff_results if r.success)
        avg_reward = sum(r.total_reward for r in diff_results) / len(diff_results)
        logger.info(
            f"  {diff:20s}: {diff_successes}/{len(diff_results)} "
            f"({100 * diff_successes / len(diff_results):.1f}%) "
            f"avg_reward={avg_reward:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline scripted agents")
    parser.add_argument(
        "--agent",
        choices=[*AGENT_CONFIGS.keys(), "all"],
        default="all",
        help="Agent to evaluate (default: all)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiments to test (default: all). Use class names like 'OxygenBottleneck'",
    )
    parser.add_argument(
        "--difficulties",
        nargs="*",
        default=None,
        help="Difficulties to test (default: agent-specific)",
    )
    parser.add_argument(
        "--cogs",
        nargs="*",
        type=int,
        default=None,
        help="Agent counts to test (default: agent-specific)",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    args = parser.parse_args()

    # Determine which agents to test
    if args.agent == "all":
        configs = list(AGENT_CONFIGS.values())
    else:
        configs = [AGENT_CONFIGS[args.agent]]

    # Determine experiments
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = list(EXPERIMENT_MAP.keys())

    # Run evaluations
    all_results = []
    for config in configs:
        # Use specified difficulties or agent-specific defaults
        difficulties = args.difficulties if args.difficulties else config.difficulties

        # Use specified cogs or agent-specific defaults
        cogs_list = args.cogs if args.cogs else config.cogs_list

        results = run_evaluation(
            agent_config=config,
            experiments=experiments,
            difficulties=difficulties,
            cogs_list=cogs_list,
            max_steps=args.steps,
            seed=args.seed,
        )
        all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
