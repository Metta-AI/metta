#!/usr/bin/env -S uv run
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
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from cogames.cogs_vs_clips.evals.difficulty_variants import DIFFICULTY_VARIANTS, get_difficulty
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import NumCogsVariant
from cogames.policy.scripted_agent.baseline_agent import BaselinePolicy
from cogames.policy.scripted_agent.types import BASELINE_HYPERPARAMETER_PRESETS
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

    agent: str
    experiment: str
    num_cogs: int
    difficulty: str
    preset: str
    clip_period: int
    total_reward: float  # Average reward per agent
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    success: bool


@dataclass
class AgentConfig:
    """Configuration for a baseline agent."""

    key: str
    label: str
    policy_class: type
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
        policy_class=BaselinePolicy,
        cogs_list=[1, 2, 4, 8],
        difficulties=[d.name for d in DIFFICULTY_VARIANTS],  # Run on all including clipping for comparison
    ),
    "unclipping": AgentConfig(
        key="unclipping",
        label="UnclippingAgent",
        policy_class=UnclippingPolicy,
        cogs_list=[1, 2, 4, 8],
        difficulties=[d.name for d in DIFFICULTY_VARIANTS],  # With and without clipping
    ),
}

# All evaluation missions
EXPERIMENT_MAP = {mission.name: mission for mission in EVAL_MISSIONS}


def run_evaluation(
    agent_config: AgentConfig,
    experiments: List[str],
    difficulties: List[str],
    cogs_list: List[int],
    max_steps: int = 1000,
    seed: int = 42,
    preset: str = "default",
) -> List[EvalResult]:
    """Run evaluation for an agent configuration."""
    results = []

    # Get hyperparameters for the preset
    base_hyperparams = BASELINE_HYPERPARAMETER_PRESETS.get(preset)
    if base_hyperparams is None:
        logger.error(f"Unknown preset: {preset}. Using 'default'.")
        base_hyperparams = BASELINE_HYPERPARAMETER_PRESETS["default"]
        preset = "default"

    # Convert to UnclippingHyperparameters if using unclipping agent
    if agent_config.policy_class == UnclippingPolicy:
        hyperparams = UnclippingHyperparameters(
            recharge_threshold_low=base_hyperparams.recharge_threshold_low,
            recharge_threshold_high=base_hyperparams.recharge_threshold_high,
            stuck_detection_enabled=base_hyperparams.stuck_detection_enabled,
            stuck_escape_distance=base_hyperparams.stuck_escape_distance,
            position_history_size=base_hyperparams.position_history_size,
            exploration_area_check_window=base_hyperparams.exploration_area_check_window,
            exploration_area_size_threshold=base_hyperparams.exploration_area_size_threshold,
            exploration_escape_duration=base_hyperparams.exploration_escape_duration,
            exploration_direction_persistence=base_hyperparams.exploration_direction_persistence,
            exploration_assembler_distance_threshold=base_hyperparams.exploration_assembler_distance_threshold,
        )
    else:
        hyperparams = base_hyperparams

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluating: {agent_config.label} (preset: {preset})")
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

        base_mission = EXPERIMENT_MAP[exp_name]

        for difficulty_name in difficulties:
            try:
                difficulty = get_difficulty(difficulty_name)
            except Exception:
                logger.error(f"Unknown difficulty: {difficulty_name}")
                continue

            for num_cogs in cogs_list:
                completed += 1
                logger.info(f"[{completed}/{total_tests}] {exp_name} | {difficulty_name} | {num_cogs} agent(s)")

                # Create mission and apply difficulty (always from base_mission)
                mission = base_mission.with_variants([difficulty, NumCogsVariant(num_cogs=num_cogs)])

                # Get clip period for metadata
                clip_period = getattr(difficulty, "extractor_clip_period", 0)

                try:
                    env_config = mission.make_env()
                    # Only override max_steps if difficulty doesn't specify it
                    if not hasattr(difficulty, "max_steps_override") or difficulty.max_steps_override is None:
                        env_config.game.max_steps = max_steps

                    # Get the actual max_steps from env_config (after all modifications)
                    actual_max_steps = env_config.game.max_steps

                    # Create policy with PolicyEnvInterface and hyperparameters
                    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_config)
                    policy = agent_config.policy_class(policy_env_info, hyperparams)
                    agent_policies = [policy.agent_policy(i) for i in range(num_cogs)]

                    # Create rollout and run episode
                    rollout = Rollout(
                        env_config,
                        agent_policies,
                        render_mode="none",
                        seed=seed,
                        pass_sim_to_policies=True,
                    )
                    rollout.run_until_done()

                    # Get results - average reward per agent
                    total_reward = float(sum(rollout._sim.episode_rewards)) / num_cogs
                    final_step = rollout._sim.current_step

                    # Record result
                    result = EvalResult(
                        agent=agent_config.label,
                        experiment=exp_name,
                        num_cogs=num_cogs,
                        difficulty=difficulty_name,
                        preset=preset,
                        clip_period=clip_period,
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
                        agent=agent_config.label,
                        experiment=exp_name,
                        num_cogs=num_cogs,
                        difficulty=difficulty_name,
                        preset=preset,
                        clip_period=0,
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

    # By preset (if multiple presets tested)
    presets = sorted(set(r.preset for r in results))
    if len(presets) > 1:
        logger.info("\n## By Hyperparameter Preset")
        for preset in presets:
            preset_results = [r for r in results if r.preset == preset]
            preset_successes = sum(1 for r in preset_results if r.success)
            avg_reward = sum(r.total_reward for r in preset_results) / len(preset_results)
            logger.info(
                f"  {preset:15s}: {preset_successes}/{len(preset_results)} "
                f"({100 * preset_successes / len(preset_results):.1f}%) "
                f"avg_reward={avg_reward:.2f}"
            )


def create_plots(results: List[EvalResult], output_dir: str = "eval_plots"):
    """Create comprehensive plots from evaluation results."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"\nGenerating plots in {output_path}/...")

    # Convert results to dicts for easier processing
    data = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r.agent, r.experiment, r.difficulty, r.num_cogs, r.preset)
        data[key]["rewards"].append(r.total_reward)
        data[key]["successes"].append(r.success)

    # Aggregate data
    aggregated = {}
    for key, vals in data.items():
        agent, experiment, difficulty, num_cogs, preset = key
        aggregated[key] = {
            "agent": agent,
            "experiment": experiment,
            "difficulty": difficulty,
            "num_cogs": num_cogs,
            "preset": preset,
            "avg_reward": np.mean(vals["rewards"]),
            "success_rate": np.mean(vals["successes"]),
        }

    # Get unique values for each dimension
    agents = sorted(set(r.agent for r in results))
    experiments = sorted(set(r.experiment for r in results))
    difficulties = sorted(set(r.difficulty for r in results))
    num_cogs_list = sorted(set(r.num_cogs for r in results))
    presets = sorted(set(r.preset for r in results))

    # 1. Average reward by agent
    _plot_by_agent(aggregated, agents, output_path)

    # 2. Average reward by num_cogs
    _plot_by_num_cogs(aggregated, num_cogs_list, agents, output_path)

    # 3. Average reward by eval environment
    _plot_by_environment(aggregated, experiments, agents, output_path)

    # 4. Average reward by difficulty
    _plot_by_difficulty(aggregated, difficulties, agents, output_path)

    # 5. Heatmap: Environment x Agent
    _plot_heatmap_env_agent(aggregated, experiments, agents, output_path)

    # 6. Heatmap: Difficulty x Agent
    _plot_heatmap_diff_agent(aggregated, difficulties, agents, output_path)

    # 7. Reward by preset (if multiple presets)
    if len(presets) > 1:
        _plot_by_preset(aggregated, presets, agents, experiments, difficulties, output_path)

    logger.info(f"✓ Plots saved to {output_path}/")


def _plot_by_agent(aggregated, agents, output_path):
    """Plot average reward grouped by agent."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_rewards = defaultdict(list)
    for _key, vals in aggregated.items():
        agent_rewards[vals["agent"]].append(vals["avg_reward"])

    avg_rewards = [np.mean(agent_rewards[agent]) for agent in agents]
    colors = plt.cm.Set2(range(len(agents)))

    bars = ax.bar(agents, avg_rewards, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward by Agent", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_agent.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_num_cogs(aggregated, num_cogs_list, agents, output_path):
    """Plot average reward by number of agents."""
    fig, ax = plt.subplots(figsize=(12, 7))

    width = 0.35
    x = np.arange(len(num_cogs_list))

    for i, agent in enumerate(agents):
        rewards = []
        for num_cogs in num_cogs_list:
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["num_cogs"] == num_cogs]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        bars = ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}", ha="center", va="bottom", fontsize=9
                )

    ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward by Number of Agents", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(num_cogs_list)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_num_cogs.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_environment(aggregated, experiments, agents, output_path):
    """Plot average reward by eval environment."""
    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.35
    x = np.arange(len(experiments))

    for i, agent in enumerate(agents):
        rewards = []
        for exp in experiments:
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["experiment"] == exp]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Eval Environment", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward by Eval Environment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_environment.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_difficulty(aggregated, difficulties, agents, output_path):
    """Plot average reward by difficulty variant."""
    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.35
    x = np.arange(len(difficulties))

    for i, agent in enumerate(agents):
        rewards = []
        for diff in difficulties:
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["difficulty"] == diff]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Difficulty Variant", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward by Difficulty Variant", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_heatmap_env_agent(aggregated, experiments, agents, output_path):
    """Create heatmap of Environment x Agent."""
    fig, ax = plt.subplots(figsize=(10, len(experiments) * 0.5 + 2))

    matrix = np.zeros((len(experiments), len(agents)))
    for i, exp in enumerate(experiments):
        for j, agent in enumerate(agents):
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["experiment"] == exp]
            matrix[i, j] = np.mean(vals) if vals else 0

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(agents)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels(agents)
    ax.set_yticklabels(experiments)

    # Rotate the x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Reward", rotation=270, labelpad=20, fontweight="bold")

    # Add text annotations
    for i in range(len(experiments)):
        for j in range(len(agents)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    ax.set_title("Average Reward: Environment × Agent", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
    ax.set_ylabel("Environment", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "heatmap_env_agent.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_heatmap_diff_agent(aggregated, difficulties, agents, output_path):
    """Create heatmap of Difficulty x Agent."""
    fig, ax = plt.subplots(figsize=(10, len(difficulties) * 0.4 + 2))

    matrix = np.zeros((len(difficulties), len(agents)))
    for i, diff in enumerate(difficulties):
        for j, agent in enumerate(agents):
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["difficulty"] == diff]
            matrix[i, j] = np.mean(vals) if vals else 0

    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(agents)))
    ax.set_yticks(np.arange(len(difficulties)))
    ax.set_xticklabels(agents)
    ax.set_yticklabels(difficulties)

    # Rotate the x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Reward", rotation=270, labelpad=20, fontweight="bold")

    # Add text annotations
    for i in range(len(difficulties)):
        for j in range(len(agents)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    ax.set_title("Average Reward: Difficulty × Agent", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
    ax.set_ylabel("Difficulty", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "heatmap_diff_agent.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_preset(aggregated, presets, agents, experiments, difficulties, output_path):
    """Plot reward by hyperparameter preset across different dimensions."""

    # Preset x Agent
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Overall by preset and agent
    ax = axes[0]
    width = 0.35
    x = np.arange(len(presets))

    for i, agent in enumerate(agents):
        rewards = []
        for preset in presets:
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["preset"] == preset]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hyperparameter Preset", fontsize=11, fontweight="bold")
    ax.set_title("Reward by Preset (Overall)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(presets, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Preset performance on top 3 environments
    ax = axes[1]
    top_envs = sorted(
        experiments,
        key=lambda e: np.mean([v["avg_reward"] for k, v in aggregated.items() if v["experiment"] == e]),
        reverse=True,
    )[:3]  # type: ignore

    x = np.arange(len(presets))
    width = 0.25

    for i, env in enumerate(top_envs):
        rewards = []
        for preset in presets:
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["preset"] == preset and v["experiment"] == env]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - 1)
        ax.bar(x + offset, rewards, width, label=env, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hyperparameter Preset", fontsize=11, fontweight="bold")
    ax.set_title("Reward by Preset (Top 3 Envs)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(presets, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Preset performance on top 3 difficulties
    ax = axes[2]
    top_diffs = sorted(
        difficulties,
        key=lambda d: np.mean([v["avg_reward"] for k, v in aggregated.items() if v["difficulty"] == d]),  # type: ignore
        reverse=True,
    )[:3]

    x = np.arange(len(presets))
    width = 0.25

    for i, diff in enumerate(top_diffs):
        rewards = []
        for preset in presets:
            vals = [v["avg_reward"] for k, v in aggregated.items() if v["preset"] == preset and v["difficulty"] == diff]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - 1)
        ax.bar(x + offset, rewards, width, label=diff, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hyperparameter Preset", fontsize=11, fontweight="bold")
    ax.set_title("Reward by Preset (Top 3 Difficulties)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(presets, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_preset.png", dpi=150, bbox_inches="tight")
    plt.close()


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
    parser.add_argument(
        "--preset",
        choices=list(BASELINE_HYPERPARAMETER_PRESETS.keys()),
        default="default",
        help="Hyperparameter preset to use (default: default)",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Run evaluation across all hyperparameter presets",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="eval_plots",
        help="Directory to save plots (default: eval_plots)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

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

    # Determine which presets to test
    if args.all_presets:
        presets = list(BASELINE_HYPERPARAMETER_PRESETS.keys())
    else:
        presets = [args.preset]

    # Run evaluations
    all_results = []
    for preset in presets:
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
                preset=preset,
            )
            all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")

    # Generate plots
    if not args.no_plots and all_results:
        create_plots(all_results, output_dir=args.plot_dir)


if __name__ == "__main__":
    main()
