#!/usr/bin/env -S uv run
"""
Evaluation Script for Policies

Tests any policy including:
- Scripted agents: baseline, ladybug
- NIM agents: thinky, nim_random, nim_race_car
- Any custom policy via full class path

Usage:
  # Evaluate all predefined agents
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py --agent all

  # Evaluate specific scripted agent (shorthand)
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py \\
      --agent baseline --experiments oxygen_bottleneck --cogs 1

  # Evaluate NIM agent (shorthand)
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py \\
      --agent thinky --experiments oxygen_bottleneck --cogs 1

  # Evaluate ladybug (unclipping agent) with specific config
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py \\
      --agent ladybug --mission-set integrated_evals --cogs 2 4

  # Evaluate with full policy path and checkpoint
  uv run python packages/cogames/scripts/evaluate_scripted_agents.py \\
      --agent cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy \\
      --checkpoint ./checkpoints/model.pt --experiments oxygen_bottleneck --cogs 1
"""

import argparse
import importlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.mission import Mission, NumCogsVariant
from cogames.cogs_vs_clips.variants import VARIANTS
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
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
    clip_period: int
    total_reward: float  # Total reward across all agents
    avg_reward_per_agent: float  # Average reward per agent
    hearts_assembled: int
    steps_taken: int
    max_steps: int
    success: bool
    run_index: int
    seed_used: int


@dataclass
class AgentConfig:
    """Configuration for an agent policy."""

    key: str
    label: str
    policy_path: str  # Fully-qualified policy class path
    data_path: Optional[str] = None  # Optional checkpoint path


def is_clipping_difficulty(name: str) -> bool:
    """Check if a difficulty involves clipping."""
    return "clipped" in name.lower() or "clipping" in name.lower()


# Available agents
AGENT_CONFIGS: Dict[str, AgentConfig] = {
    "baseline": AgentConfig(
        key="baseline",
        label="Baseline",
        policy_path="cogames.policy.scripted_agent.baseline_agent.BaselinePolicy",
    ),
    "ladybug": AgentConfig(
        key="ladybug",
        label="Ladybug",
        policy_path="cogames.policy.scripted_agent.unclipping_agent.UnclippingPolicy",
    ),
}

EXPERIMENT_MAP: Dict[str, Mission] = {}


def load_eval_missions(module_path: str):
    """Dynamically import a module and return its EVAL_MISSIONS list."""
    module = importlib.import_module(module_path)
    missions = getattr(module, "EVAL_MISSIONS", None)
    if missions is None:
        raise AttributeError(f"Module '{module_path}' does not define EVAL_MISSIONS")
    return missions


def run_evaluation(
    agent_config: AgentConfig,
    experiments: List[str],
    variants: List[str],
    cogs_list: List[int],
    max_steps: int = 1000,
    seed: int = 42,
    repeats: int = 3,
    experiment_map: Dict[str, Mission] | None = None,
) -> List[EvalResult]:
    """Run evaluation for an agent configuration."""
    results = []
    experiment_lookup = experiment_map if experiment_map is not None else EXPERIMENT_MAP

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluating: {agent_config.label}")
    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Variants: {len(variants) if variants else 0} (none = base mission)")
    logger.info(f"Agent counts: {cogs_list}")
    logger.info(f"{'=' * 80}\n")

    runs_per_case = max(1, int(repeats))
    total_cases = len(experiments) * max(1, len(variants)) * len(cogs_list)
    total_tests = total_cases * runs_per_case
    case_counter = 0
    completed_runs = 0

    for exp_name in experiments:
        if exp_name not in experiment_lookup:
            logger.error(f"Unknown experiment: {exp_name}")
            continue

        base_mission = experiment_lookup[exp_name]

        # If no variants specified, run with base mission
        variant_list = variants if variants else [None]

        for variant_name in variant_list:
            variant = None
            variant_label = "base"
            if variant_name:
                # Find the variant by name
                variant = next((v for v in VARIANTS if v.name == variant_name), None)
                if variant is None:
                    logger.error(f"Unknown variant: {variant_name}")
                    continue
                variant_label = variant_name

            for num_cogs in cogs_list:
                case_counter += 1
                logger.info(f"[{case_counter}/{total_cases}] {exp_name} | {variant_label} | {num_cogs} agent(s)")

                # Get clip period for metadata (if applicable)
                clip_period = getattr(variant, "extractor_clip_period", 0) if variant else 0

                try:
                    # Create mission and apply variant if specified
                    mission_variants = [NumCogsVariant(num_cogs=num_cogs)]
                    if variant:
                        mission_variants.insert(0, variant)

                    mission = base_mission.with_variants(mission_variants)

                    env_config = mission.make_env()
                    # Only override max_steps if variant doesn't specify it
                    has_override = (
                        (variant is not None)
                        and hasattr(variant, "max_steps_override")
                        and (variant.max_steps_override is not None)
                    )
                    if not has_override:
                        env_config.game.max_steps = max_steps

                    # Get the actual max_steps from env_config (after all modifications)
                    actual_max_steps = env_config.game.max_steps

                    # Create policy using generic initialization
                    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_config)

                    policy_spec = PolicySpec(
                        class_path=agent_config.policy_path,
                        data_path=agent_config.data_path,
                    )

                    policy = initialize_or_load_policy(policy_env_info, policy_spec)
                    agent_policies = [policy.agent_policy(i) for i in range(num_cogs)]

                    # Run repeated trials with seed offsets
                    for run_idx in range(runs_per_case):
                        run_seed = seed + run_idx
                        # Create rollout with run-specific seed
                        rollout = Rollout(
                            env_config,
                            agent_policies,
                            render_mode="none",
                            seed=run_seed,
                            pass_sim_to_policies=True,
                        )
                        rollout.run_until_done()

                        total_reward = float(sum(rollout._sim.episode_rewards))
                        avg_reward_per_agent = total_reward / max(1, num_cogs)
                        final_step = rollout._sim.current_step

                        result = EvalResult(
                            agent=agent_config.label,
                            experiment=exp_name,
                            num_cogs=num_cogs,
                            difficulty=variant_label,
                            clip_period=clip_period,
                            total_reward=total_reward,
                            avg_reward_per_agent=avg_reward_per_agent,
                            hearts_assembled=int(total_reward),
                            steps_taken=final_step + 1,
                            max_steps=actual_max_steps,
                            success=total_reward > 0,
                            seed_used=run_seed,
                            run_index=run_idx + 1,
                        )
                        results.append(result)

                        completed_runs += 1
                        status = "✓" if result.success else "✗"
                        logger.info(
                            f"  [run {run_idx + 1}/{runs_per_case}] {status} Total: {total_reward:.1f}, "
                            f"Avg/Agent: {avg_reward_per_agent:.1f}, Steps: {final_step + 1}/{actual_max_steps} "
                            f"(seed={run_seed}, progress {completed_runs}/{total_tests})"
                        )

                except Exception as e:
                    logger.error(f"  ✗ Error: {e}")
                    # Record failure
                    for run_idx in range(runs_per_case):
                        result = EvalResult(
                            agent=agent_config.label,
                            experiment=exp_name,
                            num_cogs=num_cogs,
                            difficulty=variant_label,
                            clip_period=clip_period,
                            total_reward=0.0,
                            avg_reward_per_agent=0.0,
                            hearts_assembled=0,
                            steps_taken=0,
                            max_steps=max_steps,
                            success=False,
                            run_index=run_idx + 1,
                            seed_used=seed + run_idx,
                        )
                        results.append(result)
                        completed_runs += 1

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
        avg_total_reward = sum(r.total_reward for r in agent_results) / len(agent_results)
        avg_reward_per_agent = sum(r.avg_reward_per_agent for r in agent_results) / len(agent_results)
        logger.info(
            f"  {agent}: {agent_successes}/{len(agent_results)} "
            f"({100 * agent_successes / len(agent_results):.1f}%) "
            f"avg_total={avg_total_reward:.2f} avg_per_agent={avg_reward_per_agent:.2f}"
        )

    # By agent count
    logger.info("\n## By Agent Count")
    cogs = sorted(set(r.num_cogs for r in results))
    for num_cogs in cogs:
        cogs_results = [r for r in results if r.num_cogs == num_cogs]
        cogs_successes = sum(1 for r in cogs_results if r.success)
        avg_total_reward = sum(r.total_reward for r in cogs_results) / len(cogs_results)
        avg_reward_per_agent = sum(r.avg_reward_per_agent for r in cogs_results) / len(cogs_results)
        logger.info(
            f"  {num_cogs} agent(s): {cogs_successes}/{len(cogs_results)} "
            f"({100 * cogs_successes / len(cogs_results):.1f}%) "
            f"avg_total={avg_total_reward:.2f} avg_per_agent={avg_reward_per_agent:.2f}"
        )

    # By variant
    logger.info("\n## By Variant")
    variants = sorted(set(r.difficulty for r in results))
    for var in variants:
        var_results = [r for r in results if r.difficulty == var]
        var_successes = sum(1 for r in var_results if r.success)
        avg_total_reward = sum(r.total_reward for r in var_results) / len(var_results)
        avg_reward_per_agent = sum(r.avg_reward_per_agent for r in var_results) / len(var_results)
        logger.info(
            f"  {var:20s}: {var_successes}/{len(var_results)} "
            f"({100 * var_successes / len(var_results):.1f}%) "
            f"avg_total={avg_total_reward:.2f} avg_per_agent={avg_reward_per_agent:.2f}"
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
        key = (r.agent, r.experiment, r.difficulty, r.num_cogs)
        data[key]["total_rewards"].append(r.total_reward)
        data[key]["avg_rewards"].append(r.avg_reward_per_agent)
        data[key]["successes"].append(r.success)

    # Aggregate data
    aggregated = {}
    for key, vals in data.items():
        agent, experiment, difficulty, num_cogs = key
        aggregated[key] = {
            "agent": agent,
            "experiment": experiment,
            "difficulty": difficulty,
            "num_cogs": num_cogs,
            "avg_total_reward": np.mean(vals["total_rewards"]),
            "avg_reward_per_agent": np.mean(vals["avg_rewards"]),
            "success_rate": np.mean(vals["successes"]),
        }

    # Get unique values for each dimension
    agents = sorted(set(r.agent for r in results))
    experiments = sorted(set(r.experiment for r in results))
    variants = sorted(set(r.difficulty for r in results))
    num_cogs_list = sorted(set(r.num_cogs for r in results))

    # 1. Average reward per agent by agent type
    _plot_by_agent(aggregated, agents, output_path)

    # 2. Total reward by agent type
    _plot_by_agent_total(aggregated, agents, output_path)

    # 3. Average reward per agent by num_cogs
    _plot_by_num_cogs(aggregated, num_cogs_list, agents, output_path)

    # 4. Total reward by num_cogs
    _plot_by_num_cogs_total(aggregated, num_cogs_list, agents, output_path)

    # 5. Average reward per agent by eval environment
    _plot_by_environment(aggregated, experiments, agents, output_path)

    # 6. Total reward by eval environment
    _plot_by_environment_total(aggregated, experiments, agents, output_path)

    # 7. Average reward per agent by variant
    _plot_by_difficulty(aggregated, variants, agents, output_path)

    # 8. Total reward by variant
    _plot_by_difficulty_total(aggregated, variants, agents, output_path)

    # 8.5. Average reward per agent by environment, grouped by agent count
    _plot_by_environment_by_cogs(aggregated, experiments, num_cogs_list, output_path)

    # 9. Heatmap: Environment x Agent (avg per agent)
    _plot_heatmap_env_agent(aggregated, experiments, agents, output_path)

    # 10. Heatmap: Environment x Agent (total)
    _plot_heatmap_env_agent_total(aggregated, experiments, agents, output_path)

    # 11. Heatmap: Variant x Agent (avg per agent)
    _plot_heatmap_diff_agent(aggregated, variants, agents, output_path)

    # 12. Heatmap: Variant x Agent (total)
    _plot_heatmap_diff_agent_total(aggregated, variants, agents, output_path)

    logger.info(f"✓ Plots saved to {output_path}/")


def _plot_by_agent(aggregated, agents, output_path):
    """Plot average reward per agent grouped by agent type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_rewards = defaultdict(list)
    for _key, vals in aggregated.items():
        agent_rewards[vals["agent"]].append(vals["avg_reward_per_agent"])

    avg_rewards = [np.mean(agent_rewards[agent]) for agent in agents]
    colors = plt.get_cmap("Set2")(range(len(agents)))

    bars = ax.bar(agents, avg_rewards, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Average Reward Per Agent", fontsize=12, fontweight="bold")
    ax.set_xlabel("Agent Type", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward Per Agent by Type", fontsize=14, fontweight="bold")
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


def _plot_by_agent_total(aggregated, agents, output_path):
    """Plot total reward grouped by agent type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    agent_rewards = defaultdict(list)
    for _key, vals in aggregated.items():
        agent_rewards[vals["agent"]].append(vals["avg_total_reward"])

    avg_rewards = [np.mean(agent_rewards[agent]) for agent in agents]
    colors = plt.cm.Set2(range(len(agents)))

    bars = ax.bar(agents, avg_rewards, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Agent Type", fontsize=12, fontweight="bold")
    ax.set_title("Total Reward by Agent Type", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path / "total_reward_by_agent.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_num_cogs(aggregated, num_cogs_list, agents, output_path):
    """Plot average reward per agent by number of agents."""
    fig, ax = plt.subplots(figsize=(12, 7))

    width = 0.35
    x = np.arange(len(num_cogs_list))

    for i, agent in enumerate(agents):
        rewards = []
        for num_cogs in num_cogs_list:
            vals = [
                v["avg_reward_per_agent"]
                for k, v in aggregated.items()
                if v["agent"] == agent and v["num_cogs"] == num_cogs
            ]
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

    ax.set_ylabel("Average Reward Per Agent", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward Per Agent by Team Size", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(num_cogs_list)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_num_cogs.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_num_cogs_total(aggregated, num_cogs_list, agents, output_path):
    """Plot total reward by number of agents."""
    fig, ax = plt.subplots(figsize=(12, 7))

    width = 0.35
    x = np.arange(len(num_cogs_list))

    for i, agent in enumerate(agents):
        rewards = []
        for num_cogs in num_cogs_list:
            vals = [
                v["avg_total_reward"]
                for k, v in aggregated.items()
                if v["agent"] == agent and v["num_cogs"] == num_cogs
            ]
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

    ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
    ax.set_title("Total Reward by Team Size", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(num_cogs_list)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "total_reward_by_num_cogs.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_environment(aggregated, experiments, agents, output_path):
    """Plot average reward per agent by eval environment."""
    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.35
    x = np.arange(len(experiments))

    for i, agent in enumerate(agents):
        rewards = []
        for exp in experiments:
            vals = [
                v["avg_reward_per_agent"]
                for k, v in aggregated.items()
                if v["agent"] == agent and v["experiment"] == exp
            ]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward Per Agent", fontsize=12, fontweight="bold")
    ax.set_xlabel("Eval Environment", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward by Eval Environment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_environment.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_environment_total(aggregated, experiments, agents, output_path):
    """Plot total reward by eval environment."""
    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.35
    x = np.arange(len(experiments))

    for i, agent in enumerate(agents):
        rewards = []
        for exp in experiments:
            vals = [
                v["avg_total_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["experiment"] == exp
            ]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Eval Environment", fontsize=12, fontweight="bold")
    ax.set_title("Total Reward by Eval Environment", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "total_reward_by_environment.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_environment_by_cogs(aggregated, experiments, num_cogs_list, output_path):
    """Plot average reward per agent by eval environment, grouped by agent count."""
    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.25
    x = np.arange(len(experiments))

    for i, num_cogs in enumerate(num_cogs_list):
        rewards = []
        for exp in experiments:
            vals = [
                v["avg_reward_per_agent"]
                for k, v in aggregated.items()
                if v["num_cogs"] == num_cogs and v["experiment"] == exp
            ]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(num_cogs_list) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=f"{num_cogs} agent(s)", alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward Per Agent", fontsize=12, fontweight="bold")
    ax.set_xlabel("Eval Environment", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward by Eval Environment (Grouped by Agent Count)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_environment_by_cogs.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_difficulty(aggregated, difficulties, agents, output_path):
    """Plot average reward per agent by difficulty variant."""
    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.35
    x = np.arange(len(difficulties))

    for i, agent in enumerate(agents):
        rewards = []
        for diff in difficulties:
            vals = [
                v["avg_reward_per_agent"]
                for k, v in aggregated.items()
                if v["agent"] == agent and v["difficulty"] == diff
            ]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Average Reward Per Agent", fontsize=12, fontweight="bold")
    ax.set_xlabel("Difficulty Variant", fontsize=12, fontweight="bold")
    ax.set_title("Average Reward by Difficulty Variant", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "reward_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_difficulty_total(aggregated, difficulties, agents, output_path):
    """Plot total reward by difficulty variant."""
    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.35
    x = np.arange(len(difficulties))

    for i, agent in enumerate(agents):
        rewards = []
        for diff in difficulties:
            vals = [
                v["avg_total_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["difficulty"] == diff
            ]
            rewards.append(np.mean(vals) if vals else 0)

        offset = width * (i - len(agents) / 2 + 0.5)
        ax.bar(x + offset, rewards, width, label=agent, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
    ax.set_xlabel("Difficulty Variant", fontsize=12, fontweight="bold")
    ax.set_title("Total Reward by Difficulty Variant", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "total_reward_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_heatmap_env_agent(aggregated, experiments, agents, output_path):
    """Create heatmap of Environment x Agent showing avg reward per agent."""
    fig, ax = plt.subplots(figsize=(10, len(experiments) * 0.5 + 2))

    matrix = np.zeros((len(experiments), len(agents)))
    for i, exp in enumerate(experiments):
        for j, agent in enumerate(agents):
            vals = [
                v["avg_reward_per_agent"]
                for k, v in aggregated.items()
                if v["agent"] == agent and v["experiment"] == exp
            ]
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


def _plot_heatmap_env_agent_total(aggregated, experiments, agents, output_path):
    """Create heatmap of Environment x Agent showing total reward."""
    fig, ax = plt.subplots(figsize=(10, len(experiments) * 0.5 + 2))

    matrix = np.zeros((len(experiments), len(agents)))
    for i, exp in enumerate(experiments):
        for j, agent in enumerate(agents):
            vals = [
                v["avg_total_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["experiment"] == exp
            ]
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
    cbar.set_label("Total Reward", rotation=270, labelpad=20, fontweight="bold")

    # Add text annotations
    for i in range(len(experiments)):
        for j in range(len(agents)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    ax.set_title("Total Reward: Environment × Agent", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
    ax.set_ylabel("Environment", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "heatmap_env_agent_total.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_heatmap_diff_agent(aggregated, difficulties, agents, output_path):
    """Create heatmap of Difficulty x Agent showing avg reward per agent."""
    fig, ax = plt.subplots(figsize=(10, len(difficulties) * 0.4 + 2))

    matrix = np.zeros((len(difficulties), len(agents)))
    for i, diff in enumerate(difficulties):
        for j, agent in enumerate(agents):
            vals = [
                v["avg_reward_per_agent"]
                for k, v in aggregated.items()
                if v["agent"] == agent and v["difficulty"] == diff
            ]
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


def _plot_heatmap_diff_agent_total(aggregated, difficulties, agents, output_path):
    """Create heatmap of Difficulty x Agent showing total reward."""
    fig, ax = plt.subplots(figsize=(10, len(difficulties) * 0.4 + 2))

    matrix = np.zeros((len(difficulties), len(agents)))
    for i, diff in enumerate(difficulties):
        for j, agent in enumerate(agents):
            vals = [
                v["avg_total_reward"] for k, v in aggregated.items() if v["agent"] == agent and v["difficulty"] == diff
            ]
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
    cbar.set_label("Total Reward", rotation=270, labelpad=20, fontweight="bold")

    # Add text annotations
    for i in range(len(difficulties)):
        for j in range(len(agents)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    ax.set_title("Total Reward: Difficulty × Agent", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
    ax.set_ylabel("Difficulty", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "heatmap_diff_agent_total.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies across missions")
    parser.add_argument(
        "--agent",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Agent(s)/policy to evaluate. Can specify multiple. Can be:\n"
            "  - Shorthand: 'baseline', 'ladybug', 'thinky', 'all'\n"
            "  - Full policy path: 'cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy'\n"
            "  - Other registered names: 'nim_thinky', 'nim_random', 'unclipping', etc.\n"
            "  - Default: ladybug and thinky"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for trained policies (optional)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiments to test (default: all). Use class names like 'OxygenBottleneck'",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help=(
            "Mission variants to test (default: none, runs base missions). "
            "Available variants from VARIANTS in variants.py"
        ),
    )
    parser.add_argument(
        "--cogs",
        nargs="*",
        type=int,
        default=None,
        help="Agent counts to test (default: 1, 2, 4)",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
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
    parser.add_argument(
        "--mission-set",
        choices=["eval_missions", "integrated_evals", "diagnostic_evals", "all"],
        default="all",
        help=(
            "Mission set selector. "
            "'all' runs on all three mission sets (eval_missions, integrated_evals, diagnostic_evals); "
            "Or specify individual sets: 'eval_missions', 'integrated_evals', 'diagnostic_evals'"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of runs per case with different seeds (default: 3)",
    )

    args = parser.parse_args()

    # Select mission set based on argument
    mission_set = args.mission_set

    if mission_set == "all":
        # Load all three mission sets
        missions_list = []
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.eval_missions"))
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals"))
        missions_list.extend([mission_cls() for mission_cls in DIAGNOSTIC_EVALS])  # type: ignore[call-arg]
    elif mission_set == "diagnostic_evals":
        missions_list = [mission_cls() for mission_cls in DIAGNOSTIC_EVALS]  # type: ignore[call-arg]
    elif mission_set == "eval_missions":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.eval_missions")
    elif mission_set == "integrated_evals":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals")
    else:
        raise ValueError(f"Unknown mission set: {mission_set}")

    experiment_map = {mission.name: mission for mission in missions_list}  # type: ignore[misc]
    global EXPERIMENT_MAP
    EXPERIMENT_MAP = experiment_map  # type: ignore[assignment]

    # Determine which agents to test based on --agent argument
    # Can be: "all", a shorthand like "baseline", or a full policy path
    # Default to ladybug and thinky if nothing specified
    if args.agent is None or len(args.agent) == 0:
        # Default: ladybug and thinky
        agent_keys = ["ladybug", "thinky"]
    else:
        agent_keys = args.agent

    configs = []
    for agent_key in agent_keys:
        if agent_key == "all":
            # Add all predefined configs
            configs.extend(list(AGENT_CONFIGS.values()))
        elif agent_key in AGENT_CONFIGS:
            # Use predefined config
            configs.append(AGENT_CONFIGS[agent_key])
        else:
            # Treat as a policy class path (full or shorthand)
            label = agent_key.rsplit(".", 1)[-1] if "." in agent_key else agent_key
            configs.append(
                AgentConfig(
                    key="custom",
                    label=label,
                    policy_path=agent_key,
                    data_path=args.checkpoint,
                )
            )

    # Determine experiments
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = list(experiment_map.keys())

    # Run evaluations
    all_results = []
    for config in configs:
        # Use specified variants, or default to no variants (base missions)
        variants = args.variants if args.variants else []

        # Use specified cogs or default to [1, 2, 4]
        cogs_list = args.cogs if args.cogs else [1, 2, 4]

        results = run_evaluation(
            agent_config=config,
            experiments=experiments,
            variants=variants,
            cogs_list=cogs_list,
            experiment_map=experiment_map,  # type: ignore[arg-type]
            max_steps=args.steps,
            seed=args.seed,
            repeats=args.repeats,
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
