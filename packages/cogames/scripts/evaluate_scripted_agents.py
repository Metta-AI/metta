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
from typing import Any, Callable, Dict, List, Optional, Tuple

from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.mission import Mission, MissionVariant, NumCogsVariant
from cogames.cogs_vs_clips.variants import VARIANTS
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

plt: Any | None = None
np: Any | None = None


def _ensure_vibe_supports_gear(env_cfg) -> None:
    """
    Ensure the change_vibe action space is large enough to include the 'gear' vibe
    if any assembler protocol uses it.
    """
    try:
        assembler = env_cfg.game.objects.get("assembler")
        uses_gear = False
        if assembler is not None and hasattr(assembler, "protocols"):
            for proto in assembler.protocols:
                if any(v == "gear" for v in getattr(proto, "vibes", [])):
                    uses_gear = True
                    break
        if uses_gear:
            change_vibe = env_cfg.game.actions.change_vibe
            if getattr(change_vibe, "number_of_vibes", 0) < 8:
                change_vibe.number_of_vibes = 8
    except Exception:
        # Best-effort; if anything fails, leave as-is.
        pass


@dataclass(slots=True)
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


@dataclass(slots=True)
class AgentConfig:
    """Configuration for an agent policy."""

    key: str
    label: str
    policy_path: str  # Fully-qualified policy class path
    data_path: Optional[str] = None  # Optional checkpoint path


@dataclass(slots=True)
class AggregateMetrics:
    """Rolling statistics for a grouping of EvalResults."""

    count: int = 0
    success_count: int = 0
    total_reward_sum: float = 0.0
    avg_reward_sum: float = 0.0

    def update(self, result: EvalResult) -> None:
        self.count += 1
        self.success_count += int(result.success)
        self.total_reward_sum += result.total_reward
        self.avg_reward_sum += result.avg_reward_per_agent

    def mean_total_reward(self) -> float:
        return self.total_reward_sum / self.count if self.count else 0.0

    def mean_agent_reward(self) -> float:
        return self.avg_reward_sum / self.count if self.count else 0.0

    def success_rate(self) -> float:
        return self.success_count / self.count if self.count else 0.0


@dataclass(slots=True)
class AggregatedResults:
    """Container for frequently accessed aggregations."""

    by_agent: Dict[str, AggregateMetrics]
    by_num_cogs: Dict[int, AggregateMetrics]
    by_difficulty: Dict[str, AggregateMetrics]
    by_experiment: Dict[str, AggregateMetrics]
    by_agent_num_cogs: Dict[Tuple[str, int], AggregateMetrics]
    by_agent_experiment: Dict[Tuple[str, str], AggregateMetrics]
    by_agent_difficulty: Dict[Tuple[str, str], AggregateMetrics]
    by_num_cogs_experiment: Dict[Tuple[int, str], AggregateMetrics]


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
VARIANT_LOOKUP: Dict[str, MissionVariant] = {variant.name: variant for variant in VARIANTS}


def load_eval_missions(module_path: str) -> List[Mission]:
    """Dynamically import a module and return its EVAL_MISSIONS list."""
    module = importlib.import_module(module_path)
    missions = getattr(module, "EVAL_MISSIONS", None)
    if missions is None:
        raise AttributeError(f"Module '{module_path}' does not define EVAL_MISSIONS")
    return missions


def aggregate_results(results: List[EvalResult]) -> AggregatedResults:
    """Aggregate frequently reused statistics in a single pass."""

    by_agent: defaultdict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
    by_num_cogs: defaultdict[int, AggregateMetrics] = defaultdict(AggregateMetrics)
    by_difficulty: defaultdict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
    by_experiment: defaultdict[str, AggregateMetrics] = defaultdict(AggregateMetrics)
    by_agent_num_cogs: defaultdict[Tuple[str, int], AggregateMetrics] = defaultdict(AggregateMetrics)
    by_agent_experiment: defaultdict[Tuple[str, str], AggregateMetrics] = defaultdict(AggregateMetrics)
    by_agent_difficulty: defaultdict[Tuple[str, str], AggregateMetrics] = defaultdict(AggregateMetrics)
    by_num_cogs_experiment: defaultdict[Tuple[int, str], AggregateMetrics] = defaultdict(AggregateMetrics)

    for result in results:
        by_agent[result.agent].update(result)
        by_num_cogs[result.num_cogs].update(result)
        by_difficulty[result.difficulty].update(result)
        by_experiment[result.experiment].update(result)
        by_agent_num_cogs[(result.agent, result.num_cogs)].update(result)
        by_agent_experiment[(result.agent, result.experiment)].update(result)
        by_agent_difficulty[(result.agent, result.difficulty)].update(result)
        by_num_cogs_experiment[(result.num_cogs, result.experiment)].update(result)

    return AggregatedResults(
        by_agent=dict(by_agent),
        by_num_cogs=dict(by_num_cogs),
        by_difficulty=dict(by_difficulty),
        by_experiment=dict(by_experiment),
        by_agent_num_cogs=dict(by_agent_num_cogs),
        by_agent_experiment=dict(by_agent_experiment),
        by_agent_difficulty=dict(by_agent_difficulty),
        by_num_cogs_experiment=dict(by_num_cogs_experiment),
    )


def _ensure_plot_modules_loaded() -> Tuple[Any, Any]:
    """Lazily import matplotlib/numpy so non-plot runs avoid the overhead."""

    global plt, np
    if plt is None or np is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod  # type: ignore
        import numpy as np_mod  # type: ignore

        plt = plt_mod
        np = np_mod
    return plt, np


def run_evaluation(
    agent_config: AgentConfig,
    experiments: List[str],
    variants: List[str],
    cogs_list: List[int],
    max_steps: int = 1000,
    seed: int = 42,
    repeats: int = 3,
    experiment_map: Optional[Dict[str, Mission]] = None,
) -> List[EvalResult]:
    """Run evaluation for an agent configuration."""
    results = []
    experiment_lookup = experiment_map if experiment_map is not None else EXPERIMENT_MAP
    num_cogs_variant_cache: Dict[int, NumCogsVariant] = {}

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
                variant = VARIANT_LOOKUP.get(variant_name)
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
                    mission_variants: List[MissionVariant] = [
                        num_cogs_variant_cache.setdefault(num_cogs, NumCogsVariant(num_cogs=num_cogs))
                    ]
                    if variant:
                        mission_variants.insert(0, variant)

                    mission = base_mission.with_variants(mission_variants)

                    env_config = mission.make_env()
                    # Ensure 'gear' vibe is representable in the action space when required.
                    _ensure_vibe_supports_gear(env_config)
                    # Only override max_steps if variant doesn't specify it
                    has_override = bool(
                        (variant is not None)
                        and hasattr(variant, "max_steps_override")
                        and variant.max_steps_override is not None
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


def print_summary(results: List[EvalResult]) -> Optional[AggregatedResults]:
    """Print summary statistics."""
    if not results:
        logger.info("\nNo results to summarize.")
        return None

    aggregated = aggregate_results(results)

    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}\n")

    total = len(results)
    successes = sum(metrics.success_count for metrics in aggregated.by_agent.values())
    logger.info(f"Total tests: {total}")
    logger.info(f"Successes: {successes}/{total} ({100 * successes / total:.1f}%)")

    # By agent
    logger.info("\n## By Agent")
    agents = sorted(aggregated.by_agent)
    for agent in agents:
        metrics = aggregated.by_agent[agent]
        logger.info(
            f"  {agent}: {metrics.success_count}/{metrics.count} "
            f"({100 * metrics.success_rate():.1f}%) "
            f"avg_total={metrics.mean_total_reward():.2f} avg_per_agent={metrics.mean_agent_reward():.2f}"
        )

    # By agent count
    logger.info("\n## By Agent Count")
    cogs_counts = sorted(aggregated.by_num_cogs)
    for num_cogs in cogs_counts:
        metrics = aggregated.by_num_cogs[num_cogs]
        logger.info(
            f"  {num_cogs} agent(s): {metrics.success_count}/{metrics.count} "
            f"({100 * metrics.success_rate():.1f}%) "
            f"avg_total={metrics.mean_total_reward():.2f} avg_per_agent={metrics.mean_agent_reward():.2f}"
        )

    # By variant
    logger.info("\n## By Variant")
    variant_keys = sorted(aggregated.by_difficulty)
    for variant_key in variant_keys:
        metrics = aggregated.by_difficulty[variant_key]
        logger.info(
            f"  {variant_key:20s}: {metrics.success_count}/{metrics.count} "
            f"({100 * metrics.success_rate():.1f}%) "
            f"avg_total={metrics.mean_total_reward():.2f} avg_per_agent={metrics.mean_agent_reward():.2f}"
        )

    return aggregated


def create_plots(
    results: List[EvalResult],
    output_dir: str = "eval_plots",
    aggregated: Optional[AggregatedResults] = None,
    annotate_bars: bool = True,
) -> None:
    """Create plots with minimal boilerplate via plot specs."""

    if not results:
        return

    agg = aggregated if aggregated is not None else aggregate_results(results)
    _ensure_plot_modules_loaded()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"\nGenerating plots in {output_path}/...")

    agents = sorted(agg.by_agent)
    experiments = sorted(agg.by_experiment)
    variants = sorted(agg.by_difficulty)
    num_cogs_list = sorted(agg.by_num_cogs)

    def _val(metric: Optional[AggregateMetrics], getter: Callable[[AggregateMetrics], float]) -> float:
        return getter(metric) if metric else 0.0

    group_specs = [
        # Single-series bars (series label ignored)
        {
            "filename": "reward_by_agent.png",
            "title": "Average Reward Per Agent by Type",
            "xlabel": "Agent Type",
            "ylabel": "Average Reward Per Agent",
            "x_labels": agents,
            "series": ["value"],
            "lookup": lambda _s, a: _val(agg.by_agent.get(a), AggregateMetrics.mean_agent_reward),
            "figsize": (10, 6),
            "rotation": 0,
        },
        {
            "filename": "total_reward_by_agent.png",
            "title": "Total Reward by Agent Type",
            "xlabel": "Agent Type",
            "ylabel": "Total Reward",
            "x_labels": agents,
            "series": ["value"],
            "lookup": lambda _s, a: _val(agg.by_agent.get(a), AggregateMetrics.mean_total_reward),
            "figsize": (10, 6),
            "rotation": 0,
        },
        # Grouped bars
        {
            "filename": "reward_by_num_cogs.png",
            "title": "Average Reward Per Agent by Team Size",
            "xlabel": "Number of Agents",
            "ylabel": "Average Reward Per Agent",
            "x_labels": num_cogs_list,
            "series": agents,
            "lookup": lambda agent, cogs: _val(
                agg.by_agent_num_cogs.get((agent, cogs)), AggregateMetrics.mean_agent_reward
            ),
            "figsize": (12, 7),
            "rotation": 0,
        },
        {
            "filename": "total_reward_by_num_cogs.png",
            "title": "Total Reward by Team Size",
            "xlabel": "Number of Agents",
            "ylabel": "Total Reward",
            "x_labels": num_cogs_list,
            "series": agents,
            "lookup": lambda agent, cogs: _val(
                agg.by_agent_num_cogs.get((agent, cogs)), AggregateMetrics.mean_total_reward
            ),
            "figsize": (12, 7),
            "rotation": 0,
        },
        {
            "filename": "reward_by_environment.png",
            "title": "Average Reward by Eval Environment",
            "xlabel": "Eval Environment",
            "ylabel": "Average Reward Per Agent",
            "x_labels": experiments,
            "series": agents,
            "lookup": lambda agent, exp: _val(
                agg.by_agent_experiment.get((agent, exp)), AggregateMetrics.mean_agent_reward
            ),
        },
        {
            "filename": "total_reward_by_environment.png",
            "title": "Total Reward by Eval Environment",
            "xlabel": "Eval Environment",
            "ylabel": "Total Reward",
            "x_labels": experiments,
            "series": agents,
            "lookup": lambda agent, exp: _val(
                agg.by_agent_experiment.get((agent, exp)), AggregateMetrics.mean_total_reward
            ),
        },
        {
            "filename": "reward_by_difficulty.png",
            "title": "Average Reward by Difficulty Variant",
            "xlabel": "Difficulty Variant",
            "ylabel": "Average Reward Per Agent",
            "x_labels": variants,
            "series": agents,
            "lookup": lambda agent, diff: _val(
                agg.by_agent_difficulty.get((agent, diff)), AggregateMetrics.mean_agent_reward
            ),
        },
        {
            "filename": "total_reward_by_difficulty.png",
            "title": "Total Reward by Difficulty Variant",
            "xlabel": "Difficulty Variant",
            "ylabel": "Total Reward",
            "x_labels": variants,
            "series": agents,
            "lookup": lambda agent, diff: _val(
                agg.by_agent_difficulty.get((agent, diff)), AggregateMetrics.mean_total_reward
            ),
        },
        {
            "filename": "reward_by_environment_by_cogs.png",
            "title": "Average Reward by Eval Environment (Grouped by Agent Count)",
            "xlabel": "Eval Environment",
            "ylabel": "Average Reward Per Agent",
            "x_labels": experiments,
            "series": num_cogs_list,
            "lookup": lambda cogs, exp: _val(
                agg.by_num_cogs_experiment.get((cogs, exp)), AggregateMetrics.mean_agent_reward
            ),
        },
    ]

    for spec in group_specs:
        _plot_grouped_bars(
            x_labels=spec.get("x_labels", []),
            series_labels=spec.get("series", []),
            value_lookup=spec["lookup"],
            ylabel=spec["ylabel"],
            xlabel=spec["xlabel"],
            title=spec["title"],
            filename=spec["filename"],
            output_path=output_path,
            figsize=spec.get("figsize", (16, 8)),
            rotation=spec.get("rotation", 45),
            annotate=annotate_bars,
        )

    heatmap_specs = [
        {
            "filename": "heatmap_env_agent.png",
            "title": "Average Reward: Environment x Agent",
            "cbar": "Average Reward",
            "x_labels": agents,
            "y_labels": experiments,
            "lookup": lambda agent, exp: _val(
                agg.by_agent_experiment.get((agent, exp)), AggregateMetrics.mean_agent_reward
            ),
            "cmap": "YlOrRd",
            "height": max(2, int(len(experiments) * 0.5 + 2)),
        },
        {
            "filename": "heatmap_env_agent_total.png",
            "title": "Total Reward: Environment x Agent",
            "cbar": "Total Reward",
            "x_labels": agents,
            "y_labels": experiments,
            "lookup": lambda agent, exp: _val(
                agg.by_agent_experiment.get((agent, exp)), AggregateMetrics.mean_total_reward
            ),
            "cmap": "YlOrRd",
            "height": max(2, int(len(experiments) * 0.5 + 2)),
        },
        {
            "filename": "heatmap_diff_agent.png",
            "title": "Average Reward: Difficulty x Agent",
            "cbar": "Average Reward",
            "x_labels": agents,
            "y_labels": variants,
            "lookup": lambda agent, diff: _val(
                agg.by_agent_difficulty.get((agent, diff)), AggregateMetrics.mean_agent_reward
            ),
            "cmap": "YlGnBu",
            "height": max(2, int(len(variants) * 0.4 + 2)),
        },
        {
            "filename": "heatmap_diff_agent_total.png",
            "title": "Total Reward: Difficulty x Agent",
            "cbar": "Total Reward",
            "x_labels": agents,
            "y_labels": variants,
            "lookup": lambda agent, diff: _val(
                agg.by_agent_difficulty.get((agent, diff)), AggregateMetrics.mean_total_reward
            ),
            "cmap": "YlGnBu",
            "height": max(2, int(len(variants) * 0.4 + 2)),
        },
    ]

    for spec in heatmap_specs:
        _plot_heatmap(
            x_labels=spec.get("x_labels", []),
            y_labels=spec.get("y_labels", []),
            value_lookup=spec["lookup"],
            title=spec["title"],
            cbar_label=spec["cbar"],
            filename=spec["filename"],
            output_path=output_path,
            cmap=spec.get("cmap", "YlGnBu"),
            figsize=(10, spec.get("height", 5)),
        )

    logger.info(f"✓ Plots saved to {output_path}/")


def _plot_grouped_bars(
    x_labels: List[str],
    series_labels: List[str],
    value_lookup: Callable[[str, str], float],
    ylabel: str,
    xlabel: str,
    title: str,
    filename: str,
    output_path: Path,
    width: float = 0.35,
    figsize: Tuple[int, int] = (16, 8),
    rotation: float = 45,
    annotate: bool = False,
    annotation_fmt: str = ".1f",
) -> None:
    if not x_labels or not series_labels:
        return

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(x_labels))

    for i, series in enumerate(series_labels):
        rewards: List[float] = [value_lookup(series, label) for label in x_labels]
        offset = width * (i - len(series_labels) / 2 + 0.5)
        bars = ax.bar(x + offset, rewards, width, label=series, alpha=0.8, edgecolor="black")

        if annotate:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:{annotation_fmt}}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotation, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_heatmap(
    x_labels: List[str],
    y_labels: List[str],
    value_lookup: Callable[[str, str], float],
    title: str,
    cbar_label: str,
    filename: str,
    output_path: Path,
    cmap: str,
    figsize: Tuple[int, int],
    annotation_fmt: str = ".1f",
) -> None:
    if not x_labels or not y_labels:
        return

    fig, ax = plt.subplots(figsize=figsize)

    matrix = np.zeros((len(y_labels), len(x_labels)))
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            matrix[i, j] = value_lookup(x, y)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontweight="bold")

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:{annotation_fmt}}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
    ax.set_ylabel("Environment" if "Environment" in title else "Difficulty", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_heatmap_env_agent(
    aggregated: AggregatedResults, experiments: List[str], agents: List[str], output_path: Path
) -> None:
    """Create heatmap of Environment x Agent showing avg reward per agent."""
    _plot_heatmap(
        x_labels=agents,
        y_labels=experiments,
        value_lookup=lambda agent, exp: (
            aggregated.by_agent_experiment.get((agent, exp)).mean_agent_reward()
            if aggregated.by_agent_experiment.get((agent, exp))
            else 0.0
        ),
        title="Average Reward: Environment x Agent",
        cbar_label="Average Reward",
        filename="heatmap_env_agent.png",
        output_path=output_path,
        cmap="YlOrRd",
        figsize=(10, int(len(experiments) * 0.5 + 2)),
        annotation_fmt=".1f",
    )


def _plot_heatmap_env_agent_total(
    aggregated: AggregatedResults, experiments: List[str], agents: List[str], output_path: Path
) -> None:
    """Create heatmap of Environment x Agent showing total reward."""
    _plot_heatmap(
        x_labels=agents,
        y_labels=experiments,
        value_lookup=lambda agent, exp: (
            aggregated.by_agent_experiment.get((agent, exp)).mean_total_reward()
            if aggregated.by_agent_experiment.get((agent, exp))
            else 0.0
        ),
        title="Total Reward: Environment x Agent",
        cbar_label="Total Reward",
        filename="heatmap_env_agent_total.png",
        output_path=output_path,
        cmap="YlOrRd",
        figsize=(10, int(len(experiments) * 0.5 + 2)),
        annotation_fmt=".1f",
    )


def _plot_heatmap_diff_agent(
    aggregated: AggregatedResults, difficulties: List[str], agents: List[str], output_path: Path
) -> None:
    """Create heatmap of Difficulty x Agent showing avg reward per agent."""
    _plot_heatmap(
        x_labels=agents,
        y_labels=difficulties,
        value_lookup=lambda agent, diff: (
            aggregated.by_agent_difficulty.get((agent, diff)).mean_agent_reward()
            if aggregated.by_agent_difficulty.get((agent, diff))
            else 0.0
        ),
        title="Average Reward: Difficulty x Agent",
        cbar_label="Average Reward",
        filename="heatmap_diff_agent.png",
        output_path=output_path,
        cmap="YlGnBu",
        figsize=(10, int(len(difficulties) * 0.4 + 2)),
        annotation_fmt=".1f",
    )


def _plot_heatmap_diff_agent_total(
    aggregated: AggregatedResults, difficulties: List[str], agents: List[str], output_path: Path
) -> None:
    """Create heatmap of Difficulty x Agent showing total reward."""
    _plot_heatmap(
        x_labels=agents,
        y_labels=difficulties,
        value_lookup=lambda agent, diff: (
            aggregated.by_agent_difficulty.get((agent, diff)).mean_total_reward()
            if aggregated.by_agent_difficulty.get((agent, diff))
            else 0.0
        ),
        title="Total Reward: Difficulty x Agent",
        cbar_label="Total Reward",
        filename="heatmap_diff_agent_total.png",
        output_path=output_path,
        cmap="YlGnBu",
        figsize=(10, int(len(difficulties) * 0.4 + 2)),
        annotation_fmt=".1f",
    )


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
        choices=["eval_missions", "integrated_evals", "diagnostic_evals", "spanning_evals", "all"],
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
        # Load all mission sets
        missions_list = []
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.eval_missions"))
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals"))
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.spanning_evals"))
        missions_list.extend([mission_cls() for mission_cls in DIAGNOSTIC_EVALS])  # type: ignore[call-arg]
    elif mission_set == "diagnostic_evals":
        missions_list = [mission_cls() for mission_cls in DIAGNOSTIC_EVALS]  # type: ignore[call-arg]
    elif mission_set == "eval_missions":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.eval_missions")
    elif mission_set == "integrated_evals":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals")
    elif mission_set == "spanning_evals":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.spanning_evals")
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
    aggregated_summary = print_summary(all_results)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")

    # Generate plots
    if not args.no_plots and all_results:
        create_plots(all_results, output_dir=args.plot_dir, aggregated=aggregated_summary)


if __name__ == "__main__":
    main()
