#!/usr/bin/env -S uv run
"""
Evaluation Script for Policies

Supports:
- Policy specs via --policy / -p (same format as cogames CLI, repeatable)
- Checkpoint bundle URIs (s3:// or file://) with policy_spec.json bundles
- Built-in scripted agents via --agent (legacy; `--agent all` runs all)

Usage snippets:
  uv run python packages/cogames/scripts/run_evaluation.py --policy class=ladybug
  uv run python packages/cogames/scripts/run_evaluation.py \
      --policy class=ladybug --experiments oxygen_bottleneck --cogs 1
  uv run python packages/cogames/scripts/run_evaluation.py \
      --policy s3://bucket/path/checkpoints/run:v5 --cogs 1
  uv run python packages/cogames/scripts/run_evaluation.py \
      --policy file://./train_dir/run:v5 --mission-set diagnostic_evals --cogs 1
  uv run python packages/cogames/scripts/run_evaluation.py --agent all  # legacy
"""

import argparse
import importlib
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

from cogames.cli.policy import parse_policy_spec
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from alo.pure_single_episode_runner import PureSingleEpisodeSpecJob, run_pure_single_episode_from_specs
from safetensors.torch import load_file as load_safetensors_file

from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.mission import Mission, MissionVariant, NumCogsVariant
from cogames.cogs_vs_clips.missions import MISSIONS as ALL_MISSIONS
from cogames.cogs_vs_clips.variants import VARIANTS
from mettagrid.policy.policy import PolicySpec

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Cache for policy action space sizes to avoid reloading checkpoints
_policy_action_space_cache: Dict[str, int] = {}


@dataclass
class EvalResult:
    agent: str
    experiment: str
    num_cogs: int
    difficulty: str
    clip_period: int
    total_reward: float
    avg_reward_per_agent: float
    hearts_assembled: int
    heart_gained: float  # Total heart.gained stat across all agents
    avg_heart_gained_per_agent: float  # Average heart.gained per agent
    steps_taken: int
    max_steps: int
    success: bool
    run_index: int
    seed_used: int


@dataclass
class AgentConfig:
    key: str
    label: str
    policy_path: str
    data_path: Optional[str] = None
    init_kwargs: Optional[dict[str, str]] = None
    source: Optional[str] = None


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://") if path else False


def _policy_source_is_s3(agent_config: AgentConfig) -> bool:
    return any(
        is_s3_uri(path) for path in (agent_config.source, agent_config.policy_path, agent_config.data_path) if path
    )


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
    "thinky": AgentConfig(
        key="thinky",
        label="Thinky",
        policy_path="cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy",
    ),
    "racecar": AgentConfig(
        key="racecar",
        label="RaceCar",
        policy_path="cogames.policy.nim_agents.agents.RaceCarAgentsMultiPolicy",
    ),
    "starter": AgentConfig(
        key="starter",
        label="Starter",
        policy_path="cogames.policy.scripted_agent.starter_agent.StarterPolicy",
    ),
}

EXPERIMENT_MAP: Dict[str, Mission] = {}
VARIANT_LOOKUP: Dict[str, MissionVariant] = {v.name: v for v in VARIANTS}


def load_eval_missions(module_path: str) -> List[Mission]:
    module = importlib.import_module(module_path)
    missions = getattr(module, "EVAL_MISSIONS", None)
    if missions is None:
        raise AttributeError(f"Module '{module_path}' does not define EVAL_MISSIONS")
    return missions


def _run_case(
    exp_name: str,
    variant_name: Optional[str],
    num_cogs: int,
    base_mission: Mission,
    variant: Optional[MissionVariant],
    clip_period: int,
    max_steps: int,
    seed: int,
    runs_per_case: int,
    agent_config: AgentConfig,
) -> List[EvalResult]:
    mission_variants: List[MissionVariant] = [NumCogsVariant(num_cogs=num_cogs)]
    if variant:
        mission_variants.insert(0, variant)
    mission = base_mission.with_variants(mission_variants)
    env_config = mission.make_env()

    assembler = env_config.game.objects.get("assembler")
    if assembler is not None and hasattr(assembler, "protocols"):
        for proto in assembler.protocols:
            if any(v == "gear" for v in getattr(proto, "vibes", [])):
                change_vibe = env_config.game.actions.change_vibe
                if not any(v.name == "gear" for v in change_vibe.vibes):
                    from mettagrid.config.vibes import VIBE_BY_NAME

                    change_vibe.vibes = list(change_vibe.vibes) + [VIBE_BY_NAME["gear"]]
                break

    policy_action_space = None
    policy_path = agent_config.data_path or agent_config.policy_path
    if policy_path:
        policy_action_space = _policy_action_space_cache.get(policy_path)
        if policy_action_space is None:
            try:
                spec = policy_spec_from_uri(policy_path)
                if spec.data_path:
                    for key, tensor in load_safetensors_file(str(Path(spec.data_path))).items():
                        if "actor_head" in key and "weight" in key and len(tensor.shape) == 2:
                            policy_action_space = int(tensor.shape[0])
                            _policy_action_space_cache[policy_path] = policy_action_space
                            logger.info(f"Detected policy action space: {policy_action_space} actions")
                            break
            except Exception as exc:
                logger.warning(f"Failed to detect policy action space: {exc}")

    if policy_action_space is not None:
        num_vibes = policy_action_space - 5
        if num_vibes <= 0:
            logger.warning(f"Invalid action space {policy_action_space}, skipping vibe configuration")
        else:
            from mettagrid.config import vibes as vibes_module

            if num_vibes == 16:
                vibe_names = [v.name for v in vibes_module.VIBES[:16]]
            elif num_vibes == 13:
                vibe_names = [v.name for v in vibes_module.VIBES[:13]]
            elif num_vibes <= len(vibes_module.VIBES):
                vibe_names = [v.name for v in vibes_module.VIBES[:num_vibes]]
            else:
                vibe_names = [v.name for v in vibes_module.VIBES]

            env_config.game.vibe_names = vibe_names

            if env_config.game.actions:
                if env_config.game.actions.change_vibe:
                    env_config.game.actions.change_vibe.vibes = [vibes_module.VIBE_BY_NAME[name] for name in vibe_names]
                    if env_config.game.agent.initial_vibe >= len(vibe_names):
                        env_config.game.agent.initial_vibe = 0
                if env_config.game.actions.attack:
                    env_config.game.actions.attack.enabled = False

            allowed_vibes = set(vibe_names)
            chest = env_config.game.objects.get("chest")
            if chest:
                vibe_transfers = getattr(chest, "vibe_transfers", None)
                if isinstance(vibe_transfers, dict):
                    chest.vibe_transfers = {v: t for v, t in vibe_transfers.items() if v in allowed_vibes}

    if variant is None or getattr(variant, "max_steps_override", None) is None:
        env_config.game.max_steps = max_steps

    # For evaluation, only heart rewards should count (not resource rewards)
    env_config.game.agent.rewards.stats = env_config.game.agent.rewards.stats or {}
    resource_stats = ["carbon.gained", "oxygen.gained", "germanium.gained", "silicon.gained"]
    for resource_stat in resource_stats:
        env_config.game.agent.rewards.stats[resource_stat] = 0.0
    env_config.game.agent.rewards.stats_max = env_config.game.agent.rewards.stats_max or {}
    for resource_stat in resource_stats:
        env_config.game.agent.rewards.stats_max[resource_stat] = 0.0

    actual_max_steps = env_config.game.max_steps
    policy_spec = PolicySpec(
        class_path=agent_config.policy_path,
        data_path=agent_config.data_path,
        init_kwargs=agent_config.init_kwargs or {},
    )

    out: List[EvalResult] = []
    assignments = [0] * num_cogs
    for run_idx in range(runs_per_case):
        run_seed = seed + run_idx
        job = PureSingleEpisodeSpecJob(
            policy_specs=[policy_spec],
            assignments=assignments,
            env=env_config,
            seed=run_seed,
            max_action_time_ms=10000,
        )
        results, _replay = run_pure_single_episode_from_specs(job, device="cpu")

        total_reward = float(sum(results.rewards))
        avg_reward_per_agent = total_reward / max(1, num_cogs)

        heart_gained = 0.0
        episode_stats = results.stats
        if "agent" in episode_stats:
            agent_stats_list = episode_stats["agent"]
            for agent_stats in agent_stats_list:
                heart_gained += float(agent_stats.get("heart.gained", 0.0))
        avg_heart_gained_per_agent = heart_gained / max(1, num_cogs)

        out.append(
            EvalResult(
                agent=agent_config.label,
                experiment=exp_name,
                num_cogs=num_cogs,
                difficulty=variant_name or "base",
                clip_period=clip_period,
                total_reward=total_reward,
                avg_reward_per_agent=avg_reward_per_agent,
                hearts_assembled=int(total_reward),
                heart_gained=heart_gained,
                avg_heart_gained_per_agent=avg_heart_gained_per_agent,
                steps_taken=results.steps + 1,
                max_steps=actual_max_steps,
                success=total_reward > 0,
                seed_used=run_seed,
                run_index=run_idx + 1,
            )
        )
    return out


def run_evaluation(
    agent_config: AgentConfig,
    experiments: List[str],
    variants: List[str],
    cogs_list: List[int],
    max_steps: int = 1000,
    seed: int = 42,
    repeats: int = 3,
    jobs: int = 0,
    experiment_map: Optional[Dict[str, Mission]] = None,
) -> List[EvalResult]:
    results: List[EvalResult] = []
    experiment_lookup = experiment_map if experiment_map is not None else EXPERIMENT_MAP
    runs_per_case = max(1, int(repeats))
    variant_list = variants or [None]

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Evaluating: {agent_config.label}")
    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Variants: {len(variant_list)} (none = base mission)")
    logger.info(f"Agent counts: {cogs_list}")
    logger.info(f"{'=' * 80}\n")

    cases: List[tuple[str, Optional[str], int, Mission, Optional[MissionVariant], int]] = []
    for exp_name in experiments:
        base_mission = experiment_lookup.get(exp_name)
        if base_mission is None:
            logger.error(f"Unknown experiment: {exp_name}")
            continue
        for variant_name in variant_list:
            variant = VARIANT_LOOKUP.get(variant_name) if variant_name else None
            if variant_name and variant is None:
                logger.error(f"Unknown variant: {variant_name}")
                continue
            clip_period = getattr(variant, "extractor_clip_period", 0) if variant else 0
            for num_cogs in cogs_list:
                cases.append((exp_name, variant_name, num_cogs, base_mission, variant, clip_period))

    total_cases = len(cases)
    total_tests = total_cases * runs_per_case
    completed = 0

    # Force sequential execution for S3 policies to avoid TensorDict threading issues
    # TensorDict has internal state that conflicts with Python threading
    use_threading = jobs != 1 and not _policy_source_is_s3(agent_config)
    max_workers = jobs if jobs > 0 else max(1, os.cpu_count() or 1)

    if use_threading:
        logger.info(f"Running with {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _run_case,
                    exp_name,
                    variant_name,
                    num_cogs,
                    base_mission,
                    variant,
                    clip_period,
                    max_steps,
                    seed,
                    runs_per_case,
                    agent_config,
                ): (exp_name, variant_name, num_cogs)
                for exp_name, variant_name, num_cogs, base_mission, variant, clip_period in cases
            }

            for idx, future in enumerate(as_completed(future_map), start=1):
                exp_name, variant_name, num_cogs = future_map[future]
                case_results = future.result()
                results.extend(case_results)
                completed += len(case_results)
                logger.info(
                    f"[{idx}/{total_cases}] {exp_name} | {variant_name or 'base'} | {num_cogs} agent(s) "
                    f"(progress {completed}/{total_tests})"
                )
    else:
        if _policy_source_is_s3(agent_config):
            logger.info("Running sequentially (S3 policies require sequential execution due to TensorDict)")
        else:
            logger.info("Running sequentially (--jobs 1)")

        for idx, (exp_name, variant_name, num_cogs, base_mission, variant, clip_period) in enumerate(cases, start=1):
            case_results = _run_case(
                exp_name,
                variant_name,
                num_cogs,
                base_mission,
                variant,
                clip_period,
                max_steps,
                seed,
                runs_per_case,
                agent_config,
            )
            results.extend(case_results)
            completed += len(case_results)
            logger.info(
                f"[{idx}/{total_cases}] {exp_name} | {variant_name or 'base'} | {num_cogs} agent(s) "
                f"(progress {completed}/{total_tests})"
            )

    return results


def print_summary(results: List[EvalResult]):
    if not results:
        logger.info("\nNo results to summarize.")
        return

    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}\n")

    diagnostic_results = [r for r in results if r.experiment.startswith("diagnostic_")]
    if diagnostic_results and len({r.agent for r in results}) == 1:
        logger.info("DIAGNOSTIC SUMMARY")
        grouped = defaultdict(list)
        for r in diagnostic_results:
            grouped[r.experiment].append(r)
        for experiment in sorted(grouped):
            group = grouped[experiment]
            successes = sum(1 for r in group if r.success)
            status = "PASS" if successes > 0 else "FAIL"
            logger.info(f"{status} {experiment} ({successes}/{len(group)})")
        logger.info("")

    total = len(results)
    successes = sum(1 for r in results if r.success)
    logger.info(f"Total tests: {total}")
    logger.info(f"Successes: {successes}/{total} ({100 * successes / total:.1f}%)")

    def _summarize(group_key, label):
        grouped = defaultdict(list)
        for r in results:
            grouped[getattr(r, group_key)].append(r)

        logger.info(f"\n## By {label}")
        for key in sorted(grouped):
            group = grouped[key]
            group_successes = sum(1 for r in group if r.success)
            avg_total_reward = sum(r.total_reward for r in group) / len(group)
            avg_reward_per_agent = sum(r.avg_reward_per_agent for r in group) / len(group)
            logger.info(
                f"  {key}: {group_successes}/{len(group)} "
                f"({100 * group_successes / len(group):.1f}%) "
                f"avg_total={avg_total_reward:.2f} avg_per_agent={avg_reward_per_agent:.2f}"
            )

    _summarize("agent", "Agent")
    _summarize("num_cogs", "Agent Count")
    _summarize("difficulty", "Variant")


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _bar_plot(
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_labels: List[str],
    series_labels: List[str],
    value_fn,
    output_path: Path,
    width: float = 0.35,
    rotation: int = 45,
    figsize: tuple[float, float] = (12, 7),
    annotate: bool = True,
):
    # Filter out labels that have no data for any series
    # This distinguishes between "evaluated but got 0" vs "not evaluated"
    labels_with_data = []
    for lbl in x_labels:
        has_any_data = False
        for series in series_labels:
            val = value_fn(series, lbl)
            # Check if value is not None and not NaN (None means no data)
            if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                has_any_data = True
                break
        if has_any_data:
            labels_with_data.append(lbl)

    if not labels_with_data:
        # No data at all, skip this plot
        plt.close()
        return

    # Use filtered labels
    x_labels = labels_with_data
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(x_labels))
    colors = plt.get_cmap("Set2")(range(max(1, len(series_labels))))

    if len(series_labels) == 1:
        vals = [value_fn(series_labels[0], lbl) for lbl in x_labels]
        # Filter out None values (replace with 0 for plotting, but we already filtered labels)
        vals = [v if v is not None else 0.0 for v in vals]
        bars = list(ax.bar(x, vals, color=colors[0], alpha=0.8, edgecolor="black"))
    else:
        bars = []
        for i, series in enumerate(series_labels):
            vals = [value_fn(series, lbl) for lbl in x_labels]
            # Filter out None values (replace with 0 for plotting)
            vals = [v if v is not None else 0.0 for v in vals]
            offset = width * (i - len(series_labels) / 2 + 0.5)
            bars.extend(
                ax.bar(x + offset, vals, width, label=str(series), color=colors[i], alpha=0.8, edgecolor="black")
            )
        ax.legend(fontsize=11)

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    # Adjust label alignment based on rotation
    ha = "right" if rotation < 45 else "center" if rotation == 90 else "right"
    ax.set_xticklabels(x_labels, rotation=rotation, ha=ha)
    ax.grid(axis="y", alpha=0.3)

    if annotate:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


def _heatmap(
    filename: str,
    title: str,
    x_labels: List[str],
    y_labels: List[str],
    value_fn,
    output_path: Path,
    figsize: tuple[float, float],
    xlabel: str,
    ylabel: str,
    x_rotation: int = 0,
    y_rotation: int = 0,
):
    # Build matrix, replacing None with 0.0 for plotting
    matrix_data = []
    for y in y_labels:
        row = []
        for x in x_labels:
            val = value_fn(x, y)
            # Convert None to 0.0, keep other values as float
            if val is None:
                row.append(0.0)
            else:
                row.append(float(val))
        matrix_data.append(row)
    matrix = np.array(matrix_data)
    fig, ax = plt.subplots(figsize=figsize)
    # Set vmax=10 for better gradient visualization
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Set rotation for x-axis labels
    if x_rotation != 0:
        plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha="right" if x_rotation > 45 else "center")
    else:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Set rotation for y-axis labels if needed
    if y_rotation != 0:
        plt.setp(ax.get_yticklabels(), rotation=y_rotation, ha="right")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value", rotation=270, labelpad=20, fontweight="bold")

    for i, _y in enumerate(y_labels):
        for j, _x in enumerate(x_labels):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150, bbox_inches="tight")
    plt.close()


def create_plots(results: List[EvalResult], output_dir: str = "eval_plots") -> None:
    if not results:
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"\nGenerating plots in {output_path}/...")

    data: defaultdict[tuple[str, str, str, int], defaultdict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r.agent, r.experiment, r.difficulty, r.num_cogs)
        data[key]["total_rewards"].append(r.total_reward)
        data[key]["avg_rewards"].append(r.avg_reward_per_agent)
        data[key]["heart_gained"].append(r.heart_gained)
        data[key]["avg_heart_gained"].append(r.avg_heart_gained_per_agent)
        data[key]["successes"].append(r.success)

    aggregated: Dict[tuple[str, str, str, int], Dict[str, float | str | int]] = {}
    for key, vals in data.items():
        agent, experiment, difficulty, num_cogs = key
        aggregated[key] = {
            "agent": agent,
            "experiment": experiment,
            "difficulty": difficulty,
            "num_cogs": num_cogs,
            "avg_total_reward": _mean(vals["total_rewards"]),
            "avg_reward_per_agent": _mean(vals["avg_rewards"]),
            "avg_heart_gained": _mean(vals["heart_gained"]),
            "avg_heart_gained_per_agent": _mean(vals["avg_heart_gained"]),
            "success_rate": _mean(vals["successes"]),
        }

    agents = sorted(set(r.agent for r in results))
    experiments = sorted(set(r.experiment for r in results))  # Only experiments with actual results
    variants = sorted(set(r.difficulty for r in results))  # Only variants with actual results
    num_cogs_list = sorted(set(r.num_cogs for r in results))  # Only agent counts with actual results

    def lookup(agent: str | None, exp: str | None, diff: str | None, num_cogs: int | None, field: str):
        """Lookup aggregated value. Returns None if no data exists, 0.0 if data exists but is zero."""
        vals: List[float] = [
            float(v[field])
            for v in aggregated.values()
            if (agent is None or v["agent"] == agent)
            and (exp is None or v["experiment"] == exp)
            and (diff is None or v["difficulty"] == diff)
            and (num_cogs is None or v["num_cogs"] == num_cogs)
        ]
        return float(np.mean(vals)) if vals else None  # Return None if no data (not evaluated)

    def has_data(agent: str | None, exp: str | None, diff: str | None, num_cogs: int | None) -> bool:
        """Check if there's any data for the given combination."""
        return any(
            (agent is None or v["agent"] == agent)
            and (exp is None or v["experiment"] == exp)
            and (diff is None or v["difficulty"] == diff)
            and (num_cogs is None or v["num_cogs"] == num_cogs)
            for v in aggregated.values()
        )

    def filter_comparison_environments(envs: List[str], field: str) -> List[str]:
        """Filter environments for comparison plots: exclude envs where ALL policies got zero reward."""
        filtered = []
        for exp in envs:
            # Check if any policy has non-zero reward for this environment
            has_nonzero = False
            for agent in agents:
                val = lookup(agent, exp, None, None, field)
                if val is not None and val > 0.0:
                    has_nonzero = True
                    break
            if has_nonzero:
                filtered.append(exp)
        return filtered

    bar_specs = [
        {
            "filename": "reward_by_agent.png",
            "title": "Average Reward Per Agent by Type",
            "xlabel": "Agent Type",
            "ylabel": "Average Reward Per Agent",
            "x_labels": agents,
            "series": ["value"],
            "fn": lambda _s, a: lookup(a, None, None, None, "avg_reward_per_agent"),
            "rotation": 45,
            "figsize": (max(12, len(agents) * 1.2), 6),  # Wider figure for rotated labels
        },
        {
            "filename": "total_reward_by_agent.png",
            "title": "Total Reward by Agent Type",
            "xlabel": "Agent Type",
            "ylabel": "Total Reward",
            "x_labels": agents,
            "series": ["value"],
            "fn": lambda _s, a: lookup(a, None, None, None, "avg_total_reward"),
            "rotation": 45,
            "figsize": (max(12, len(agents) * 1.2), 6),  # Wider figure for rotated labels
        },
        {
            "filename": "reward_by_num_cogs.png",
            "title": "Average Reward Per Agent by Team Size",
            "xlabel": "Number of Agents",
            "ylabel": "Average Reward Per Agent",
            "x_labels": [str(c) for c in num_cogs_list],
            "series": agents,
            "fn": lambda agent, c: lookup(agent, None, None, int(c), "avg_reward_per_agent"),
            "rotation": 0,
        },
        {
            "filename": "total_reward_by_num_cogs.png",
            "title": "Total Reward by Team Size",
            "xlabel": "Number of Agents",
            "ylabel": "Total Reward",
            "x_labels": [str(c) for c in num_cogs_list],
            "series": agents,
            "fn": lambda agent, c: lookup(agent, None, None, int(c), "avg_total_reward"),
            "rotation": 0,
        },
        {
            "filename": "reward_by_environment.png",
            "title": "Average Reward by Eval Environment (Comparison)",
            "xlabel": "Eval Environment",
            "ylabel": "Average Reward Per Agent",
            "x_labels": filter_comparison_environments(experiments, "avg_reward_per_agent"),
            # Filter: exclude envs where all policies got zero
            "series": agents,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_reward_per_agent"),
            "figsize": (
                max(16, len(filter_comparison_environments(experiments, "avg_reward_per_agent")) * 0.4),
                7,
            ),
            "rotation": 90,  # Vertical labels to prevent overlap
            "width": 0.15,  # Very thin bars to prevent overlap with 4 agents
        },
        {
            "filename": "total_reward_by_environment.png",
            "title": "Total Reward by Eval Environment (Comparison)",
            "xlabel": "Eval Environment",
            "ylabel": "Total Reward",
            "x_labels": filter_comparison_environments(experiments, "avg_total_reward"),
            # Filter: exclude envs where all policies got zero
            "series": agents,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_total_reward"),
            "figsize": (
                max(16, len(filter_comparison_environments(experiments, "avg_total_reward")) * 0.4),
                7,
            ),
            "rotation": 90,  # Vertical labels to prevent overlap
            "width": 0.15,  # Very thin bars to prevent overlap with 4 agents
        },
        {
            "filename": "reward_by_difficulty.png",
            "title": "Average Reward by Difficulty Variant",
            "xlabel": "Difficulty Variant",
            "ylabel": "Average Reward Per Agent",
            "x_labels": variants,
            "series": agents,
            "fn": lambda agent, diff: lookup(agent, None, diff, None, "avg_reward_per_agent"),
        },
        {
            "filename": "total_reward_by_difficulty.png",
            "title": "Total Reward by Difficulty Variant",
            "xlabel": "Difficulty Variant",
            "ylabel": "Total Reward",
            "x_labels": variants,
            "series": agents,
            "fn": lambda agent, diff: lookup(agent, None, diff, None, "avg_total_reward"),
        },
        {
            "filename": "reward_by_environment_by_cogs.png",
            "title": "Average Reward by Eval Environment (Grouped by Agent Count)",
            "xlabel": "Eval Environment",
            "ylabel": "Average Reward Per Agent",
            "x_labels": experiments,
            "series": [str(c) for c in num_cogs_list],
            "fn": lambda cogs, exp: lookup(None, exp, None, int(cogs), "avg_reward_per_agent"),
        },
        # Heart.gained plots mirroring reward plots
        {
            "filename": "heart_gained_by_agent.png",
            "title": "Average Heart Gained Per Agent by Type",
            "xlabel": "Agent Type",
            "ylabel": "Average Heart Gained Per Agent",
            "x_labels": agents,
            "series": ["value"],
            "fn": lambda _s, a: lookup(a, None, None, None, "avg_heart_gained_per_agent"),
            "rotation": 45,
            "figsize": (max(12, len(agents) * 1.2), 6),  # Wider figure for rotated labels
        },
        {
            "filename": "total_heart_gained_by_agent.png",
            "title": "Total Heart Gained by Agent Type",
            "xlabel": "Agent Type",
            "ylabel": "Total Heart Gained",
            "x_labels": agents,
            "series": ["value"],
            "fn": lambda _s, a: lookup(a, None, None, None, "avg_heart_gained"),
            "rotation": 45,
            "figsize": (max(12, len(agents) * 1.2), 6),  # Wider figure for rotated labels
        },
        {
            "filename": "heart_gained_by_num_cogs.png",
            "title": "Average Heart Gained Per Agent by Team Size",
            "xlabel": "Number of Agents",
            "ylabel": "Average Heart Gained Per Agent",
            "x_labels": [str(c) for c in num_cogs_list],
            "series": agents,
            "fn": lambda agent, c: lookup(agent, None, None, int(c), "avg_heart_gained_per_agent"),
            "rotation": 0,
        },
        {
            "filename": "total_heart_gained_by_num_cogs.png",
            "title": "Total Heart Gained by Team Size",
            "xlabel": "Number of Agents",
            "ylabel": "Total Heart Gained",
            "x_labels": [str(c) for c in num_cogs_list],
            "series": agents,
            "fn": lambda agent, c: lookup(agent, None, None, int(c), "avg_heart_gained"),
            "rotation": 0,
        },
        {
            "filename": "heart_gained_by_environment.png",
            "title": "Average Heart Gained Per Agent by Eval Environment (Comparison)",
            "xlabel": "Eval Environment",
            "ylabel": "Average Heart Gained Per Agent",
            "x_labels": filter_comparison_environments(experiments, "avg_heart_gained_per_agent"),
            # Filter: exclude envs where all policies got zero
            "series": agents,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_heart_gained_per_agent"),
            "figsize": (
                max(16, len(filter_comparison_environments(experiments, "avg_heart_gained_per_agent")) * 0.4),
                7,
            ),
            "rotation": 90,  # Vertical labels to prevent overlap
            "width": 0.15,  # Very thin bars to prevent overlap with 4 agents
        },
        {
            "filename": "total_heart_gained_by_environment.png",
            "title": "Total Heart Gained by Eval Environment (Comparison)",
            "xlabel": "Eval Environment",
            "ylabel": "Total Heart Gained",
            "x_labels": filter_comparison_environments(experiments, "avg_heart_gained"),
            # Filter: exclude envs where all policies got zero
            "series": agents,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_heart_gained"),
            "figsize": (
                max(16, len(filter_comparison_environments(experiments, "avg_heart_gained")) * 0.4),
                7,
            ),
            "rotation": 90,  # Vertical labels to prevent overlap
            "width": 0.15,  # Very thin bars to prevent overlap with 4 agents
        },
        {
            "filename": "heart_gained_by_difficulty.png",
            "title": "Average Heart Gained Per Agent by Difficulty Variant",
            "xlabel": "Difficulty Variant",
            "ylabel": "Average Heart Gained Per Agent",
            "x_labels": variants,
            "series": agents,
            "fn": lambda agent, diff: lookup(agent, None, diff, None, "avg_heart_gained_per_agent"),
        },
        {
            "filename": "total_heart_gained_by_difficulty.png",
            "title": "Total Heart Gained by Difficulty Variant",
            "xlabel": "Difficulty Variant",
            "ylabel": "Total Heart Gained",
            "x_labels": variants,
            "series": agents,
            "fn": lambda agent, diff: lookup(agent, None, diff, None, "avg_heart_gained"),
        },
    ]

    for spec in bar_specs:
        _bar_plot(
            filename=spec["filename"],
            title=spec["title"],
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            x_labels=spec["x_labels"],
            series_labels=spec["series"],
            value_fn=spec["fn"],
            output_path=output_path,
            rotation=spec.get("rotation", 45),
            figsize=spec.get("figsize", (12, 7)),
            width=spec.get("width", 0.35),  # Use custom width if specified, otherwise default
        )

    # Filter environments for comparison heatmaps (exclude envs where all policies got zero)
    comparison_experiments_reward = filter_comparison_environments(experiments, "avg_reward_per_agent")

    heatmap_specs = [
        {
            "filename": "heatmap_env_agent.png",
            "title": "Average Reward: Environment × Agent (Comparison)",
            "x_labels": agents,
            "y_labels": comparison_experiments_reward,  # Filtered: exclude envs where all policies got zero
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_reward_per_agent"),
            "figsize": (
                max(12, len(agents) * 1.2),
                len(comparison_experiments_reward) * 0.5 + 2,
            ),  # Wider for rotated labels
            "xlabel": "Agent",
            "ylabel": "Environment",
            "x_rotation": 45,  # Rotate agent labels to prevent overlap
        },
        {
            "filename": "heatmap_env_agent_total.png",
            "title": "Total Reward: Environment × Agent (Comparison)",
            "x_labels": agents,
            "y_labels": comparison_experiments_reward,  # Filtered: exclude envs where all policies got zero
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_total_reward"),
            "figsize": (
                max(12, len(agents) * 1.2),
                len(comparison_experiments_reward) * 0.5 + 2,
            ),  # Wider for rotated labels
            "xlabel": "Agent",
            "ylabel": "Environment",
            "x_rotation": 45,  # Rotate agent labels to prevent overlap
        },
        {
            "filename": "heatmap_diff_agent.png",
            "title": "Average Reward: Difficulty × Agent",
            "x_labels": agents,
            "y_labels": variants,
            "fn": lambda agent, diff: lookup(agent, None, diff, None, "avg_reward_per_agent"),
            "figsize": (10, len(variants) * 0.4 + 2),
            "xlabel": "Agent",
            "ylabel": "Difficulty",
        },
        {
            "filename": "heatmap_diff_agent_total.png",
            "title": "Total Reward: Difficulty × Agent",
            "x_labels": agents,
            "y_labels": variants,
            "fn": lambda agent, diff: lookup(agent, None, diff, None, "avg_total_reward"),
            "figsize": (10, len(variants) * 0.4 + 2),
            "xlabel": "Agent",
            "ylabel": "Difficulty",
        },
        {
            "filename": "heatmap_env_diff.png",
            "title": "Average Reward: Environment × Difficulty",
            "x_labels": experiments,
            "y_labels": variants,
            "fn": lambda exp, diff: lookup(None, exp, diff, None, "avg_reward_per_agent"),
            "figsize": (max(12, len(experiments) * 0.6), len(variants) * 0.4 + 2),
            "xlabel": "Environment",
            "ylabel": "Difficulty",
        },
        {
            "filename": "heatmap_env_diff_total.png",
            "title": "Total Reward: Environment × Difficulty",
            "x_labels": experiments,
            "y_labels": variants,
            "fn": lambda exp, diff: lookup(None, exp, diff, None, "avg_total_reward"),
            "figsize": (max(12, len(experiments) * 0.6), len(variants) * 0.4 + 2),
            "xlabel": "Environment",
            "ylabel": "Difficulty",
        },
    ]

    for spec in heatmap_specs:
        _heatmap(
            filename=spec["filename"],
            title=spec["title"],
            x_labels=spec["x_labels"],
            y_labels=spec["y_labels"],
            value_fn=spec["fn"],
            output_path=output_path,
            figsize=spec["figsize"],
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            x_rotation=spec.get("x_rotation", 0),
            y_rotation=spec.get("y_rotation", 0),
        )

    # Create individual policy plots (one per policy, showing all environments that policy was evaluated on)
    per_policy_dir = output_path / "per_policy"
    per_policy_dir.mkdir(exist_ok=True)
    logger.info(f"\nGenerating per-policy plots in {per_policy_dir}/...")

    for agent in agents:
        # Get all environments this policy was evaluated on
        agent_experiments = sorted(set(r.experiment for r in results if r.agent == agent))

        if not agent_experiments:
            continue

        # Sanitize agent name for filename
        safe_agent_name = agent.replace("/", "_").replace(" ", "_").replace(":", "_")

        # Plot 1: Average reward per agent by environment
        _bar_plot(
            filename=f"{safe_agent_name}_reward_by_environment.png",
            title=f"Average Reward Per Agent by Environment: {agent}",
            xlabel="Environment",
            ylabel="Average Reward Per Agent",
            x_labels=agent_experiments,  # All environments this policy was evaluated on
            series_labels=["value"],
            value_fn=lambda _s, exp_name, agent_name=agent: lookup(
                agent_name, exp_name, None, None, "avg_reward_per_agent"
            ),
            output_path=per_policy_dir,
            rotation=90,
            figsize=(max(16, len(agent_experiments) * 0.4), 7),
        )

        # Plot 2: Total reward by environment
        _bar_plot(
            filename=f"{safe_agent_name}_total_reward_by_environment.png",
            title=f"Total Reward by Environment: {agent}",
            xlabel="Environment",
            ylabel="Total Reward",
            x_labels=agent_experiments,  # All environments this policy was evaluated on
            series_labels=["value"],
            value_fn=lambda _s, exp_name, agent_name=agent: lookup(
                agent_name, exp_name, None, None, "avg_total_reward"
            ),
            output_path=per_policy_dir,
            rotation=90,
            figsize=(max(16, len(agent_experiments) * 0.4), 7),
        )

        # Plot 3: Heart gained per agent by environment
        _bar_plot(
            filename=f"{safe_agent_name}_heart_gained_by_environment.png",
            title=f"Average Heart Gained Per Agent by Environment: {agent}",
            xlabel="Environment",
            ylabel="Average Heart Gained Per Agent",
            x_labels=agent_experiments,  # All environments this policy was evaluated on
            series_labels=["value"],
            value_fn=lambda _s, exp_name, agent_name=agent: lookup(
                agent_name, exp_name, None, None, "avg_heart_gained_per_agent"
            ),
            output_path=per_policy_dir,
            rotation=90,
            figsize=(max(16, len(agent_experiments) * 0.4), 7),
        )

        # Plot 4: Total heart gained by environment
        _bar_plot(
            filename=f"{safe_agent_name}_total_heart_gained_by_environment.png",
            title=f"Total Heart Gained by Environment: {agent}",
            xlabel="Environment",
            ylabel="Total Heart Gained",
            x_labels=agent_experiments,  # All environments this policy was evaluated on
            series_labels=["value"],
            value_fn=lambda _s, exp_name, agent_name=agent: lookup(
                agent_name, exp_name, None, None, "avg_heart_gained"
            ),
            output_path=per_policy_dir,
            rotation=90,
            figsize=(max(16, len(agent_experiments) * 0.4), 7),
        )

    logger.info(f"✓ Comparison plots saved to {output_path}/")
    logger.info(f"✓ Per-policy plots saved to {per_policy_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate scripted or custom agents.")
    parser.add_argument(
        "--policy",
        "-p",
        action="append",
        default=None,
        help="Policy specification (same format as cogames CLI policy args). Repeat to evaluate multiple policies.",
    )
    parser.add_argument(
        "--agent",
        nargs="*",
        default=None,
        help="Legacy: agent key, class path, or S3 URI (prefer --policy)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Legacy: checkpoint path for --agent (prefer --policy class=...,data=...)",
    )
    parser.add_argument("--experiments", "-m", nargs="*", default=None, help="Experiments to run")
    parser.add_argument("--variants", nargs="*", default=None, help="Variants to apply")
    parser.add_argument("--cogs", "-c", nargs="*", type=int, default=None, help="Agent counts to test")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--plot-dir", type=str, default="eval_plots", help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--mission-set",
        "-S",
        choices=["integrated_evals", "spanning_evals", "diagnostic_evals", "tournament", "all"],
        default="all",
    )
    parser.add_argument("--diagnostic", action="store_true", help="Shortcut for --mission-set diagnostic_evals")
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Shortcut for integrated + diagnostic evals",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per case")
    parser.add_argument("--jobs", type=int, default=0, help="Max parallel cases (0 = CPU count)")

    args = parser.parse_args()

    if args.diagnostic and args.tournament:
        parser.error("Use only one of --diagnostic or --tournament.")
    if args.diagnostic:
        args.mission_set = "diagnostic_evals"
    if args.tournament:
        args.mission_set = "tournament"

    if args.mission_set == "all":
        missions_list = []
        # Skip eval_missions - they are deprecated
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals"))
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.spanning_evals"))
        missions_list.extend([mission_cls() for mission_cls in DIAGNOSTIC_EVALS])  # type: ignore[call-arg]
        eval_mission_names = {m.name for m in missions_list}
        for mission in ALL_MISSIONS:
            if mission.name not in eval_mission_names:
                missions_list.append(mission)
    elif args.mission_set == "diagnostic_evals":
        missions_list = [mission_cls() for mission_cls in DIAGNOSTIC_EVALS]  # type: ignore[call-arg]
    elif args.mission_set == "tournament":
        missions_list = []
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals"))
        missions_list.extend([mission_cls() for mission_cls in DIAGNOSTIC_EVALS])  # type: ignore[call-arg]
    elif args.mission_set == "integrated_evals":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals")
    elif args.mission_set == "spanning_evals":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.spanning_evals")
    else:
        raise ValueError(f"Unknown mission set: {args.mission_set}")

    experiment_map = {mission.name: mission for mission in missions_list}
    for mission in ALL_MISSIONS:
        experiment_map.setdefault(mission.name, mission)
    global EXPERIMENT_MAP
    EXPERIMENT_MAP = experiment_map

    configs: List[AgentConfig] = []
    if args.policy:
        if args.agent:
            logger.warning("Ignoring --agent because --policy was provided.")
        if args.checkpoint:
            logger.warning("Ignoring --checkpoint because --policy was provided.")

        for policy_arg in args.policy:
            try:
                policy_spec = parse_policy_spec(policy_arg).to_policy_spec()
            except (ValueError, ModuleNotFoundError) as exc:
                parser.error(f"Invalid --policy '{policy_arg}': {exc}")
            configs.append(
                AgentConfig(
                    key="custom",
                    label=policy_spec.name,
                    policy_path=policy_spec.class_path,
                    data_path=policy_spec.data_path,
                    init_kwargs=policy_spec.init_kwargs,
                    source=policy_arg,
                )
            )
    else:
        if args.agent:
            logger.warning("--agent is deprecated; use --policy instead.")
        if args.checkpoint and args.agent:
            logger.warning("--checkpoint is deprecated; use --policy class=...,data=... instead.")

        agent_keys = args.agent if args.agent else ["ladybug"]
        for agent_key in agent_keys:
            if agent_key == "all":
                configs.extend(list(AGENT_CONFIGS.values()))
            elif agent_key in AGENT_CONFIGS:
                configs.append(AGENT_CONFIGS[agent_key])
            elif is_s3_uri(agent_key):
                label = Path(agent_key).stem if "/" in agent_key else agent_key
                configs.append(
                    AgentConfig(
                        key="custom",
                        label=f"s3_{label}",
                        policy_path=agent_key,
                        data_path=None,
                        source=agent_key,
                    )
                )
            else:
                label = agent_key.rsplit(".", 1)[-1] if "." in agent_key else agent_key
                configs.append(
                    AgentConfig(
                        key="custom",
                        label=label,
                        policy_path=agent_key,
                        data_path=args.checkpoint,
                    )
                )

    experiments = args.experiments if args.experiments else list(experiment_map.keys())

    all_results: List[EvalResult] = []
    for config in configs:
        variants = args.variants if args.variants else []
        cogs_list = args.cogs if args.cogs else [1, 2, 4]

        # Clear policy cache between policies
        global _policy_action_space_cache
        _policy_action_space_cache = {}

        policy_results = run_evaluation(
            agent_config=config,
            experiments=experiments,
            variants=variants,
            cogs_list=cogs_list,
            experiment_map=experiment_map,
            max_steps=args.steps,
            seed=args.seed,
            repeats=args.repeats,
            jobs=args.jobs,
        )
        all_results.extend(policy_results)

        # Save incremental results after each policy
        if args.output and policy_results:
            with open(args.output, "w") as f:
                json.dump([asdict(r) for r in all_results], f, indent=2)
            logger.info(f"\n[INCREMENTAL] Results saved to: {args.output} ({len(all_results)} total tests)")

        # Generate per-policy plots
        if not args.no_plots and policy_results:
            policy_plot_dir = f"{args.plot_dir}_{config.label}" if args.plot_dir else f"eval_plots_{config.label}"
            logger.info(f"[INCREMENTAL] Generating plots for {config.label} in {policy_plot_dir}")
            create_plots(policy_results, output_dir=policy_plot_dir)

    print_summary(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        logger.info(f"\nFinal results saved to: {args.output}")

    if not args.no_plots and all_results:
        create_plots(all_results, output_dir=args.plot_dir)


if __name__ == "__main__":
    main()
