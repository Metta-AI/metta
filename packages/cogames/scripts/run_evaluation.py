#!/usr/bin/env -S uv run
"""
Evaluation Script for Policies

Supports:
- Built-in shorthands: baseline, ladybug (`--agent all` runs both)
- Any policy via full class path
- Local or S3 checkpoints when CheckpointManager is available

Usage snippets:
  uv run python packages/cogames/scripts/run_evaluation.py --agent all
  uv run python packages/cogames/scripts/run_evaluation.py \
      --agent baseline --experiments oxygen_bottleneck --cogs 1
  uv run python packages/cogames/scripts/run_evaluation.py \
      --agent cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy --cogs 1
  uv run python packages/cogames/scripts/run_evaluation.py \
      --agent cogames.policy.lstm.LSTMPolicy --checkpoint s3://bucket/path/model.mpt --cogs 1
  uv run python packages/cogames/scripts/run_evaluation.py \
      --agent s3://bucket/path/model.mpt --cogs 1
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.mission import Mission, MissionVariant, NumCogsVariant
from cogames.cogs_vs_clips.missions import MISSIONS as ALL_MISSIONS
from cogames.cogs_vs_clips.variants import VARIANTS
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

try:
    from metta.rl.checkpoint_manager import CheckpointManager

    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CHECKPOINT_MANAGER_AVAILABLE = False
    CheckpointManager = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_vibe_supports_gear(env_cfg) -> None:
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


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://") if path else False


def load_policy(
    policy_env_info: PolicyEnvInterface,
    policy_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    device = device or torch.device("cpu")

    if checkpoint_path and is_s3_uri(checkpoint_path):
        if not CHECKPOINT_MANAGER_AVAILABLE or CheckpointManager is None:
            raise ImportError("CheckpointManager not available. Install metta package to use S3 checkpoints.")
        logger.info(f"Loading policy from S3 URI: {checkpoint_path}")
        policy_spec = CheckpointManager.policy_spec_from_uri(checkpoint_path, device=device)
        return initialize_or_load_policy(policy_env_info, policy_spec)

    if is_s3_uri(policy_path):
        if not CHECKPOINT_MANAGER_AVAILABLE or CheckpointManager is None:
            raise ImportError("CheckpointManager not available. Install metta package to use S3 checkpoints.")
        logger.info(f"Loading policy from S3 URI: {policy_path}")
        policy_spec = CheckpointManager.policy_spec_from_uri(policy_path, device=device)
        return initialize_or_load_policy(policy_env_info, policy_spec)

    # Otherwise, use the standard initialization path
    policy_spec = PolicySpec(class_path=policy_path, data_path=checkpoint_path)
    return initialize_or_load_policy(policy_env_info, policy_spec)


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
    try:
        mission = base_mission.with_variants(mission_variants)
        env_config = mission.make_env()
        _ensure_vibe_supports_gear(env_config)
        if variant is None or getattr(variant, "max_steps_override", None) is None:
            env_config.game.max_steps = max_steps

        actual_max_steps = env_config.game.max_steps
        policy_env_info = PolicyEnvInterface.from_mg_cfg(env_config)
        policy = load_policy(policy_env_info, agent_config.policy_path, agent_config.data_path)
        agent_policies = [policy.agent_policy(i) for i in range(num_cogs)]

        out: List[EvalResult] = []
        for run_idx in range(runs_per_case):
            run_seed = seed + run_idx
            rollout = Rollout(
                env_config,
                agent_policies,
                render_mode="none",
                seed=run_seed,
                pass_sim_to_policies=True,
            )
            rollout.run_until_done()

            total_reward = float(sum(rollout._sim.episode_rewards))
            final_step = rollout._sim.current_step
            out.append(
                EvalResult(
                    agent=agent_config.label,
                    experiment=exp_name,
                    num_cogs=num_cogs,
                    difficulty=variant_name or "base",
                    clip_period=clip_period,
                    total_reward=total_reward,
                    avg_reward_per_agent=total_reward / max(1, num_cogs),
                    hearts_assembled=int(total_reward),
                    steps_taken=final_step + 1,
                    max_steps=actual_max_steps,
                    success=total_reward > 0,
                    seed_used=run_seed,
                    run_index=run_idx + 1,
                )
            )
        return out
    except Exception:
        # Use a fresh index to avoid referencing run_idx when the failure occurs before the loop above runs.
        return [
            EvalResult(
                agent=agent_config.label,
                experiment=exp_name,
                num_cogs=num_cogs,
                difficulty=variant_name or "base",
                clip_period=clip_period,
                total_reward=0.0,
                avg_reward_per_agent=0.0,
                hearts_assembled=0,
                steps_taken=0,
                max_steps=max_steps,
                success=False,
                seed_used=seed + i,
                run_index=i + 1,
            )
            for i in range(runs_per_case)
        ]


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
    max_workers = jobs if jobs > 0 else max(1, os.cpu_count() or 1)

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

    return results


def print_summary(results: List[EvalResult]):
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

    logger.info("\n## By Variant")
    variants_present = sorted(set(r.difficulty for r in results))
    for variant_key in variants_present:
        var_results = [r for r in results if r.difficulty == variant_key]
        var_successes = sum(1 for r in var_results if r.success)
        avg_total_reward = sum(r.total_reward for r in var_results) / len(var_results)
        avg_reward_per_agent = sum(r.avg_reward_per_agent for r in var_results) / len(var_results)
        logger.info(
            f"  {variant_key:20s}: {var_successes}/{len(var_results)} "
            f"({100 * var_successes / len(var_results):.1f}%) "
            f"avg_total={avg_total_reward:.2f} avg_per_agent={avg_reward_per_agent:.2f}"
        )


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
    figsize: tuple[int, int] = (12, 7),
    annotate: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(x_labels))
    # Always materialize at least one color so single-series plots have an explicit color
    colors = plt.get_cmap("Set2")(range(max(1, len(series_labels))))

    if len(series_labels) == 1:
        vals = [value_fn(series_labels[0], lbl) for lbl in x_labels]
        bars = list(ax.bar(x, vals, color=colors[0], alpha=0.8, edgecolor="black"))
    else:
        bars = []
        for i, series in enumerate(series_labels):
            vals = [value_fn(series, lbl) for lbl in x_labels]
            offset = width * (i - len(series_labels) / 2 + 0.5)
            bars.extend(
                ax.bar(x + offset, vals, width, label=str(series), color=colors[i], alpha=0.8, edgecolor="black")
            )
        ax.legend(fontsize=11)

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotation, ha="right")
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
):
    matrix = np.array([[value_fn(x, y) for x in x_labels] for y in y_labels])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

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
            "success_rate": _mean(vals["successes"]),
        }

    agents = sorted(set(r.agent for r in results))
    experiments = sorted(set(r.experiment for r in results))
    variants = sorted(set(r.difficulty for r in results))
    num_cogs_list = sorted(set(r.num_cogs for r in results))

    def lookup(agent: str | None, exp: str | None, diff: str | None, num_cogs: int | None, field: str) -> float:
        vals = [
            v[field]
            for v in aggregated.values()
            if (agent is None or v["agent"] == agent)
            and (exp is None or v["experiment"] == exp)
            and (diff is None or v["difficulty"] == diff)
            and (num_cogs is None or v["num_cogs"] == num_cogs)
        ]
        return float(np.mean(vals)) if vals else 0.0

    # Bar plots configured declaratively to keep styling consistent and code short.
    bar_specs = [
        {
            "filename": "reward_by_agent.png",
            "title": "Average Reward Per Agent by Type",
            "xlabel": "Agent Type",
            "ylabel": "Average Reward Per Agent",
            "x_labels": agents,
            "series": ["value"],
            "fn": lambda _s, a: lookup(a, None, None, None, "avg_reward_per_agent"),
            "rotation": 0,
            "figsize": (10, 6),
        },
        {
            "filename": "total_reward_by_agent.png",
            "title": "Total Reward by Agent Type",
            "xlabel": "Agent Type",
            "ylabel": "Total Reward",
            "x_labels": agents,
            "series": ["value"],
            "fn": lambda _s, a: lookup(a, None, None, None, "avg_total_reward"),
            "rotation": 0,
            "figsize": (10, 6),
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
            "title": "Average Reward by Eval Environment",
            "xlabel": "Eval Environment",
            "ylabel": "Average Reward Per Agent",
            "x_labels": experiments,
            "series": agents,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_reward_per_agent"),
        },
        {
            "filename": "total_reward_by_environment.png",
            "title": "Total Reward by Eval Environment",
            "xlabel": "Eval Environment",
            "ylabel": "Total Reward",
            "x_labels": experiments,
            "series": agents,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_total_reward"),
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
        )

    heatmap_specs = [
        {
            "filename": "heatmap_env_agent.png",
            "title": "Average Reward: Environment × Agent",
            "x_labels": agents,
            "y_labels": experiments,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_reward_per_agent"),
            "figsize": (10, len(experiments) * 0.5 + 2),
            "xlabel": "Agent",
            "ylabel": "Environment",
        },
        {
            "filename": "heatmap_env_agent_total.png",
            "title": "Total Reward: Environment × Agent",
            "x_labels": agents,
            "y_labels": experiments,
            "fn": lambda agent, exp: lookup(agent, exp, None, None, "avg_total_reward"),
            "figsize": (10, len(experiments) * 0.5 + 2),
            "xlabel": "Agent",
            "ylabel": "Environment",
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
        )

    logger.info(f"✓ Plots saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate scripted or custom agents.")
    parser.add_argument("--agent", nargs="*", default=None, help="Agent key, class path, or S3 URI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (or S3 URI)")
    parser.add_argument("--experiments", nargs="*", default=None, help="Experiments to run")
    parser.add_argument("--variants", nargs="*", default=None, help="Variants to apply")
    parser.add_argument("--cogs", nargs="*", type=int, default=None, help="Agent counts to test")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--plot-dir", type=str, default="eval_plots", help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--mission-set",
        choices=["eval_missions", "integrated_evals", "spanning_evals", "diagnostic_evals", "all"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per case")
    parser.add_argument("--jobs", type=int, default=0, help="Max parallel cases (0 = CPU count)")

    args = parser.parse_args()

    if args.mission_set == "all":
        missions_list = []
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.eval_missions"))
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.integrated_evals"))
        missions_list.extend(load_eval_missions("cogames.cogs_vs_clips.evals.spanning_evals"))
        missions_list.extend([mission_cls() for mission_cls in DIAGNOSTIC_EVALS])
        eval_mission_names = {m.name for m in missions_list}
        for mission in ALL_MISSIONS:
            if mission.name not in eval_mission_names:
                missions_list.append(mission)
    elif args.mission_set == "diagnostic_evals":
        missions_list = [mission_cls() for mission_cls in DIAGNOSTIC_EVALS]
    elif args.mission_set == "eval_missions":
        missions_list = load_eval_missions("cogames.cogs_vs_clips.evals.eval_missions")
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

    agent_keys = args.agent if args.agent else ["ladybug"]
    configs: List[AgentConfig] = []
    for agent_key in agent_keys:
        if agent_key == "all":
            configs.extend(list(AGENT_CONFIGS.values()))
        elif agent_key in AGENT_CONFIGS:
            configs.append(AGENT_CONFIGS[agent_key])
        elif is_s3_uri(agent_key):
            label = Path(agent_key).stem if "/" in agent_key else agent_key
            configs.append(AgentConfig(key="custom", label=f"s3_{label}", policy_path=agent_key, data_path=None))
        else:
            label = agent_key.rsplit(".", 1)[-1] if "." in agent_key else agent_key
            configs.append(AgentConfig(key="custom", label=label, policy_path=agent_key, data_path=args.checkpoint))

    experiments = args.experiments if args.experiments else list(experiment_map.keys())

    all_results: List[EvalResult] = []
    for config in configs:
        variants = args.variants if args.variants else []
        cogs_list = args.cogs if args.cogs else [1, 2, 4]
        all_results.extend(
            run_evaluation(
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
        )

    print_summary(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        logger.info(f"\nResults saved to: {args.output}")

    if not args.no_plots and all_results:
        create_plots(all_results, output_dir=args.plot_dir)


if __name__ == "__main__":
    main()
