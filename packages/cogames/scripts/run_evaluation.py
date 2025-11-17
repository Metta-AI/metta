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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
        return CheckpointManager.load_from_uri(checkpoint_path, policy_env_info, device)

    if is_s3_uri(policy_path):
        if not CHECKPOINT_MANAGER_AVAILABLE or CheckpointManager is None:
            raise ImportError("CheckpointManager not available. Install metta package to use S3 checkpoints.")
        return CheckpointManager.load_from_uri(policy_path, policy_env_info, device)

    return initialize_or_load_policy(
        policy_env_info,
        PolicySpec(class_path=policy_path, data_path=checkpoint_path),
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


def _lazy_plot_imports():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_mod  # type: ignore
    import numpy as np_mod  # type: ignore

    return plt_mod, np_mod


def create_plots(results: List[EvalResult], output_dir: str = "eval_plots"):
    if not results:
        return
    plt, np = _lazy_plot_imports()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"\nGenerating plots in {output_path}/...")

    agents = sorted(set(r.agent for r in results))
    experiments = sorted(set(r.experiment for r in results))
    variants = sorted(set(r.difficulty for r in results))
    num_cogs_list = sorted(set(r.num_cogs for r in results))

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def plot_bar(filename, title, xlabel, ylabel, labels, series, lookup, rotation=45, figsize=(12, 7)):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(labels))
        if len(series) == 1:
            vals = [lookup(series[0], lbl) for lbl in labels]
            ax.bar(x, vals)
        else:
            width = 0.8 / len(series)
            for i, s in enumerate(series):
                vals = [lookup(s, lbl) for lbl in labels]
                ax.bar(x + i * width, vals, width, label=str(s))
            ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xticks(x + (0 if len(series) == 1 else (0.4)))
        ax.set_xticklabels(labels, rotation=rotation)
        fig.tight_layout()
        fig.savefig(output_path / filename)
        plt.close(fig)

    def lookup_agent(agent, key, fn):
        vals = [r for r in results if r.agent == agent and key(r)]
        return mean([fn(r) for r in vals])

    def lookup_pair(a, b, fn):
        vals = [r for r in results if a(r) and b(r)]
        return mean([fn(r) for r in vals])

    plot_bar(
        "reward_by_agent.png",
        "Average Reward Per Agent by Type",
        "Agent Type",
        "Average Reward Per Agent",
        agents,
        ["value"],
        lambda _s, a: lookup_agent(a, lambda _r: True, lambda r: r.avg_reward_per_agent),
        rotation=0,
        figsize=(10, 6),
    )

    plot_bar(
        "total_reward_by_agent.png",
        "Total Reward by Agent Type",
        "Agent Type",
        "Total Reward",
        agents,
        ["value"],
        lambda _s, a: lookup_agent(a, lambda _r: True, lambda r: r.total_reward),
        rotation=0,
        figsize=(10, 6),
    )

    plot_bar(
        "reward_by_num_cogs.png",
        "Average Reward Per Agent by Team Size",
        "Number of Agents",
        "Average Reward Per Agent",
        num_cogs_list,
        agents,
        lambda agent, cogs: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.num_cogs == cogs,
            lambda r: r.avg_reward_per_agent,
        ),
        rotation=0,
    )

    plot_bar(
        "total_reward_by_num_cogs.png",
        "Total Reward by Team Size",
        "Number of Agents",
        "Total Reward",
        num_cogs_list,
        agents,
        lambda agent, cogs: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.num_cogs == cogs,
            lambda r: r.total_reward,
        ),
        rotation=0,
    )

    plot_bar(
        "reward_by_environment.png",
        "Average Reward by Eval Environment",
        "Eval Environment",
        "Average Reward Per Agent",
        experiments,
        agents,
        lambda agent, exp: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.experiment == exp,
            lambda r: r.avg_reward_per_agent,
        ),
    )

    plot_bar(
        "total_reward_by_environment.png",
        "Total Reward by Eval Environment",
        "Eval Environment",
        "Total Reward",
        experiments,
        agents,
        lambda agent, exp: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.experiment == exp,
            lambda r: r.total_reward,
        ),
    )

    plot_bar(
        "reward_by_difficulty.png",
        "Average Reward by Difficulty Variant",
        "Difficulty Variant",
        "Average Reward Per Agent",
        variants,
        agents,
        lambda agent, diff: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.difficulty == diff,
            lambda r: r.avg_reward_per_agent,
        ),
    )

    plot_bar(
        "total_reward_by_difficulty.png",
        "Total Reward by Difficulty Variant",
        "Difficulty Variant",
        "Total Reward",
        variants,
        agents,
        lambda agent, diff: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.difficulty == diff,
            lambda r: r.total_reward,
        ),
    )

    plot_bar(
        "reward_by_environment_by_cogs.png",
        "Average Reward by Eval Environment (Grouped by Agent Count)",
        "Eval Environment",
        "Average Reward Per Agent",
        experiments,
        num_cogs_list,
        lambda cogs, exp: lookup_pair(
            lambda r: r.num_cogs == cogs,
            lambda r: r.experiment == exp,
            lambda r: r.avg_reward_per_agent,
        ),
    )

    # Heatmaps
    def heatmap(filename, title, x_labels, y_labels, lookup_fn, figsize):
        data = np.array([[lookup_fn(x, y) for x in x_labels] for y in y_labels])
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, cmap="YlOrRd")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticklabels(y_labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_path / filename)
        plt.close(fig)

    heatmap(
        "heatmap_env_agent.png",
        "Average Reward: Environment x Agent",
        agents,
        experiments,
        lambda agent, exp: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.experiment == exp,
            lambda r: r.avg_reward_per_agent,
        ),
        figsize=(10, len(experiments) * 0.5 + 2),
    )

    heatmap(
        "heatmap_env_agent_total.png",
        "Total Reward: Environment x Agent",
        agents,
        experiments,
        lambda agent, exp: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.experiment == exp,
            lambda r: r.total_reward,
        ),
        figsize=(10, len(experiments) * 0.5 + 2),
    )

    heatmap(
        "heatmap_diff_agent.png",
        "Average Reward: Difficulty x Agent",
        agents,
        variants,
        lambda agent, diff: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.difficulty == diff,
            lambda r: r.avg_reward_per_agent,
        ),
        figsize=(10, len(variants) * 0.4 + 2),
    )

    heatmap(
        "heatmap_diff_agent_total.png",
        "Total Reward: Difficulty x Agent",
        agents,
        variants,
        lambda agent, diff: lookup_pair(
            lambda r: r.agent == agent,
            lambda r: r.difficulty == diff,
            lambda r: r.total_reward,
        ),
        figsize=(10, len(variants) * 0.4 + 2),
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
