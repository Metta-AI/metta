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
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from cogames.cogs_vs_clips.evals import (
    CANONICAL_DIFFICULTY_ORDER,
    DIFFICULTY_LEVELS,
    apply_difficulty,
)
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.missions import make_game
from cogames.policy.coordinating_policy import CoordinatingPolicy
from cogames.policy.scripted_agent import HYPERPARAMETER_PRESETS, Hyperparameters, ScriptedAgentPolicy
from cogames.policy.simple_baseline_policy import SimpleBaselinePolicy
from cogames.policy.unclipping_policy import UnclippingPolicy
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
    agent: str
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
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class AgentEvalConfig:
    """Configuration for running a scripted agent variant."""

    key: str
    label: str
    policy_factory: Callable[[MettaGridEnv], Any]
    cogs_list: List[int]
    difficulties: List[str]
    preset_name: Optional[str] = None


def is_clipping_difficulty(name: str) -> bool:
    """Return True if the difficulty definition introduces clipping."""

    level = DIFFICULTY_LEVELS[name]
    clip_rate = float(level.clip_rate or 0.0)
    clip_target = level.clip_target
    return clip_rate > 0.0 or (clip_target is not None and clip_target.lower() != "none")


def partition_difficulties(names: List[str]) -> tuple[List[str], List[str]]:
    """Split difficulty names into (clipping, non_clipping)."""

    clipping: List[str] = []
    non_clipping: List[str] = []
    for name in names:
        if is_clipping_difficulty(name):
            clipping.append(name)
        else:
            non_clipping.append(name)
    return clipping, non_clipping


ALL_DIFFICULTIES: List[str] = [name for name in CANONICAL_DIFFICULTY_ORDER if name in DIFFICULTY_LEVELS]
CLIPPING_DIFFICULTIES, NON_CLIPPING_DIFFICULTIES = partition_difficulties(ALL_DIFFICULTIES)


AGENT_CONFIGS: Dict[str, AgentEvalConfig] = {
    "simple": AgentEvalConfig(
        key="simple",
        label="SimpleBaseline",
        policy_factory=lambda env: SimpleBaselinePolicy(env),
        cogs_list=[1],
        difficulties=NON_CLIPPING_DIFFICULTIES,
        preset_name="simple_defaults",
    ),
    "unclipping": AgentEvalConfig(
        key="unclipping",
        label="UnclippingAgent",
        policy_factory=lambda env: UnclippingPolicy(env),
        cogs_list=[1],
        difficulties=CLIPPING_DIFFICULTIES,
        preset_name="unclipping_defaults",
    ),
    "coordinating": AgentEvalConfig(
        key="coordinating",
        label="CoordinatingAgent",
        policy_factory=lambda env: CoordinatingPolicy(env),
        cogs_list=[1, 2, 4, 8],
        difficulties=ALL_DIFFICULTIES,
        preset_name="coordinating_defaults",
    ),
}


# =============================================================================
# Feature Toggle Helpers
# =============================================================================


FEATURE_BUNDLES: Dict[str, Dict[str, bool]] = {
    "baseline": {},
    "no_coordination": {
        "enable_assembly_coordination": False,
        "enable_target_reservation": False,
    },
    "no_reservation": {
        "enable_target_reservation": False,
    },
    "lightweight_nav": {
        "enable_navigation_cache": False,
        "enable_target_reservation": False,
        "enable_visit_scoring": False,
    },
    "minimal": {
        "enable_assembly_coordination": False,
        "enable_target_reservation": False,
        "enable_resource_focus_limits": False,
        "enable_navigation_cache": False,
        "enable_probe_module": False,
        "enable_visit_scoring": False,
        "use_probes": False,
    },
}


def clone_hyperparameters(preset_name: str) -> Hyperparameters:
    base = HYPERPARAMETER_PRESETS[preset_name]
    hp = replace(base)
    if base.resource_focus_limits is not None:
        hp.resource_focus_limits = dict(base.resource_focus_limits)
    return hp


def build_feature_overrides(bundle: Optional[str], explicit: Dict[str, bool]) -> Dict[str, bool]:
    overrides: Dict[str, bool] = {}
    if bundle:
        bundle_overrides = FEATURE_BUNDLES.get(bundle)
        if bundle_overrides is None:
            raise ValueError(f"Unknown feature bundle: {bundle}")
        overrides.update(bundle_overrides)
    overrides.update(explicit)
    return overrides


def apply_feature_overrides(hp: Hyperparameters, overrides: Dict[str, bool]) -> Dict[str, bool]:
    resolved: Dict[str, bool] = {}
    for attr, value in overrides.items():
        if hasattr(hp, attr):
            setattr(hp, attr, value)
            resolved[attr] = value
    if not getattr(hp, "enable_probe_module", True):
        hp.use_probes = False
        resolved["enable_probe_module"] = False
        resolved["use_probes"] = False
    if not getattr(hp, "enable_resource_focus_limits", True):
        hp.resource_focus_limits = None
        resolved["enable_resource_focus_limits"] = False
    return resolved


def collect_feature_overrides(args: Any) -> Dict[str, bool]:
    explicit: Dict[str, bool] = {}
    if getattr(args, "disable_probes", False):
        explicit["enable_probe_module"] = False
        explicit["use_probes"] = False
    if getattr(args, "disable_visit_scoring", False):
        explicit["enable_visit_scoring"] = False
    if getattr(args, "disable_target_reservation", False):
        explicit["enable_target_reservation"] = False
    if getattr(args, "disable_resource_focus", False):
        explicit["enable_resource_focus_limits"] = False
    if getattr(args, "disable_assembly_coordination", False):
        explicit["enable_assembly_coordination"] = False
    if getattr(args, "disable_nav_cache", False):
        explicit["enable_navigation_cache"] = False

    bundle = getattr(args, "feature_bundle", None)
    if bundle == "baseline":
        bundle = None

    return build_feature_overrides(bundle, explicit)


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
                agent="ScriptedAgentPolicy",
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
    agent_config: AgentEvalConfig,
    experiments: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    cogs_list: Optional[List[int]] = None,
    max_steps: int = 1000,
    feature_overrides: Optional[Dict[str, bool]] = None,
) -> List[EvalResult]:
    """Run comprehensive evaluation for a specific scripted agent variant."""

    print("\n" + "=" * 80)
    print("FULL EVALUATION SUITE")
    print("=" * 80)

    # Defaults / filtering
    if experiments is None:
        experiments = [name for name, _ in EXPERIMENTS]

    available_difficulties = agent_config.difficulties
    if not available_difficulties:
        logger.warning("Agent %s has no configured difficulties", agent_config.label)
        return []

    if difficulties is None:
        selected_difficulties = available_difficulties
    else:
        selected_difficulties = [d for d in difficulties if d in available_difficulties]
        dropped = sorted(set(difficulties) - set(selected_difficulties))
        if dropped:
            logger.warning("Dropping unsupported difficulties %s for agent %s", dropped, agent_config.label)
        if not selected_difficulties:
            logger.warning("No valid difficulties left for agent %s", agent_config.label)
            return []

    if cogs_list is None:
        selected_cogs = agent_config.cogs_list
    else:
        if set(cogs_list) != set(agent_config.cogs_list):
            logger.warning(
                "Ignoring --cogs override %s for agent %s; enforcing %s",
                cogs_list,
                agent_config.label,
                agent_config.cogs_list,
            )
        selected_cogs = agent_config.cogs_list

    clip_rates_summary = sorted({float(DIFFICULTY_LEVELS[d].clip_rate or 0.0) for d in selected_difficulties})
    total_tests = len(experiments) * len(selected_difficulties) * len(selected_cogs)

    print(f"\nAgent Variant: {agent_config.label} ({agent_config.key})")
    print(f"Experiments: {len(experiments)}")
    print(f"Difficulties: {len(selected_difficulties)}")
    print(f"Agent counts: {selected_cogs}")
    print(f"Hyperparams preset: {agent_config.preset_name or 'n/a'}")
    print(f"Difficulty clip rates: {clip_rates_summary}")
    print(f"Total tests: {total_tests}\n")

    results: List[EvalResult] = []
    success_count = 0
    test_count = 0

    overrides = feature_overrides or {}

    for exp_name in experiments:
        if exp_name not in EXPERIMENT_MAP:
            logger.error(f"Unknown experiment: {exp_name}")
            continue

        mission_class = EXPERIMENT_MAP[exp_name]

        for difficulty_name in selected_difficulties:
            if difficulty_name not in DIFFICULTY_LEVELS:
                logger.error(f"Unknown difficulty: {difficulty_name}")
                continue

            difficulty_level = DIFFICULTY_LEVELS[difficulty_name]
            clip_mode = "difficulty" if is_clipping_difficulty(difficulty_name) else "none"
            clip_rate = float(difficulty_level.clip_rate or 0.0)

            for num_cogs in selected_cogs:
                test_name = f"{agent_config.key}_{exp_name}_{difficulty_name}_cogs{num_cogs}"
                print(f"\n[{test_count}/{total_tests}] {test_name}")

                try:
                    mission = mission_class()
                    apply_difficulty(mission, difficulty_level)

                    map_builder = mission.site.map_builder if mission.site else None
                    mission = mission.instantiate(map_builder, num_cogs=num_cogs)
                    env_config = mission.make_env()
                    env_config.game.max_steps = max_steps

                    env = MettaGridEnv(env_config)
                    policy = agent_config.policy_factory(env)

                    obs, info = env.reset()
                    policy.reset(obs, info)
                    policy_impl = getattr(policy, "_impl", None)
                    if policy_impl is not None and hasattr(policy_impl, "reset_metrics"):
                        policy_impl.reset_metrics()
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
                    if agent_state is None and policy_impl is not None and hasattr(policy_impl, "_agent_states"):
                        agent_state = policy_impl._agent_states.get(0)  # type: ignore[attr-defined]

                    hearts_assembled = 0
                    final_energy = 0
                    if agent_state is not None:
                        hearts_assembled = int(
                            getattr(
                                agent_state,
                                "hearts_assembled",
                                getattr(agent_state, "hearts", 0),
                            )
                        )
                        final_energy = int(getattr(agent_state, "energy", 0))

                    steps_taken = last_step + 1
                    success = total_reward > 0

                    metrics: Optional[Dict[str, Any]] = None
                    if policy_impl is not None and hasattr(policy_impl, "export_metrics"):
                        metrics = policy_impl.export_metrics()  # type: ignore[assignment]
                        if overrides:
                            metrics = dict(metrics)
                            metrics.setdefault("feature_overrides", dict(overrides))

                    env.close()

                    result = EvalResult(
                        suite="full",
                        agent=agent_config.label,
                        test_name=test_name,
                        num_cogs=num_cogs,
                        preset=agent_config.preset_name,
                        difficulty=difficulty_name,
                        clip_mode=clip_mode,
                        clip_rate=clip_rate,
                        total_reward=float(total_reward),
                        hearts_assembled=int(hearts_assembled),
                        steps_taken=int(steps_taken),
                        max_steps=max_steps,
                        final_energy=int(final_energy),
                        success=success,
                        metrics=metrics,
                    )
                    results.append(result)
                    test_count += 1
                    if success:
                        success_count += 1

                    print(f"  Reward: {total_reward:.1f}")
                    print(f"  Steps: {steps_taken}/{max_steps}")
                    print(f"  {'✅ SUCCESS' if success else '❌ FAILED'}")
                    if metrics:
                        print(
                            "  Metrics: nav_calls={nav} cache_hits={cache} "
                            "astar={astar} assembly_latencies={latencies}".format(
                                nav=metrics.get("navigation_calls"),
                                cache=metrics.get("navigation_cache_hits"),
                                astar=metrics.get("navigation_astar_calls"),
                                latencies=metrics.get("assembly_latencies"),
                            )
                        )

                except Exception as e:
                    logger.error(f"Error in {test_name}: {e}")
                    test_count += 1

    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {success_count}/{test_count} successful")
    print(f"{'=' * 80}")

    return results


# =============================================================================
# Trace utilities
# =============================================================================


def _make_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {k: _make_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_serializable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _summarize_extractors(policy_impl: Any, step_count: int) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    resources = ["germanium", "silicon", "carbon", "oxygen", "charger"]
    for resource in resources:
        extractors = policy_impl.extractor_memory.get_by_type(resource)
        known = len(extractors)
        if known == 0:
            summary[resource] = {"known": 0, "available": 0, "clipped": 0, "depleted": 0}
            continue

        available = 0
        clipped = 0
        depleted = 0
        for ex in extractors:
            if ex.is_clipped:
                clipped += 1
            if ex.permanently_depleted or ex.is_depleted():
                depleted += 1
            if not ex.is_clipped and not ex.is_depleted() and policy_impl.cooldown_remaining(ex, step_count) <= 0:
                available += 1
        summary[resource] = {
            "known": known,
            "available": available,
            "clipped": clipped,
            "depleted": depleted,
        }
    return summary


def run_trace_episode(
    experiment: str,
    difficulty_name: str,
    preset_name: str,
    num_cogs: int,
    max_steps: int,
    seed: int = 42,
    feature_overrides: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    if experiment not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown experiment: {experiment}")
    if difficulty_name not in DIFFICULTY_LEVELS:
        raise ValueError(f"Unknown difficulty: {difficulty_name}")
    if preset_name not in HYPERPARAMETER_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    mission_class = EXPERIMENT_MAP[experiment]
    mission = mission_class()
    difficulty = DIFFICULTY_LEVELS[difficulty_name]
    apply_difficulty(mission, difficulty)

    map_builder = mission.site.map_builder if mission.site else None
    mission = mission.instantiate(map_builder, num_cogs=num_cogs)
    env_config = mission.make_env()
    env_config.game.max_steps = max_steps

    env = MettaGridEnv(env_config)
    hyperparams_obj = clone_hyperparameters(preset_name)
    resolved_flags = apply_feature_overrides(hyperparams_obj, feature_overrides or {})
    policy = ScriptedAgentPolicy(env, hyperparams=hyperparams_obj)

    obs, info = env.reset(seed=seed)
    policy.reset(obs, info)
    policy_impl = policy._impl  # type: ignore[attr-defined]
    if policy_impl is not None:
        policy_impl.reset_metrics()
    agents = [policy.agent_policy(i) for i in range(num_cogs)]

    trace: Dict[str, Any] = {
        "experiment": experiment,
        "difficulty": difficulty_name,
        "preset": preset_name,
        "num_cogs": num_cogs,
        "max_steps": max_steps,
        "seed": seed,
        "steps": [],
    }

    cumulative_reward = 0.0

    for step in range(max_steps):
        actions = np.zeros(num_cogs, dtype=dtype_actions)
        step_record: Dict[str, Any] = {
            "step": step,
            "agents": [],
        }

        # Capture per-agent decision context before env.step()
        for agent_id in range(num_cogs):
            current_obs = obs[agent_id]
            obs_summary = {
                "shape": list(current_obs.shape),
                "nonzero": int(np.count_nonzero(current_obs)),
                "max": int(current_obs.max()) if current_obs.size else 0,
                "min": int(current_obs.min()) if current_obs.size else 0,
            }

            action = int(agents[agent_id].step(current_obs))
            actions[agent_id] = action

            agent_state = getattr(agents[agent_id], "_state", None)
            if agent_state is None:
                step_record["agents"].append(
                    {
                        "agent_id": agent_id,
                        "action_index": action,
                        "action_name": env.action_names[action] if action < len(env.action_names) else str(action),
                        "observation_summary": obs_summary,
                    }
                )
                continue

            inventory = {
                "carbon": int(agent_state.carbon),
                "oxygen": int(agent_state.oxygen),
                "germanium": int(agent_state.germanium),
                "silicon": int(agent_state.silicon),
                "energy": int(agent_state.energy),
                "heart": int(agent_state.heart),
            }

            deficits = {k: int(v) for k, v in policy_impl._resource_deficits(agent_state).items()}

            agent_entry = {
                "agent_id": agent_id,
                "step_count": int(agent_state.step_count),
                "phase": agent_state.current_phase.name if agent_state.current_phase else None,
                "position": [int(agent_state.agent_row), int(agent_state.agent_col)],
                "action_index": action,
                "action_name": env.action_names[action] if action < len(env.action_names) else str(action),
                "inventory": inventory,
                "deficits": deficits,
                "active_target": agent_state.active_resource_target,
                "explore_goal": agent_state.explore_goal,
                "wait_target": list(agent_state.wait_target) if agent_state.wait_target else None,
                "waiting_since_step": int(agent_state.waiting_since_step),
                "recharge_total_gained": int(agent_state.recharge_total_gained),
                "recharge_ticks_without_gain": int(agent_state.recharge_ticks_without_gain),
                "known_extractors": {
                    res: len(policy_impl.extractor_memory.get_by_type(res))
                    for res in ["germanium", "silicon", "carbon", "oxygen", "charger"]
                },
                "observation_summary": obs_summary,
            }
            step_record["agents"].append(agent_entry)

        obs, rewards, dones, truncated, info = env.step(actions)
        cumulative_reward += float(rewards.sum())

        step_record["reward_sum"] = float(rewards.sum())
        step_record["rewards"] = rewards.astype(float).tolist()
        step_record["dones"] = [bool(x) for x in dones]
        step_record["truncated"] = [bool(x) for x in truncated]
        step_counts = [agent_entry.get("step_count", 0) for agent_entry in step_record["agents"]]
        step_record["extractors"] = _summarize_extractors(policy_impl, max(step_counts) if step_counts else 0)
        step_record["env_info"] = _make_serializable(info)

        trace["steps"].append(step_record)

        if all(dones) or all(truncated):
            break

    metrics: Optional[Dict[str, Any]] = None
    if policy_impl is not None:
        metrics = policy_impl.export_metrics()
        if resolved_flags:
            metrics = dict(metrics)
            metrics.setdefault("feature_overrides", dict(resolved_flags))

    env.close()

    agent_state = getattr(agents[0], "_state", None)
    trace["final_energy"] = int(getattr(agent_state, "energy", 0)) if agent_state else 0
    trace["hearts_assembled"] = int(getattr(agent_state, "hearts_assembled", 0)) if agent_state else 0
    trace["total_reward"] = cumulative_reward
    trace["metrics"] = metrics

    return trace


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

    # Group by agent variant
    by_agent: Dict[str, List[EvalResult]] = {}
    for r in results:
        by_agent.setdefault(r.agent, []).append(r)

    print("\n## Success Rate by Agent")
    for agent_label in sorted(by_agent.keys()):
        agent_results = by_agent[agent_label]
        agent_success = sum(1 for r in agent_results if r.success)
        print(
            f"  {agent_label}: {agent_success}/{len(agent_results)} ({100 * agent_success / len(agent_results):.1f}%)"
        )

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
    full_parser.add_argument("--cogs", nargs="*", type=int, default=None, help="Agent counts (default: 1 2 4 8)")
    full_parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    full_parser.add_argument(
        "--agent",
        choices=[*AGENT_CONFIGS.keys(), "all"],
        default="all",
        help="Scripted agent variant to evaluate (default: all)",
    )
    full_parser.add_argument(
        "--feature-bundle",
        choices=list(FEATURE_BUNDLES.keys()),
        default="baseline",
        help="Preset feature toggle bundle (default: baseline)",
    )
    full_parser.add_argument("--disable-probes", action="store_true", help="Disable probe exploration module")
    full_parser.add_argument(
        "--disable-visit-scoring",
        action="store_true",
        help="Disable visit-count based frontier scoring",
    )
    full_parser.add_argument(
        "--disable-target-reservation",
        action="store_true",
        help="Disable shared target reservation/assignment",
    )
    full_parser.add_argument(
        "--disable-resource-focus",
        action="store_true",
        help="Disable resource focus limits (all agents can pursue any resource)",
    )
    full_parser.add_argument(
        "--disable-assembly-coordination",
        action="store_true",
        help="Disable assembly coordination and slot management",
    )
    full_parser.add_argument(
        "--disable-nav-cache",
        action="store_true",
        help="Disable navigation path caching and reuse",
    )

    # Trace single episode
    trace_parser = subparsers.add_parser("trace", help="Record detailed trajectory for a single configuration")
    trace_parser.add_argument(
        "--experiment", required=True, choices=list(EXPERIMENT_MAP.keys()), help="Experiment/mission name"
    )
    trace_parser.add_argument(
        "--difficulty", required=True, choices=list(DIFFICULTY_LEVELS.keys()), help="Difficulty variant"
    )
    trace_parser.add_argument(
        "--preset", required=True, choices=list(HYPERPARAMETER_PRESETS.keys()), help="Hyperparameter preset"
    )
    trace_parser.add_argument("--cogs", type=int, default=2, help="Number of agents (default: 2)")
    trace_parser.add_argument("--steps", type=int, default=600, help="Max steps for the trace (default: 600)")
    trace_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    trace_parser.add_argument(
        "--feature-bundle",
        choices=list(FEATURE_BUNDLES.keys()),
        default="baseline",
        help="Preset feature toggle bundle (default: baseline)",
    )
    trace_parser.add_argument("--disable-probes", action="store_true", help="Disable probe exploration module")
    trace_parser.add_argument(
        "--disable-visit-scoring",
        action="store_true",
        help="Disable visit-count based frontier scoring",
    )
    trace_parser.add_argument(
        "--disable-target-reservation",
        action="store_true",
        help="Disable shared target reservation/assignment",
    )
    trace_parser.add_argument(
        "--disable-resource-focus",
        action="store_true",
        help="Disable resource focus limits",
    )
    trace_parser.add_argument(
        "--disable-assembly-coordination",
        action="store_true",
        help="Disable assembly coordination",
    )
    trace_parser.add_argument(
        "--disable-nav-cache",
        action="store_true",
        help="Disable navigation path caching",
    )

    # All suites
    subparsers.add_parser("all", help="Run all evaluation suites")

    args = parser.parse_args()

    if args.suite is None:
        parser.print_help()
        return

    if args.suite == "trace":
        if not args.output:
            parser.error("Trace mode requires --output to specify the trace file path")
        feature_overrides = collect_feature_overrides(args)
        trace_data = run_trace_episode(
            experiment=args.experiment,
            difficulty_name=args.difficulty,
            preset_name=args.preset,
            num_cogs=args.cogs,
            max_steps=args.steps,
            seed=args.seed,
            feature_overrides=feature_overrides,
        )
        with open(args.output, "w") as f:
            json.dump(_make_serializable(trace_data), f, indent=2)
        print(f"Trace saved to: {args.output}")
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
        if args.suite == "full":
            agent_selection = args.agent
            experiments = args.experiments
            difficulties = args.difficulties
            steps = args.steps
            feature_overrides = collect_feature_overrides(args)
        else:
            agent_selection = "all"
            experiments = None
            difficulties = None
            steps = 1000
            feature_overrides = None

        if agent_selection == "all":
            selected_configs = list(AGENT_CONFIGS.values())
        else:
            selected_configs = [AGENT_CONFIGS[agent_selection]]

        full_results: List[EvalResult] = []
        for config in selected_configs:
            config_results = run_full_evaluation_suite(
                agent_config=config,
                experiments=experiments,
                difficulties=difficulties,
                cogs_list=args.cogs if args.suite == "full" else None,
                max_steps=steps,
                feature_overrides=feature_overrides,
            )
            full_results.extend(config_results)

        all_results.extend(full_results)
        print_summary(full_results, "Full Evaluation")

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
