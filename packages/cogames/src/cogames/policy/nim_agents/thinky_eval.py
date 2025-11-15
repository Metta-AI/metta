# much simpler evaluator for thinky agents.

from __future__ import annotations

import logging
import time
import warnings
from typing import Dict, List, Tuple

from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.mission import Mission, NumCogsVariant
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

# Agent to evaluate
AGENT_PATH = "cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy"

# Defaults (keep simple)
NUM_COGS = 4
MAX_STEPS = 1000
SEED = 42

# Add/modify your evals here over time
EVALS: List[Tuple[str, str, int]] = [
    # Regular evals
    ("energy_starved", "n/a", NUM_COGS),
    ("oxygen_bottleneck", "n/a", NUM_COGS),
    ("extractor_hub_30", "n/a", NUM_COGS),
    ("extractor_hub_50", "n/a", NUM_COGS),
    ("extractor_hub_70", "n/a", NUM_COGS),
    ("extractor_hub_80", "n/a", NUM_COGS),
    ("extractor_hub_100", "n/a", NUM_COGS),
    ("collect_resources_classic", "n/a", NUM_COGS),
    ("collect_resources_spread", "n/a", NUM_COGS),
    ("collect_far", "slow", NUM_COGS),
    ("divide_and_conquer", "n/a", NUM_COGS),
    ("go_together", "n/a", NUM_COGS),
    ("single_use_swarm", "n/a", NUM_COGS),
    # Diagnostic evals
    ("diagnostic_chest_navigation1", "slow", 1),
    ("diagnostic_chest_navigation2", "slow", 1),
    ("diagnostic_chest_navigation3", "slow", 1),
    ("diagnostic_chest_deposit_near", "slow", 1),
    ("diagnostic_chest_deposit_search", "slow", 1),
    ("diagnostic_charge_up", "n/a", 1),
    ("diagnostic_memory", "slow", 1),
    ("diagnostic_assemble_seeded_near", "n/a", 1),
    ("diagnostic_assemble_seeded_search", "n/a", 1),
    ("diagnostic_extract_missing_carbon", "n/a", 1),
    ("diagnostic_extract_missing_oxygen", "n/a", 1),
    ("diagnostic_extract_missing_germanium", "n/a", 1),
    ("diagnostic_extract_missing_silicon", "n/a", 1),
    ("diagnostic_unclip_craft", "n/a", 1),
    ("diagnostic_unclip_preseed", "n/a", 1),
    ("diagnostic_agile", "n/a", 1),
    ("diagnostic_radial", "n/a", 1),
    # Hello World evals
    ("distant_resources", "n/a", NUM_COGS),
    ("quadrant_buildings", "n/a", NUM_COGS),
    ("single_use_swarm", "n/a", NUM_COGS),
    ("vibe_check", "n/a", NUM_COGS),
    ("easy_hearts", "n/a", NUM_COGS),
    ("oxygen_bottleneck_easy", "n/a", NUM_COGS),
    ("oxygen_bottleneck_standard", "n/a", NUM_COGS),
    ("oxygen_bottleneck_hard", "n/a", NUM_COGS),
    ("energy_starved_easy", "n/a", NUM_COGS),
    ("energy_starved_standard", "n/a", NUM_COGS),
    ("energy_starved_hard", "n/a", NUM_COGS),
    ("unclipping_easy", "n/a", NUM_COGS),
    ("unclipping_standard", "n/a", NUM_COGS),
    ("unclipping_hard", "n/a", NUM_COGS),
    ("distant_resources_easy", "n/a", NUM_COGS),
    ("distant_resources_standard", "n/a", NUM_COGS),
    ("distant_resources_hard", "n/a", NUM_COGS),
    ("quadrant_buildings_easy", "n/a", NUM_COGS),
    ("quadrant_buildings_standard", "n/a", NUM_COGS),
    ("quadrant_buildings_hard", "n/a", NUM_COGS),
    ("single_use_swarm_easy", "n/a", NUM_COGS),
    ("single_use_swarm_standard", "n/a", NUM_COGS),
    ("single_use_swarm_hard", "n/a", NUM_COGS),
    ("vibe_check_easy", "n/a", NUM_COGS),
    ("vibe_check_standard", "n/a", NUM_COGS),
    ("vibe_check_hard", "n/a", NUM_COGS),
    # Hearts evals
    ("easy_large_hearts", "slow", NUM_COGS),
    ("easy_medium_hearts", "n/a", NUM_COGS),
    ("easy_small_hearts", "n/a", NUM_COGS),
    ("easy_hearts_training", "n/a", NUM_COGS),
]


def fix_logger() -> None:
    # Silence torch elastic redirect note and similar warnings
    for name in (
        "torch.distributed.elastic.multiprocessing.redirects",
        "torch.distributed.elastic",
        "torch.distributed",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*Redirects are currently not supported in Windows or MacOs.*",
    )


def _load_all_missions() -> Dict[str, Mission]:
    # Minimal loader: merge all known mission sets
    from importlib import import_module

    missions: List[Mission] = []
    for mod_name in (
        "cogames.cogs_vs_clips.evals.eval_missions",
        "cogames.cogs_vs_clips.evals.integrated_evals",
        "cogames.cogs_vs_clips.evals.spanning_evals",
    ):
        try:
            mod = import_module(mod_name)
            eval_list = getattr(mod, "EVAL_MISSIONS", [])
            missions.extend(eval_list)
        except Exception:
            pass

    # Diagnostic evals are a list of classes; instantiate them
    try:
        missions.extend([cls() for cls in DIAGNOSTIC_EVALS])  # type: ignore[misc]
    except Exception:
        pass

    # Build name -> mission instance map
    mission_map: Dict[str, Mission] = {}
    for m in missions:
        # Items in EVAL_MISSIONS may be classes or instances; normalize to instances
        try:
            mission: Mission = m() if isinstance(m, type) else m  # type: ignore[call-arg,assignment]
        except Exception:
            continue
        mission_map[mission.name] = mission
    return mission_map


def _ensure_vibe_supports_gear(env_cfg) -> None:
    # Keep minimal and silent if anything fails
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
        pass


def run_eval(experiment_name: str, tag: str, mission_map: Dict[str, Mission], num_cogs: int) -> None:
    start = time.perf_counter()
    try:
        if experiment_name not in mission_map:
            return 0.0

        base_mission = mission_map[experiment_name]
        mission = base_mission.with_variants([NumCogsVariant(num_cogs=num_cogs)])

        env_cfg = mission.make_env()
        _ensure_vibe_supports_gear(env_cfg)
        env_cfg.game.max_steps = MAX_STEPS

        # Create policy and rollout
        pei = PolicyEnvInterface.from_mg_cfg(env_cfg)
        policy = initialize_or_load_policy(
            pei,
            PolicySpec(class_path=AGENT_PATH, data_path=None),
        )
        agent_policies = [policy.agent_policy(i) for i in range(num_cogs)]

        rollout = Rollout(
            env_cfg,
            agent_policies,
            render_mode="none",
            seed=SEED,
            pass_sim_to_policies=True,
        )
        rollout.run_until_done()

        total_reward = float(sum(rollout._sim.episode_rewards))
        hearts_per_agent = total_reward / max(1, num_cogs)
        elapsed = time.perf_counter() - start

        # One simple line per eval
        hpa = f"{hearts_per_agent:.2f}"
        tm = f"{elapsed:.2f}"
        print(f"{tag:<6} {experiment_name:<40} {hpa:>6}h {tm:>6}s")
        return hearts_per_agent
    except Exception as e:
        elapsed = time.perf_counter() - start
        error_message = str(e)
        print(f"{tag:<6} {experiment_name:<40} {error_message}")
        return 0.0


def main() -> None:
    fix_logger()
    mission_map = _load_all_missions()
    print("tag .. map name ............................... harts/A .. time")
    start = time.perf_counter()
    total_hpa = 0.0
    successful_evals = 0
    for experiment_name, tag, num_cogs in EVALS:
        hpa = run_eval(experiment_name, tag, mission_map, num_cogs)
        if hpa > 0:
            successful_evals += 1
        total_hpa += hpa
    success_rate = successful_evals / len(EVALS)
    elapsed = time.perf_counter() - start
    total_evals = f"{len(EVALS)} evals {success_rate * 100:.1f}% successful"
    hpa = f"{total_hpa:.2f}"
    tm = f"{elapsed:.2f}"
    tag = "total"
    print(f"{tag:<6} {total_evals:<40} {hpa:>6}h {tm:>6}s")


if __name__ == "__main__":
    main()
