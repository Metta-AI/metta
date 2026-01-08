# much simpler evaluator for thinky agents.

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import cogames.policy.nim_agents.agents as na
from cogames.cli.utils import suppress_noisy_logs
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
MAX_STEPS = 10000
SEED = 42

# Add/modify your evals here over time
EVALS: List[Tuple[str, str, int]] = [
    # Regular evals
    (
        "energy_starved",
        "buggy",
        NUM_COGS,
    ),  # E is very hard, max E is 256, but agents think its 100.
    ("oxygen_bottleneck", "", NUM_COGS),
    ("collect_resources_classic", "", NUM_COGS),
    ("collect_resources_spread", "", NUM_COGS),
    ("collect_far", "", NUM_COGS),
    ("divide_and_conquer", "", NUM_COGS),
    ("go_together", "", NUM_COGS),
    ("single_use_swarm", "flakey", NUM_COGS),
    # Diagnostic evals
    ("diagnostic_chest_navigation1", "", 1),
    ("diagnostic_chest_navigation2", "", 1),
    ("diagnostic_chest_navigation3", "", 1),
    ("diagnostic_chest_deposit_near", "", 1),
    ("diagnostic_chest_deposit_search", "", 1),
    ("diagnostic_charge_up", "buggy", 1),  # The cog needs to sacrifice itself to make hart.
    ("diagnostic_memory", "", 1),
    ("diagnostic_assemble_seeded_near", "", 1),
    ("diagnostic_assemble_seeded_search", "", 1),
    ("diagnostic_extract_missing_carbon", "", 1),
    ("diagnostic_extract_missing_oxygen", "", 1),
    ("diagnostic_extract_missing_germanium", "", 1),
    ("diagnostic_extract_missing_silicon", "", 1),
    ("diagnostic_unclip_craft", "", 1),
    ("diagnostic_unclip_preseed", "", 1),
    ("diagnostic_agile", "", 1),
    ("diagnostic_radial", "", 1),
    # Hello World evals
    ("distant_resources", "buggy", NUM_COGS),  # Not enough time for such distances.
    ("quadrant_buildings", "buggy", NUM_COGS),  # Not enough charger for such distances.
    ("vibe_check", "", NUM_COGS),
    ("oxygen_bottleneck_easy", "", NUM_COGS),
    ("oxygen_bottleneck_standard", "", NUM_COGS),
    ("oxygen_bottleneck_hard", "buggy", NUM_COGS),  # Not enough charger for such distances.
    ("energy_starved_easy", "", NUM_COGS),
    ("energy_starved_standard", "buggy", NUM_COGS),  # E drain too high.
    ("energy_starved_hard", "buggy", NUM_COGS),  # E drain too high.
    ("unclipping_easy", "n/a", NUM_COGS),
    ("unclipping_standard", "n/a", NUM_COGS),
    ("unclipping_hard", "n/a", NUM_COGS),
    ("distant_resources_easy", "", NUM_COGS),
    ("distant_resources_standard", "flakey", NUM_COGS),  # Not enough time for such distances.
    ("distant_resources_hard", "buggy", NUM_COGS),  # Not enough time for such distances.
    ("quadrant_buildings_easy", "", NUM_COGS),
    ("quadrant_buildings_standard", "buggy", NUM_COGS),  # Not enough charger for such distances.
    ("quadrant_buildings_hard", "buggy", NUM_COGS),  # Not enough charger for such distances.
    ("single_use_swarm_easy", "buggy", NUM_COGS),
    ("single_use_swarm_standard", "buggy", NUM_COGS),  # Not enough time for such distances.
    ("single_use_swarm_hard", "buggy", NUM_COGS),  # E drain too high.
    ("vibe_check_easy", "buggy", NUM_COGS),  # No/invalid recipes available.
    ("vibe_check_standard", "", NUM_COGS),
    ("vibe_check_hard", "flakey", NUM_COGS),  # Not enough time for such distances.
    # Hearts evals
    ("easy_large_hearts", "slow", NUM_COGS),
    ("easy_medium_hearts", "", NUM_COGS),
    ("easy_small_hearts", "flakey", NUM_COGS),
    # Missions from missions.py
    ("harvest", "", NUM_COGS),
    ("repair", "", 2),  # repair uses 2 cogs
    ("hello_world_unclip", "", NUM_COGS),
]


def _load_all_missions() -> Dict[str, Mission]:
    # Minimal loader: merge all known mission sets
    from importlib import import_module

    missions: List[Mission] = []
    for mod_name in (
        "cogames.cogs_vs_clips.evals.eval_missions",
        "cogames.cogs_vs_clips.evals.integrated_evals",
        "cogames.cogs_vs_clips.evals.spanning_evals",
        "cogames.cogs_vs_clips.missions",
    ):
        try:
            mod = import_module(mod_name)
            # missions.py uses MISSIONS, others use EVAL_MISSIONS
            eval_list = getattr(mod, "MISSIONS", getattr(mod, "EVAL_MISSIONS", []))
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
            has_gear = any(v.name == "gear" for v in change_vibe.vibes)
            if not has_gear:
                from mettagrid.config.vibes import VIBE_BY_NAME

                change_vibe.vibes = list(change_vibe.vibes) + [VIBE_BY_NAME["gear"]]
    except Exception:
        pass


def run_eval(experiment_name: str, tag: str, mission_map: Dict[str, Mission], num_cogs: int, seed: int) -> float:
    start = time.perf_counter()
    try:
        if experiment_name not in mission_map:
            print(f"{tag:<6} {experiment_name:<40} {'MISSION NOT FOUND':>6}")
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
            seed=seed,
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
    suppress_noisy_logs()
    na.start_measure()
    mission_map = _load_all_missions()
    print(f"Loaded {len(mission_map)} missions")
    print("tag .. map name ............................... harts/A .. time")
    start = time.perf_counter()
    total_hpa = 0.0
    successful_evals = 0
    num_evals = 0
    for experiment_name, tag, num_cogs in EVALS:
        num_evals += 1
        if tag == "flakey":
            for i in range(10):
                hpa = run_eval(experiment_name, tag, mission_map, num_cogs, SEED + i)
                if hpa > 0:
                    successful_evals += 1
                    total_hpa += hpa
                    break
        else:
            hpa = run_eval(experiment_name, tag, mission_map, num_cogs, SEED)
            if hpa > 0:
                successful_evals += 1
                total_hpa += hpa
    success_rate = successful_evals / num_evals
    elapsed = time.perf_counter() - start
    total_evals = f"{num_evals} evals {success_rate * 100:.1f}% successful"
    hpa = f"{total_hpa:.2f}"
    tm = f"{elapsed:.2f}"
    tag = "total"
    print(f"{tag:<6} {total_evals:<40} {hpa:>6}h {tm:>6}s")
    na.end_measure()


if __name__ == "__main__":
    main()
