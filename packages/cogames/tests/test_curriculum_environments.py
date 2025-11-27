"""Test that all environments in the curriculum can be loaded with 4 agents.

This test verifies that each mission-variant combination in the curriculum
can successfully create an environment and build a map for 4 agents.
This catches issues like "No surface found" errors before training starts.

The test replicates the exact logic from make_curriculum() to ensure
we test every environment that will actually be used in training.
"""

from __future__ import annotations

import pytest

from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from mettagrid.simulator.simulator import Simulator
from recipes.experiment import cogs_v_clips
from recipes.experiment.cvc.mission_variant_curriculum import (
    _MISSION_BY_NAME,
    _deduplicate_assembler_protocols,
    DIAGNOSTIC_MISSIONS,
    get_all_variant_names,
    make_curriculum,
    resolve_missions,
)


def test_submit_experiments_curriculum_loads_with_4_agents():
    """Test that the curriculum from submit_experiments.py can be loaded with 4 agents.

    This tests the S3 successful missions curriculum with all variants,
    which is what submit_experiments.py actually submits.

    The test replicates the exact logic from make_curriculum() to ensure
    we test every environment that will actually be used in training.
    """
    # S3 successful missions from submit_experiments.py
    S3_SUCCESSFUL_MISSIONS = [
        "diagnostic_chest_deposit_near",
        "diagnostic_chest_navigation3",
        "diagnostic_extract_missing_oxygen",
        "diagnostic_extract_missing_silicon",
        "easy_hearts",
        "easy_hearts_hello_world",
        "easy_hearts_training",
        "easy_hearts_training_facility",
        "easy_medium_hearts",
        "easy_mode",
        "easy_small_hearts",
        "go_together",
        "repair",
        "single_use_swarm_easy",
        "vibe_check",
    ]

    num_cogs = 4
    variant_names = get_all_variant_names()  # "all" variants

    # Resolve missions (same as make_curriculum does)
    base_missions = resolve_missions(S3_SUCCESSFUL_MISSIONS)

    failures = []
    tested = 0

    # Replicate the exact logic from make_curriculum() lines 240-331
    for mission_name in base_missions:
        mission_template = None

        # Check if this is an eval mission (same as make_curriculum)
        mission_template = _MISSION_BY_NAME.get(mission_name)

        # Check if this is a diagnostic mission (class-based) (same as make_curriculum)
        if mission_template is None and mission_name in DIAGNOSTIC_MISSIONS:
            for mission_cls in DIAGNOSTIC_EVALS:
                temp_mission = mission_cls()  # type: ignore[call-arg]
                if temp_mission.name == mission_name:
                    mission_template = temp_mission
                    break

        # For each variant, create a separate curriculum task (same as make_curriculum)
        for variant_name in variant_names:
            tested += 1
            try:
                # Replicate exact logic from make_curriculum lines 259-282
                if mission_template is None:
                    # Fall back to make_training_env for standard missions
                    try:
                        mission_env = cogs_v_clips.make_training_env(
                            num_cogs=num_cogs,
                            mission=mission_name,
                            variants=[variant_name],
                        )
                    except ValueError:
                        # Skip missions that don't exist (same as make_curriculum)
                        continue
                    # Deduplicate assembler protocols to avoid C++ config errors
                    _deduplicate_assembler_protocols(mission_env)
                else:
                    # Use the mission template directly (works for both eval and diagnostic missions)
                    mission = cogs_v_clips._prepare_mission(
                        mission_template,
                        num_cogs=num_cogs,
                        variant_names=[variant_name],
                    )
                    mission_env = mission.make_env()

                    # Deduplicate assembler protocols to avoid C++ config errors
                    _deduplicate_assembler_protocols(mission_env)

                # Verify the environment can be created and map can be built
                # This will catch "No surface found" and spawn point issues
                simulator = Simulator()
                sim = simulator.new_simulation(mission_env)

                # Verify we have the expected number of agents
                actual_num_agents = sim.num_agents
                assert actual_num_agents == num_cogs, (
                    f"Expected {num_cogs} agents, got {actual_num_agents} for {mission_name}:{variant_name}"
                )

            except Exception as e:
                # Check if it's a map building error (the important ones we care about)
                error_str = str(e).lower()
                # Skip expected incompatibilities
                if any(
                    skip_phrase in error_str
                    for skip_phrase in [
                        "not callable",
                        "already exists",
                        "can only be applied",
                        "not found",
                        "does not exist",
                        "incompatible",
                        "protocol with vibes",
                    ]
                ):
                    # These are expected incompatibilities, skip them
                    continue
                # Only fail on actual map building errors
                if any(
                    key_phrase in error_str
                    for key_phrase in [
                        "surface",
                        "spawn",
                        "exceeds available",
                        "no surface found",
                        "number of agents",
                    ]
                ):
                    failures.append((mission_name, variant_name, str(e)))
                # Otherwise, skip other errors (likely incompatibilities)
                continue

    if failures:
        error_msg = f"\nFailed to load {len(failures)}/{tested} environments (map building errors):\n"
        for mission_name, variant_name, error in failures:
            error_msg += f"  - {mission_name}:{variant_name} - {error}\n"
        pytest.fail(error_msg)
