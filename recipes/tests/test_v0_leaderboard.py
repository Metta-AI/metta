from __future__ import annotations

from metta.app_backend.leaderboard_constants import (
    LEADERBOARD_CANDIDATE_COUNT_KEY,
    LEADERBOARD_LADYBUG_COUNT_KEY,
    LEADERBOARD_SCENARIO_KEY,
    LEADERBOARD_SCENARIO_KIND_KEY,
    LEADERBOARD_SIM_NAME_EPISODE_KEY,
    LEADERBOARD_THINKY_COUNT_KEY,
)
from recipes.experiment import v0_leaderboard as leaderboard


def test_generate_scenarios_covers_all_partitions() -> None:
    """Verify _generate_scenarios produces the expected 15 partitions for 4 cogs."""
    scenarios = leaderboard._generate_scenarios(leaderboard.NUM_COGS)

    assert len(scenarios) == 15  # C(4+3-1, 3-1) = 15 ways to distribute 4 agents among 3 types

    for scenario in scenarios:
        assert scenario.candidate_count + scenario.thinky_count + scenario.ladybug_count == leaderboard.NUM_COGS, (
            "each scenario must allocate all cogs"
        )

    expected_special = {
        ("machina1-c4-t0-l0", "candidate_self_play"),
        ("machina1-c0-t4-l0", "thinky_self_play"),
        ("machina1-c0-t0-l4", "ladybug_self_play"),
        ("machina1-c0-t2-l2", "thinky_ladybug_dual"),
    }
    actual = {(scenario.sim_name, scenario.scenario_kind) for scenario in scenarios}
    assert expected_special.issubset(actual)


def test_simulations_builds_configs_for_all_scenarios() -> None:
    """Verify simulations() produces a config per scenario with correct structure."""
    configs = leaderboard.simulations(num_episodes=1, map_seed=42)
    scenarios = leaderboard._generate_scenarios(leaderboard.NUM_COGS)

    assert len(configs) == len(scenarios), "simulations() should produce one config per scenario"

    for config, scenario in zip(configs, scenarios, strict=True):
        # Check proportions match scenario counts
        assert config.proportions == scenario.proportions, (
            f"proportions mismatch for {scenario.sim_name}: {config.proportions} != {scenario.proportions}"
        )

        # Check tags contain required keys with correct values
        tags = config.episode_tags
        assert tags[LEADERBOARD_CANDIDATE_COUNT_KEY] == str(scenario.candidate_count)
        assert tags[LEADERBOARD_THINKY_COUNT_KEY] == str(scenario.thinky_count)
        assert tags[LEADERBOARD_LADYBUG_COUNT_KEY] == str(scenario.ladybug_count)
        assert tags[LEADERBOARD_SCENARIO_KEY] == scenario.sim_name
        assert tags[LEADERBOARD_SCENARIO_KIND_KEY] == scenario.scenario_kind
        assert tags[LEADERBOARD_SIM_NAME_EPISODE_KEY] == scenario.sim_name


def test_simulations_minimal_excludes_replacement_scenarios() -> None:
    """Verify minimal=True only includes candidate scenarios (candidate_count > 0)."""
    full_configs = leaderboard.simulations(num_episodes=1, map_seed=42, minimal=False)
    minimal_configs = leaderboard.simulations(num_episodes=1, map_seed=42, minimal=True)

    # minimal should have fewer configs (excludes 5 replacement scenarios where c=0)
    assert len(minimal_configs) == len(full_configs) - 5

    # All minimal configs should have candidate_count > 0
    for config in minimal_configs:
        candidate_count = int(config.episode_tags[LEADERBOARD_CANDIDATE_COUNT_KEY])
        assert candidate_count > 0, f"minimal mode should exclude replacement scenarios, got {candidate_count}"


def test_episode_tags_roundtrip() -> None:
    """Ensure tags can be parsed back to original counts (critical for VOR computation)."""
    scenario = leaderboard.LeaderboardScenario(
        candidate_count=2, thinky_count=1, ladybug_count=1, scenario_kind="candidate_mix"
    )
    tags = scenario.episode_tags()

    # Simulate what VOR code does when reading tags from episode metadata
    parsed_candidate = int(tags[LEADERBOARD_CANDIDATE_COUNT_KEY])
    parsed_thinky = int(tags[LEADERBOARD_THINKY_COUNT_KEY])
    parsed_ladybug = int(tags[LEADERBOARD_LADYBUG_COUNT_KEY])
    parsed_scenario_name = tags[LEADERBOARD_SCENARIO_KEY]
    parsed_scenario_kind = tags[LEADERBOARD_SCENARIO_KIND_KEY]

    assert parsed_candidate == scenario.candidate_count
    assert parsed_thinky == scenario.thinky_count
    assert parsed_ladybug == scenario.ladybug_count
    assert parsed_scenario_name == scenario.sim_name
    assert parsed_scenario_kind == scenario.scenario_kind


def test_scenario_proportions_match_counts() -> None:
    """Verify proportions list matches the individual count fields."""
    for scenario in leaderboard._generate_scenarios(leaderboard.NUM_COGS):
        expected = [
            float(scenario.candidate_count),
            float(scenario.thinky_count),
            float(scenario.ladybug_count),
        ]
        assert scenario.proportions == expected, f"proportions mismatch for {scenario.sim_name}"


def test_scenario_kind_classification() -> None:
    """Verify scenario_kind is correctly assigned based on agent counts."""
    scenarios = leaderboard._generate_scenarios(leaderboard.NUM_COGS)

    for scenario in scenarios:
        if scenario.candidate_count == leaderboard.NUM_COGS:
            assert scenario.scenario_kind == "candidate_self_play"
        elif scenario.candidate_count == 0:
            if scenario.thinky_count == leaderboard.NUM_COGS:
                assert scenario.scenario_kind == "thinky_self_play"
            elif scenario.ladybug_count == leaderboard.NUM_COGS:
                assert scenario.scenario_kind == "ladybug_self_play"
            else:
                assert scenario.scenario_kind == "thinky_ladybug_dual"
        else:
            assert scenario.scenario_kind == "candidate_mix"
