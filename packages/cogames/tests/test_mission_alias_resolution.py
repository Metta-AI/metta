"""Tests for mission and curriculum alias resolution helpers."""

from io import StringIO

from rich.console import Console

from cogames import game, utils
from cogames.cogs_vs_clips.missions import CurriculumUserMap


def test_get_mission_config_handles_curriculum_alias() -> None:
    console = Console(file=StringIO())

    resolved_name, config, user_map = utils.get_mission_config(console, "training_rotation_easy_shaped")

    assert resolved_name == "training_rotation_easy_shaped"
    assert config.game.num_agents > 0

    assert isinstance(user_map, CurriculumUserMap)

    next_config = user_map.generate_env(user_map.default_mission)
    assert next_config.game.num_agents > 0
    # Shaped curriculum should add shaped rewards
    assert config.game.agent.rewards.stats is not None
    assert "heart.gained" in config.game.agent.rewards.stats


def test_curriculum_maps_listed_with_missions() -> None:
    missions = game.get_all_missions()

    assert "training_rotation_easy_shaped" in missions


def test_training_facility_easy_variant_extends_episode() -> None:
    default_cfg, _, _ = game.get_mission("training_facility_1")
    easy_cfg, _, mission_name = game.get_mission("training_facility_1", "easy")

    assert mission_name == "easy"
    default_max_steps = default_cfg.game.max_steps
    easy_max_steps = easy_cfg.game.max_steps
    if default_max_steps is not None and easy_max_steps is not None:
        assert easy_max_steps > default_max_steps

    assembler = easy_cfg.game.objects["assembler"]
    first_recipe = assembler.recipes[0][1]
    assert first_recipe.input_resources == {"energy": 1}
    assert first_recipe.output_resources == {"heart": 1}


def test_training_facility_shaped_variant_has_rewards() -> None:
    shaped_cfg, _, _ = game.get_mission("training_facility_1", "shaped")
    stats = shaped_cfg.game.agent.rewards.stats
    assert stats is not None
    assert stats["heart.gained"] == 5.0


def test_training_facility_suffix_alias_resolves() -> None:
    cfg, map_name, mission_name = game.get_mission("training_facility_1_easy")
    assert map_name == "training_facility_1"
    assert mission_name == "easy"
    assembler = cfg.game.objects["assembler"]
    assert assembler.recipes[0][1].input_resources == {"energy": 1}
