"""Tests for mission and curriculum alias resolution helpers."""

from io import StringIO

from rich.console import Console

from cogames import game, utils
from mettagrid.config.mettagrid_config import MettaGridConfig


def test_get_mission_config_handles_curriculum_alias() -> None:
    console = Console(file=StringIO())

    resolved_name, config = utils.get_mission_config(console, "training_rotation_easy_shaped")

    assert resolved_name == "training_rotation_easy_shaped"
    assert isinstance(config, MettaGridConfig)
    assert config.game.num_agents > 0


def test_curriculum_aliases_listed_with_missions() -> None:
    missions = game.get_all_missions()

    assert "training_rotation_easy_shaped" in missions
