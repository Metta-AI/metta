"""Lightweight checks for mission alias support."""

from io import StringIO
from typing import Iterable

import pytest
from rich.console import Console

from cogames import game, utils
from cogames.cogs_vs_clips.missions import CurriculumUserMap


def _console() -> Console:
    return Console(file=StringIO())


def test_curriculum_alias_uses_registered_map() -> None:
    resolved, cfg, user_map = utils.get_mission_config(_console(), "training_rotation_easy_shaped")

    assert resolved == "training_rotation_easy_shaped"
    assert isinstance(user_map, CurriculumUserMap)
    assert cfg.game.num_agents > 0


@pytest.mark.parametrize(
    ("alias", "expected_map", "expected_mission"),
    [
        ("training_facility_1_easy", "training_facility_1", "easy"),
        ("training_facility_1_shaped", "training_facility_1", "shaped"),
    ],
)
def test_suffix_aliases_share_base_maps(alias: str, expected_map: str, expected_mission: str) -> None:
    cfg, map_name, mission_name = game.get_mission(alias)

    assert map_name == expected_map
    assert mission_name == expected_mission
    assert cfg.game.max_steps is None or cfg.game.max_steps > 0


def test_aliases_listed_in_catalog() -> None:
    missions = set(game.get_all_missions())

    required: Iterable[str] = ("training_rotation_easy_shaped", "training_facility_1:easy")
    for name in required:
        assert name in missions
