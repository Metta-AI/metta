"""Lightweight checks for mission alias support."""

from typing import Iterable

import pytest

from cogames import game
from cogames.cogs_vs_clips.missions import CurriculumUserMap
from cogames.mission_aliases import get_user_map


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


def test_curriculum_alias_resolves_to_curriculum_map() -> None:
    cfg, map_name, mission_name = game.get_mission("training_rotation_easy_shaped")

    assert mission_name == "default"
    curriculum_map = get_user_map(map_name)
    assert isinstance(curriculum_map, CurriculumUserMap)
    assert cfg.game.num_agents > 0
