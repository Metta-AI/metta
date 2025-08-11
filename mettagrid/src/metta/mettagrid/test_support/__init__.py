# mettagrid/src/metta/mettagrid/test_support/__init__.py

from dataclasses import dataclass

from .environment_builder import DefaultEnvConfig as EnvConfig
from .environment_builder import TestEnvironmentBuilder
from .make_test_config import env_cfg_builder, make_test_config, make_test_level_map
from .observation_helper import ObservationHelper


# Constants from C++ code
@dataclass
class TokenTypes:
    # Observation features
    TYPE_ID_FEATURE: int = 0
    GROUP: int = 1
    HP: int = 2
    FROZEN: int = 3
    ORIENTATION: int = 4
    COLOR: int = 5
    CONVERTING_OR_COOLING_DOWN: int = 6
    SWAPPABLE: int = 7
    EPISODE_COMPLETION_PCT: int = 8
    LAST_ACTION: int = 9
    LAST_ACTION_ARG: int = 10
    LAST_REWARD: int = 11
    GLYPH: int = 12
    RESOURCE_REWARDS: int = 13
    VISITATION_COUNTS: int = 14

    # Observation dimensions
    OBS_TOKEN_SIZE: int = 3

    # Empty token
    EMPTY_TOKEN = [0xFF, 0xFF, 0xFF]

    # Object type IDs
    WALL_TYPE_ID: int = 1
    ALTAR_TYPE_ID: int = 10


__all__ = [
    "EnvConfig",
    "TestEnvironmentBuilder",
    "TokenTypes",
    "ObservationHelper",
    "env_cfg_builder",
    "make_test_config",
    "make_test_level_map",
]
