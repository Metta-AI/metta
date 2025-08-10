from dataclasses import dataclass


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

    # Object type IDs
    WALL_TYPE_ID: int = 1
    ALTAR_TYPE_ID: int = 10

    # empty token
    EMPTY_TOKEN = [0xFF, 0xFF, 0xFF]

    # three bytes per token
    OBS_TOKEN_SIZE = 3
