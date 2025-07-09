from typing import Dict, List, Optional, Tuple, TypeAlias, TypedDict

import gymnasium as gym
import numpy as np

# Type alias for clarity
StatsDict: TypeAlias = Dict[str, float]

class EpisodeStats(TypedDict):
    game: StatsDict
    agent: List[StatsDict]
    converter: List[StatsDict]

class PackedCoordinate:
    """Packed coordinate encoding utilities."""

    MAX_PACKABLE_COORD: int

    @staticmethod
    def pack(row: int, col: int) -> int:
        """Pack (row, col) coordinates into a single byte.

        Args:
            row: Row coordinate (0-15)
            col: Column coordinate (0-15)

        Returns:
            Packed byte value

        Note:
            The value 0xFF is reserved to indicate 'empty', so the
            coordinate (15, 15) cannot be encoded.

        Raises:
            ValueError: If row or col > 15, or if attempting to pack (15, 15)
        """
        ...

    @staticmethod
    def unpack(packed: int) -> Optional[Tuple[int, int]]:
        """Unpack byte into (row, col) tuple or None if empty.

        Args:
            packed: Packed coordinate byte

        Returns:
            (row, col) tuple or None if empty location
        """
        ...

    @staticmethod
    def is_empty(packed: int) -> bool:
        """Check if packed value represents empty location."""
        ...

class MettaGrid:
    obs_width: int
    obs_height: int
    max_steps: int
    current_step: int
    map_width: int
    map_height: int
    num_agents: int
    action_space: gym.spaces.MultiDiscrete
    observation_space: gym.spaces.Box
    initial_grid_hash: int

    def __init__(self, env_cfg: dict, map: list, seed: int) -> None: ...
    def reset(self) -> Tuple[np.ndarray, dict]: ...
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]: ...
    def set_buffers(
        self, observations: np.ndarray, terminals: np.ndarray, truncations: np.ndarray, rewards: np.ndarray
    ) -> None: ...
    def grid_objects(self) -> Dict[int, dict]: ...
    def action_names(self) -> List[str]: ...
    def get_episode_rewards(self) -> np.ndarray: ...
    def get_episode_stats(self) -> EpisodeStats: ...
    def action_success(self) -> List[bool]: ...
    def max_action_args(self) -> List[int]: ...
    def object_type_names(self) -> List[str]: ...
    def inventory_item_names(self) -> List[str]: ...
    def get_agent_groups(self) -> np.ndarray: ...
    def feature_normalizations(self) -> Dict[int, float]: ...
