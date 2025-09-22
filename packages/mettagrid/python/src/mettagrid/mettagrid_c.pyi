from typing import Optional, Tuple, TypeAlias, TypedDict

import gymnasium as gym
import numpy as np

# Type alias for clarity
StatsDict: TypeAlias = dict[str, float]

# Data types exported from C++
dtype_observations: np.dtype
dtype_terminals: np.dtype
dtype_truncations: np.dtype
dtype_rewards: np.dtype
dtype_actions: np.dtype
dtype_masks: np.dtype
dtype_success: np.dtype

class EpisodeStats(TypedDict):
    game: StatsDict
    agent: list[StatsDict]
    converter: list[StatsDict]

class PackedCoordinate:
    """Packed coordinate encoding utilities."""

    MAX_PACKABLE_COORD: int

    @staticmethod
    def pack(row: int, col: int) -> int:
        """Pack (row, col) coordinates into a single byte.
        Args:
            row: Row coordinate (0-14)
            col: Column coordinate (0-14)
        Returns:
            Packed byte value
        Note:
            The value 0xFF is reserved to indicate 'empty'.
        Raises:
            ValueError: If row or col > 14
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

class GridObjectConfig: ...

class WallConfig(GridObjectConfig):
    def __init__(self, type_id: int, type_name: str, swappable: bool = False): ...
    type_id: int
    type_name: str
    swappable: bool

class AgentConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str = "agent",
        group_id: int = ...,
        group_name: str = ...,
        freeze_duration: int = 0,
        action_failure_penalty: float = 0,
        resource_limits: dict[int, int] = {},
        resource_rewards: dict[int, float] = {},
        resource_reward_max: dict[int, float] = {},
        stat_rewards: dict[str, float] = {},
        stat_reward_max: dict[str, float] = {},
        group_reward_pct: float = 0,
        initial_inventory: dict[int, int] = {},
    ) -> None: ...
    type_id: int
    type_name: str
    group_id: int
    group_name: str
    freeze_duration: int
    action_failure_penalty: float
    resource_limits: dict[int, int]
    resource_rewards: dict[int, float]
    resource_reward_max: dict[int, float]
    stat_rewards: dict[str, float]  # Added this
    stat_reward_max: dict[str, float]  # Added this
    group_reward_pct: float
    initial_inventory: dict[int, int]

class ConverterConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str,
        input_resources: dict[int, int],
        output_resources: dict[int, int],
        max_output: int,
        max_conversions: int,
        conversion_ticks: int,
        cooldown: int,
        initial_resource_count: int = 0,
        color: int = 0,
        recipe_details_obs: bool = False,
    ) -> None: ...
    type_id: int
    type_name: str
    input_resources: dict[int, int]
    output_resources: dict[int, int]
    max_output: int
    max_conversions: int
    conversion_ticks: int
    cooldown: int
    initial_resource_count: int
    color: int
    recipe_details_obs: bool

class ActionConfig:
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, float] = {},
    ) -> None: ...
    required_resources: dict[int, int]
    consumed_resources: dict[int, float]

class AttackActionConfig(ActionConfig):
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, float] = {},
        defense_resources: dict[int, int] = {},
    ) -> None: ...
    defense_resources: dict[int, int]

class ChangeGlyphActionConfig(ActionConfig):
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, float] = {},
        number_of_glyphs: int = ...,
    ) -> None: ...
    number_of_glyphs: int

class GlobalObsConfig:
    def __init__(
        self,
        episode_completion_pct: bool = True,
        last_action: bool = True,
        last_reward: bool = True,
        resource_rewards: bool = False,
        visitation_counts: bool = False,
    ) -> None: ...
    episode_completion_pct: bool
    last_action: bool
    last_reward: bool
    resource_rewards: bool
    visitation_counts: bool

class GameConfig:
    def __init__(
        self,
        num_agents: int,
        max_steps: int,
        episode_truncates: bool,
        obs_width: int,
        obs_height: int,
        resource_names: list[str],
        num_observation_tokens: int,
        global_obs: GlobalObsConfig,
        actions: dict[str, ActionConfig],
        objects: dict[str, GridObjectConfig],
        resource_loss_prob: float = 0.0,
        track_movement_metrics: bool = False,
        recipe_details_obs: bool = False,
        allow_diagonals: bool = False,
        reward_estimates: Optional[dict[str, float]] = None,
    ) -> None: ...
    num_agents: int
    max_steps: int
    episode_truncates: bool
    obs_width: int
    obs_height: int
    resource_names: list[str]
    num_observation_tokens: int
    global_obs: GlobalObsConfig
    resource_loss_prob: float
    # FEATURE FLAGS
    track_movement_metrics: bool
    recipe_details_obs: bool
    allow_diagonals: bool
    reward_estimates: Optional[dict[str, float]]

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

    def __init__(self, env_cfg: GameConfig, map: list, seed: int) -> None: ...
    def reset(self) -> Tuple[np.ndarray, dict]: ...
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]: ...
    def set_buffers(
        self, observations: np.ndarray, terminals: np.ndarray, truncations: np.ndarray, rewards: np.ndarray
    ) -> None: ...
    def grid_objects(self) -> dict[int, dict]: ...
    def action_names(self) -> list[str]: ...
    def get_episode_rewards(self) -> np.ndarray: ...
    def get_episode_stats(self) -> EpisodeStats: ...
    def action_success(self) -> list[bool]: ...
    def max_action_args(self) -> list[int]: ...
    def object_type_names(self) -> list[str]: ...
    def resource_names(self) -> list[str]: ...
    def feature_spec(self) -> dict[str, dict[str, float | int]]: ...
