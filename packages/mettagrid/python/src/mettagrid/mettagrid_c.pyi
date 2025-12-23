from typing import Optional, TypeAlias, TypedDict

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
    def unpack(packed: int) -> Optional[tuple[int, int]]:
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
    def __init__(self, type_id: int, type_name: str): ...
    type_id: int
    type_name: str

class AgentConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str = "agent",
        group_id: int = ...,
        group_name: str = ...,
        freeze_duration: int = 0,
        resource_limits: dict[int, int] = {},
        stat_rewards: dict[str, float] = {},
        stat_reward_max: dict[str, float] = {},
        initial_inventory: dict[int, int] = {},
        soul_bound_resources: list[int] | None = None,
        inventory_regen_amounts: dict[int, int] | None = None,
        diversity_tracked_resources: list[int] | None = None,
        vibe_transfers: dict[int, dict[int, int]] | None = None,
    ) -> None: ...
    type_id: int
    type_name: str
    tag_ids: list[int]
    group_id: int
    group_name: str
    freeze_duration: int
    resource_limits: dict[int, int]
    stat_rewards: dict[str, float]  # Added this
    stat_reward_max: dict[str, float]  # Added this
    initial_inventory: dict[int, int]
    soul_bound_resources: list[int]
    inventory_regen_amounts: dict[int, int]
    diversity_tracked_resources: list[int]
    vibe_transfers: dict[int, dict[int, int]]

class ActionConfig:
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, int] = {},
    ) -> None: ...
    required_resources: dict[int, int]
    consumed_resources: dict[int, int]

class Protocol:
    def __init__(self) -> None: ...
    min_agents: int
    vibes: list[int]
    input_resources: dict[int, int]
    output_resources: dict[int, int]
    cooldown: int

class ClipperConfig:
    def __init__(self) -> None: ...
    unclipping_protocols: list[Protocol]
    length_scale: int
    scaled_cutoff_distance: int
    clip_period: int

class AttackActionConfig(ActionConfig):
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, int] = {},
        defense_resources: dict[int, int] = {},
        enabled: bool = True,
    ) -> None: ...
    defense_resources: dict[int, int]
    enabled: bool

class ChangeVibeActionConfig(ActionConfig):
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, int] = {},
        number_of_vibes: int = ...,
    ) -> None: ...
    number_of_vibes: int

class GlobalObsConfig:
    def __init__(
        self,
        episode_completion_pct: bool = True,
        last_action: bool = True,
        last_reward: bool = True,
        compass: bool = False,
    ) -> None: ...
    episode_completion_pct: bool
    last_action: bool
    last_reward: bool
    compass: bool

class GameConfig:
    def __init__(
        self,
        num_agents: int,
        max_steps: int,
        episode_truncates: bool,
        obs_width: int,
        obs_height: int,
        resource_names: list[str],
        vibe_names: list[str],
        num_observation_tokens: int,
        global_obs: GlobalObsConfig,
        actions: dict[str, ActionConfig],
        objects: dict[str, GridObjectConfig],
        tag_id_map: dict[int, str] | None = None,
        protocol_details_obs: bool = True,
        allow_diagonals: bool = False,
        reward_estimates: Optional[dict[str, float]] = None,
        inventory_regen_amounts: dict[int, int] | None = None,
        inventory_regen_interval: int = 0,
        clipper: Optional[ClipperConfig] = None,
    ) -> None: ...
    num_agents: int
    max_steps: int
    episode_truncates: bool
    obs_width: int
    obs_height: int
    resource_names: list[str]
    vibe_names: list[str]
    num_observation_tokens: int
    global_obs: GlobalObsConfig
    # FEATURE FLAGS
    protocol_details_obs: bool
    allow_diagonals: bool
    reward_estimates: Optional[dict[str, float]]
    tag_id_map: dict[int, str]
    inventory_regen_amounts: dict[int, int]
    inventory_regen_interval: int
    clipper: Optional[ClipperConfig]

class MettaGrid:
    obs_width: int
    obs_height: int
    max_steps: int
    current_step: int
    map_width: int
    map_height: int
    num_agents: int
    object_type_names: list[str]

    def __init__(self, env_cfg: GameConfig, map: list, seed: int) -> None: ...
    def step(self) -> None: ...
    def set_buffers(
        self,
        observations: np.ndarray,
        terminals: np.ndarray,
        truncations: np.ndarray,
        rewards: np.ndarray,
        actions: np.ndarray,
    ) -> None: ...
    def grid_objects(
        self,
        min_row: int = -1,
        max_row: int = -1,
        min_col: int = -1,
        max_col: int = -1,
        ignore_types: list[str] = [],
    ) -> dict[int, dict]: ...
    def observations(self) -> np.ndarray: ...
    def terminals(self) -> np.ndarray: ...
    def truncations(self) -> np.ndarray: ...
    def rewards(self) -> np.ndarray: ...
    def masks(self) -> np.ndarray: ...
    def actions(self) -> np.ndarray: ...
    def get_episode_rewards(self) -> np.ndarray: ...
    def get_episode_stats(self) -> EpisodeStats: ...
    def action_success(self) -> list[bool]: ...
    def set_inventory(self, agent_id: int, inventory: dict[int, int]) -> None: ...
