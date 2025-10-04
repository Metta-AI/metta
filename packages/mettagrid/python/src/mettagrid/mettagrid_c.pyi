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
    def __init__(
        self,
        type_id: int,
        type_name: str,
        swappable: bool = False,
        tag_ids: list[int] = ...,
    ): ...
    type_id: int
    type_name: str
    swappable: bool
    tag_ids: list[int]

class AgentConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str = "agent",
        group_id: int = ...,
        group_name: str = ...,
        freeze_duration: int = 0,
        action_failure_penalty: float = 0,
        inventory_config: "InventoryConfig" = ...,
        stat_rewards: dict[str, float] = {},
        stat_reward_max: dict[str, float] = {},
        group_reward_pct: float = 0,
        initial_inventory: dict[int, int] = {},
        tag_ids: list[int] = ...,
        soul_bound_resources: list[int] = ...,
        shareable_resources: list[int] = ...,
        inventory_regen_amounts: dict[int, int] = {},
    ) -> None: ...
    type_id: int
    type_name: str
    group_id: int
    group_name: str
    freeze_duration: int
    action_failure_penalty: float
    inventory_config: "InventoryConfig"
    stat_rewards: dict[str, float]
    stat_reward_max: dict[str, float]
    group_reward_pct: float
    initial_inventory: dict[int, int]
    tag_ids: list[int]
    soul_bound_resources: list[int]
    shareable_resources: list[int]
    inventory_regen_amounts: dict[int, int]

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
        recipe_details_obs: bool = False,
        tag_ids: list[int] = ...,
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
    recipe_details_obs: bool
    tag_ids: list[int]

class Recipe:
    def __init__(
        self,
        input_resources: dict[int, int] = {},
        output_resources: dict[int, int] = {},
        cooldown: int = 0,
    ) -> None: ...
    input_resources: dict[int, int]
    output_resources: dict[int, int]
    cooldown: int

class AssemblerConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str,
        tag_ids: list[int] = ...,
    ) -> None: ...
    type_id: int
    type_name: str
    tag_ids: list[int]
    recipes: list[Recipe | None]
    recipe_details_obs: bool
    allow_partial_usage: bool
    max_uses: int
    exhaustion: float
    clip_immune: bool
    start_clipped: bool

class ChestConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str,
        resource_type: int,
        position_deltas: dict[int, int],
        initial_inventory: int = ...,
        max_inventory: int = ...,
        tag_ids: list[int] = ...,
    ) -> None: ...
    type_id: int
    type_name: str
    resource_type: int
    position_deltas: dict[int, int]
    initial_inventory: int
    max_inventory: int
    tag_ids: list[int]

class ActionConfig:
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, float] = {},
    ) -> None: ...
    required_resources: dict[int, int]
    consumed_resources: dict[int, float]

class ClipperConfig:
    def __init__(
        self,
        unclipping_recipes: list[Recipe],
        length_scale: float,
        cutoff_distance: float,
        clip_rate: float,
    ) -> None: ...
    unclipping_recipes: list[Recipe]
    length_scale: float
    cutoff_distance: float
    clip_rate: float

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

class ResourceModConfig(ActionConfig):
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, float] = {},
        modifies: dict[int, float] = {},
        agent_radius: int = 0,
        converter_radius: int = 0,
        scales: bool = False,
    ) -> None: ...
    modifies: dict[int, float]
    agent_radius: int
    converter_radius: int
    scales: bool

class GlobalObsConfig:
    def __init__(
        self,
        episode_completion_pct: bool = True,
        last_action: bool = True,
        last_reward: bool = True,
        visitation_counts: bool = False,
    ) -> None: ...
    episode_completion_pct: bool
    last_action: bool
    last_reward: bool
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
        tag_id_map: dict[int, str] = {},
        track_movement_metrics: bool = False,
        recipe_details_obs: bool = False,
        allow_diagonals: bool = False,
        reward_estimates: Optional[dict[str, float]] = None,
        inventory_regen_interval: int = 0,
        clipper: Optional[ClipperConfig] = None,
    ) -> None: ...
    num_agents: int
    max_steps: int
    episode_truncates: bool
    obs_width: int
    obs_height: int
    resource_names: list[str]
    num_observation_tokens: int
    global_obs: GlobalObsConfig
    objects: dict[str, GridObjectConfig]  # Readonly - for inspection only
    resource_loss_prob: float
    # FEATURE FLAGS
    track_movement_metrics: bool
    recipe_details_obs: bool
    allow_diagonals: bool
    reward_estimates: Optional[dict[str, float]]
    tag_id_map: dict[int, str]
    inventory_regen_interval: int
    clipper: Optional[ClipperConfig]

class InventoryConfig:
    def __init__(
        self,
        limits: list[tuple[list[int], int]] = ...,
    ) -> None: ...
    limits: list[tuple[list[int], int]]

class MettaGrid:
    obs_width: int
    obs_height: int
    max_steps: int
    current_step: int
    map_width: int
    map_height: int
    num_agents: int
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    initial_grid_hash: int

    def __init__(self, env_cfg: GameConfig, map: list, seed: int) -> None: ...
    def reset(self) -> Tuple[np.ndarray, dict]: ...
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]: ...
    def set_buffers(
        self, observations: np.ndarray, terminals: np.ndarray, truncations: np.ndarray, rewards: np.ndarray
    ) -> None: ...
    def grid_objects(
        self,
        min_row: int = -1,
        max_row: int = -1,
        min_col: int = -1,
        max_col: int = -1,
        ignore_types: list[str] = [],
    ) -> dict[int, dict]: ...
    def action_names(self) -> list[str]: ...
    def get_episode_rewards(self) -> np.ndarray: ...
    def get_episode_stats(self) -> EpisodeStats: ...
    def action_success(self) -> list[bool]: ...
    def object_type_names(self) -> list[str]: ...
    def resource_names(self) -> list[str]: ...
    def feature_spec(self) -> dict[str, dict[str, float | int]]: ...
    def set_inventory(self, agent_id: int, inventory: dict[int, int]) -> None: ...
