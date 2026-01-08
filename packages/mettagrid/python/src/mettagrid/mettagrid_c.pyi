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

class LimitDef:
    def __init__(
        self,
        resources: list[int] = [],
        base_limit: int = 0,
        modifiers: dict[int, int] = {},
    ) -> None: ...
    resources: list[int]
    base_limit: int
    modifiers: dict[int, int]

class InventoryConfig:
    def __init__(self) -> None: ...
    limit_defs: list[LimitDef]

class DamageConfig:
    def __init__(self) -> None: ...
    threshold: dict[int, int]
    resources: dict[int, int]
    def enabled(self) -> bool: ...

class WallConfig(GridObjectConfig):
    def __init__(self, type_id: int, type_name: str, initial_vibe: int = 0): ...
    type_id: int
    type_name: str
    tag_ids: list[int]
    initial_vibe: int

class AgentConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str = "agent",
        group_id: int = ...,
        group_name: str = ...,
        freeze_duration: int = 0,
        initial_vibe: int = 0,
        inventory_config: InventoryConfig = ...,
        stat_rewards: dict[str, float] = {},
        stat_reward_max: dict[str, float] = {},
        initial_inventory: dict[int, int] = {},
        inventory_regen_amounts: dict[int, dict[int, int]] | None = None,
        diversity_tracked_resources: list[int] | None = None,
        initial_vibe: int = 0,
        damage_config: DamageConfig = ...,
    ) -> None: ...
    type_id: int
    type_name: str
    tag_ids: list[int]
    initial_vibe: int
    group_id: int
    group_name: str
    freeze_duration: int
    inventory_config: InventoryConfig
    stat_rewards: dict[str, float]
    stat_reward_max: dict[str, float]
    initial_inventory: dict[int, int]
    inventory_regen_amounts: dict[int, dict[int, int]]
    diversity_tracked_resources: list[int]
    initial_vibe: int
    damage_config: DamageConfig

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

class InventoryConfig:
    def __init__(
        self,
        limits: list[tuple[list[int], int]] = [],
    ) -> None: ...
    limits: list[tuple[list[int], int]]

class AssemblerConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str,
        initial_vibe: int = 0,
    ) -> None: ...
    type_id: int
    type_name: str
    tag_ids: list[int]
    protocols: list[Protocol]
    allow_partial_usage: bool
    max_uses: int
    clip_immune: bool
    start_clipped: bool
    chest_search_distance: int
    initial_vibe: int

class ChestConfig(GridObjectConfig):
    def __init__(
        self,
        type_id: int,
        type_name: str,
        initial_vibe: int = 0,
    ) -> None: ...
    type_id: int
    type_name: str
    tag_ids: list[int]
    vibe_transfers: dict[int, dict[int, int]]
    initial_inventory: dict[int, int]
    inventory_config: InventoryConfig
    initial_vibe: int

class ClipperConfig:
    def __init__(self) -> None: ...
    unclipping_protocols: list[Protocol]
    length_scale: int
    scaled_cutoff_distance: int
    clip_period: int

class CollectiveConfig:
    def __init__(self, name: str = "") -> None: ...
    name: str
    inventory_config: InventoryConfig
    initial_inventory: dict[int, int]

class AttackOutcome:
    def __init__(
        self,
        actor_inv_delta: dict[int, int] = {},
        target_inv_delta: dict[int, int] = {},
        loot: list[int] = [],
        freeze: int = 0,
    ) -> None: ...
    actor_inv_delta: dict[int, int]
    target_inv_delta: dict[int, int]
    loot: list[int]
    freeze: int

class AttackActionConfig(ActionConfig):
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, int] = {},
        defense_resources: dict[int, int] = {},
        armor_resources: dict[int, int] = {},
        weapon_resources: dict[int, int] = {},
        success: AttackOutcome = ...,
        enabled: bool = True,
        vibes: list[int] = [],
        vibe_bonus: dict[int, int] = {},
    ) -> None: ...
    defense_resources: dict[int, int]
    armor_resources: dict[int, int]
    weapon_resources: dict[int, int]
    success: AttackOutcome
    enabled: bool
    vibes: list[int]
    vibe_bonus: dict[int, int]

class MoveActionConfig(ActionConfig):
    def __init__(
        self,
        allowed_directions: list[str] = ["north", "south", "west", "east"],
        required_resources: dict[int, int] = {},
        consumed_resources: dict[int, int] = {},
    ) -> None: ...
    allowed_directions: list[str]

class VibeTransferEffect:
    def __init__(
        self,
        target_deltas: dict[int, int] = {},
        actor_deltas: dict[int, int] = {},
    ) -> None: ...
    target_deltas: dict[int, int]
    actor_deltas: dict[int, int]

class TransferActionConfig(ActionConfig):
    def __init__(
        self,
        required_resources: dict[int, int] = {},
        vibe_transfers: dict[int, VibeTransferEffect] = {},
        enabled: bool = True,
    ) -> None: ...
    vibe_transfers: dict[int, VibeTransferEffect]
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
        goal_obs: bool = False,
    ) -> None: ...
    episode_completion_pct: bool
    last_action: bool
    last_reward: bool
    compass: bool
    goal_obs: bool

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
        feature_ids: dict[str, int],
        actions: dict[str, ActionConfig],
        objects: dict[str, GridObjectConfig],
        tag_id_map: dict[int, str] | None = None,
        collectives: dict[str, CollectiveConfig] | None = None,
        protocol_details_obs: bool = True,
        reward_estimates: Optional[dict[str, float]] = None,
        inventory_regen_interval: int = 0,
        clipper: Optional[ClipperConfig] = None,
        token_value_base: int = 256,
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
    feature_ids: dict[str, int]
    tag_id_map: dict[int, str]
    collectives: dict[str, CollectiveConfig]
    # FEATURE FLAGS
    protocol_details_obs: bool
    reward_estimates: Optional[dict[str, float]]
    inventory_regen_interval: int
    clipper: Optional[ClipperConfig]
    token_value_base: int

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
