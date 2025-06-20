from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, RootModel


class InventoryItemReward(BaseModel):
    """Reward configuration for an inventory item."""

    reward: float
    # the maximum number of items that can be collected for this reward
    max_reward: int


class AgentRewards(BaseModel):
    """Agent reward configuration."""

    action_failure_penalty: Optional[float] = None
    inventory_item_rewards: Dict[str, InventoryItemReward]


class AgentConfig(BaseModel):
    """Agent configuration."""

    default_item_max: Optional[int] = None
    freeze_duration: Optional[int] = None
    inventory_size: Optional[int] = None
    rewards: Optional[AgentRewards] = None


class GroupProps(RootModel[Dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig(BaseModel):
    """Group configuration."""

    id: int
    sprite: Optional[int] = None
    group_reward_pct: Optional[float] = None
    props: Optional[GroupProps] = None


class ActionConfig(BaseModel):
    """Action configuration."""

    enabled: bool


class ActionsConfig(BaseModel):
    """Actions configuration."""

    noop: ActionConfig
    move: ActionConfig
    rotate: ActionConfig
    put_items: ActionConfig
    get_items: ActionConfig
    attack: ActionConfig
    swap: ActionConfig
    change_color: ActionConfig


class WallConfig(BaseModel):
    """Wall/Block configuration."""

    swappable: Optional[bool] = None


class ConverterConfig(BaseModel):
    """Converter configuration for objects that convert items."""

    input_items: Dict[str, int]
    output_items: Dict[str, int]

    # Converter properties
    max_output: int
    conversion_ticks: int
    cooldown: int
    initial_items: int
    color: Optional[int] = None


class ObjectsConfig(BaseModel):
    """Objects configuration."""

    altar: Optional[ConverterConfig] = None
    mine_red: Optional[ConverterConfig] = Field(default=None, alias="mine.red")
    mine_blue: Optional[ConverterConfig] = Field(default=None, alias="mine.blue")
    mine_green: Optional[ConverterConfig] = Field(default=None, alias="mine.green")
    generator_red: Optional[ConverterConfig] = Field(default=None, alias="generator.red")
    generator_blue: Optional[ConverterConfig] = Field(default=None, alias="generator.blue")
    generator_green: Optional[ConverterConfig] = Field(default=None, alias="generator.green")
    armory: Optional[ConverterConfig] = None
    lasery: Optional[ConverterConfig] = None
    lab: Optional[ConverterConfig] = None
    factory: Optional[ConverterConfig] = None
    temple: Optional[ConverterConfig] = None
    wall: Optional[WallConfig] = None
    block: Optional[WallConfig] = None


class RewardSharingGroup(RootModel[Dict[str, float]]):
    """Reward sharing configuration for a group."""

    pass


class RewardSharingConfig(BaseModel):
    """Reward sharing configuration."""

    groups: Optional[Dict[str, RewardSharingGroup]] = None


class GameConfig(BaseModel):
    """Game configuration."""

    num_agents: int
    max_steps: int
    obs_width: int
    obs_height: int
    num_observation_tokens: int
    agent: AgentConfig
    groups: Dict[str, GroupConfig]
    actions: ActionsConfig
    objects: ObjectsConfig
    reward_sharing: Optional[RewardSharingConfig] = None

    class Config:
        extra = "forbid"
