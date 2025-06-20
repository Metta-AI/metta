import copy
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, RootModel

from mettagrid.mettagrid_config import GameConfig as MettaGridGameConfig


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

    default_item_max: int
    freeze_duration: int
    inventory_size: int
    group_id: int
    rewards: AgentRewards


class GroupProps(RootModel[Dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig(BaseModel):
    """Group configuration."""

    id: int
    sprite: Optional[int] = None
    group_reward_pct: Optional[float] = None


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


def agent_rewards_dict_from_flat_dict(flat_rewards_dict: Dict[str, float]) -> Dict[str, float]:
    """Converts from a dictionary like
      {
        "invalid_action_penalty": 0,
        "ore.red": 0.005,
        "ore.red_max": 4,
        "battery.red": 0.01,
        "battery.red_max": 5
      }
    to a dictionary like
      {
        "action_failure_penalty": 0,
        "ore.red": {
          "reward": 0.005,
          "max_reward": 4,
        },
        "battery.red": {
          "reward": 0.01,
          "max_reward": 5,
        }
      }
    """

    result = {}
    for k, v in flat_rewards_dict.items():
        if k == "invalid_action_penalty":
            result["action_failure_penalty"] = v
        elif k.endswith("_max"):
            inventory_item_name = k.replace("_max", "")
            result.setdefault(inventory_item_name, {})["max_reward"] = v
        else:
            result.setdefault(k, {})["reward"] = v

    return result


def from_mettagrid_config(mettagrid_config: MettaGridGameConfig) -> GameConfig:
    """Convert a mettagrid_config.GameConfig to a mettagrid_c_config.GameConfig."""

    agent_configs = {}

    # these are the baseline settings for all agents
    agent_default_config_dict = mettagrid_config.agent.model_dump(by_alias=True, exclude_unset=True)

    # Group information is more specific than the defaults, so it should override
    for group_name, group_config in mettagrid_config.groups.items():
        group_config_dict = group_config.model_dump(by_alias=True, exclude_unset=True)
        agent_group_config = copy.deepcopy(agent_default_config_dict)
        # update, but in a nested way
        for key, value in group_config_dict.get("props", {}).items():
            if isinstance(value, dict):
                # At the time of writing, this should only be the rewards field
                agent_group_config[key] = value
            else:
                agent_group_config[key] = value
        agent_group_config["group_id"] = group_config.id
        agent_group_config["rewards"] = agent_rewards_dict_from_flat_dict(agent_group_config.get("rewards", {}))
        agent_configs["agent." + group_name] = AgentConfig(**agent_group_config)

    return GameConfig(
        num_agents=mettagrid_config.num_agents,
        max_steps=mettagrid_config.max_steps,
        obs_width=mettagrid_config.obs_width,
        obs_height=mettagrid_config.obs_height,
        num_observation_tokens=mettagrid_config.num_observation_tokens,
        agent=AgentConfig(
            default_item_max=mettagrid_config.agent.default_item_max,
            freeze_duration=mettagrid_config.agent.freeze_duration,
            inventory_size=mettagrid_config.agent.inventory_size,
            rewards=mettagrid_config.agent.rewards,
        ),
        groups=mettagrid_config.groups,
    )
