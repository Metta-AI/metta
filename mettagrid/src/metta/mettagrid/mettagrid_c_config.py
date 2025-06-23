import copy
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, RootModel

from mettagrid.mettagrid_config import GameConfig as GameConfig_py


class BaseModelWithForbidExtra(BaseModel):
    class Config:
        extra = "forbid"


class InventoryItemReward_cpp(BaseModelWithForbidExtra):
    """Reward configuration for an inventory item."""

    reward: float
    # the maximum number of items that can be collected for this reward
    max_reward: int


class AgentRewards_cpp(BaseModelWithForbidExtra):
    """Agent reward configuration."""

    action_failure_penalty: Optional[float] = None
    inventory_item_rewards: Dict[str, InventoryItemReward_cpp]


class AgentConfig_cpp(BaseModelWithForbidExtra):
    """Agent configuration."""

    default_item_max: int
    freeze_duration: int
    inventory_size: int
    group_id: int
    rewards: AgentRewards_cpp


class GroupProps_cpp(RootModel[Dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig_cpp(BaseModelWithForbidExtra):
    """Group configuration."""

    id: int
    sprite: Optional[int] = None
    group_reward_pct: Optional[float] = None


class ActionConfig_cpp(BaseModelWithForbidExtra):
    """Action configuration."""

    enabled: bool


class ActionsConfig_cpp(BaseModelWithForbidExtra):
    """Actions configuration."""

    noop: ActionConfig_cpp
    move: ActionConfig_cpp
    rotate: ActionConfig_cpp
    put_items: ActionConfig_cpp
    get_items: ActionConfig_cpp
    attack: ActionConfig_cpp
    swap: ActionConfig_cpp
    change_color: ActionConfig_cpp


class WallConfig_cpp(BaseModelWithForbidExtra):
    """Wall/Block configuration."""

    swappable: Optional[bool] = None


class ConverterConfig_cpp(BaseModelWithForbidExtra):
    """Converter configuration for objects that convert items."""

    input_items: Dict[str, int]
    output_items: Dict[str, int]

    # Converter properties
    max_output: int = Field(ge=0)
    conversion_ticks: int = Field(ge=0)
    cooldown: int = Field(ge=0)
    initial_items: int = Field(ge=0)
    color: Optional[int] = Field(default=None, ge=0, le=255)


class ObjectsConfig_cpp(BaseModelWithForbidExtra):
    """Objects configuration."""

    altar: Optional[ConverterConfig_cpp] = None
    mine_red: Optional[ConverterConfig_cpp] = Field(default=None, alias="mine.red")
    mine_blue: Optional[ConverterConfig_cpp] = Field(default=None, alias="mine.blue")
    mine_green: Optional[ConverterConfig_cpp] = Field(default=None, alias="mine.green")
    generator_red: Optional[ConverterConfig_cpp] = Field(default=None, alias="generator.red")
    generator_blue: Optional[ConverterConfig_cpp] = Field(default=None, alias="generator.blue")
    generator_green: Optional[ConverterConfig_cpp] = Field(default=None, alias="generator.green")
    armory: Optional[ConverterConfig_cpp] = None
    lasery: Optional[ConverterConfig_cpp] = None
    lab: Optional[ConverterConfig_cpp] = None
    factory: Optional[ConverterConfig_cpp] = None
    temple: Optional[ConverterConfig_cpp] = None
    wall: Optional[WallConfig_cpp] = None
    block: Optional[WallConfig_cpp] = None


class RewardSharingGroup_cpp(RootModel[Dict[str, float]]):
    """Reward sharing configuration for a group."""

    pass


class RewardSharingConfig_cpp(BaseModelWithForbidExtra):
    """Reward sharing configuration."""

    groups: Optional[Dict[str, RewardSharingGroup_cpp]] = None


class GameConfig_cpp(BaseModelWithForbidExtra):
    """Game configuration."""

    num_agents: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    obs_width: int = Field(ge=1)
    obs_height: int = Field(ge=1)
    num_observation_tokens: int = Field(ge=1)
    agent: AgentConfig_cpp
    groups: Dict[str, GroupConfig_cpp] = Field(min_length=1)
    actions: ActionsConfig_cpp
    objects: ObjectsConfig_cpp
    reward_sharing: Optional[RewardSharingConfig_cpp] = None


def agent_rewards_dict_from_flat_dict(flat_rewards_dict: Dict[str, float]) -> Dict[str, float]:
    """Converts from a dictionary like
      {
        "action_failure_penalty": 0,
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

    result = {
        "inventory_item_rewards": {},
    }
    for k, v in flat_rewards_dict.items():
        if k == "action_failure_penalty":
            result["action_failure_penalty"] = v
        elif k.endswith("_max"):
            inventory_item_name = k.replace("_max", "")
            result["inventory_item_rewards"].setdefault(inventory_item_name, {})["max_reward"] = v
        else:
            result["inventory_item_rewards"].setdefault(k, {})["reward"] = v

    return result


def from_mettagrid_config(mettagrid_config: GameConfig_py) -> GameConfig_cpp:
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
        agent_group_config["rewards"] = agent_rewards_dict_from_flat_dict(agent_group_config["rewards"])

        agent_configs["agent." + group_name] = AgentConfig_cpp(**agent_group_config)

    return GameConfig_cpp(
        num_agents=mettagrid_config.num_agents,
        max_steps=mettagrid_config.max_steps,
        obs_width=mettagrid_config.obs_width,
        obs_height=mettagrid_config.obs_height,
        num_observation_tokens=mettagrid_config.num_observation_tokens,
        agent=agent_configs,
        groups=mettagrid_config.groups,
    )
