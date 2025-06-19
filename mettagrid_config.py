from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, RootModel


class AgentRewards(BaseModel):
    """Agent reward configuration."""

    action_failure_penalty: Optional[float] = 0.0
    ore_red: Optional[float] = Field(default=0.005, alias="ore.red")
    ore_blue: Optional[float] = Field(default=0.005, alias="ore.blue")
    ore_green: Optional[float] = Field(default=0.005, alias="ore.green")
    battery_red: Optional[float] = Field(default=0.01, alias="battery.red")
    battery_blue: Optional[float] = Field(default=0.01, alias="battery.blue")
    battery_green: Optional[float] = Field(default=0.01, alias="battery.green")
    battery_red_max: Optional[float] = Field(default=5.0, alias="battery.red_max")
    battery_blue_max: Optional[float] = Field(default=5.0, alias="battery.blue_max")
    battery_green_max: Optional[float] = Field(default=5.0, alias="battery.green_max")
    heart: Optional[float] = 1.0
    heart_max: Optional[float] = 1000.0


class AgentConfig(BaseModel):
    """Agent configuration."""

    default_item_max: Optional[int] = 50
    freeze_duration: Optional[int] = 10
    inventory_size: Optional[int] = 0
    rewards: Optional[AgentRewards] = None


class GroupProps(RootModel[Dict[str, Any]]):
    """Group properties configuration."""

    pass


class GroupConfig(BaseModel):
    """Group configuration."""

    id: int
    sprite: Optional[int] = 0
    group_reward_pct: Optional[float] = 0.0
    props: Optional[GroupProps] = Field(default_factory=lambda: GroupProps({}))


class ActionConfig(BaseModel):
    """Action configuration."""

    enabled: bool = True


class ActionsConfig(BaseModel):
    """Actions configuration."""

    noop: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    move: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    rotate: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    put_items: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    get_items: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    attack: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    swap: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())
    change_color: Optional[ActionConfig] = Field(default_factory=lambda: ActionConfig())


class ObjectConfig(RootModel[Dict[str, Union[int, bool, float]]]):
    """Object configuration - flexible dict for object-specific properties."""

    pass


class ObjectsConfig(BaseModel):
    """Objects configuration."""

    altar: Optional[ObjectConfig] = None
    mine_red: Optional[ObjectConfig] = Field(default=None, alias="mine.red")
    mine_blue: Optional[ObjectConfig] = Field(default=None, alias="mine.blue")
    mine_green: Optional[ObjectConfig] = Field(default=None, alias="mine.green")
    generator_red: Optional[ObjectConfig] = Field(default=None, alias="generator.red")
    generator_blue: Optional[ObjectConfig] = Field(default=None, alias="generator.blue")
    generator_green: Optional[ObjectConfig] = Field(default=None, alias="generator.green")
    armory: Optional[ObjectConfig] = None
    lasery: Optional[ObjectConfig] = None
    lab: Optional[ObjectConfig] = None
    factory: Optional[ObjectConfig] = None
    temple: Optional[ObjectConfig] = None
    wall: Optional[ObjectConfig] = None
    block: Optional[ObjectConfig] = None


class RewardSharingGroup(RootModel[Dict[str, float]]):
    """Reward sharing configuration for a group."""

    pass


class RewardSharingConfig(BaseModel):
    """Reward sharing configuration."""

    groups: Optional[Dict[str, RewardSharingGroup]] = None


class RoomObjectsConfig(BaseModel):
    """Room objects configuration."""

    mine: Optional[int] = 0
    generator: Optional[int] = 0
    altar: Optional[int] = 0
    armory: Optional[int] = 0
    lasery: Optional[int] = 0
    lab: Optional[int] = 0
    factory: Optional[int] = 0
    temple: Optional[int] = 0
    block: Optional[int] = 0
    wall: Optional[int] = 0


class RoomConfig(BaseModel):
    """Room configuration."""

    width: Optional[int] = 25
    height: Optional[int] = 25
    border_width: Optional[int] = 0
    agents: Optional[int] = 6
    objects: Optional[RoomObjectsConfig] = Field(default_factory=RoomObjectsConfig)


class MapBuilderConfig(BaseModel):
    """Map builder configuration."""

    num_rooms: Optional[int] = 4
    border_width: Optional[int] = 6
    room: Optional[RoomConfig] = Field(default_factory=RoomConfig)
    # For ASCII map builders
    uri: Optional[str] = None
    # For terrain map builders
    _target_: Optional[str] = None


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


# Example usage and validation
if __name__ == "__main__":
    # Example configuration
    example_config = {
        "num_agents": 24,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 100,
        "max_steps": 1000,
        "agent": {
            "default_item_max": 50,
            "freeze_duration": 10,
            "rewards": {
                "action_failure_penalty": 0,
                "ore.red": 0.005,
                "ore.blue": 0.005,
                "ore.green": 0.005,
                "battery.red": 0.01,
                "battery.blue": 0.01,
                "battery.green": 0.01,
                "battery.red_max": 5,
                "battery.blue_max": 5,
                "battery.green_max": 5,
                "heart": 1,
                "heart_max": 1000,
            },
        },
        "groups": {
            "agent": {"id": 0, "sprite": 0, "props": {}},
            "team_1": {"id": 1, "sprite": 1, "group_reward_pct": 0.5, "props": {}},
        },
        "objects": {
            "altar": {
                "input_battery.red": 3,
                "output_heart": 1,
                "max_output": 5,
                "conversion_ticks": 1,
                "cooldown": 10,
                "initial_items": 1,
            },
            "mine.red": {
                "output_ore.red": 1,
                "color": 0,
                "max_output": 5,
                "conversion_ticks": 1,
                "cooldown": 50,
                "initial_items": 1,
            },
            "wall": {"swappable": False},
        },
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "change_color": {"enabled": True},
        },
        "map_builder": {
            "num_rooms": 4,
            "border_width": 6,
            "room": {
                "width": 25,
                "height": 25,
                "border_width": 0,
                "agents": 6,
                "objects": {
                    "mine": 10,
                    "generator": 2,
                    "altar": 1,
                    "armory": 1,
                    "lasery": 1,
                    "lab": 1,
                    "factory": 1,
                    "temple": 1,
                    "block": 20,
                    "wall": 20,
                },
            },
        },
    }

    # Validate the configuration
    config = GameConfig(**example_config)
    print("Configuration is valid!")
    print(f"Number of agents: {config.num_agents}")
    print(f"Max steps: {config.max_steps}")
    print(f"Observation dimensions: {config.obs_width}x{config.obs_height}")
