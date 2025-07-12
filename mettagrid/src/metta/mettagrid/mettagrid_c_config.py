import copy
from typing import Any

from metta.mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from metta.mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from metta.mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from metta.mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from metta.mettagrid.mettagrid_c import GameConfig as CppGameConfig
from metta.mettagrid.mettagrid_c import WallConfig as CppWallConfig
from metta.mettagrid.mettagrid_config import PyConverterConfig, PyGameConfig, PyWallConfig


def from_mettagrid_config(mettagrid_config_dict: dict[str, Any]) -> CppGameConfig:
    """Convert a PyGameConfig to a CppGameConfig."""

    mettagrid_config = PyGameConfig(**mettagrid_config_dict)

    resource_names = list(mettagrid_config.inventory_item_names)
    resource_ids = dict((name, i) for i, name in enumerate(resource_names))

    cpp_object_configs = {}

    # These are the baseline settings for all agents
    agent_default_config_dict = mettagrid_config.agent.model_dump(by_alias=True, exclude_unset=True)

    # Group information is more specific than the defaults, so it should override
    for group_name, cfg in mettagrid_config.groups.items():
        group_config_dict = cfg.model_dump(by_alias=True, exclude_unset=True)
        merged_config = copy.deepcopy(agent_default_config_dict)

        # Update, but in a nested way
        for key, value in group_config_dict.get("props", {}).items():
            if isinstance(value, dict):
                # At the time of writing, this should only be the rewards field
                merged_config[key] = value
            else:
                merged_config[key] = value

        default_resource_limit = merged_config.get("default_resource_limit", 0)

        # Handle rewards conversion
        rewards_dict = {}
        if merged_config.get("rewards"):
            rewards_dict = merged_config["rewards"]

        cpp_agent_group_params = {
            "freeze_duration": merged_config.get("freeze_duration", 0),
            "group_id": cfg.id,
            "group_name": group_name,
            "action_failure_penalty": merged_config.get("action_failure_penalty", 0),
            "resource_limits": dict(
                (resource_id, merged_config.get("resource_limits", {}).get(resource_name, default_resource_limit))
                for (resource_id, resource_name) in enumerate(resource_names)
            ),
            "resource_rewards": dict((resource_ids[k], v) for k, v in rewards_dict.items() if not k.endswith("_max")),
            "resource_reward_max": dict(
                (resource_ids[k[:-4]], v) for k, v in rewards_dict.items() if k.endswith("_max") and v is not None
            ),
            "group_reward_pct": cfg.group_reward_pct or 0,
        }

        # HardCodedConfig
        cpp_agent_group_params["type_id"] = 0
        cpp_agent_group_params["type_name"] = "agent"
        cpp_object_configs["agent." + group_name] = CppAgentConfig(**cpp_agent_group_params)

    # Convert other objects
    for object_type, cfg in mettagrid_config.objects.items():
        if isinstance(cfg, PyConverterConfig):
            cpp_converter_config = CppConverterConfig(
                type_id=cfg.type_id,
                type_name=object_type,
                input_resources=dict((resource_ids[k], v) for k, v in cfg.input_resources.items() if v > 0),
                output_resources=dict((resource_ids[k], v) for k, v in cfg.output_resources.items() if v > 0),
                max_output=cfg.max_output,
                conversion_ticks=cfg.conversion_ticks,
                cooldown=cfg.cooldown,
                initial_resource_count=cfg.initial_resource_count,
                color=cfg.color,
            )
            cpp_object_configs[object_type] = cpp_converter_config
        elif isinstance(cfg, PyWallConfig):
            cpp_wall_config = CppWallConfig(
                type_id=cfg.type_id,
                type_name=object_type,
                swappable=cfg.swappable,
            )
            cpp_object_configs[object_type] = cpp_wall_config
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    game_config = mettagrid_config.model_dump(by_alias=True, exclude_none=True)

    cpp_actions_config = {}
    # Add required and consumed resources to the attack action
    for action_name, cfg in game_config["actions"].items():
        if not cfg["enabled"]:
            continue

        cpp_action_params = {}
        cpp_action_params["consumed_resources"] = dict(
            (resource_ids[k], v) for k, v in cfg["consumed_resources"].items()
        )
        if cfg.get("required_resources", None) is not None:
            cpp_action_params["required_resources"] = dict(
                (resource_ids[k], v) for k, v in cfg["required_resources"].items()
            )
        else:
            cpp_action_params["required_resources"] = cpp_action_params["consumed_resources"]

        if action_name == "attack":
            cpp_action_params["defense_resources"] = dict(
                (resource_ids[k], v) for k, v in cfg["defense_resources"].items()
            )
            cpp_actions_config[action_name] = CppAttackActionConfig(**cpp_action_params)
        else:
            cpp_actions_config[action_name] = CppActionConfig(**cpp_action_params)

    game_config["actions"] = cpp_actions_config

    del game_config["agent"]
    del game_config["groups"]
    game_config["objects"] = cpp_object_configs

    return CppGameConfig(**game_config)
