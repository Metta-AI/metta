import copy

from metta.mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from metta.mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from metta.mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from metta.mettagrid.mettagrid_c import ChangeGlyphActionConfig as CppChangeGlyphActionConfig
from metta.mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from metta.mettagrid.mettagrid_c import GameConfig as CppGameConfig
from metta.mettagrid.mettagrid_c import GlobalObsConfig as CppGlobalObsConfig
from metta.mettagrid.mettagrid_c import WallConfig as CppWallConfig
from metta.mettagrid.mettagrid_config import PyConverterConfig, PyGameConfig, PyWallConfig


def convert_to_cpp_game_config(mettagrid_config_dict: dict):
    """Convert a PyGameConfig to a CppGameConfig."""

    game_config = PyGameConfig(**mettagrid_config_dict)

    resource_names = list(game_config.inventory_item_names)
    resource_name_to_id = {name: i for i, name in enumerate(resource_names)}

    objects_cpp_params = {}  # params for CppConverterConfig or CppWallConfig

    # These are the baseline settings for all agents
    default_agent_config_dict = game_config.agent.model_dump()
    default_resource_limit = default_agent_config_dict["default_resource_limit"]

    # Group information is more specific than the defaults, so it should override
    for group_name, group_config in game_config.groups.items():
        agent_group_props = copy.deepcopy(default_agent_config_dict)

        # Update, but in a nested way
        for key, value in group_config.props.model_dump(exclude_unset=True).items():
            agent_group_props[key] = value

        # Extract inventory rewards - handle both old and new format for backward compatibility
        inventory_rewards = {}
        inventory_reward_max = {}

        rewards_config = agent_group_props.get("rewards", {})
        if rewards_config:
            inventory_rewards_dict = rewards_config.get("inventory", {})

            # Process inventory rewards
            for k, v in inventory_rewards_dict.items():
                if v is not None and not k.endswith("_max"):
                    if k in resource_name_to_id:
                        inventory_rewards[resource_name_to_id[k]] = v
                elif k.endswith("_max") and v is not None:
                    item_name = k[:-4]
                    if item_name in resource_name_to_id:
                        inventory_reward_max[resource_name_to_id[item_name]] = v

        # Process stats rewards
        stat_rewards = {}
        stat_reward_max = {}
        stats_rewards_dict = rewards_config.get("stats", {}) if rewards_config else {}

        for k, v in stats_rewards_dict.items():
            if v is not None and not k.endswith("_max"):
                stat_rewards[k] = v
            elif k.endswith("_max") and v is not None:
                stat_name = k[:-4]
                stat_reward_max[stat_name] = v

        agent_cpp_params = {
            "freeze_duration": agent_group_props["freeze_duration"],
            "group_id": group_config.id,
            "group_name": group_name,
            "action_failure_penalty": agent_group_props["action_failure_penalty"],
            "resource_limits": {
                resource_id: agent_group_props["resource_limits"].get(resource_name, default_resource_limit)
                for resource_id, resource_name in enumerate(resource_names)
            },
            "resource_rewards": inventory_rewards,
            "resource_reward_max": inventory_reward_max,
            "stat_rewards": stat_rewards,
            "stat_reward_max": stat_reward_max,
            "group_reward_pct": group_config.group_reward_pct,
            "type_id": 0,
            "type_name": "agent",
        }

        objects_cpp_params["agent." + group_name] = CppAgentConfig(**agent_cpp_params)

    # Convert other objects
    for object_type, object_config in game_config.objects.items():
        if isinstance(object_config, PyConverterConfig):
            cpp_converter_config = CppConverterConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                input_resources={
                    resource_name_to_id[k]: v
                    for k, v in object_config.input_resources.items()
                    if v > 0 and k in resource_name_to_id
                },
                output_resources={
                    resource_name_to_id[k]: v
                    for k, v in object_config.output_resources.items()
                    if v > 0 and k in resource_name_to_id
                },
                max_output=object_config.max_output,
                conversion_ticks=object_config.conversion_ticks,
                cooldown=object_config.cooldown,
                initial_resource_count=object_config.initial_resource_count,
                color=object_config.color,
                recipe_details_obs=game_config.recipe_details_obs,
            )
            objects_cpp_params[object_type] = cpp_converter_config
        elif isinstance(object_config, PyWallConfig):
            cpp_wall_config = CppWallConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                swappable=object_config.swappable,
            )
            objects_cpp_params[object_type] = cpp_wall_config
        else:
            raise ValueError(f"Unknown object type: {object_type}")

    game_cpp_params = game_config.model_dump(exclude_none=True)
    del game_cpp_params["agent"]
    del game_cpp_params["groups"]

    # Convert global_obs configuration
    global_obs_config = game_config.global_obs
    global_obs_cpp = CppGlobalObsConfig(
        episode_completion_pct=global_obs_config.episode_completion_pct,
        last_action=global_obs_config.last_action,
        last_reward=global_obs_config.last_reward,
        resource_rewards=global_obs_config.resource_rewards,
    )
    game_cpp_params["global_obs"] = global_obs_cpp

    actions_cpp_params = {}
    for action_name, action_config in game_cpp_params["actions"].items():
        if not action_config["enabled"]:
            continue

        # Check if any consumed resources are not in inventory_item_names
        missing_consumed = []
        for resource in action_config["consumed_resources"].keys():
            if resource not in resource_name_to_id:
                missing_consumed.append(resource)

        if missing_consumed:
            raise ValueError(
                f"Action '{action_name}' has consumed_resources {missing_consumed} that are not in "
                f"inventory_item_names. These resources will be ignored, making the action free! "
                f"Either add these resources to inventory_item_names or disable the action."
            )

        action_cpp_params = {
            "consumed_resources": {
                resource_name_to_id[k]: v
                for k, v in action_config["consumed_resources"].items()
                if k in resource_name_to_id
            },
            "required_resources": {
                resource_name_to_id[k]: v
                for k, v in (action_config.get("required_resources") or action_config["consumed_resources"]).items()
                if k in resource_name_to_id
            },
        }

        if action_name == "attack":
            action_cpp_params["defense_resources"] = {
                resource_name_to_id[k]: v
                for k, v in action_config["defense_resources"].items()
                if k in resource_name_to_id
            }
            actions_cpp_params[action_name] = CppAttackActionConfig(**action_cpp_params)
        elif action_name == "change_glyph":
            action_cpp_params["number_of_glyphs"] = action_config["number_of_glyphs"]
            actions_cpp_params[action_name] = CppChangeGlyphActionConfig(**action_cpp_params)
        else:
            actions_cpp_params[action_name] = CppActionConfig(**action_cpp_params)

    game_cpp_params["actions"] = actions_cpp_params
    game_cpp_params["objects"] = objects_cpp_params

    # Add recipe_details_obs flag
    game_cpp_params["recipe_details_obs"] = game_config.recipe_details_obs

    return CppGameConfig(**game_cpp_params)


# Alias for backward compatibility
from_mettagrid_config = convert_to_cpp_game_config
