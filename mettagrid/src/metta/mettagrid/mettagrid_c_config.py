from metta.mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from metta.mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from metta.mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from metta.mettagrid.mettagrid_c import ChangeGlyphActionConfig as CppChangeGlyphActionConfig
from metta.mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from metta.mettagrid.mettagrid_c import GameConfig as CppGameConfig
from metta.mettagrid.mettagrid_c import GlobalObsConfig as CppGlobalObsConfig
from metta.mettagrid.mettagrid_c import WallConfig as CppWallConfig
from metta.mettagrid.mettagrid_config import AgentConfig, ConverterConfig, GameConfig, WallConfig


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def convert_to_cpp_game_config(mettagrid_config: dict | GameConfig):
    """Convert a GameConfig to a CppGameConfig."""
    if isinstance(mettagrid_config, GameConfig):
        # If it's already a GameConfig instance, convert to dict
        game_config = mettagrid_config
    else:
        # If it's a dict, instantiate a GameConfig from it
        # mettagrid_config needs special handling for map_builder
        game_config = GameConfig(**mettagrid_config)

    # Set up resource mappings
    resource_names = list(game_config.resource_names)
    resource_name_to_id = {name: i for i, name in enumerate(resource_names)}

    objects_cpp_params = {}  # params for CppConverterConfig or CppWallConfig

    # These are the baseline settings for all agents
    default_agent_config_dict = game_config.agent.model_dump()
    default_resource_limit = default_agent_config_dict["default_resource_limit"]

    # If no agents specified, create default agents with appropriate team IDs
    if not game_config.agents:
        # Create default agents that inherit from game_config.agent
        base_agent_dict = game_config.agent.model_dump()
        game_config.agents = []
        for _ in range(game_config.num_agents):
            agent_dict = base_agent_dict.copy()
            agent_dict["team_id"] = 0  # All default agents are on team 0
            game_config.agents.append(AgentConfig(**agent_dict))

    # Group agents by team_id to create groups
    team_groups = {}
    for agent_idx, agent_config in enumerate(game_config.agents):
        team_id = agent_config.team_id
        if team_id not in team_groups:
            team_groups[team_id] = []
        team_groups[team_id].append((agent_idx, agent_config))

    # Create a group for each team
    for team_id, team_agents in team_groups.items():
        # Use the first agent in the team as the template for the group
        _, first_agent = team_agents[0]
        agent_props = first_agent.model_dump()

        rewards_config = agent_props.get("rewards", {})
        inventory_rewards = {
            resource_name_to_id[k]: v
            for k, v in rewards_config.get("inventory", {}).items()
            if k in resource_name_to_id
        }
        inventory_reward_max = {
            resource_name_to_id[k]: v
            for k, v in rewards_config.get("inventory_max", {}).items()
            if k in resource_name_to_id
        }

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

        # Process potential initial inventory
        initial_inventory = {}
        for k, v in agent_props["initial_inventory"].items():
            initial_inventory[resource_name_to_id[k]] = v

        # Map team IDs to conventional group names
        team_names = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "purple", 5: "orange"}
        group_name = team_names.get(team_id, f"team_{team_id}")
        agent_cpp_params = {
            "freeze_duration": agent_props["freeze_duration"],
            "group_id": team_id,
            "group_name": group_name,
            "action_failure_penalty": agent_props["action_failure_penalty"],
            "resource_limits": {
                resource_id: agent_props["resource_limits"].get(resource_name, default_resource_limit)
                for resource_id, resource_name in enumerate(resource_names)
            },
            "resource_rewards": inventory_rewards,
            "resource_reward_max": inventory_reward_max,
            "stat_rewards": stat_rewards,
            "stat_reward_max": stat_reward_max,
            "group_reward_pct": 0.0,  # Default to 0 for direct agents
            "type_id": 0,
            "type_name": "agent",
            "initial_inventory": initial_inventory,
        }

        objects_cpp_params["agent." + group_name] = CppAgentConfig(**agent_cpp_params)

        # Also register team_X naming convention for maps that use it
        objects_cpp_params[f"agent.team_{team_id}"] = CppAgentConfig(**agent_cpp_params)

        # Also register aliases for team 0 for backward compatibility
        if team_id == 0:
            objects_cpp_params["agent.default"] = CppAgentConfig(**agent_cpp_params)
            objects_cpp_params["agent.agent"] = CppAgentConfig(**agent_cpp_params)

    # Convert other objects
    for object_type, object_config in game_config.objects.items():
        if isinstance(object_config, ConverterConfig):
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
                max_conversions=object_config.max_conversions,
                conversion_ticks=object_config.conversion_ticks,
                cooldown=object_config.cooldown,
                initial_resource_count=object_config.initial_resource_count,
                color=object_config.color,
                recipe_details_obs=game_config.recipe_details_obs,
            )
            objects_cpp_params[object_type] = cpp_converter_config
        elif isinstance(object_config, WallConfig):
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
    if "agents" in game_cpp_params:
        del game_cpp_params["agents"]
    if "params" in game_cpp_params:
        del game_cpp_params["params"]
    if "map_builder" in game_cpp_params:
        del game_cpp_params["map_builder"]

    # Convert global_obs configuration
    global_obs_config = game_config.global_obs
    global_obs_cpp = CppGlobalObsConfig(
        episode_completion_pct=global_obs_config.episode_completion_pct,
        last_action=global_obs_config.last_action,
        last_reward=global_obs_config.last_reward,
        resource_rewards=global_obs_config.resource_rewards,
        visitation_counts=global_obs_config.visitation_counts,
    )
    game_cpp_params["global_obs"] = global_obs_cpp

    actions_cpp_params = {}
    for action_name, action_config in game_cpp_params["actions"].items():
        if not action_config["enabled"]:
            continue

        # Check if any consumed resources are not in resource_names
        missing_consumed = []
        for resource in action_config["consumed_resources"].keys():
            if resource not in resource_name_to_id:
                missing_consumed.append(resource)

        if missing_consumed:
            raise ValueError(
                f"Action '{action_name}' has consumed_resources {missing_consumed} that are not in "
                f"resource_names. These resources will be ignored, making the action free! "
                f"Either add these resources to resource_names or disable the action."
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
            # Extract the specific parameters needed for ChangeGlyphActionConfig
            change_glyph_params = {
                "required_resources": action_cpp_params.get("required_resources", {}),
                "consumed_resources": action_cpp_params.get("consumed_resources", {}),
                "number_of_glyphs": action_config["number_of_glyphs"],
            }
            actions_cpp_params[action_name] = CppChangeGlyphActionConfig(**change_glyph_params)
        else:
            actions_cpp_params[action_name] = CppActionConfig(**action_cpp_params)

    game_cpp_params["actions"] = actions_cpp_params
    game_cpp_params["objects"] = objects_cpp_params

    # Add resource_loss_prob
    game_cpp_params["resource_loss_prob"] = game_config.resource_loss_prob

    # Set feature flags
    game_cpp_params["recipe_details_obs"] = game_config.recipe_details_obs
    game_cpp_params["allow_diagonals"] = game_config.allow_diagonals
    game_cpp_params["track_movement_metrics"] = game_config.track_movement_metrics

    return CppGameConfig(**game_cpp_params)


# Alias for backward compatibility
from_mettagrid_config = convert_to_cpp_game_config
