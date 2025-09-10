import re

from metta.mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from metta.mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from metta.mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from metta.mettagrid.mettagrid_c import BoxConfig as CppBoxConfig
from metta.mettagrid.mettagrid_c import ChangeGlyphActionConfig as CppChangeGlyphActionConfig
from metta.mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from metta.mettagrid.mettagrid_c import GameConfig as CppGameConfig
from metta.mettagrid.mettagrid_c import GlobalObsConfig as CppGlobalObsConfig
from metta.mettagrid.mettagrid_c import WallConfig as CppWallConfig
from metta.mettagrid.mettagrid_config import AgentConfig, BoxConfig, ConverterConfig, GameConfig, WallConfig


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# Validate tag names (alphanumeric and underscore only)
_tag_pattern = re.compile(r"^[A-Za-z0-9_]+$")


def parse_object_with_tags(object_name: str) -> tuple[str, list[str]]:
    """Parse an object name that may include tags using dot notation."""

    parts = object_name.split(".")
    base_type = parts[0]
    tags = parts[1:] if len(parts) > 1 else []

    # Check maximum tags per object limit
    MAX_TAGS_PER_OBJECT = 10
    if len(tags) > MAX_TAGS_PER_OBJECT:
        raise ValueError(
            f"Object '{object_name}' has {len(tags)} tags, exceeding the maximum of "
            f"{MAX_TAGS_PER_OBJECT} tags per object. Consider reducing the number of "
            f"tags or combining them into composite tags."
        )

    for tag in tags:
        if not _tag_pattern.match(tag):
            raise ValueError(
                f"Invalid tag name '{tag}': tags must contain only alphanumeric characters and underscores"
            )

    return base_type, tags


def apply_tag_overrides(base_config: dict, tags: list[str], game_config: GameConfig, model_class=None) -> dict:
    """Apply tag overrides to a base configuration."""
    result_config = base_config.copy()

    # Get allowed fields if model class is provided
    allowed_fields = None
    if model_class:
        allowed_fields = set(model_class.model_fields.keys())

    for tag in tags:
        if tag in game_config.tags:
            tag_config = game_config.tags[tag]
            if tag_config.overrides:
                # Filter overrides to allowed fields if model class provided
                if allowed_fields:
                    filtered_overrides = {}
                    invalid_fields = []
                    for key, value in tag_config.overrides.items():
                        if key in allowed_fields:
                            filtered_overrides[key] = value
                        else:
                            invalid_fields.append(key)

                    if invalid_fields:
                        model_name = model_class.__name__
                        raise ValueError(
                            f"Tag '{tag}' contains invalid fields for {model_name}: {', '.join(invalid_fields)}. "
                            f"Allowed fields are: {', '.join(sorted(allowed_fields))}"
                        )

                    result_config = recursive_update(result_config, filtered_overrides)
                else:
                    result_config = recursive_update(result_config, tag_config.overrides)

    return result_config


def convert_to_cpp_game_config(mettagrid_config: dict | GameConfig, map_data: list[list[str]] | None = None):
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

    # Collect all tags for deterministic pre-registration
    all_tags = set()

    # Add tags from config
    all_tags.update(game_config.tags.keys())

    # Add tags discovered from map data
    if map_data:
        for row in map_data:
            for cell in row:
                if "." in cell:
                    # Parse tags from object names
                    parts = cell.split(".")
                    if cell.startswith("agent.agent."):
                        # Special case for agent.agent.tag1.tag2
                        agent_tags = parts[2:]
                        # Validate tag names before adding
                        for tag in agent_tags:
                            if not _tag_pattern.match(tag):
                                raise ValueError(
                                    f"Invalid tag name '{tag}' in object '{cell}': "
                                    f"tags must contain only alphanumeric characters and underscores"
                                )
                        all_tags.update(agent_tags)
                    elif cell.startswith("agent."):
                        # Could be agent.subtype or agent.subtype.tag1.tag2
                        # Skip the first part (agent) and check if second is a subtype
                        valid_subtypes = {"team_1", "team_2", "team_3", "team_4", "prey", "predator", "agent"}
                        agent_tags = []
                        if len(parts) > 1 and parts[1] not in valid_subtypes:
                            # It's a tag, not a subtype
                            agent_tags = parts[1:]
                        elif len(parts) > 2:
                            # Has subtype and tags
                            agent_tags = parts[2:]

                        # Validate tag names before adding
                        for tag in agent_tags:
                            if not _tag_pattern.match(tag):
                                raise ValueError(
                                    f"Invalid tag name '{tag}' in object '{cell}': "
                                    f"tags must contain only alphanumeric characters and underscores"
                                )
                        all_tags.update(agent_tags)
                    else:
                        # Regular object with tags
                        object_tags = parts[1:]
                        # Validate tag names before adding
                        for tag in object_tags:
                            if not _tag_pattern.match(tag):
                                raise ValueError(
                                    f"Invalid tag name '{tag}' in object '{cell}': "
                                    f"tags must contain only alphanumeric characters and underscores"
                                )
                        all_tags.update(object_tags)

    # Sort tags deterministically for consistent feature ID assignment
    sorted_tags = sorted(all_tags)

    # Note: Feature space validation is handled by C++ ObservationEncoder::register_tag
    # which knows the exact layout of feature IDs and can accurately detect overflow.
    # We don't duplicate that logic here to avoid maintaining two sources of truth.

    # If map data is provided, scan for tagged objects and create configs for them
    if map_data:
        tagged_objects = set()
        for row in map_data:
            for cell in row:
                if "." in cell and not cell.startswith("agent"):
                    # This is a tagged object like "converter.red.fast"
                    tagged_objects.add(cell)

        # Track all used type_ids to avoid conflicts
        # Also track which object uses each ID for better error messages
        used_type_ids = {}  # type_id -> object_name mapping
        max_type_id = 0

        # Collect type IDs from existing game config objects
        for obj_name, obj_config in game_config.objects.items():
            if hasattr(obj_config, "type_id"):
                used_type_ids[obj_config.type_id] = obj_name
                if obj_config.type_id > max_type_id:
                    max_type_id = obj_config.type_id

        # Create configs for tagged objects with overrides applied
        next_type_id = max_type_id + 1
        for tagged_obj in tagged_objects:
            base_type, tags = parse_object_with_tags(tagged_obj)

            # Skip if base type doesn't exist in objects
            if base_type not in game_config.objects:
                continue

            # Get base config and apply tag overrides
            base_config = game_config.objects[base_type]
            base_dict = base_config.model_dump()

            # Determine the model class for filtering
            if isinstance(base_config, ConverterConfig):
                model_class = ConverterConfig
            elif isinstance(base_config, WallConfig):
                model_class = WallConfig
            elif isinstance(base_config, BoxConfig):
                model_class = BoxConfig
            else:
                model_class = None

            overridden_dict = apply_tag_overrides(base_dict, tags, game_config, model_class)

            # Handle type_id: use override if provided, otherwise allocate next available
            if "type_id" in overridden_dict and overridden_dict["type_id"] != base_dict.get("type_id"):
                # Tag override specifies a type_id
                override_type_id = overridden_dict["type_id"]

                # Validate it's in valid range
                if not 0 <= override_type_id <= 255:
                    raise ValueError(
                        f"Invalid type_id={override_type_id} for tagged object '{tagged_obj}'. "
                        f"Type ID must be in range [0, 255] (ObservationType limit)."
                    )

                # Check for conflicts with all used type_ids
                if override_type_id in used_type_ids:
                    conflicting_obj = used_type_ids[override_type_id]
                    raise ValueError(
                        f"Type ID conflict: Tagged object '{tagged_obj}' tries to use type_id={override_type_id}, "
                        f"but it's already used by '{conflicting_obj}'. Each object must have a unique type_id."
                    )

                used_type_ids[override_type_id] = tagged_obj
            else:
                # No override specified, allocate next available type_id
                if next_type_id > 255:
                    raise ValueError(
                        f"Type ID overflow: Cannot create tagged object '{tagged_obj}' with type_id={next_type_id}. "
                        f"Maximum allowed type_id is 255 (ObservationType limit). "
                        f"Consider reducing the number of unique object types or tagged variants."
                    )

                overridden_dict["type_id"] = next_type_id
                used_type_ids[next_type_id] = tagged_obj
                next_type_id += 1

            # Create new config object of the same type with overrides
            if isinstance(base_config, ConverterConfig):
                game_config.objects[tagged_obj] = ConverterConfig(**overridden_dict)
            elif isinstance(base_config, WallConfig):
                game_config.objects[tagged_obj] = WallConfig(**overridden_dict)
            elif isinstance(base_config, BoxConfig):
                game_config.objects[tagged_obj] = BoxConfig(**overridden_dict)

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

    # Process all objects including those with tags (e.g., "converter.red.fast")
    for object_type, object_config in game_config.objects.items():
        # For base objects, just convert them directly
        # Tagged versions will be created dynamically when encountered in maps
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
        elif isinstance(object_config, BoxConfig):
            returned_resources = game_config.actions.place_box.consumed_resources
            cpp_box_config = CppBoxConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                returned_resources={
                    resource_name_to_id[k]: v for k, v in returned_resources.items() if k in resource_name_to_id
                },
            )
            objects_cpp_params[object_type] = cpp_box_config
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
    if "tags" in game_cpp_params:
        del game_cpp_params["tags"]

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

    # Add sorted tags for deterministic pre-registration
    game_cpp_params["tags"] = sorted_tags

    # Set feature flags
    game_cpp_params["recipe_details_obs"] = game_config.recipe_details_obs
    game_cpp_params["allow_diagonals"] = game_config.allow_diagonals
    game_cpp_params["track_movement_metrics"] = game_config.track_movement_metrics

    return CppGameConfig(**game_cpp_params)


# Alias for backward compatibility
def from_mettagrid_config(mettagrid_config: dict | GameConfig, map_data: list[list[str]] | None = None):
    return convert_to_cpp_game_config(mettagrid_config, map_data)
