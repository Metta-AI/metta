import math
from typing import Sequence

from mettagrid.config.mettagrid_config import (
    AgentConfig,
    AssemblerConfig,
    ChestConfig,
    ClipperConfig,
    ConverterConfig,
    GameConfig,
    Position,
    WallConfig,
)
from mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from mettagrid.mettagrid_c import AssemblerConfig as CppAssemblerConfig
from mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from mettagrid.mettagrid_c import ChangeGlyphActionConfig as CppChangeGlyphActionConfig
from mettagrid.mettagrid_c import ChestConfig as CppChestConfig
from mettagrid.mettagrid_c import ClipperConfig as CppClipperConfig
from mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from mettagrid.mettagrid_c import GameConfig as CppGameConfig
from mettagrid.mettagrid_c import GlobalObsConfig as CppGlobalObsConfig
from mettagrid.mettagrid_c import InventoryConfig as CppInventoryConfig
from mettagrid.mettagrid_c import Recipe as CppRecipe
from mettagrid.mettagrid_c import ResourceModConfig as CppResourceModConfig
from mettagrid.mettagrid_c import WallConfig as CppWallConfig

# Note that these are left to right, top to bottom.
FIXED_POSITIONS: list[Position] = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]
FIXED_POSITION_TO_BITMASK = {pos: 1 << i for i, pos in enumerate(FIXED_POSITIONS)}


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def expand_position_patterns(positions: Sequence[Position]) -> list[int]:
    """Convert from a list of string positions to a list of matching bit patterns.

    Args:
        positions: List of position strings like ["N", "Any"]
        "Any" means exactly one agent in any position
        Other positions mean exactly one agent in that specific position

    Returns:
        List of bit patterns that match the position requirements
    """

    fix_positions_byte = 0
    has_any = False
    for pos in positions:
        if pos == "Any":
            has_any = True
        else:
            assert pos in FIXED_POSITIONS, f"Invalid position: {pos}"
            position_bit = FIXED_POSITION_TO_BITMASK[pos]
            assert fix_positions_byte & position_bit == 0, (
                f"Position {pos} already set. Only one agent per position is allowed."
            )
            fix_positions_byte |= position_bit

    if not has_any:
        return [fix_positions_byte]

    result = []
    # Not the most elegant solution, but there are only 8 positions, so it's not too bad.
    # We're just iterating over all possible bit patterns and seeing which ones
    # (a) have the right fixed positions, and (b) have the right number of total agents (which would be fixed + "Any")
    for i in range(256):
        if i & fix_positions_byte != fix_positions_byte:
            continue
        if bin(i).count("1") == len(positions):
            result.append(i)
    return result


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

    # Build tag mappings - collect all unique tags from all objects
    # Note: This must happen AFTER default agents are created, so their tags are included
    all_tags = set()
    for obj_config in game_config.objects.values():
        all_tags.update(obj_config.tags)

    # Also collect tags from agents
    for agent_config in game_config.agents:
        all_tags.update(agent_config.tags)

    tag_id_offset = 0  # Start tag IDs at 0
    sorted_tags = sorted(all_tags)

    # Validate tag count doesn't exceed uint8 max (255)
    if len(sorted_tags) > 256:
        raise ValueError(f"Too many unique tags ({len(sorted_tags)}). Maximum supported is 256 due to uint8 limit.")

    tag_name_to_id = {tag: tag_id_offset + i for i, tag in enumerate(sorted_tags)}
    tag_id_to_name = {id: name for name, id in tag_name_to_id.items()}

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

        # Validate that all agents in the team have identical tags
        # Currently tags are applied per-team, not per-agent
        first_agent_tags = set(first_agent.tags)
        for agent_idx, agent_config in team_agents[1:]:
            if set(agent_config.tags) != first_agent_tags:
                raise ValueError(
                    f"All agents in team {team_id} must have identical tags. "
                    f"Agent 0 has tags {sorted(first_agent_tags)}, "
                    f"but agent {agent_idx} has tags {sorted(agent_config.tags)}. "
                    f"Tags are currently applied per-team, not per-agent."
                )

        rewards_config = agent_props.get("rewards", {})

        # Process stats rewards
        stat_rewards = rewards_config.get("stats", {})
        stat_reward_max = rewards_config.get("stats_max", {})

        for k, v in rewards_config.get("inventory", {}).items():
            assert k in resource_name_to_id, f"Inventory reward {k} not in resource_names"
            stat_name = k + ".amount"
            assert stat_name not in stat_rewards, f"Stat reward {stat_name} already exists"
            stat_rewards[stat_name] = v
        for k, v in rewards_config.get("inventory_max", {}).items():
            assert k in resource_name_to_id, f"Inventory reward max {k} not in resource_names"
            stat_name = k + ".amount"
            assert stat_name not in stat_reward_max, f"Stat reward max {stat_name} already exists"
            stat_reward_max[stat_name] = v

        # Process potential initial inventory
        initial_inventory = {}
        for k, v in agent_props["initial_inventory"].items():
            initial_inventory[resource_name_to_id[k]] = v

        # Map team IDs to conventional group names
        team_names = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "purple", 5: "orange"}
        group_name = team_names.get(team_id, f"team_{team_id}")
        # Convert tag names to IDs for first agent in team
        tag_ids = [tag_name_to_id[tag] for tag in first_agent.tags]

        # Convert soul bound resources from names to IDs
        soul_bound_resources = [
            resource_name_to_id[resource_name] for resource_name in agent_props.get("soul_bound_resources", [])
        ]

        # Convert shareable resources from names to IDs
        shareable_resources = [
            resource_name_to_id[resource_name] for resource_name in agent_props.get("shareable_resources", [])
        ]

        # Convert inventory regeneration amounts from names to IDs
        inventory_regen_amounts = {}
        for resource_name, amount in agent_props.get("inventory_regen_amounts", {}).items():
            inventory_regen_amounts[resource_name_to_id[resource_name]] = amount

        # Build inventory config with support for grouped limits
        limits_list = []

        # First, handle explicitly configured limits (both individual and grouped)
        configured_resources = set()
        for key, limit_value in agent_props["resource_limits"].items():
            if isinstance(key, str):
                # Single resource limit
                limits_list.append([[resource_name_to_id[key]], limit_value])
                configured_resources.add(key)
            elif isinstance(key, tuple):
                # Grouped resources with shared limit
                resource_ids = [resource_name_to_id[name] for name in key]
                if resource_ids:
                    limits_list.append([resource_ids, limit_value])
                    configured_resources.update(key)

        # Add default limits for unconfigured resources
        for resource_name in resource_names:
            if resource_name not in configured_resources:
                limits_list.append([[resource_name_to_id[resource_name]], default_resource_limit])

        inventory_config = CppInventoryConfig(limits=limits_list)

        agent_cpp_params = {
            "freeze_duration": agent_props["freeze_duration"],
            "group_id": team_id,
            "group_name": group_name,
            "action_failure_penalty": agent_props["action_failure_penalty"],
            "inventory_config": inventory_config,
            "stat_rewards": stat_rewards,
            "stat_reward_max": stat_reward_max,
            "group_reward_pct": 0.0,  # Default to 0 for direct agents
            "type_id": 0,
            "type_name": "agent",
            "initial_inventory": initial_inventory,
            "tag_ids": tag_ids,
            "soul_bound_resources": soul_bound_resources,
            "shareable_resources": shareable_resources,
            "inventory_regen_amounts": inventory_regen_amounts,
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
            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            cpp_converter_config = CppConverterConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                input_resources={resource_name_to_id[k]: v for k, v in object_config.input_resources.items()},
                output_resources={resource_name_to_id[k]: v for k, v in object_config.output_resources.items()},
                max_output=object_config.max_output,
                max_conversions=object_config.max_conversions,
                conversion_ticks=object_config.conversion_ticks,
                cooldown_time=list(object_config.cooldown),
                initial_resource_count=object_config.initial_resource_count,
                recipe_details_obs=game_config.recipe_details_obs,
                tag_ids=tag_ids,
            )
            objects_cpp_params[object_type] = cpp_converter_config
        elif isinstance(object_config, WallConfig):
            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            cpp_wall_config = CppWallConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                swappable=object_config.swappable,
                tag_ids=tag_ids,
            )
            objects_cpp_params[object_type] = cpp_wall_config
        elif isinstance(object_config, AssemblerConfig):
            # Convert recipes with position patterns to C++ recipes
            # Create a mapping from byte patterns to recipes
            recipe_map = {}  # byte_pattern -> CppRecipe

            for position_pattern, recipe_config in reversed(object_config.recipes):
                # Expand position patterns to byte patterns
                bit_patterns = expand_position_patterns(position_pattern)

                # Create C++ recipe
                cpp_recipe = CppRecipe(
                    input_resources={resource_name_to_id[k]: v for k, v in recipe_config.input_resources.items()},
                    output_resources={resource_name_to_id[k]: v for k, v in recipe_config.output_resources.items()},
                    cooldown=recipe_config.cooldown,
                )

                # Map this recipe to all matching byte patterns
                recipe_has_active_pattern = False
                for bit_pattern in bit_patterns:
                    if bit_pattern not in recipe_map:
                        recipe_map[bit_pattern] = cpp_recipe
                        recipe_has_active_pattern = True
                if not recipe_has_active_pattern and not object_config.fully_overlapping_recipes_allowed:
                    raise ValueError(
                        f"Recipe {recipe_config} has no valid cog patterns not already claimed by other recipes."
                    )

            # Create a vector of 256 Recipe pointers (indexed by byte pattern)
            cpp_recipes = [None] * 256
            for byte_pattern, recipe in recipe_map.items():
                cpp_recipes[byte_pattern] = recipe

            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            cpp_assembler_config = CppAssemblerConfig(
                type_id=object_config.type_id, type_name=object_type, tag_ids=tag_ids
            )
            cpp_assembler_config.recipes = cpp_recipes
            cpp_assembler_config.allow_partial_usage = object_config.allow_partial_usage
            cpp_assembler_config.max_uses = object_config.max_uses
            cpp_assembler_config.exhaustion = object_config.exhaustion
            cpp_assembler_config.clip_immune = object_config.clip_immune
            cpp_assembler_config.start_clipped = object_config.start_clipped
            objects_cpp_params[object_type] = cpp_assembler_config
        elif isinstance(object_config, ChestConfig):
            # Convert resource type name to ID
            resource_type_id = resource_name_to_id.get(object_config.resource_type, 0)

            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            # Convert position_deltas from (FixedPosition, delta) to (position_index, delta)
            position_deltas_map = {}
            for pos, delta in object_config.position_deltas:
                position_index = FIXED_POSITIONS.index(pos)
                position_deltas_map[position_index] = delta

            cpp_chest_config = CppChestConfig(
                type_id=object_config.type_id,
                type_name=object_type,
                resource_type=resource_type_id,
                position_deltas=position_deltas_map,
                initial_inventory=object_config.initial_inventory,
                max_inventory=object_config.max_inventory,
                tag_ids=tag_ids,
            )
            objects_cpp_params[object_type] = cpp_chest_config
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

        consumed_resources = {resource_name_to_id[k]: float(v) for k, v in action_config["consumed_resources"].items()}

        required_source = action_config.get("required_resources")
        if not required_source:
            required_source = {k: math.ceil(v) for k, v in action_config["consumed_resources"].items()}

        required_resources = {resource_name_to_id[k]: int(math.ceil(v)) for k, v in required_source.items()}

        action_cpp_params = {
            "consumed_resources": consumed_resources,
            "required_resources": required_resources,
        }

        if action_name == "attack":
            action_cpp_params["defense_resources"] = {
                resource_name_to_id[k]: v for k, v in action_config["defense_resources"].items()
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
        elif action_name == "resource_mod":
            # Extract the specific parameters needed for ResourceModConfig
            modifies_dict = action_config.get("modifies", {})
            unknown_modifies = set(modifies_dict.keys()) - set(resource_name_to_id.keys())
            if unknown_modifies:
                unknown_list = sorted(unknown_modifies)
                raise ValueError(f"Unknown resource names in modifies for action '{action_name}': {unknown_list}")

            resource_mod_params = {
                "required_resources": action_cpp_params.get("required_resources", {}),
                "consumed_resources": action_cpp_params.get("consumed_resources", {}),
                "modifies": {resource_name_to_id[k]: float(v) for k, v in modifies_dict.items()},
                "agent_radius": action_config.get("agent_radius", 0),
                "converter_radius": action_config.get("converter_radius", 0),
                "scales": action_config.get("scales", False),
            }
            actions_cpp_params[action_name] = CppResourceModConfig(**resource_mod_params)
        else:
            actions_cpp_params[action_name] = CppActionConfig(**action_cpp_params)

    # Convert actions_cpp_params dict to an ordered list of (name, config) pairs
    # Ensure "noop" is always at index 0 if present
    action_pairs = []
    if "noop" in actions_cpp_params:
        action_pairs.append(("noop", actions_cpp_params["noop"]))

    # Add remaining actions in their original order
    for action_name, action_config in actions_cpp_params.items():
        if action_name != "noop":
            action_pairs.append((action_name, action_config))

    game_cpp_params["actions"] = action_pairs
    game_cpp_params["objects"] = objects_cpp_params

    # Add resource_loss_prob
    game_cpp_params["resource_loss_prob"] = game_config.resource_loss_prob

    # Add inventory regeneration interval
    game_cpp_params["inventory_regen_interval"] = game_config.inventory_regen_interval

    # Add clipper if configured
    if game_config.clipper is not None:
        clipper: ClipperConfig = game_config.clipper
        clipper_recipes = []
        for recipe_config in clipper.unclipping_recipes:
            cpp_recipe = CppRecipe(
                input_resources={resource_name_to_id[k]: v for k, v in recipe_config.input_resources.items()},
                output_resources={resource_name_to_id[k]: v for k, v in recipe_config.output_resources.items()},
                cooldown=recipe_config.cooldown,
            )
            clipper_recipes.append(cpp_recipe)
        game_cpp_params["clipper"] = CppClipperConfig(
            clipper_recipes, clipper.length_scale, clipper.cutoff_distance, clipper.clip_rate
        )

    # Set feature flags
    game_cpp_params["recipe_details_obs"] = game_config.recipe_details_obs
    game_cpp_params["allow_diagonals"] = game_config.allow_diagonals
    game_cpp_params["track_movement_metrics"] = game_config.track_movement_metrics

    # Add tag mappings for C++ debugging/display
    game_cpp_params["tag_id_map"] = tag_id_to_name

    return CppGameConfig(**game_cpp_params)


# Alias for backward compatibility
from_mettagrid_config = convert_to_cpp_game_config
