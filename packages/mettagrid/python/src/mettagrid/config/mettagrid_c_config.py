import math

from mettagrid.config.mettagrid_config import (
    AgentConfig,
    AssemblerConfig,
    ChestConfig,
    ClipperConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.config.vibes import VIBES
from mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from mettagrid.mettagrid_c import AssemblerConfig as CppAssemblerConfig
from mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from mettagrid.mettagrid_c import ChangeVibeActionConfig as CppChangeVibeActionConfig
from mettagrid.mettagrid_c import ChestConfig as CppChestConfig
from mettagrid.mettagrid_c import ClipperConfig as CppClipperConfig
from mettagrid.mettagrid_c import GameConfig as CppGameConfig
from mettagrid.mettagrid_c import GlobalObsConfig as CppGlobalObsConfig
from mettagrid.mettagrid_c import InventoryConfig as CppInventoryConfig
from mettagrid.mettagrid_c import MoveActionConfig as CppMoveActionConfig
from mettagrid.mettagrid_c import Protocol as CppProtocol
from mettagrid.mettagrid_c import ResourceModConfig as CppResourceModConfig
from mettagrid.mettagrid_c import WallConfig as CppWallConfig


def convert_to_cpp_game_config(mettagrid_config: dict | GameConfig):
    """Convert a GameConfig to a CppGameConfig."""
    if isinstance(mettagrid_config, GameConfig):
        # If it's already a GameConfig instance, use it directly
        game_config = mettagrid_config
    else:
        # If it's a dict, remove computed fields before instantiating GameConfig
        # features is a computed field and can't be set during __init__
        config_dict = mettagrid_config.copy()
        if "obs" in config_dict and "features" in config_dict["obs"]:
            config_dict["obs"] = config_dict["obs"].copy()
            config_dict["obs"].pop("features", None)
        game_config = GameConfig(**config_dict)

    # Ensure type IDs are assigned even if objects were added/modified after construction
    # This mirrors the behavior documented in GameConfig._resolve_object_type_ids.
    try:
        game_config._resolve_object_type_ids()
    except Exception:
        # Best-effort; if this fails for any reason, let downstream code surface errors
        pass

    # Set up resource mappings
    resource_names = list(game_config.resource_names)
    resource_name_to_id = {name: i for i, name in enumerate(resource_names)}

    # Set up vibe mappings from the change_vibe action config.
    # The C++ bindings expect dense uint8 identifiers, so keep a name->id lookup.
    num_vibes = game_config.actions.change_vibe.number_of_vibes
    supported_vibes = VIBES[:num_vibes]
    if not game_config.vibe_names:
        game_config.vibe_names = [vibe.name for vibe in supported_vibes]
    vibe_name_to_id = {vibe.name: i for i, vibe in enumerate(supported_vibes)}

    objects_cpp_params = {}  # params for CppWallConfig

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

        diversity_tracked_resources = [
            resource_name_to_id[resource_name]
            for resource_name in agent_props.get("diversity_tracked_resources", [])
            if resource_name in resource_name_to_id
        ]

        # Build inventory config with support for grouped limits
        limits_list = []

        # First, handle explicitly configured limits (both individual and grouped)
        configured_resources = set()
        for key, limit_value in agent_props["resource_limits"].items():
            if isinstance(key, str):
                # Single resource limit
                limits_list.append(([resource_name_to_id[key]], limit_value))
                configured_resources.add(key)
            elif isinstance(key, tuple):
                # Grouped resources with shared limit
                resource_ids = [resource_name_to_id[name] for name in key]
                if resource_ids:
                    limits_list.append((resource_ids, limit_value))
                    configured_resources.update(key)

        # Add default limits for unconfigured resources
        for resource_name in resource_names:
            if resource_name not in configured_resources:
                limits_list.append(([resource_name_to_id[resource_name]], default_resource_limit))

        inventory_config = CppInventoryConfig(limits=limits_list)

        cpp_agent_config = CppAgentConfig(
            type_id=0,
            type_name="agent",
            group_id=team_id,
            group_name=group_name,
            freeze_duration=agent_props["freeze_duration"],
            action_failure_penalty=agent_props["action_failure_penalty"],
            inventory_config=inventory_config,
            stat_rewards=stat_rewards,
            stat_reward_max=stat_reward_max,
            group_reward_pct=0.0,
            initial_inventory=initial_inventory,
            soul_bound_resources=soul_bound_resources,
            shareable_resources=shareable_resources,
            inventory_regen_amounts=inventory_regen_amounts,
            diversity_tracked_resources=diversity_tracked_resources,
        )
        cpp_agent_config.tag_ids = tag_ids

        objects_cpp_params["agent." + group_name] = cpp_agent_config

        # Also register team_X naming convention for maps that use it
        objects_cpp_params[f"agent.team_{team_id}"] = cpp_agent_config

        # Also register aliases for team 0 for backward compatibility
        if team_id == 0:
            objects_cpp_params["agent.default"] = cpp_agent_config
            objects_cpp_params["agent.agent"] = cpp_agent_config

    # Convert other objects
    for object_type, object_config in game_config.objects.items():
        if isinstance(object_config, WallConfig):
            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            cpp_wall_config = CppWallConfig(
                type_id=object_config.type_id, type_name=object_type, initial_vibe=object_config.vibe
            )
            cpp_wall_config.swappable = object_config.swappable
            cpp_wall_config.tag_ids = tag_ids
            objects_cpp_params[object_type] = cpp_wall_config
        elif isinstance(object_config, AssemblerConfig):
            protocols = []
            seen_vibes = []

            for protocol_config in reversed(object_config.protocols):
                # Convert vibe names to IDs
                vibe_ids = sorted([vibe_name_to_id[vibe] for vibe in protocol_config.vibes])
                # Check for duplicate vibes
                if vibe_ids in seen_vibes:
                    raise ValueError(f"Protocol with vibes {protocol_config.vibes} already exists in {object_type}")
                seen_vibes.append(vibe_ids)
                input_res = {resource_name_to_id[k]: int(v) for k, v in protocol_config.input_resources.items()}
                output_res = {resource_name_to_id[k]: int(v) for k, v in protocol_config.output_resources.items()}
                cpp_protocol = CppProtocol()
                cpp_protocol.vibes = vibe_ids
                cpp_protocol.input_resources = input_res
                cpp_protocol.output_resources = output_res
                cpp_protocol.cooldown = protocol_config.cooldown
                protocols.append(cpp_protocol)

            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            cpp_assembler_config = CppAssemblerConfig(
                type_id=object_config.type_id, type_name=object_type, initial_vibe=object_config.vibe
            )
            cpp_assembler_config.tag_ids = tag_ids
            cpp_assembler_config.protocols = protocols
            cpp_assembler_config.allow_partial_usage = object_config.allow_partial_usage
            cpp_assembler_config.max_uses = object_config.max_uses
            cpp_assembler_config.exhaustion = object_config.exhaustion
            cpp_assembler_config.clip_immune = object_config.clip_immune
            cpp_assembler_config.start_clipped = object_config.start_clipped
            objects_cpp_params[object_type] = cpp_assembler_config
        elif isinstance(object_config, ChestConfig):
            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            # Convert vibe_transfers: vibe -> resource -> delta
            vibe_transfers_map = {}
            for vibe_name, resource_deltas in object_config.vibe_transfers.items():
                vibe_id = vibe_name_to_id[vibe_name]
                resource_deltas_cpp = {
                    resource_name_to_id[resource]: delta for resource, delta in resource_deltas.items()
                }
                vibe_transfers_map[vibe_id] = resource_deltas_cpp

            # Convert initial inventory
            initial_inventory_cpp = {}
            for resource, amount in object_config.initial_inventory.items():
                resource_id = resource_name_to_id[resource]
                initial_inventory_cpp[resource_id] = amount

            # Create inventory config with limits
            limits_list = []
            for resource, limit in object_config.resource_limits.items():
                resource_id = resource_name_to_id[resource]
                limits_list.append([[resource_id], limit])

            inventory_config = CppInventoryConfig(limits=limits_list)

            cpp_chest_config = CppChestConfig(
                type_id=object_config.type_id, type_name=object_type, initial_vibe=object_config.vibe
            )
            cpp_chest_config.vibe_transfers = vibe_transfers_map
            cpp_chest_config.initial_inventory = initial_inventory_cpp
            cpp_chest_config.inventory_config = inventory_config
            cpp_chest_config.tag_ids = tag_ids
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

    # Extract obs config to top level for C++ compatibility
    if "obs" in game_cpp_params:
        obs_config = game_cpp_params.pop("obs")
        game_cpp_params["obs_width"] = obs_config["width"]
        game_cpp_params["obs_height"] = obs_config["height"]
        game_cpp_params["num_observation_tokens"] = obs_config["num_tokens"]
        # Note: token_dim is not used by C++ GameConfig, it's only used in Python

    # Convert observation features from Python to C++
    # Use id_map to get feature_ids
    id_map = game_config.id_map()
    game_cpp_params["feature_ids"] = {feature.name: feature.id for feature in id_map.features()}

    # Convert global_obs configuration
    global_obs_config = game_config.global_obs
    global_obs_cpp = CppGlobalObsConfig(
        episode_completion_pct=global_obs_config.episode_completion_pct,
        last_action=global_obs_config.last_action,
        last_reward=global_obs_config.last_reward,
        visitation_counts=global_obs_config.visitation_counts,
    )
    game_cpp_params["global_obs"] = global_obs_cpp

    # Process actions using new typed config structure
    actions_config = game_config.actions
    actions_cpp_params = {}

    # Helper function to process common action config fields
    def process_action_config(action_name: str, action_config):
        # If disabled, return empty config (C++ code checks enabled status)
        if not action_config.enabled:
            return {
                "consumed_resources": {},
                "required_resources": {},
            }

        # Only validate resources for enabled actions
        # Check if any consumed resources are not in resource_names
        missing_consumed = []
        for resource in action_config.consumed_resources.keys():
            if resource not in resource_name_to_id:
                missing_consumed.append(resource)

        if missing_consumed:
            raise ValueError(
                f"Action '{action_name}' has consumed_resources {missing_consumed} that are not in "
                f"resource_names. These resources will be ignored, making the action free! "
                f"Either add these resources to resource_names or disable the action."
            )

        consumed_resources = {resource_name_to_id[k]: float(v) for k, v in action_config.consumed_resources.items()}

        required_source = action_config.required_resources
        if not required_source:
            required_source = {k: math.ceil(v) for k, v in action_config.consumed_resources.items()}

        required_resources = {resource_name_to_id[k]: int(math.ceil(v)) for k, v in required_source.items()}

        return {
            "consumed_resources": consumed_resources,
            "required_resources": required_resources,
        }

    # Process noop - always add to map
    action_params = process_action_config("noop", actions_config.noop)
    actions_cpp_params["noop"] = CppActionConfig(**action_params)

    # Process move - always add to map
    action_params = process_action_config("move", actions_config.move)
    action_params["allowed_directions"] = actions_config.move.allowed_directions
    actions_cpp_params["move"] = CppMoveActionConfig(**action_params)

    # Process attack - always add to map
    action_params = process_action_config("attack", actions_config.attack)
    if actions_config.attack.enabled:
        action_params["defense_resources"] = {
            resource_name_to_id[k]: v for k, v in actions_config.attack.defense_resources.items()
        }
    else:
        action_params["defense_resources"] = {}
    action_params["enabled"] = actions_config.attack.enabled
    actions_cpp_params["attack"] = CppAttackActionConfig(**action_params)

    # Process change_vibe - always add to map
    action_params = process_action_config("change_vibe", actions_config.change_vibe)
    action_params["number_of_vibes"] = (
        actions_config.change_vibe.number_of_vibes if actions_config.change_vibe.enabled else 0
    )
    actions_cpp_params["change_vibe"] = CppChangeVibeActionConfig(**action_params)

    # Process resource_mod - always add to map (required by C++)
    action_params = process_action_config("resource_mod", actions_config.resource_mod)
    if actions_config.resource_mod.enabled:
        modifies_dict = actions_config.resource_mod.modifies
        unknown_modifies = set(modifies_dict.keys()) - set(resource_name_to_id.keys())
        if unknown_modifies:
            unknown_list = sorted(unknown_modifies)
            raise ValueError(f"Unknown resource names in modifies for action 'resource_mod': {unknown_list}")
        action_params["modifies"] = {resource_name_to_id[k]: float(v) for k, v in modifies_dict.items()}
        action_params["agent_radius"] = actions_config.resource_mod.agent_radius
        action_params["scales"] = actions_config.resource_mod.scales
    else:
        action_params["modifies"] = {}
        action_params["agent_radius"] = 0
        action_params["scales"] = False
    actions_cpp_params["resource_mod"] = CppResourceModConfig(**action_params)

    game_cpp_params["actions"] = actions_cpp_params
    game_cpp_params["objects"] = objects_cpp_params

    # Add resource_loss_prob
    game_cpp_params["resource_loss_prob"] = game_config.resource_loss_prob

    # Add inventory regeneration interval
    game_cpp_params["inventory_regen_interval"] = game_config.inventory_regen_interval

    # Add clipper if configured
    if game_config.clipper is not None:
        clipper: ClipperConfig = game_config.clipper
        clipper_protocols = []
        for protocol_config in clipper.unclipping_protocols:
            cpp_protocol = CppProtocol()
            cpp_protocol.vibes = sorted([vibe_name_to_id[vibe] for vibe in protocol_config.vibes])
            cpp_protocol.input_resources = {
                resource_name_to_id[k]: v for k, v in protocol_config.input_resources.items()
            }
            cpp_protocol.output_resources = {
                resource_name_to_id[k]: v for k, v in protocol_config.output_resources.items()
            }
            cpp_protocol.cooldown = protocol_config.cooldown
            clipper_protocols.append(cpp_protocol)
        game_cpp_params["clipper"] = CppClipperConfig(
            clipper_protocols, clipper.length_scale, clipper.cutoff_distance, clipper.clip_rate
        )

    # Set feature flags
    game_cpp_params["protocol_details_obs"] = game_config.protocol_details_obs
    game_cpp_params["track_movement_metrics"] = game_config.track_movement_metrics

    # Add tag mappings for C++ debugging/display
    game_cpp_params["tag_id_map"] = tag_id_to_name

    return CppGameConfig(**game_cpp_params)
