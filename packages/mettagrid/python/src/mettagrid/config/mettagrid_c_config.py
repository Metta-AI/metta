from mettagrid.config.mettagrid_config import (
    AgentConfig,
    AssemblerConfig,
    ChestConfig,
    ClipperConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from mettagrid.mettagrid_c import AssemblerConfig as CppAssemblerConfig
from mettagrid.mettagrid_c import AttackActionConfig as CppAttackActionConfig
from mettagrid.mettagrid_c import AttackOutcome as CppAttackOutcome
from mettagrid.mettagrid_c import ChangeVibeActionConfig as CppChangeVibeActionConfig
from mettagrid.mettagrid_c import ChestConfig as CppChestConfig
from mettagrid.mettagrid_c import ClipperConfig as CppClipperConfig
from mettagrid.mettagrid_c import CollectiveConfig as CppCollectiveConfig
from mettagrid.mettagrid_c import DamageConfig as CppDamageConfig
from mettagrid.mettagrid_c import GameConfig as CppGameConfig
from mettagrid.mettagrid_c import GlobalObsConfig as CppGlobalObsConfig
from mettagrid.mettagrid_c import InventoryConfig as CppInventoryConfig
from mettagrid.mettagrid_c import LimitDef as CppLimitDef
from mettagrid.mettagrid_c import MoveActionConfig as CppMoveActionConfig
from mettagrid.mettagrid_c import Protocol as CppProtocol
from mettagrid.mettagrid_c import TransferActionConfig as CppTransferActionConfig
from mettagrid.mettagrid_c import VibeTransferEffect as CppVibeTransferEffect
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
        # Keep vibe_names in sync with vibes; favor the vibes list.
        config_dict.pop("vibe_names", None)
        game_config = GameConfig(**config_dict)

    # Ensure runtime object has consistent vibes.
    game_config.vibe_names = [vibe.name for vibe in game_config.actions.change_vibe.vibes]

    # Set up resource mappings
    resource_names = list(game_config.resource_names)
    resource_name_to_id = {name: i for i, name in enumerate(resource_names)}

    # Compute deterministic type_id mapping for C++ (Python never exposes these)
    type_names_sorted = sorted(game_config.objects.keys())
    type_id_by_type_name = {name: (i + 1) for i, name in enumerate(type_names_sorted)}  # 0 reserved for agents

    # Set up vibe mappings from the change_vibe action config.
    # The C++ bindings expect dense uint8 identifiers, so keep a name->id lookup.
    supported_vibes = game_config.actions.change_vibe.vibes
    vibe_name_to_id = {vibe.name: i for i, vibe in enumerate(supported_vibes)}

    objects_cpp_params = {}  # params for CppWallConfig

    # These are the baseline settings for all agents
    default_agent_config_dict = game_config.agent.model_dump()
    default_resource_limit = default_agent_config_dict["inventory"]["default_limit"]

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

        # Get inventory config
        inv_config = agent_props.get("inventory", {})

        # Process potential initial inventory
        initial_inventory = {resource_name_to_id[k]: v for k, v in inv_config.get("initial", {}).items()}

        # Map team IDs to conventional group names
        team_names = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "purple", 5: "orange"}
        group_name = team_names.get(team_id, f"team_{team_id}")
        # Convert tag names to IDs for first agent in team
        tag_ids = [tag_name_to_id[tag] for tag in first_agent.tags]

        # Convert vibe-keyed inventory regeneration amounts from names to IDs
        # Format: {vibe_name: {resource_name: amount}} -> {vibe_id: {resource_id: amount}}
        inventory_regen_amounts = {}
        for vibe_name, resource_amounts in inv_config.get("regen_amounts", {}).items():
            vibe_id = vibe_name_to_id[vibe_name]
            resource_amounts_cpp = {
                resource_name_to_id[resource_name]: amount for resource_name, amount in resource_amounts.items()
            }
            inventory_regen_amounts[vibe_id] = resource_amounts_cpp

        diversity_tracked_resources = [
            resource_name_to_id[resource_name]
            for resource_name in agent_props.get("diversity_tracked_resources", [])
            if resource_name in resource_name_to_id
        ]

        # Build damage config if present
        damage_config_py = agent_props.get("damage")
        cpp_damage_config = CppDamageConfig()
        if damage_config_py is not None:
            # Convert threshold resource names to IDs
            for resource_name in damage_config_py.get("threshold", {}).keys():
                assert resource_name in resource_name_to_id, (
                    f"Threshold resource '{resource_name}' not in resource_names"
                )
            cpp_damage_config.threshold = {
                resource_name_to_id[resource_name]: threshold_value
                for resource_name, threshold_value in damage_config_py.get("threshold", {}).items()
            }
            # Convert resources map (name -> minimum) to IDs
            for resource_name in damage_config_py.get("resources", {}).keys():
                assert resource_name in resource_name_to_id, f"Damage resource '{resource_name}' not in resource_names"
            cpp_damage_config.resources = {
                resource_name_to_id[resource_name]: minimum_value
                for resource_name, minimum_value in damage_config_py.get("resources", {}).items()
            }

        # Build inventory config with support for grouped limits and modifiers
        limit_defs = []

        # First, handle explicitly configured limits (both individual and grouped)
        configured_resources = set()
        for resource_limit in inv_config.get("limits", {}).values():
            # Convert resource names to IDs
            resource_ids = [resource_name_to_id[name] for name in resource_limit["resources"]]
            # Convert modifier names to IDs
            modifiers_dict = resource_limit.get("modifiers", {})
            modifier_ids = {
                resource_name_to_id[name]: bonus
                for name, bonus in modifiers_dict.items()
                if name in resource_name_to_id
            }
            base_limit = resource_limit["limit"]
            limit_defs.append(CppLimitDef(resource_ids, base_limit, modifier_ids))
            configured_resources.update(resource_limit["resources"])

        # Add default limits for unconfigured resources
        for resource_name in resource_names:
            if resource_name not in configured_resources:
                limit_defs.append(CppLimitDef([resource_name_to_id[resource_name]], default_resource_limit))

        inventory_config = CppInventoryConfig()
        inventory_config.limit_defs = limit_defs

        cpp_agent_config = CppAgentConfig(
            type_id=0,
            type_name="agent",
            group_id=team_id,
            group_name=group_name,
            freeze_duration=agent_props["freeze_duration"],
            initial_vibe=agent_props["initial_vibe"],
            inventory_config=inventory_config,
            stat_rewards=stat_rewards,
            stat_reward_max=stat_reward_max,
            initial_inventory=initial_inventory,
            inventory_regen_amounts=inventory_regen_amounts,
            diversity_tracked_resources=diversity_tracked_resources,
            damage_config=cpp_damage_config,
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
                type_id=type_id_by_type_name[object_type], type_name=object_type, initial_vibe=object_config.vibe
            )
            cpp_wall_config.tag_ids = tag_ids
            # Key by map_name so map grid (which uses map_name) resolves directly.
            objects_cpp_params[object_config.map_name or object_type] = cpp_wall_config
        elif isinstance(object_config, AssemblerConfig):
            protocols = []
            seen_vibes_and_min_agents = []

            for protocol_config in reversed(object_config.protocols):
                # Convert vibe names to IDs (validate all vibe names exist)
                for vibe in protocol_config.vibes:
                    if vibe not in vibe_name_to_id:
                        raise ValueError(f"Unknown vibe name '{vibe}' in assembler '{object_type}' protocol")
                vibe_ids = sorted([vibe_name_to_id[vibe] for vibe in protocol_config.vibes])
                # Check for duplicate vibes
                if (vibe_ids, protocol_config.min_agents) in seen_vibes_and_min_agents:
                    raise ValueError(
                        f"Protocol with vibes {protocol_config.vibes} and min_agents {protocol_config.min_agents} "
                        f"already exists in {object_type}"
                    )
                seen_vibes_and_min_agents.append((vibe_ids, protocol_config.min_agents))
                # Ensure keys and values are explicitly Python ints for C++ binding
                # Build dict item-by-item to ensure pybind11 recognizes it as dict[int, int]
                input_res = {}
                for k, v in protocol_config.input_resources.items():
                    key = int(resource_name_to_id[k])
                    val = int(v)
                    input_res[key] = val
                output_res = {}
                for k, v in protocol_config.output_resources.items():
                    key = int(resource_name_to_id[k])
                    val = int(v)
                    output_res[key] = val
                cpp_protocol = CppProtocol()
                cpp_protocol.min_agents = protocol_config.min_agents
                cpp_protocol.vibes = vibe_ids
                cpp_protocol.input_resources = input_res
                cpp_protocol.output_resources = output_res
                cpp_protocol.cooldown = protocol_config.cooldown
                protocols.append(cpp_protocol)

            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            cpp_assembler_config = CppAssemblerConfig(
                type_id=type_id_by_type_name[object_type], type_name=object_type, initial_vibe=object_config.vibe
            )
            cpp_assembler_config.tag_ids = tag_ids
            cpp_assembler_config.protocols = protocols
            cpp_assembler_config.allow_partial_usage = object_config.allow_partial_usage
            cpp_assembler_config.max_uses = object_config.max_uses
            cpp_assembler_config.clip_immune = object_config.clip_immune
            cpp_assembler_config.start_clipped = object_config.start_clipped
            cpp_assembler_config.chest_search_distance = object_config.chest_search_distance
            # Key by map_name so map grid (which uses map_name) resolves directly.
            objects_cpp_params[object_config.map_name or object_type] = cpp_assembler_config
        elif isinstance(object_config, ChestConfig):
            # Convert tag names to IDs
            tag_ids = [tag_name_to_id[tag] for tag in object_config.tags]

            # Convert vibe_transfers: vibe -> resource -> delta
            vibe_transfers_map = {}
            for vibe_name, resource_deltas in object_config.vibe_transfers.items():
                if vibe_name not in vibe_name_to_id:
                    raise ValueError(f"Unknown vibe name '{vibe_name}' in chest '{object_type}' vibe_transfers")
                vibe_id = vibe_name_to_id[vibe_name]
                resource_deltas_cpp = {
                    resource_name_to_id[resource]: delta for resource, delta in resource_deltas.items()
                }
                vibe_transfers_map[vibe_id] = resource_deltas_cpp

            # Convert initial inventory from nested inventory config
            initial_inventory_cpp = {}
            for resource, amount in object_config.inventory.initial.items():
                resource_id = resource_name_to_id[resource]
                initial_inventory_cpp[resource_id] = amount

            # Create inventory config with limits and modifiers
            limit_defs = []
            for resource_limit in object_config.inventory.limits.values():
                # resources is always a list of strings
                resource_list = resource_limit.resources

                # Convert resource names to IDs
                resource_ids = [resource_name_to_id[name] for name in resource_list if name in resource_name_to_id]
                if resource_ids:
                    # Convert modifier names to IDs
                    modifier_ids = {
                        resource_name_to_id[name]: bonus
                        for name, bonus in resource_limit.modifiers.items()
                        if name in resource_name_to_id
                    }
                    limit_defs.append(CppLimitDef(resource_ids, resource_limit.limit, modifier_ids))

            inventory_config = CppInventoryConfig()
            inventory_config.limit_defs = limit_defs

            cpp_chest_config = CppChestConfig(
                type_id=type_id_by_type_name[object_type], type_name=object_type, initial_vibe=object_config.vibe
            )
            cpp_chest_config.vibe_transfers = vibe_transfers_map
            cpp_chest_config.initial_inventory = initial_inventory_cpp
            cpp_chest_config.inventory_config = inventory_config
            cpp_chest_config.tag_ids = tag_ids
            # Key by map_name so map grid (which uses map_name) resolves directly.
            objects_cpp_params[object_config.map_name or object_type] = cpp_chest_config
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
        game_cpp_params["token_value_base"] = obs_config.get("token_value_base", 256)
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
        compass=global_obs_config.compass,
        goal_obs=global_obs_config.goal_obs,
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

        consumed_resources = {resource_name_to_id[k]: int(v) for k, v in action_config.consumed_resources.items()}

        required_source = action_config.required_resources
        if not required_source:
            required_source = action_config.consumed_resources

        required_resources = {resource_name_to_id[k]: int(v) for k, v in required_source.items()}

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
    attack_cfg = actions_config.attack
    # Always convert full attack config (enabled only controls standalone actions, not vibe-triggered)
    action_params["defense_resources"] = {resource_name_to_id[k]: v for k, v in attack_cfg.defense_resources.items()}
    action_params["armor_resources"] = {resource_name_to_id[k]: v for k, v in attack_cfg.armor_resources.items()}
    action_params["weapon_resources"] = {resource_name_to_id[k]: v for k, v in attack_cfg.weapon_resources.items()}
    # Convert success outcome
    success_actor = {resource_name_to_id[k]: v for k, v in attack_cfg.success.actor_inv_delta.items()}
    success_target = {resource_name_to_id[k]: v for k, v in attack_cfg.success.target_inv_delta.items()}
    success_loot = [resource_name_to_id[name] for name in attack_cfg.success.loot]
    action_params["success"] = CppAttackOutcome(
        success_actor,
        success_target,
        success_loot,
        attack_cfg.success.freeze,
    )
    action_params["enabled"] = attack_cfg.enabled
    # Convert vibes from names to IDs (validate all vibe names exist)
    for vibe in attack_cfg.vibes:
        if vibe not in vibe_name_to_id:
            raise ValueError(f"Unknown vibe name '{vibe}' in attack.vibes")
    action_params["vibes"] = [vibe_name_to_id[vibe] for vibe in attack_cfg.vibes]
    # Convert vibe_bonus from names to IDs
    for vibe in attack_cfg.vibe_bonus:
        if vibe not in vibe_name_to_id:
            raise ValueError(f"Unknown vibe name '{vibe}' in attack.vibe_bonus")
    action_params["vibe_bonus"] = {vibe_name_to_id[vibe]: bonus for vibe, bonus in attack_cfg.vibe_bonus.items()}
    actions_cpp_params["attack"] = CppAttackActionConfig(**action_params)

    # Process transfer - vibes are derived from vibe_transfers keys in C++
    transfer_cfg = actions_config.transfer
    vibe_transfers_cpp = {}
    seen_vibes: set[str] = set()
    for vt in transfer_cfg.vibe_transfers:
        if vt.vibe not in vibe_name_to_id:
            raise ValueError(f"Unknown vibe name '{vt.vibe}' in transfer.vibe_transfers")
        if vt.vibe in seen_vibes:
            raise ValueError(f"Duplicate vibe name '{vt.vibe}' in transfer.vibe_transfers")
        seen_vibes.add(vt.vibe)
        vibe_id = vibe_name_to_id[vt.vibe]
        target_deltas = {resource_name_to_id[k]: v for k, v in vt.target.items()}
        actor_deltas = {resource_name_to_id[k]: v for k, v in vt.actor.items()}
        vibe_transfers_cpp[vibe_id] = CppVibeTransferEffect(target_deltas, actor_deltas)
    actions_cpp_params["transfer"] = CppTransferActionConfig(
        required_resources={resource_name_to_id[k]: int(v) for k, v in transfer_cfg.required_resources.items()},
        vibe_transfers=vibe_transfers_cpp,
        enabled=transfer_cfg.enabled,
    )

    # Process change_vibe - always add to map
    action_params = process_action_config("change_vibe", actions_config.change_vibe)
    num_vibes = len(actions_config.change_vibe.vibes) if actions_config.change_vibe.enabled else 0
    action_params["number_of_vibes"] = num_vibes
    actions_cpp_params["change_vibe"] = CppChangeVibeActionConfig(**action_params)

    game_cpp_params["actions"] = actions_cpp_params
    game_cpp_params["objects"] = objects_cpp_params

    # Add inventory regeneration interval
    game_cpp_params["inventory_regen_interval"] = game_config.inventory_regen_interval

    # Add clipper if configured
    if game_config.clipper is not None:
        clipper: ClipperConfig = game_config.clipper
        clipper_protocols = []
        for protocol_config in clipper.unclipping_protocols:
            cpp_protocol = CppProtocol()
            cpp_protocol.min_agents = protocol_config.min_agents
            # Validate all vibe names exist
            for vibe in protocol_config.vibes:
                if vibe not in vibe_name_to_id:
                    raise ValueError(f"Unknown vibe name '{vibe}' in clipper unclipping_protocols")
            cpp_protocol.vibes = sorted([vibe_name_to_id[vibe] for vibe in protocol_config.vibes])
            # Ensure keys and values are explicitly Python ints for C++ binding
            # Build dict item-by-item to ensure pybind11 recognizes it as dict[int, int]
            input_res = {}
            for k, v in protocol_config.input_resources.items():
                key = int(resource_name_to_id[k])
                val = int(v)
                input_res[key] = val
            cpp_protocol.input_resources = input_res
            output_res = {}
            for k, v in protocol_config.output_resources.items():
                key = int(resource_name_to_id[k])
                val = int(v)
                output_res[key] = val
            cpp_protocol.output_resources = output_res
            cpp_protocol.cooldown = protocol_config.cooldown
            clipper_protocols.append(cpp_protocol)
        clipper_config = CppClipperConfig()
        clipper_config.unclipping_protocols = clipper_protocols
        clipper_config.length_scale = clipper.length_scale
        clipper_config.scaled_cutoff_distance = clipper.scaled_cutoff_distance
        clipper_config.clip_period = clipper.clip_period
        game_cpp_params["clipper"] = clipper_config

    # Add tag mappings for C++ debugging/display
    game_cpp_params["tag_id_map"] = tag_id_to_name

    # Convert collective configurations
    collectives_cpp = {}
    for collective_cfg in game_config.collectives:
        # Build inventory config with limits
        limit_defs = []
        for resource_limit in collective_cfg.inventory.limits.values():
            resource_list = resource_limit.resources
            resource_ids = [resource_name_to_id[name] for name in resource_list if name in resource_name_to_id]
            if resource_ids:
                modifier_ids = {
                    resource_name_to_id[name]: bonus
                    for name, bonus in resource_limit.modifiers.items()
                    if name in resource_name_to_id
                }
                limit_defs.append(CppLimitDef(resource_ids, resource_limit.limit, modifier_ids))

        inventory_config = CppInventoryConfig()
        inventory_config.limit_defs = limit_defs

        # Convert initial inventory
        initial_inventory_cpp = {}
        for resource, amount in collective_cfg.inventory.initial.items():
            if resource in resource_name_to_id:
                resource_id = resource_name_to_id[resource]
                initial_inventory_cpp[resource_id] = amount

        cpp_collective_config = CppCollectiveConfig(collective_cfg.name)
        cpp_collective_config.inventory_config = inventory_config
        cpp_collective_config.initial_inventory = initial_inventory_cpp
        collectives_cpp[collective_cfg.name] = cpp_collective_config

    game_cpp_params["collectives"] = collectives_cpp

    return CppGameConfig(**game_cpp_params)
