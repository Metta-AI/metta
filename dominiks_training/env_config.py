"""
Standalone environment configuration for arena advanced training.
No imports from the complex metta config system - just plain Python dictionaries.
"""


def create_arena_advanced_config():
    """Creates a complete environment configuration for arena advanced training."""

    return {
        "desync_episodes": True,
        "game": {
            "num_agents": 24,
            "obs_width": 11,
            "obs_height": 11,
            "num_observation_tokens": 200,
            "max_steps": 1000,
            "track_movement_metrics": True,
            "no_agent_interference": False,
            # Global observation tokens
            "global_obs": {
                "episode_completion_pct": True,
                "last_action": True,
                "last_reward": True,
                "resource_rewards": False,
                "visitation_counts": False,
            },
            # Inventory items
            "inventory_item_names": [
                "ore_red",
                "ore_blue",
                "ore_green",
                "battery_red",
                "battery_blue",
                "battery_green",
                "heart",
                "armor",
                "laser",
                "blueprint",
            ],
            "resource_loss_prob": 0.0,
            "recipe_details_obs": False,
            # Actions
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": True},
                "place_box": {"enabled": False},
                "get_items": {"enabled": True},
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
                "swap": {"enabled": True},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            # Agent configuration
            "agent": {"rewards": {"inventory": {"heart": 1}}},
            # Groups configuration (solo agents)
            "groups": {"agent": {"id": 0, "sprite": 0, "props": {}}},
            # Map builder
            "map_builder": {
                "_target_": "metta.map.mapgen.MapGen",
                "width": 25,
                "height": 25,
                "instances": 4,  # num_agents / 6 = 24 / 6 = 4
                "border_width": 6,
                "instance_border_width": 0,
                "root": {
                    "type": "metta.map.scenes.random.Random",
                    "params": {
                        "agents": 6,
                        "objects": {
                            # Basic objects
                            "mine_red": 10,
                            "generator_red": 5,
                            "altar": 5,
                            "block": 20,
                            "wall": 20,
                            # Combat objects
                            "lasery": 1,
                            "armory": 1,
                            # Advanced objects
                            "lab": 1,
                            "factory": 1,
                            "temple": 1,
                        },
                    },
                },
            },
            # Object definitions
            "objects": {
                # Basic objects
                "altar": {
                    "type_id": 8,
                    "input_resources": {"battery_red": 3},
                    "output_resources": {"heart": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_resource_count": 0,
                },
                "wall": {
                    "type_id": 1,
                    "swappable": False,
                },
                "block": {
                    "type_id": 14,
                    "swappable": True,
                },
                # Mine and generator objects
                "mine_red": {
                    "type_id": 2,
                    "output_resources": {"ore_red": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
                    "initial_resource_count": 0,
                },
                "generator_red": {
                    "type_id": 5,
                    "output_resources": {"battery_red": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
                    "initial_resource_count": 0,
                },
                # Combat objects
                "armory": {
                    "type_id": 9,
                    "input_resources": {"ore_red": 3},
                    "output_resources": {"armor": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_resource_count": 0,
                },
                "lasery": {
                    "type_id": 10,
                    "input_resources": {"ore_red": 1, "battery_red": 2},
                    "output_resources": {"laser": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_resource_count": 0,
                },
                # Advanced objects
                "lab": {
                    "type_id": 11,
                    "input_resources": {"ore_red": 3, "battery_red": 3},
                    "output_resources": {"blueprint": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
                    "initial_resource_count": 0,
                },
                "factory": {
                    "type_id": 12,
                    "input_resources": {"blueprint": 1, "ore_red": 5, "battery_red": 5},
                    "output_resources": {"armor": 5, "laser": 5},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
                    "initial_resource_count": 0,
                },
                "temple": {
                    "type_id": 13,
                    "input_resources": {"heart": 1, "blueprint": 1},
                    "output_resources": {"heart": 5},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 5,
                    "initial_resource_count": 0,
                },
            },
        },
    }


def create_simple_arena_config():
    """Creates a simpler version with fewer agents for faster training."""
    config = create_arena_advanced_config()

    # Reduce complexity for faster training
    config["game"]["num_agents"] = 4
    config["game"]["max_steps"] = 200
    config["game"]["map_builder"]["width"] = 15
    config["game"]["map_builder"]["height"] = 15
    config["game"]["map_builder"]["instances"] = 1
    config["game"]["map_builder"]["root"]["params"]["agents"] = 4
    config["game"]["map_builder"]["root"]["params"]["objects"] = {
        "mine_red": 3,
        "generator_red": 2,
        "altar": 2,
        "block": 10,
        "wall": 10,
        "armory": 1,
    }

    return config
